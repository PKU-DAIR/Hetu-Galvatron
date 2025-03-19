from functools import reduce
from torch.nn import ModuleList
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch import distributed as dist
from .dataloader import compile_helpers
import torch.distributed
import megatron
from megatron.core import mpu
from functools import partial
from torch.distributed.fsdp._common_utils import _named_parameters_with_duplicates
from megatron.training.global_vars import rebuild_tokenizer
from megatron.core.optimizer.clip_grads import clip_grad_norm_fp32
import os
import json
# utility functions, support on nested attributes for getattr, setattr, and setattr
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
# https://stackoverflow.com/questions/24779483/hasattr-for-nested-attributes
def rgetattr(obj, attr):
    if attr == "":
        return obj
    def _getattr_fsdp(obj, attr):
        if isinstance(obj, FSDP):
            return getattr(obj._fsdp_wrapped_module, attr)
        else:
            return getattr(obj, attr)
    return reduce(_getattr_fsdp, [obj] + attr.split('.'))

def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rhasattr(obj, attr):
    try:
        rgetattr(obj, attr)
        return True
    except AttributeError:
        return False

# def get_vtp_tensor_model_parallel_rank(vtp_tensor_group):
#     print("rank",torch.cuda.current_device(),vtp_tensor_group.ranks)
#     return torch.distributed.get_rank(group=vtp_tensor_group.group)

# def get_vtp_tensor_model_parallel_src_rank(vtp_tensor_group):
#     print("src",torch.cuda.current_device(),vtp_tensor_group.ranks)
#     return vtp_tensor_group.ranks[0]

# def get_vtp_tensor_model_parallel_group(vtp_tensor_group):
#     print("group",torch.cuda.current_device(),vtp_tensor_group.ranks)
#     return vtp_tensor_group.group

# def get_vtp_data_parallel_rank(vtp_data_group):
#     return torch.distributed.get_rank(group=vtp_data_group.group)

# def get_vtp_data_parallel_world_size(vtp_data_group):
#     return torch.distributed.get_world_size(group=vtp_data_group.group)

def set_megatron_args_for_dataset(args, hp_model, vtp_tensor_group, vtp_data_group):
    if torch.distributed.get_rank() == 0:
        compile_helpers()
    torch.distributed.barrier()
    
    world_size = torch.distributed.get_world_size()
    assert world_size // args.pp_deg // args.vocab_tp == len(vtp_data_group.ranks)
    args.micro_batch_size = args.global_train_batch_size // len(vtp_data_group.ranks) # // args.chunks
    args.global_batch_size = args.global_train_batch_size
    if args.load_iteration != 0:
        assert args.distributed_checkpoint == True, "Checkpoint iteration > 0 requires distributed checkpoint"
        args.iteration = args.load_iteration
    else:
        args.iteration = 0
    
    args.pipeline_model_parallel_size = hp_model.model.group_size
    mpu.set_pipeline_model_parallel_rank(hp_model.model.group_rank)
    mpu.set_pipeline_model_parallel_world_size(hp_model.model.group_size)
    mpu.set_tensor_model_parallel_group(vtp_tensor_group.group)
    mpu.set_tensor_model_parallel_rank(torch.distributed.get_rank(group=vtp_tensor_group.group))
    mpu.set_data_parallel_group(vtp_data_group.group)
    mpu.set_tensor_model_parallel_src_rank(vtp_tensor_group.ranks[0])
    # mpu.is_pipeline_first_stage = hp_model.model.is_pipeline_first_stage
    # mpu.is_pipeline_last_stage = hp_model.model.is_pipeline_last_stage
    # mpu.get_tensor_model_parallel_rank = partial(get_vtp_tensor_model_parallel_rank, vtp_tensor_group)
    # mpu.get_data_parallel_rank = partial(get_vtp_data_parallel_rank, vtp_data_group)
    # mpu.get_data_parallel_world_size = partial(get_vtp_data_parallel_world_size, vtp_data_group)
    # mpu.get_tensor_model_parallel_src_rank = partial(get_vtp_tensor_model_parallel_src_rank, vtp_tensor_group)
    # mpu.get_tensor_model_parallel_group = partial(get_vtp_tensor_model_parallel_group, vtp_tensor_group)
    rebuild_tokenizer(args)

def get_layernorm_offset(model, layernorm_name=[]):
    total_ln_offset = []
    total_ln_size = []
    for module in model:
        ln_offset = []
        ln_size = []
        offset = 0
        for submodule_name, submodule in module.named_modules(remove_duplicate=False):
            is_ln = False
            for ln_name in layernorm_name:
                if ln_name in submodule_name:
                    is_ln = True
                    break
            for param_name, param in _named_parameters_with_duplicates(
                submodule, recurse=False
            ):
                if is_ln or getattr(param, 'sequence_parallel', False):
                    ln_offset.append(offset)
                    ln_size.append(param.numel())
                offset += param.numel()
        total_ln_offset.append(ln_offset)
        total_ln_size.append(ln_size)
    
    return total_ln_offset, total_ln_size

def clip_grad_norm(model, max_norm, norm_type=2):
    parameters = []
    grads_for_norm = []
    for name, params in model.named_parameters():
        parameters.append(params)
        grads_for_norm.append(params.grad)
    
    total_norm = clip_grad_norm_fp32(parameters, grads_for_norm, max_norm, norm_type)

    return total_norm

def get_optimizer_and_param_scheduler(model, args):
    # from torch.optim import Adam
    from apex.optimizers import FusedAdam as Adam
    from megatron.training.training import get_optimizer_param_scheduler

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.adam_weight_decay, betas=(args.adam_beta1, args.adam_beta2), eps=args.adam_eps)
    
    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    if args.distributed_checkpoint:
        rank = torch.distributed.get_rank()
        if rank == 0:
            print("Begin to load optimizer and param scheduler")
        optimizer.load_state_dict(torch.load(os.path.join(args.load, f"iter_{args.load_iteration}", "optimizer", f"{rank}.pt")))
        opt_param_scheduler.load_state_dict(json.load(open(os.path.join(args.load, f"iter_{args.load_iteration}", "opt_param_scheduler.json"))))
        torch.distributed.barrier()
        if rank == 0:
            print("Finish loading optimizer and param scheduler")

    return optimizer, opt_param_scheduler