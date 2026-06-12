import torch
import os
import json
from galvatron.core.runtime.optimizer.clip_grads import get_grad_norm_fp32, clip_grad_by_total_norm_fp32
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from galvatron.core.runtime.optimizer.param_scheduler import get_optimizer_param_scheduler
# from torch.optim import Adam
try:
    from apex.optimizers import FusedAdam as Adam
except ImportError:
    from torch.optim import AdamW as Adam


def clip_grad_norm(model, max_norm, norm_type=2):
    parameters = []
    grads_for_norm = []
    with torch.no_grad():
        for name, module in model.named_modules():
            # TODO: find a better way to keep the correctness
            if isinstance(module, FSDP) and hasattr(module, "scaling_groups"):
                if module._handle.flat_param.grad is not None:
                    module._handle.flat_param.grad *= 1 / (
                        torch.distributed.get_world_size(module.scaling_groups[0])
                        / torch.distributed.get_world_size(module.scaling_groups[1])
                    )
    
    for name, params in model.named_parameters():
        if params.grad is None:
            continue
        parameters.append(params)
        grads_for_norm.append(params.grad)

    # Profiling / forward-only style runs may legitimately have no gradients.
    if not grads_for_norm:
        return 0.0

    total_norm = get_grad_norm_fp32(grads_for_norm, norm_type)
    if max_norm > 0:
        clip_grad_by_total_norm_fp32(parameters, max_norm, total_norm)

    return total_norm


def get_optimizer_and_param_scheduler(model, args):

    train_args = args.train
    optimizer = Adam(
        model.parameters(),
        lr=train_args.lr,
        weight_decay=getattr(train_args, "adam_weight_decay", train_args.weight_decay),
        betas=(train_args.adam_beta1, train_args.adam_beta2),
        eps=train_args.adam_eps,
    )

    opt_param_scheduler = get_optimizer_param_scheduler(optimizer)

    ckpt_args = args.ckpt
    if ckpt_args.distributed_checkpoint:
        rank = torch.distributed.get_rank()
        if rank == 0:
            print("Begin to load optimizer and param scheduler")
        optimizer.load_state_dict(
            torch.load(os.path.join(ckpt_args.load, f"iter_{ckpt_args.load_iteration}", "optimizer", f"{rank}.pt"))
        )
        opt_param_scheduler.load_state_dict(
            json.load(open(os.path.join(ckpt_args.load, f"iter_{ckpt_args.load_iteration}", "opt_param_scheduler.json")))
        )
        torch.distributed.barrier()
        if rank == 0:
            print("Finish loading optimizer and param scheduler")

    return optimizer, opt_param_scheduler