import collections
from functools import partial
from typing import List, Set, Tuple

import torch
import torch.distributed
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import CheckpointImpl, checkpoint_wrapper
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import _get_module_fsdp_state
from torch.distributed.fsdp.api import BackwardPrefetch, CPUOffload, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import _recursive_wrap, lambda_auto_wrap_policy
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp._fully_shard._fully_shard import fully_shard, FSDPModule

from .redistribute import fused_split_allgather, split_to_group, gather_from_group
from .comm_groups import CommGroup
from galvatron.core.runtime.checkpoint.types import LoadModuleFunc
from galvatron.core.runtime.parallel_state import fsdp2_enabled
from galvatron.core.runtime.data_parallel.fsdp2_adapter import _mesh_from_group, _mixed_precision_policy, _get_matched_modules, apply_fsdp2_patch
from galvatron.core.runtime.data_parallel.fsdp1_adapter import apply_fsdp1_patch

def _get_modules_to_materialize(root_module: nn.Module) -> List[nn.Module]:
    # Run BFS to collect the modules to materialize via `reset_parameters()`,
    # stopping at any module with FSDP already applied
    module_names_to_materialize: List[nn.Module] = []
    modules_to_materialize: List[nn.Module] = []
    queue = collections.deque([("", root_module)])
    visited_modules: Set[nn.Module] = {root_module}
    while queue:
        name, module = queue.popleft()
        module_names_to_materialize.append(name)
        modules_to_materialize.append(module)
        for child_name, child_module in module.named_children():
            if child_module not in visited_modules and _get_module_fsdp_state(child_module) is None:
                visited_modules.add(child_module)
                if name == "":
                    queue.append((child_name, child_module))
                else:
                    queue.append((name + "." + child_name, child_module))

    return module_names_to_materialize, modules_to_materialize


def wrap_data_parallel(
    module,
    dp_type=None,
    dp_group=None,
    module_type="bert_enc",
    dp_of_ep_groups=None,
    pp_device=None,
    mixed_precision=torch.bfloat16,
    pp_on=False,
    wrap_block_name=None,
    wrap_other_block_name=None,
    tp_groups=None,
    tp_of_ep_groups=None,
    ep_groups=None,
    all_block_name=None,
    load_module_func=None,
    is_moe_model=False,
):
    if dp_type is None:
        return module
    else:
        assert pp_device is not None
        from galvatron.core.runtime.parallel_state import get_args

        fsdp_type_dict = {0: get_args().parallel.default_dp_type, 1: "zero3"}
        assert dp_type in fsdp_type_dict.keys()
        return wrap_module_fsdp_manually(
            module,
            pp_device,
            module_type,
            dp_group,
            dp_of_ep_groups,
            fsdp_type=fsdp_type_dict[dp_type],
            mixed_precision=mixed_precision,
            pp_on=pp_on,
            wrap_block_name=wrap_block_name,
            wrap_other_block_name=wrap_other_block_name,
            tp_groups=tp_groups,
            tp_of_ep_groups=tp_of_ep_groups,
            ep_groups=ep_groups,
            all_block_name=all_block_name,
            load_module_func=load_module_func,
            is_moe_model=is_moe_model,
        )


def param_init_fn(all_block_name, load, distributed_checkpoint, tp_groups, ep_groups, load_module_func, module):
    m = module
    if isinstance(m, tuple(all_block_name)):
        m.to_empty(device=torch.device("cuda"))
        module_names_to_materialize, modules_to_materialize = _get_modules_to_materialize(m)
        for name, submodule in zip(module_names_to_materialize, modules_to_materialize):
            if callable(getattr(submodule, "reset_parameters", None)):
                if load == None:
                    submodule.reset_parameters()
                else:
                    load_module_func(load, tp_groups, name, submodule, m, distributed_checkpoint, ep_groups)


def wrap_module_fsdp_manually(
    module,
    pp_device,
    module_type="bert_enc",
    dp_group=None,
    dp_of_ep_groups=None,
    fsdp_type="zero3",
    mixed_precision=torch.bfloat16,
    pp_on=False,
    wrap_block_name=None,
    wrap_other_block_name=None,
    tp_groups=None,
    tp_of_ep_groups=None,
    ep_groups=None,
    all_block_name=None,
    load_module_func=None,
    is_moe_model=False,
):
    comm_group = None if dp_group is None else dp_group.group
    sharding_strategy = {
        "ddp": ShardingStrategy.NO_SHARD,
        "zero2": ShardingStrategy.SHARD_GRAD_OP,
        "zero3": ShardingStrategy.FULL_SHARD,
    }[fsdp_type]
    from galvatron.core.runtime.parallel_state import get_args

    args = get_args()

    mixed_precision_policy = MixedPrecision(
        param_dtype=mixed_precision,  # Param precision
        reduce_dtype=torch.float if args.parallel.reduce_in_fp32 else mixed_precision,  # Gradient communication precision
        buffer_dtype=mixed_precision,  # Buffer precision
        cast_forward_inputs=False,
        cast_root_forward_inputs=False,
    )
    forward_prefetch = True # Always explicitly prefetch
    # backward_prefetch = None if pp_on else BackwardPrefetch.BACKWARD_PRE
    fsdp_args = dict(
        process_group=comm_group,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        forward_prefetch=forward_prefetch,
        # backward_prefetch=backward_prefetch,
        device_id=pp_device,
        param_init_fn=(
            partial(
                param_init_fn, all_block_name, args.ckpt.load, args.ckpt.distributed_checkpoint, tp_groups.group, None, load_module_func
            )
            if args.model.initialize_on_meta
            else None
        ),
        limit_all_gathers=True,
    )

    # Wrap given block
    if wrap_block_name is not None:
        if "enc" in module_type or "dec" in module_type:
            if is_moe_model:
                moe_fsdp_args = dict(
                    process_group=dp_of_ep_groups.group,
                    sharding_strategy=sharding_strategy,
                    mixed_precision=mixed_precision_policy,
                    forward_prefetch=forward_prefetch,
                    device_id=pp_device,
                    param_init_fn=(
                        partial(
                            param_init_fn, all_block_name, args.ckpt.load, args.ckpt.distributed_checkpoint, tp_of_ep_groups.group, ep_groups.group, load_module_func
                        )
                        if args.model.initialize_on_meta
                        else None
                    ),
                    limit_all_gathers=True,
                )
                # Wrap MoE layer first
                module = apply_fsdp(module, moe_fsdp_args, [wrap_block_name[1]], True)
                for name, mod in module.named_modules():
                    if isinstance(mod, FSDP):
                        # Add gradient scaling for expert parameters.
                        # Will be scaled before grad norm.
                        # (Reference: megatron/core/distributed/distributed_data_parallel.py)
                        # TODO: check the correctnees with fine-grained parallelism
                        setattr(mod, "scaling_groups", (comm_group, dp_of_ep_groups.group))
                module = apply_fsdp(module, fsdp_args, [wrap_block_name[0]], True)
            else:
                module = apply_fsdp(module, fsdp_args, wrap_block_name)
        else:
            module = apply_fsdp(module, fsdp_args, wrap_other_block_name)
        return module
    
    assert False


def apply_fsdp(model, fsdp_args, wrap_block_name, need_ignore=False):
    if need_ignore:
        ignored_modules = set()
        for name, module in model.named_modules():
            if isinstance(module, FSDP):
                ignored_modules.add(module)
    else:
        ignored_modules = set()
    check_fn = lambda submodule: (any(isinstance(submodule, block) for block in wrap_block_name))
    _recursive_wrap(
        module=model,
        auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=check_fn),
        wrapper_cls=FSDP,
        ignored_modules=ignored_modules,
        ignored_params=set(),
        only_wrap_children=True,
        **fsdp_args
    )
    return model


def apply_ckpt(model, checkpoint_wrapper_fn, wrap_block_name):
    check_fn = lambda submodule: (any(isinstance(submodule, block) for block in wrap_block_name))
    _recursive_wrap(
        module=model,
        auto_wrap_policy=partial(lambda_auto_wrap_policy, lambda_fn=check_fn),
        wrapper_cls=checkpoint_wrapper_fn,
        ignored_modules=set(),
        ignored_params=set(),
        only_wrap_children=True,
    )
    return model


def wrap_modules_checkpoint(module_list, checkpoint_flags, wrap_block_name=None):
    m = module_list
    if isinstance(m, FSDP):
        m = m._fsdp_wrapped_module
    assert len(m) == len(checkpoint_flags)
    for i in range(len(m)):
        if checkpoint_flags[i]:
            if wrap_block_name is not None:
                m[i] = apply_ckpt(m[i], checkpoint_wrapper, wrap_block_name)
            else:
                m[i] = checkpoint_wrapper(m[i])
    return module_list


def wrap_model_checkpoint(model, wrap_block_names=[]):
    model_ = model._fsdp_wrapped_module if isinstance(model, FSDP) else model
    apply_ckpt(model_, checkpoint_wrapper, wrap_block_names)
    return model


def relocate_activations(input, allgather_cp_group, allgather_tp_sp_cp_group, 
    split_cp_group, split_tp_sp_cp_group,
    fused_allgather_group, fused_split_group, is_input):
    #if fused_allgather_group is not None or fused_split_group is not None:
    input = fused_split_allgather(
        input,
        is_input,
        getattr(allgather_cp_group, "group", None),
        getattr(allgather_tp_sp_cp_group, "group", None),
        getattr(split_cp_group, "group", None),
        getattr(split_tp_sp_cp_group, "group", None),
        getattr(fused_allgather_group, "group", None),
        getattr(fused_split_group, "group", None),
    )
    # else:
    #     input = split_to_group(input, 
    #         getattr(split_cp_group, "group", None), 
    #         getattr(split_tp_sp_cp_group, "group", None), 
    #         is_input)
    #     input = gather_from_group(input, 
    #         getattr(allgather_cp_group, "group", None), 
    #         getattr(allgather_tp_sp_cp_group, "group", None), is_input)

    return input


class Module_with_relocation(nn.Module):
    def __init__(self, module, allgather_cp_group, allgather_tp_sp_cp_group, 
        split_cp_group, split_tp_sp_cp_group,
        fused_allgather_group, fused_split_group):
        super().__init__()
        self.module = module
        self.allgather_cp_group = allgather_cp_group
        self.allgather_tp_sp_cp_group = allgather_tp_sp_cp_group
        self.split_cp_group = split_cp_group
        self.split_tp_sp_cp_group = split_tp_sp_cp_group
        self.fused_allgather_group = fused_allgather_group
        self.fused_split_group = fused_split_group
        self.relocate_activations = lambda x, y: relocate_activations(
            x, self.allgather_cp_group, self.allgather_tp_sp_cp_group, 
            self.split_cp_group, self.split_tp_sp_cp_group,
            self.fused_allgather_group, self.fused_split_group, y
        )
        if hasattr(module, "get_extended_attention_mask"):
            self.get_extended_attention_mask = module.get_extended_attention_mask

    def forward(self, *inputs, **kwargs):
        if isinstance(inputs, (Tuple, List)):
            inputs_relocated = []
            for input in inputs:
                if input.dtype in [torch.float16, torch.float32, torch.float64, torch.bfloat16]:
                    inputs_relocated.append(self.relocate_activations(input, True))
                else:
                    inputs_relocated.append(self.relocate_activations(input, False))
            inputs_relocated = tuple(inputs_relocated)
            return self.module(*inputs_relocated, **kwargs)
        else:
            input_relocated = self.relocate_activations(inputs)
            return self.module(input_relocated, **kwargs)


def wrap_modules_data_parallel(
    module_list,
    dp_types,
    dp_groups,
    module_types,
    dp_of_ep_groups=None,
    pp_devices=None,
    mixed_precision=torch.bfloat16,
    default_process_group=None,
    wrap_block_name=None,
    wrap_other_block_name=None,
    tp_groups=None,
    tp_of_ep_groups=None,
    ep_groups=None,
    all_block_name=None,
    load_module_func=None,
):
    assert len(module_list) == len(dp_types)
    assert len(module_list) == len(dp_groups)

    process_group = default_process_group.group if default_process_group is not None else dp_groups[0].group
    from galvatron.core.runtime.parallel_state import get_args

    args = get_args()
    if fsdp2_enabled():
        return wrap_data_parallel_fsdp2(
            module_list=module_list,
            dp_types=dp_types,
            dp_groups=dp_groups,
            module_types=module_types,
            dp_of_ep_groups=dp_of_ep_groups,
            pp_devices=pp_devices,
            mixed_precision=mixed_precision,
            default_process_group=default_process_group,
            wrap_block_name=wrap_block_name,
            wrap_other_block_name=wrap_other_block_name,
            tp_groups=tp_groups,
            tp_of_ep_groups=tp_of_ep_groups,
            ep_groups=ep_groups,
            all_block_name=all_block_name,
            load_module_func=load_module_func,
        )

    apply_fsdp1_patch()

    pp_on = True if args.parallel.pp_deg > 1 else False
    # pp_on = True if process_group.size < torch.distributed.get_world_size() else False

    if pp_devices is not None:
        assert len(pp_devices) == len(module_list)
    for i in range(len(module_list)):
        pp_device = None if pp_devices is None else pp_devices[i]
        module_list[i] = wrap_data_parallel(
            module_list[i],
            dp_types[i],
            dp_groups[i],
            module_type=module_types[i],
            dp_of_ep_groups=dp_of_ep_groups[i] if dp_of_ep_groups is not None else None,
            pp_device=pp_device,
            mixed_precision=mixed_precision,
            pp_on=pp_on,
            wrap_block_name=wrap_block_name,
            wrap_other_block_name=wrap_other_block_name,
            tp_groups=tp_groups[i],
            tp_of_ep_groups=tp_of_ep_groups[i] if tp_of_ep_groups is not None else None,
            ep_groups=ep_groups[i] if ep_groups is not None else None,
            all_block_name=all_block_name,
            load_module_func=load_module_func,
            is_moe_model=args.model.is_moe_model,
        )
    args = get_args()
    sharding_strategy = {
        "ddp": ShardingStrategy.NO_SHARD,
        "zero2": ShardingStrategy.SHARD_GRAD_OP,
        "zero3": ShardingStrategy.FULL_SHARD,
    }[args.parallel.default_dp_type]
    mixed_precision_policy = MixedPrecision(
        param_dtype=mixed_precision,  # Param precision
        reduce_dtype=torch.float if args.parallel.reduce_in_fp32 else mixed_precision,  # Gradient communication precision
        buffer_dtype=mixed_precision,  # Buffer precision
        cast_forward_inputs=False,
        cast_root_forward_inputs=False, # For rotary embedding
    )
    forward_prefetch = True# Always explicitly prefetch
    # backward_prefetch = None if pp_on else BackwardPrefetch.BACKWARD_PRE
    # Wrap router paramter into root FSDP with WORLD process group so that the grad of router can be reduce-scatter correctly
    fsdp_args = dict(
        process_group=process_group,
        sharding_strategy=sharding_strategy,
        mixed_precision=mixed_precision_policy,
        forward_prefetch=forward_prefetch,
        # backward_prefetch=backward_prefetch,
        device_id=pp_devices[0],
        param_init_fn=(
            partial(param_init_fn, all_block_name, args.ckpt.load, args.ckpt.distributed_checkpoint, None, None, load_module_func)
            if args.model.initialize_on_meta
            else None
        ),
        limit_all_gathers=True,
    )
    module_list = FSDP(module_list, **fsdp_args)
    return module_list


def modules_to_devices(module_list, pp_devices):
    assert len(module_list) == len(pp_devices)
    for i in range(len(module_list)):
        module_list[i].to("cuda:%d" % pp_devices[i])


def wrap_modules_relocation(module_list, allgather_cp_groups, allgather_tp_sp_cp_groups, 
    split_cp_groups, split_tp_sp_cp_groups, fused_allgather_groups, fused_split_groups):
    assert len(module_list) == len(allgather_cp_groups)
    assert len(module_list) == len(allgather_tp_sp_cp_groups)
    assert len(module_list) == len(split_cp_groups)
    assert len(module_list) == len(split_tp_sp_cp_groups)
    assert len(module_list) == len(fused_allgather_groups)
    assert len(module_list) == len(fused_split_groups)
    for i in range(len(module_list)):
        module_list[i] = Module_with_relocation(
            module_list[i], allgather_cp_groups[i], allgather_tp_sp_cp_groups[i], 
            split_cp_groups[i], split_tp_sp_cp_groups[i], 
            fused_allgather_groups[i], fused_split_groups[i]
        )
    return module_list


def wrap_data_parallel_fsdp2(
    module_list,
    dp_types: List[int],
    dp_groups: List[CommGroup],
    module_types: List[str],
    dp_of_ep_groups:List[CommGroup]=None,
    pp_devices=None,
    mixed_precision=torch.bfloat16,
    default_process_group=None,
    wrap_block_name=None,
    wrap_other_block_name=None,
    tp_groups=None,
    tp_of_ep_groups=None,
    ep_groups=None,
    all_block_name=None,
    load_module_func:LoadModuleFunc=None,
):
    assert len(module_list) == len(dp_types)
    assert len(module_list) == len(dp_groups)
    assert len(module_list) == len(module_types)

    from galvatron.core.runtime.parallel_state import get_args
    args = get_args()

    # [Step 0] apply patch for fsdp2
    apply_fsdp2_patch()

    # [Step 1] fully_shard every module in module_list
    for idx, module in enumerate(module_list):
        # 1.1 determine FSDP type and corresponding args
        fsdp_type = args.parallel.default_dp_type if dp_types[idx] == 0 else 'zero3'
        assert fsdp_type in ("ddp", "zero2", "zero3"), f"unsupported dp_type {fsdp_type} for fsdp2"

        reshard_after_forward = True if fsdp_type == 'zero3' else False
        reshard_after_backward = True if fsdp_type == 'zero3' else False

        fsdp_args = {
            'mesh': _mesh_from_group(dp_groups[idx].group, fsdp_type),
            'mp_policy': _mixed_precision_policy(mixed_precision, args.parallel.reduce_in_fp32),
            'reshard_after_forward': reshard_after_forward,
        }
        moe_fsdp_args = None if not args.model.is_moe_model else {
            'mesh': _mesh_from_group(dp_of_ep_groups[idx].group, fsdp_type),
            'mp_policy': _mixed_precision_policy(mixed_precision, args.parallel.reduce_in_fp32),
            'reshard_after_forward': reshard_after_forward,
        }

        # 1.2 find submodules to wrap based on module_type and wrap_block_name
        target_block_classes = wrap_block_name if ("enc" in module_types[idx] or "dec" in module_types[idx]) else wrap_other_block_name
        modules_to_wrap = _get_matched_modules(module, target_block_classes)

        # 1.3 apply fully_shard, for MoE model, wrap MoE layer with dp_of_ep_groups and then wrap the whole block with dp_groups
        if args.model.is_moe_model and ("enc" in module_types[idx] or "dec" in module_types[idx]):
            assert len(modules_to_wrap) == 2, f"expected exactly 2 submodules to wrap for MoE model, but got {len(modules_to_wrap)} in module_list[{idx}]"
            fully_shard(modules_to_wrap[1], **moe_fsdp_args)
            fully_shard(modules_to_wrap[0], **fsdp_args)
            setattr(modules_to_wrap[1], "scaling_groups", (dp_groups[idx].group, dp_of_ep_groups[idx].group))
        else:
            assert len(modules_to_wrap) == 1, f"expected exactly 1 submodule to wrap, but got {len(modules_to_wrap)} in module_list[{idx}]"
            fully_shard(modules_to_wrap[0], **fsdp_args)

        # 1.4 set reshard_after_backward
        for module in modules_to_wrap:
            module.set_reshard_after_backward(reshard_after_backward, recurse=False)

    # [Step 2] fully_shard the root module_list
    root_fsdp_args = {
        'mesh': _mesh_from_group(dp_groups[0].group, args.parallel.default_dp_type), # use the vocab group as the root group
        'mp_policy': _mixed_precision_policy(mixed_precision, args.parallel.reduce_in_fp32),
        'reshard_after_forward': False, # For the root module, it's recommended to set False according to the fully_shard docstring.
    }
    module_list = fully_shard(module_list, **root_fsdp_args)
    module_list: FSDPModule
    module_list.set_reshard_after_backward(False, recurse=False) # Keep root params resident after backward, same reason as reshard_after_forward=False above.

    # [Step 3] materialize parameters and move to target device
    module_list = module_list.to_empty(device='cuda')
    if args.ckpt.load is None:
        module: nn.Module
        for module in module_list.modules():
            if callable(getattr(module, "reset_parameters", None)):
                module.reset_parameters()
    else:
        for idx in range(len(module_list)):
            target_block_classes = wrap_block_name if ("enc" in module_types[idx] or "dec" in module_types[idx]) else wrap_other_block_name
            modules_to_materialize = _get_matched_modules(module_list[idx], target_block_classes)
            for idx, module in enumerate(modules_to_materialize):
                if idx == 1 and args.model.is_moe_model and ("enc" in module_types[idx] or "dec" in module_types[idx]):
                    tp_group = tp_of_ep_groups[idx].group
                else:
                    tp_group = tp_groups[idx].group
                for name, param in module.named_parameters():
                    module_path, _ , param_name = name.rpartition(".")
                    submodule = module.get_submodule(module_path)
                    load_module_func(
                        load=args.ckpt.load,
                        tp_groups=tp_group,
                        name=module_path,
                        submodule=submodule,
                        module=module,
                        distributed_checkpoint=args.ckpt.distributed_checkpoint,
                        ep_groups=ep_groups[idx].group if ep_groups is not None else None,
                    )

    # [Step 4] explicit forward prefetch and backward prefetch
    all_fsdp2_module_list:List[FSDPModule] = []
    for idx in range(len(module_list)):
        for module in module_list[idx].modules():
            if isinstance(module, FSDPModule):
                all_fsdp2_module_list.append(module)

    num_to_forward_prefetch = 1
    for idx, module in enumerate(all_fsdp2_module_list):
        module:FSDPModule
        if idx >= len(all_fsdp2_module_list) - num_to_forward_prefetch:
            break

        modules_to_prefetch = [
            all_fsdp2_module_list[idx + j] for j in range(1, num_to_forward_prefetch + 1)
        ]
        module.set_modules_to_forward_prefetch(modules_to_prefetch)

    num_to_backward_prefetch = 1
    for idx, module in enumerate(all_fsdp2_module_list):
        module:FSDPModule
        if idx < num_to_backward_prefetch:
            state = module._get_fsdp_state()
            # These modules run last in backward, so nothing is left to prefetch.
            # `[None]` is non-empty (disables FSDP2's default PP-unsafe prefetch)
            # but skipped by the patched `_pre_backward`, so no prefetch is issued.
            state._states_to_backward_prefetch = [None]
        else:
            modules_to_prefetch = [
                all_fsdp2_module_list[idx - j] for j in range(1, num_to_backward_prefetch + 1)
            ]
            module.set_modules_to_backward_prefetch(modules_to_prefetch)


    return module_list
