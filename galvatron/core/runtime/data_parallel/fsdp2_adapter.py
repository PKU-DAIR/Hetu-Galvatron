import logging
from typing import Any, Dict, List

import torch
import torch.nn as nn
from torch.profiler import record_function
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
from torch.distributed.fsdp._fully_shard._fsdp_api import MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState, compiled_autograd_enabled, DDPMeshInfo, FSDPMeshInfo
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup, ReduceScatterState, AllReduceState
from torch.distributed.fsdp._fully_shard._fsdp_param import FSDPParam
from torch.distributed.fsdp._fully_shard._fsdp_collectives import foreach_reduce


# ============================== fsdp2 wrapper utils ==============================
_MESH_CACHE: Dict[object, DeviceMesh] = {}


def _mesh_from_group(
    group: torch.distributed.ProcessGroup,
    fsdp_type: str = "zero3",
    device_type: str = "cuda",
) -> DeviceMesh:
    """Build the DeviceMesh for a fully_shard call.

    - 'ddp': a 2D HSDP mesh ``(replicate=group.size(), shard=1)``. With the shard
      dim set to 1 each rank keeps the full parameter (no sharding) and gradients
      are only all-reduced over the replicate dim -- i.e. it degenerates to DDP.
    - 'zero2' / 'zero3': a 1D mesh that shards parameters over the whole group.
    """
    is_ddp = fsdp_type == "ddp"
    key = (group, is_ddp)
    mesh = _MESH_CACHE.get(key)
    if mesh is None:
        if is_ddp:
            # Build the HSDP mesh (replicate=group, shard=1) from existing groups so no
            # world-collective new_group is issued: the replicate dim reuses `group`, and
            # the shard dim is a per-rank singleton created with local synchronization.
            rank = torch.distributed.get_rank()
            dp_ranks = torch.distributed.get_process_group_ranks(group)
            shard_cache = _mesh_from_group.__dict__.setdefault("_shard_group_cache", {})
            shard_group = shard_cache.get(rank)
            if shard_group is None:
                shard_group = torch.distributed.new_group([rank], use_local_synchronization=True)
                shard_cache[rank] = shard_group
            mesh = DeviceMesh.from_group(
                [group, shard_group],
                device_type,
                mesh=torch.tensor(dp_ranks, dtype=torch.int).reshape(len(dp_ranks), 1),
                mesh_dim_names=("replicate", "shard"),
            )
        else:
            mesh = DeviceMesh.from_group(group, device_type)
        _MESH_CACHE[key] = mesh
    return mesh


def _mixed_precision_policy(mixed_precision: torch.dtype, reduce_in_fp32: bool) -> MixedPrecisionPolicy:
    reduce_dtype = torch.float if reduce_in_fp32 else mixed_precision
    return MixedPrecisionPolicy(
        param_dtype=mixed_precision,
        reduce_dtype=reduce_dtype,
        output_dtype=mixed_precision,
        cast_forward_inputs=False,
    )


def _get_matched_modules(module, target_block_classes) -> List[nn.Module]:
    modules_to_wrap = []
    for name, submodule in module.named_modules():
        if any(isinstance(submodule, block) for block in target_block_classes):
            modules_to_wrap.append(submodule)
    return modules_to_wrap


# ============================== fsdp2 patches ==============================
def apply_fsdp2_patch():
    patches = []

    FSDPState._pre_backward = _pre_backward_patch
    patches.append("FSDPState._pre_backward -> _pre_backward_patch")

    FSDPParamGroup.post_backward = post_backward_patch
    patches.append("FSDPParamGroup.post_backward -> post_backward_patch")

    from galvatron.core.runtime.data_parallel.fsdp1_adapter import _print_patch_info
    _print_patch_info("FSDP v2", patches)
    

def _pre_backward_patch(self, grad: torch.Tensor) -> torch.Tensor:
    self._training_state = TrainingState.PRE_BACKWARD
    self._register_root_post_backward_final_callback()
    if self._fsdp_param_group:
        default_prefetch = len(self._states_to_backward_prefetch) == 0
        self._fsdp_param_group.pre_backward(default_prefetch)
    for fsdp_state in self._states_to_backward_prefetch:
        if fsdp_state is None: # modified by galvatron
            continue
        if (target_param_group := fsdp_state._fsdp_param_group) is not None:
            FSDPParamGroup._prefetch_unshard(target_param_group, "backward")
    return grad


def post_backward_patch(self:FSDPParamGroup, *unused: Any):
    logger = logging.getLogger("torch.distributed.fsdp.fully_shard") # modified by galvatron

    # This method should be idempotent and safe to call even when this
    # FSDP parameter group was not used in backward (should be a no-op)
    if not compiled_autograd_enabled():
        logger.debug("%s", self._with_fqn("FSDP::post_backward"))
    self._training_state = TrainingState.POST_BACKWARD
    with record_function(self._with_fqn("FSDP::post_backward_accumulate")):
        for fsdp_param in self.fsdp_params:
            fsdp_param.accumulate_unsharded_grad_if_needed()
    with record_function(self._with_fqn("FSDP::post_backward_reshard")):
        if not self.reduce_grads:
            if self.reshard_after_backward or ( # modified by galvatron
                self.reshard_after_backward == False and hasattr(self, 'last_batch') and getattr(self, 'last_batch') == True
            ):
                self.reshard()
            for fsdp_param in self.fsdp_params:
                fsdp_param.to_accumulated_grad_if_needed()
            return
        # Save the autograd-computed gradients before resharding to only
        # access the unsharded parameters when their data is present
        fsdp_params_with_grad: list[FSDPParam] = []
        unsharded_grads: list[torch.Tensor] = []
        for fsdp_param in self.fsdp_params:
            if not hasattr(fsdp_param, "_unsharded_param"):
                continue
            # May have an accumulated gradient of the reduce dtype if the
            # previous backward did not reduce-scatter
            if fsdp_param.unsharded_accumulated_grad is not None:
                fsdp_params_with_grad.append(fsdp_param)
                unsharded_grads.append(fsdp_param.unsharded_accumulated_grad_data)
                fsdp_param.unsharded_accumulated_grad = None
            elif fsdp_param.unsharded_param.grad is not None:
                fsdp_params_with_grad.append(fsdp_param)
                unsharded_grads.append(fsdp_param.unsharded_grad_data)
                fsdp_param.unsharded_param.grad = None
        if self.reshard_after_backward or ( # modified by galvatron
                self.reshard_after_backward == False and hasattr(self, 'last_batch') and getattr(self, 'last_batch') == True
            ):
            self.reshard()
    if len(fsdp_params_with_grad) == 0:
        return

    # megatron-sp: all-reduce norm grads over the sp group on the unsharded
    # gradients, before reduce-scatter (modified by galvatron). norm params are
    # detected via FSDPParam._param_fqn (populated during lazy-init, so it is
    # available by the time post-backward runs).
    sp_group = getattr(self, "megatron_sp_group", None)
    if sp_group is not None and len(sp_group.ranks) > 1: # megatron-sp world_size > 1, reduce norm grads
        for fsdp_param, grad in zip(fsdp_params_with_grad, unsharded_grads):
            fqn = (getattr(fsdp_param, "_param_fqn", None) or "").lower()
            if 'norm' in fqn and 'q_layernorm' not in fqn and 'k_layernorm' not in fqn: # norm param, reduce its grad
                torch.distributed.all_reduce(grad, group=sp_group.group)

    with record_function(self._with_fqn("FSDP::post_backward_reduce")):
        if (
            self.comm_ctx.reduce_scatter_state is not None
            and self.comm_ctx.reduce_scatter_state.event is not None
        ):
            self.device_handle.current_stream().wait_event(
                self.comm_ctx.reduce_scatter_state.event
            )
        self.comm_ctx.reduce_scatter_state = None
        all_reduce_pg = (
            self._all_reduce_process_group
            if isinstance(self.mesh_info, DDPMeshInfo)
            else None
        )
        all_reduce_stream: torch.cuda.Stream
        if all_reduce_pg is None and self._all_reduce_hook_stream is not None:
            # this means the native HSDP is not enabled,
            # but user may want to have a custom HSDP setup
            if self._all_reduce_hook is None:
                raise AssertionError(
                    "all reduce hook stream is specified but hook itself is missing."
                )
            all_reduce_stream = self._all_reduce_hook_stream
        else:
            all_reduce_stream = self.comm_ctx.all_reduce_stream

        self._wait_for_post_backward()
        (
            reduce_scatter_input,
            reduce_scatter_event,
            self._post_reduce_event,
            all_reduce_input,
            all_reduce_event,
            self._partial_reduce_output,
        ) = foreach_reduce(
            fsdp_params_with_grad,
            unsharded_grads,
            (
                self._reduce_scatter_process_group
                if isinstance(self.mesh_info, FSDPMeshInfo)
                else None  # pyre-fixme[6]
            ),
            self.comm_ctx.reduce_scatter_stream,
            self._reduce_scatter_comm,
            self._orig_dtype,
            self._reduce_dtype,
            self.device,
            self.gradient_divide_factor,
            (
                self._all_reduce_process_group
                if isinstance(self.mesh_info, DDPMeshInfo)
                else None
            ),
            all_reduce_stream,
            self.all_reduce_grads,
            self._partial_reduce_output,
            self._all_reduce_hook,
            self.force_sum_reduction_for_comms,
        )
        self.comm_ctx.reduce_scatter_state = ReduceScatterState(
            reduce_scatter_input, reduce_scatter_event
        )
        if all_reduce_input is not None:
            if self.device.type != "cpu":
                if all_reduce_event is None:
                    raise AssertionError(
                        "Expected all_reduce_event to be set for non-CPU device"
                    )
            self._all_reduce_state = AllReduceState(
                all_reduce_input, all_reduce_event
                )



