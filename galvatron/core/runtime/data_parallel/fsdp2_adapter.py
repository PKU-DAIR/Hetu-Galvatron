"""FSDP2 (fully_shard) adapter for Galvatron.

Requires torch >= 2.6 (guarded by ``assert`` in ``apply_fsdp2_patch``).

FSDP2 internals are patched at import time via ``apply_fsdp2_patch()``.
Changes vs upstream are marked with ``# <-- Galvatron:`` inline.

Patches
-------
- **_pre_backward**: skip ``None`` entries in ``_states_to_backward_prefetch``
  when explicit backward prefetch is disabled for certain layers.

Standalone utils
----------------
- ``set_reshard_after_backward_per_microbatch``: under zero2 semantics, keep
  params unsharded across grad-accumulation microbatches and reshard only on
  the last microbatch.
- ``fsdp2_reduce_megatron_sp_norm_grads``: flatten-bucket all-reduce of
  LayerNorm / RMSNorm grads across the Megatron SP (TP) group, done once
  after backward and before the optimizer step.
"""

from typing import Dict, List

import torch
import torch.nn as nn
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp import FSDPModule
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState
from torch.distributed.fsdp._fully_shard._fsdp_api import MixedPrecisionPolicy
from torch.distributed.fsdp._fully_shard._fsdp_common import TrainingState
from torch.distributed.fsdp._fully_shard._fsdp_param_group import FSDPParamGroup

from galvatron.core.runtime.comm_groups import CommGroup


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


def set_reshard_after_backward_per_microbatch(module:FSDPModule, last_batch:bool):
    # Initial reshard_after_backward: True for zero3, False for zero2/ddp.
    # Keep zero3 always True; reshard zero2/ddp only after the last micro_batch.
    state = module._get_fsdp_state()
    if fsdp_param_group := state._fsdp_param_group:
        # Cache the original value so our own writes don't clobber it.
        if not hasattr(fsdp_param_group, '_galvatron_orig_reshard_after_backward'):
            fsdp_param_group._galvatron_orig_reshard_after_backward = fsdp_param_group.reshard_after_backward
        orig = fsdp_param_group._galvatron_orig_reshard_after_backward
        fsdp_param_group.reshard_after_backward = orig or last_batch


# ============================== fsdp2 patches ==============================
def apply_fsdp2_patch():
    assert tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2]) >= (2, 6), \
    f"FSDP2 requires torch >= 2.6, found {torch.__version__}"

    patches = []

    FSDPState._pre_backward = _pre_backward_patch
    patches.append("FSDPState._pre_backward -> _pre_backward_patch")

    from galvatron.core.runtime.data_parallel.fsdp1_adapter import _print_patch_info
    _print_patch_info("FSDP v2", patches)
    

def _pre_backward_patch(self, grad: torch.Tensor) -> torch.Tensor:
    """Upstream ``FSDPState._pre_backward`` with one change:

    **Galvatron**: ``_states_to_backward_prefetch`` may contain ``None``
    entries when explicit backward prefetch is disabled for certain layers.
    Skip them instead of crashing with ``AttributeError``.
    """
    self._training_state = TrainingState.PRE_BACKWARD
    self._register_root_post_backward_final_callback()
    if self._fsdp_param_group:
        default_prefetch = len(self._states_to_backward_prefetch) == 0
        self._fsdp_param_group.pre_backward(default_prefetch)
    for fsdp_state in self._states_to_backward_prefetch:
        if fsdp_state is None:          # <-- Galvatron: explicit prefetch None guard
            continue
        if (target_param_group := fsdp_state._fsdp_param_group) is not None:
            FSDPParamGroup._prefetch_unshard(target_param_group, "backward")
    return grad


# ============================== fsdp2 grad reduce utils ==============================

def _is_sp_norm_param(fqn: str) -> bool:
    fqn = (fqn or "").lower()
    return "norm" in fqn and "q_layernorm" not in fqn and "k_layernorm" not in fqn


def fsdp2_reduce_megatron_sp_norm_grads(module_list):
    """All-reduce norm grads across their SP groups via flatten+reduce+scatter.

    Buckets norm gradients by SP group, flattens each bucket into one
    contiguous tensor to do a single ``all_reduce`` per group, then scatters
    the result back into the original local shards.
    """
    buckets: Dict[int, tuple] = {}

    for idx in range(len(module_list)):
        for module in module_list[idx].modules():
            if not isinstance(module, FSDPModule):
                continue
            state = module._get_fsdp_state()
            param_group = getattr(state, "_fsdp_param_group", None)
            if param_group is None or not hasattr(param_group, "megatron_sp_group"):
                continue
            sp_group: CommGroup = param_group.megatron_sp_group
            if sp_group is None or sp_group.size <= 1:
                continue
            key = id(sp_group.group)

            for fsdp_param in param_group.fsdp_params:
                fqn = getattr(fsdp_param, "_param_fqn", None)
                if not _is_sp_norm_param(fqn):
                    continue
                grad = fsdp_param.sharded_param.grad
                if grad is None:
                    continue
                buckets.setdefault(key, (sp_group.group, []))[1].append(grad.to_local())

    for group, grads in buckets.values():
        if not grads:
            continue
        flat = torch.cat([g.reshape(-1) for g in grads])
        torch.distributed.all_reduce(flat, group=group)
        offset = 0
        for g in grads:
            n = g.numel()
            g.copy_(flat[offset : offset + n].view_as(g))
            offset += n

