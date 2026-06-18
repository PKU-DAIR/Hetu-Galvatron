import functools
from typing import Optional, no_type_check

import torch
from torch.distributed.utils import _p_assert
from torch.distributed.fsdp._common_utils import _FSDPState
import torch.distributed.fsdp._runtime_utils as _runtime_utils
from galvatron.core.runtime.pipeline.sp_grad_reduce import _post_backward_hook_sp as _post_backward_hook

from galvatron.core.runtime.utils.utils import is_torch_min_version

if is_torch_min_version("2.5.0"):
    from torch.distributed.fsdp._flat_param import FlatParamHandle
else:
    from torch.distributed.fsdp.flat_param import FlatParamHandle


# ============================== fsdp1 patches ==============================
def _print_patch_info(patch_type: str, patches: list[str]) -> None:
    if torch.distributed.is_initialized() and torch.distributed.get_rank() != 0:
        return

    import inspect
    from pathlib import Path

    caller = inspect.stack()[2]
    patches_msg = "\n  - ".join(patches) if patches else "<none>"

    print(
        f"[Galvatron] {patch_type} patches applied\n"
        f"  called from: {Path(caller.filename).resolve()}:{caller.lineno} in {caller.function}()\n"
        f"  patches:\n"
        f"  - {patches_msg}"
    )

def apply_fsdp1_patch():
    assert is_torch_min_version("2.0.1"), "FSDP v1 patches are only needed for torch 2.0.1"

    patches = []
    _runtime_utils._register_post_backward_hook = _register_post_backward_hook_patch
    patches.append("_runtime_utils._register_post_backward_hook -> _register_post_backward_hook_patch")
    
    _runtime_utils._finalize_params = _finalize_params_patch
    patches.append("_runtime_utils._finalize_params -> _finalize_params_patch")

    _print_patch_info("FSDP v1", patches)


def _register_post_backward_hook_patch(
    state: _FSDPState,
    handle: Optional[FlatParamHandle],
) -> None:
    """
    Registers post-backward hooks on the ``FlatParameter`` s'
    ``AccumulateGrad`` objects to reshard and to reduce-scatter gradients.

    The ``AccumulateGrad`` object represents the last function that finalizes
    the ``FlatParameter`` 's gradient, so it only runs after its entire
    gradient computation has finished.

    We register the post-backward hook only once in the *first* forward that a
    ``FlatParameter`` participates in. This relies on the ``AccumulateGrad``
    object being preserved through multiple forwards.

    NOTE: We follow this heuristic to prefer the *first* forward to target the
    parameter mixed precision case, where there are *separate*
    ``AccumulateGrad`` objects across the different forwards. (Without
    parameter mixed precision, the ``AccumulateGrad`` objects are the same.) If
    we instead prefer the *last* forward, then the hook runs early.
    """
    # If there is no gradient computation, then there is no need for
    # post-backward logic
    if not torch.is_grad_enabled():
        return
    if not handle:
        return
    flat_param = handle.flat_param
    already_registered = hasattr(flat_param, "_post_backward_hook_state")
    # if already_registered or not flat_param.requires_grad:
    #     return
    if not already_registered:
        flat_param._post_backward_hook_state = []
    # Get the `AccumulateGrad` object
    temp_flat_param = flat_param.expand_as(flat_param)
    _p_assert(
        temp_flat_param.grad_fn is not None,
        "The `grad_fn` is needed to access the `AccumulateGrad` and " "register the post-backward hook",
    )
    acc_grad = temp_flat_param.grad_fn.next_functions[0][0]  # type: ignore[union-attr]
    assert acc_grad is not None
    hook_handle = acc_grad.register_hook(functools.partial(_post_backward_hook, state, handle))
    flat_param._post_backward_hook_state.append((acc_grad, hook_handle))  # type: ignore[attr-defined]


@no_type_check
def _finalize_params_patch(
    state: _FSDPState,
) -> None:
    """Finalizes the parameters before the next iteration."""
    handle = state._handle
    if not handle:
        return
    flat_param = handle.flat_param
    if hasattr(flat_param, "_post_backward_hook_state"):
        # post_backward_hook_state_len = len(flat_param._post_backward_hook_state)
        # expected_post_backward_hook_state_len = int(flat_param.requires_grad) + 1
        # _p_assert(
        #     post_backward_hook_state_len == expected_post_backward_hook_state_len,
        #     f"Invalid: ``_post_backward_hook_state``: {flat_param._post_backward_hook_state}",
        # )
        if len(flat_param._post_backward_hook_state) > 0:
            flat_param._post_backward_hook_state[0][-1].remove()
            flat_param._post_backward_hook_state.pop(0)
        # delattr(flat_param, "_post_backward_hook_state")
    if flat_param.requires_grad:
        if not state._sync_gradients:
            # Preserve the gradient accumulation state if not synchronizing
            # gradients: `.grad` remains the unsharded gradient  from prior
            # `no_sync()` iterations, and `_saved_grad_shard` remains the
            # sharded gradient from the last synchronized iteration
            return
        if not handle._has_optim_in_backward:
            handle.prepare_gradient_for_optim()
        _p_assert(
            hasattr(flat_param, "_post_backward_called"),
            "Expects `_post_backward_called` to be set on the `FlatParameter`",
        )
        flat_param._post_backward_called = False
