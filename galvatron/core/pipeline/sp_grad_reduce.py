import logging
from typing import Any, Callable, Dict, List, no_type_check, Optional, Set, Tuple
import torch
import torch.distributed as dist
from torch.distributed.fsdp._common_utils import (
    _assert_in_training_states,
    _FSDPState,
    _get_module_fsdp_state,
    _is_composable,
    _log_post_backward_hook,
    _no_dispatch_record_stream,
    clean_tensor_name,
    TrainingState,
)
from torch.distributed.fsdp.flat_param import (
    FlatParameter,
    FlatParamHandle,
    HandleShardingStrategy,
    HandleTrainingState,
    RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES,
)
from torch.distributed.utils import (
    _apply_to_tensors,
    _cast_forward_inputs,
    _p_assert,
    _to_kwargs,
)

from torch.distributed.fsdp._runtime_utils import (
    _reduce_grad, 
    _reduce_grad_no_shard, 
    _post_backward_reshard, 
    _low_precision_hook_enabled
)

from megatron.core import parallel_state

log = logging.getLogger(__name__)

@no_type_check
@torch.no_grad()
def _post_backward_hook_sp(
    state: _FSDPState,
    handle: FlatParamHandle,
    *unused: Any,
):
    """
    Reduce-scatters the gradient of ``handle`` 's ``FlatParameter``.

    Precondition: The ``FlatParameter`` 's ``.grad`` attribute contains the
    unsharded gradient for the local batch.

    Postcondition:
    - If using ``NO_SHARD``, then the ``.grad`` attribute is the reduced
    unsharded gradient.
    - Otherwise, the ``_saved_grad_shard`` attribute is the reduced sharded
    gradient (accumulating with any existing gradient).
    """
    _log_post_backward_hook(state, handle, log)
    flat_param = handle.flat_param
    flat_param._post_backward_called = True
    with torch.autograd.profiler.record_function(
        "FullyShardedDataParallel._post_backward_hook"
    ):
        _assert_in_training_states(state, [TrainingState.FORWARD_BACKWARD])
        # For multiple applications of reentrant AC across submodules sharing
        # the same `FlatParameter`, the post-backward hook may run multiple
        # times in one backward, in which case we permit the state to already
        # be in `BACKWARD_POST`.
        _p_assert(
            handle._training_state
            in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.BACKWARD_POST),
            f"Expects `BACKWARD_PRE` or `BACKWARD_POST` state but got {handle._training_state}",
        )
        handle._training_state = HandleTrainingState.BACKWARD_POST
        if flat_param.grad is None:
            return
        if flat_param.grad.requires_grad:
            raise RuntimeError("FSDP does not support gradients of gradients")
        _post_backward_reshard(state, handle)
        if not state._sync_gradients:
            if handle._use_orig_params:
                handle._use_unsharded_grad_views()
            return

        # Wait for all ops in the current stream (e.g. gradient computation) to
        # finish before reduce-scattering the gradient
        state._post_backward_stream.wait_stream(state._device_handle.current_stream())

        with state._device_handle.stream(state._post_backward_stream):
            autograd_computed_grad = flat_param.grad.data
            if (
                not _low_precision_hook_enabled(state)
                and flat_param.grad.dtype != handle._reduce_dtype
                # If we are forcing full precision but communicating grads
                # (i.e. model.eval() + full precision in eval was configured), don't downcast gradient.
                and not handle._force_full_precision
            ):
                flat_param.grad.data = flat_param.grad.to(handle._reduce_dtype)
            
            if hasattr(state, "sp_group") and hasattr(state, "ln_offset") and len(state.ln_offset) > 0 and len(state.sp_group.ranks) > 1 and hasattr(state, "last_batch") and state.last_batch:
                all_ln_data = parallel_state.get_global_memory_buffer().get_tensor([sum(state.ln_size)],flat_param.grad.data.dtype,"reduce_grad")
                idx = 0
                for offset, size in zip(state.ln_offset, state.ln_size):
                    all_ln_data[idx:idx+size].copy_(flat_param.grad.data[offset:offset+size])
                    idx += size
                dist.all_reduce(all_ln_data, group=state.sp_group.group)
                idx = 0
                for offset, size in zip(state.ln_offset, state.ln_size):
                    flat_param.grad.data[offset:offset+size].copy_(all_ln_data[idx:idx+size])
                    idx += size
                
            if handle.uses_sharded_strategy:
                _reduce_grad(state, handle)
            else:
                _reduce_grad_no_shard(state, handle)
            # Since the unsharded gradient is produced in the computation
            # stream and consumed in the post-backward stream, inform the
            # caching allocator (before it goes out of scope)
            _no_dispatch_record_stream(
                autograd_computed_grad, state._post_backward_stream
            )
