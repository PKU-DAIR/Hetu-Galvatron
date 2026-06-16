import functools
from typing import Any, Callable, List, Optional, no_type_check

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import HandleTrainingState, TrainingState, _FSDPState
from galvatron.core.runtime.utils.utils import is_torch_min_version

if is_torch_min_version("2.5.0"):
    from torch.distributed.fsdp._flat_param import (
        RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES,
        FlatParameter,
        FlatParamHandle,
        HandleShardingStrategy,
        HandleTrainingState,
    )
else:
    from torch.distributed.fsdp.flat_param import (
        FlatParameter,
        FlatParamHandle,
        HandleShardingStrategy,
        HandleTrainingState,
        RESHARD_AFTER_FORWARD_HANDLE_STRATEGIES,
    )

from torch.distributed.fsdp._runtime_utils import _post_backward_final_callback, _unshard
from torch.distributed.utils import _p_assert

from galvatron.core.runtime.utils.utils import rgetattr, rhasattr
from .sp_grad_reduce import _post_backward_hook_sp as _post_backward_hook


def _send_backward_hook(
    input_tensor_grad: List[torch.Tensor],
    position: int,
    send_backward_partial: Callable,
    check_finish_partial: Callable,
    grad_output: Any,
) -> None:
    input_tensor_grad[position] = grad_output
    if check_finish_partial():
        send_backward_partial(input_tensor_grad)


def fsdp_reduce_gradients(model):
    for m in model.modules():
        if isinstance(m, FSDP):
            m.training_state = TrainingState.FORWARD_BACKWARD
            if hasattr(m, "_handles"):
                for handle in m._handles:
                    handle._training_state = HandleTrainingState.BACKWARD_PRE
                    _unshard(m, m._handles, m._streams["unshard"], m._streams["pre_unshard"])
                    _post_backward_hook(m, handle, None)
            else:
                if m._handle != None:
                    m._handle._training_state = HandleTrainingState.BACKWARD_PRE
                    _unshard(m, m._handle, m._unshard_stream, m._pre_unshard_stream)
                    _post_backward_hook(m, m._handle, None)

    for m in model.modules():
        if isinstance(m, FSDP) and m._is_root:
            _post_backward_final_callback(m, m)


@torch.no_grad()
def _allreduce_word_embedding_no_pipeline(wte_model, wte_attr_name, lmhead_model, lmhead_attr_name):
    wte = wte_model.module
    lmhead = lmhead_model.module
    if hasattr(wte, "_handles"):
        for wte_handle, lmhead_handle in zip(wte._handles, lmhead._handles):
            assert wte_handle.flat_param.data is not None
            assert lmhead_handle.flat_param.data is not None
            avg = (wte_handle.flat_param.data + lmhead_handle.flat_param.data) / 2
            wte_handle.flat_param.data.copy_(avg)
            lmhead_handle.flat_param.data.copy_(avg)
    else:
        assert wte._handle.flat_param.data is not None
        assert lmhead._handle.flat_param.data is not None
        avg = (wte._handle.flat_param.data + lmhead._handle.flat_param.data) / 2
        wte._handle.flat_param.data.copy_(avg)
        lmhead._handle.flat_param.data.copy_(avg)


@torch.no_grad()
def _allreduce_word_embedding(module, tied_wte_attr_name, group):
    word_embedding = module.module
    if hasattr(word_embedding, "_handles"):
        for handle in word_embedding._handles:
            assert handle.flat_param.data is not None
            dist.all_reduce(handle.flat_param.data, op=dist.ReduceOp.AVG, group=group)
    else:
        assert word_embedding._handle.flat_param.data is not None
        dist.all_reduce(word_embedding._handle.flat_param.data, op=dist.ReduceOp.AVG, group=group)


# For Finalization of Model Parameters, unsupport for learned position embedding weights
@torch.no_grad()
def _allreduce_word_embedding_grads_no_pipeline(wte_model, wte_attr_name, lmhead_model, lmhead_attr_name):
    wte = wte_model.module
    lmhead = lmhead_model.module
    if hasattr(wte, "_handles"):
        for wte_handle, lmhead_handle in zip(wte._handles, lmhead._handles):
            assert wte_handle.flat_param.grad is not None
            assert lmhead_handle.flat_param.grad is not None
            avg = (wte_handle.flat_param.grad + lmhead_handle.flat_param.grad) / 2
            wte_handle.flat_param.grad.copy_(avg)
            lmhead_handle.flat_param.grad.copy_(avg)
    else:
        assert wte._handle.flat_param.grad is not None
        assert lmhead._handle.flat_param.grad is not None
        avg = (wte._handle.flat_param.grad + lmhead._handle.flat_param.grad) / 2
        wte._handle.flat_param.grad.copy_(avg)
        lmhead._handle.flat_param.grad.copy_(avg)


# For Finalization of Model Gradients
@torch.no_grad()
def _allreduce_word_embedding_grads(module, tied_wte_attr_name, group):
    word_embedding = module.module
    if hasattr(word_embedding, "_handles"):
        for handle in word_embedding._handles:
            assert handle.flat_param.grad is not None
            dist.all_reduce(handle.flat_param.grad, group=group)
    else:
        assert word_embedding._handle.flat_param.grad is not None
        dist.all_reduce(word_embedding._handle.flat_param.grad, group=group)


def enter_no_sync_context(model):
    if isinstance(model, FSDP):
        model.no_sync_context = model.no_sync()
        model.no_sync_context.__enter__()
    elif isinstance(model, nn.Sequential):
        for block in model:
            for m in block.modules():
                if isinstance(m, FSDP):
                    m.no_sync_context = m.no_sync()
                    m.no_sync_context.__enter__()
                    break


def exit_no_sync_context(model):
    if isinstance(model, FSDP):
        model.no_sync_context.__exit__(None, None, None)
    elif isinstance(model, nn.Sequential):
        for block in model:
            for m in block.modules():
                if isinstance(m, FSDP) and hasattr(m, "no_sync_context"):
                    m.no_sync_context.__exit__(None, None, None)
                    break


def _register_post_backward_hook_bf16(
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
def _finalize_params_bf16(
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
