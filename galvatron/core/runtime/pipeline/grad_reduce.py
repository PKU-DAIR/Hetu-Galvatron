import functools
from typing import Any, Callable, List, Optional, no_type_check

import torch
import torch.distributed as dist
import torch.nn as nn
from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._common_utils import HandleTrainingState, TrainingState, _FSDPState
from torch.distributed.fsdp._fully_shard._fully_shard import FSDPModule # fsdp2
from torch.distributed.fsdp._fully_shard._fsdp_state import FSDPState as FSDPv2State
from torch.distributed.tensor import DTensor # fsdp2
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
from galvatron.core.runtime.parallel_state import fsdp2_enabled
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
    if fsdp2_enabled():
        assert isinstance(model, FSDPModule), "Expect the model to be an instance of FSDPModule when fsdp2 is enabled."
        root_states: List[FSDPv2State] = []
        for m in model.modules():
            if not isinstance(m, FSDPModule):
                continue
            state = m._get_fsdp_state()
            param_group = state._fsdp_param_group
            if param_group is not None:
                assert param_group.reduce_grads == True, f'Expect to reduce gradients for param group {param_group} in module {m}'
                assert param_group.all_reduce_grads == True, f'Expect to all-reduce gradients for param group {param_group} in module {m}'
                param_group.post_backward()  # real reduce-scatter of accumulated grad
            state._state_ctx.is_last_backward = True
            if state._is_root:
                root_states.append(state)
        for state in root_states:
            state._root_post_backward_final_callback()
        return

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


def _is_dtensor(tensor) -> bool:
    return DTensor is not None and isinstance(tensor, DTensor)


def _local_datas(module: nn.Module, attr: str) -> List[torch.Tensor]:
    tensors = []
    for param in module.parameters():
        value = getattr(param, attr, None)
        if value is None:
            continue
        tensors.append(value.to_local() if _is_dtensor(value) else value)
    return tensors


@torch.no_grad()
def _allreduce_word_embedding_no_pipeline(wte_model, wte_attr_name, lmhead_model, lmhead_attr_name):
    wte = wte_model.module
    lmhead = lmhead_model.module
    if fsdp2_enabled():
        wte_tensors = _local_datas(wte, attr='data')
        lmhead_tensors = _local_datas(lmhead, attr='data')
        assert len(wte_tensors) == len(lmhead_tensors), f"tied embedding modules have mismatched tensors: {len(wte_tensors)} vs {len(lmhead_tensors)}"
        for wte_t, lm_t in zip(wte_tensors, lmhead_tensors):
            avg = (wte_t + lm_t) / 2
            wte_t.copy_(avg)
            lm_t.copy_(avg)
        return

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
    if fsdp2_enabled():
        for tensor in _local_datas(word_embedding, attr='data'):
            dist.all_reduce(tensor, group=group, op=dist.ReduceOp.AVG)
        return

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

    if fsdp2_enabled():
        wte_tensors = _local_datas(wte, attr='grad')
        lmhead_tensors = _local_datas(lmhead, attr='grad')
        assert len(wte_tensors) == len(lmhead_tensors), f"tied embedding modules have mismatched tensors: {len(wte_tensors)} vs {len(lmhead_tensors)}"
        for wte_t, lm_t in zip(wte_tensors, lmhead_tensors):
            avg = (wte_t + lm_t) / 2
            wte_t.copy_(avg)
            lm_t.copy_(avg)
        return
    
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

    if fsdp2_enabled():
        for tensor in _local_datas(word_embedding, attr='grad'):
            dist.all_reduce(tensor, group=group)
        return
    
    if hasattr(word_embedding, "_handles"):
        for handle in word_embedding._handles:
            assert handle.flat_param.grad is not None
            dist.all_reduce(handle.flat_param.grad, group=group)
    else:
        assert word_embedding._handle.flat_param.grad is not None
        dist.all_reduce(word_embedding._handle.flat_param.grad, group=group)


def enter_no_sync_context(model):
    if fsdp2_enabled():
        assert isinstance(model, FSDPModule), "Expect the model to be an instance of FSDPModule when fsdp2 is enabled."
        model.set_requires_gradient_sync(False, recurse=True)
        return

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
    if fsdp2_enabled():
        assert isinstance(model, FSDPModule), "Expect the model to be an instance of FSDPModule when fsdp2 is enabled."
        model.set_requires_gradient_sync(True, recurse=True)
        return

    if isinstance(model, FSDP):
        model.no_sync_context.__exit__(None, None, None)
    elif isinstance(model, nn.Sequential):
        for block in model:
            for m in block.modules():
                if isinstance(m, FSDP) and hasattr(m, "no_sync_context"):
                    m.no_sync_context.__exit__(None, None, None)
                    break

