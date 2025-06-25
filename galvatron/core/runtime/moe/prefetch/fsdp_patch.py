import logging
import torch
import os
from torch import Tensor, nn
from typing import List, Union, Dict, Union, Any, Tuple, cast, Optional, Sequence
import torch.distributed as dist
from torch.distributed.utils import (
    _p_assert,
    _free_storage,
)
from torch.distributed.fsdp.flat_param import (
    FlatParamHandle, 
    FlatParameter,
    ParamInfo,
    SharedParamInfo,
    _construct_padding_tensor,
    _ext_pre_flatten_transform,
    _named_parameters_with_duplicates,
    _convert_to_params,
    HandleShardingStrategy,
    _FSDP_SKIP_WRITEBACK_CHECK,
    _FSDP_USE_FULL_PREC_IN_EVAL,
    _warn_skip_writeback_check,
    _get_aligned_numel,
)
from torch.distributed.fsdp._common_utils import (
    HandleTrainingState,
    _FSDPDeviceHandle,
    _FSDPState,
    _no_dispatch_record_stream
)
from torch.distributed.fsdp import _runtime_utils
from torch.distributed.fsdp._runtime_utils import (
    _div_if_needed,
    _accumulate_sharded_grad,
    _post_reduce_grad_callback,
)

log = logging.getLogger(__name__)

original_init = FlatParamHandle.__init__
original_init_flat_param_and_metadata = FlatParamHandle._init_flat_param_and_metadata
original_init_flat_param_attributes = FlatParamHandle.init_flat_param_attributes
original_flatten_tensors_into_flat_param = FlatParamHandle.flatten_tensors_into_flat_param
original_all_gather_flat_param = FlatParamHandle._all_gather_flat_param
original_use_unsharded_views = FlatParamHandle._use_unsharded_views
original_reduce_grad = _runtime_utils._reduce_grad
original_prepare_gradient_for_backward = FlatParamHandle.prepare_gradient_for_backward

def _new_init(
    self,
    params: Sequence[Union[nn.Parameter, Tensor]],
    fully_sharded_module: nn.Module,
    device: torch.device,
    sharding_strategy: HandleShardingStrategy,
    offload_params: bool,
    mp_param_dtype: Optional[torch.dtype],
    mp_reduce_dtype: Optional[torch.dtype],
    keep_low_precision_grads: bool,
    process_group: dist.ProcessGroup,
    use_orig_params: bool,
):
    super(FlatParamHandle, self).__init__()
    params = list(params)
    if len(params) == 0:
        raise ValueError(
            f"Cannot construct a {self.__class__.__name__} with an empty parameter list"
        )
    self._init_setattr_fns()
    self._skip_writeback_check = (
        os.environ.get(_FSDP_SKIP_WRITEBACK_CHECK, "") == "1"
    )
    self._use_full_prec_in_eval = (
        os.environ.get(_FSDP_USE_FULL_PREC_IN_EVAL, "") == "1"
    )
    if self._skip_writeback_check:
        _warn_skip_writeback_check(
            log,
            f"Since {_FSDP_SKIP_WRITEBACK_CHECK}=1, FSDP will not check "
            "for parameter or gradient writeback. Changing parameter or "
            "gradient storages may lead to silent correctness errors.",
        )
    # Only align addresses for `use_orig_params=True` (for now)
    align_addresses = use_orig_params
    self._init_get_unflat_views_fn(align_addresses)
    self.device = device
    self._device_handle = _FSDPDeviceHandle.from_device(self.device)
    self.process_group = process_group
    self.rank = process_group.rank()
    self.world_size = process_group.size()
    self._sharding_strategy = sharding_strategy
    self._offload_params = offload_params
    self._use_orig_params = use_orig_params
    self._keep_low_precision_grads = keep_low_precision_grads
    self._training_state = HandleTrainingState.IDLE
    self._debug_level = dist.get_debug_level()
    self._fully_sharded_module = fully_sharded_module
    # For strategies that do not free after forward, we skip using sharded
    # views after forward since the unsharded data exists. We still switch
    # `self.flat_param` to point to the sharded flat parameter since what
    # it points to parameterizes behavior. We use the following attribute
    # to track which tensor data the parameters are unsharded views into.
    self._unsharded_flat_param_for_skipped_views: Optional[Tensor] = None
    # The index in the state's `all_handles`, which must be the
    # same across ranks for the execution order validation to work
    self._handle_index: Optional[int] = None
    # Index in handles_to_pre_forward_order
    self._pre_forward_order_index: Optional[int] = None
    # Index in `handles_post_forward_order`
    self._post_forward_index: Optional[int] = None
    # Used for guarding against mistargeted forward prefetches
    self._needs_pre_forward_unshard = False
    # Used for guarding against mistargeted backward prefetches
    self._needs_pre_backward_unshard = False
    # Was the handle prefetched? Set on successful _prefetch_handle and unshard
    self._prefetched = False
    # Optimistically assume a valid input `params` and set dtype attributes
    # before `_init_flat_param()`, which performs the actual validation
    self._orig_param_dtype = params[0].dtype
    self._init_param_reduce_dtypes(mp_param_dtype, mp_reduce_dtype)
    assert self._fwd_bwd_param_dtype is not None  # mypy
    self._aligned_numel = (
        _get_aligned_numel(unsharded_dtype=self._fwd_bwd_param_dtype)
        if align_addresses
        else 0
    )
    self._init_flat_param_and_metadata(
        params, fully_sharded_module, self._aligned_numel, use_orig_params  # type: ignore[arg-type]
    )
    # FSEP UPD: Do not change views, delete the count of original params
    # self._use_unsharded_views(as_params=False)
    for submodule_name, submodule in fully_sharded_module.named_modules(remove_duplicate=False):
        for param_name, param in _named_parameters_with_duplicates(
            submodule, recurse=False
        ):
            submodule._parameters[param_name] = None
    self.flat_param._unpadded_unsharded_size = torch.Size((self.flat_param._unpadded_unsharded_size[0] // self.world_size,))

def new_init(self, *args, **kwargs):

    _fully_sharded_module = args[1]

    if getattr(_fully_sharded_module, "is_moe_layer", False):
        self.is_moe_layer = True
        self.global_expert_num = _fully_sharded_module.num_global_experts
        self.local_expert_num = _fully_sharded_module.num_local_experts
        # [world_size * local_expert_num]
        self.global_placement = _fully_sharded_module.global_expert_indices
    else:
        self.is_moe_layer = False

    if not self.is_moe_layer:
        original_init(self, *args, **kwargs)
    else:
        _new_init(self, *args, **kwargs)

def new_init_flat_param_and_metadata(
    self,
    params: List[Union[Tensor, nn.Parameter]],
    module: nn.Module,
    aligned_numel: int,
    use_orig_params: bool,
) -> None:
    if not self.is_moe_layer:
        original_init_flat_param_and_metadata(self, params, module, aligned_numel, use_orig_params)
    else:
        if len(params) == 0:
            raise ValueError("Expects non-empty `params`")
        if aligned_numel < 0:
            raise ValueError(
                f"Expects non-negative `aligned_numel` but got {aligned_numel}"
            )
        (
            dtype,
            flat_param_requires_grad,
            device,
        ) = self._validate_tensors_to_flatten(params)
        params_set = set(params)
        # For alignment padding, only `numels` gets strictly non-`None`
        # elements, and all other lists get `None` elements for padding.
        param_infos: List[ParamInfo] = []
        numels: List[int] = []
        shapes: List[torch.Size] = []
        fqns: List[str] = []
        shared_param_infos: List[SharedParamInfo] = []
        shared_param_memo: Dict[
            Union[Tensor, nn.Parameter], Tuple[nn.Module, str, str]
        ] = {}
        params_to_flatten: List[Union[Tensor, nn.Parameter]] = []
        shared_params: List[Union[Tensor, nn.Parameter]] = []
        param_extensions: List[Any] = []
        is_padding_mask: List[bool] = []
        total_numel = total_numel_without_padding = 0
        for submodule_name, submodule in module.named_modules(remove_duplicate=False):
            for param_name, param in _named_parameters_with_duplicates(
                submodule, recurse=False
            ):
                if param not in params_set:
                    continue
                if param in shared_param_memo:  # shared reference
                    prim_module, prim_module_name, prim_param_name = shared_param_memo[
                        param
                    ]
                    shared_params.append(param)
                    shared_param_infos.append(
                        SharedParamInfo(
                            param_name,
                            submodule,
                            submodule_name,
                            prim_param_name,
                            prim_module,
                            prim_module_name,
                        )
                    )
                else:
                    if aligned_numel > 0:
                        # Do not support orig params mode.
                        numel_to_pad = aligned_numel - (total_numel % aligned_numel)
                        if numel_to_pad > 0 and numel_to_pad < aligned_numel:
                            padding_tensor = _construct_padding_tensor(
                                numel_to_pad, dtype, False, device
                            )
                            params_to_flatten.append(padding_tensor)
                            is_padding_mask.append(True)
                            numels.append(numel_to_pad)
                            total_numel += numel_to_pad
                    if "real" in submodule_name:
                        transform_t, extension = _ext_pre_flatten_transform(param)
                        param = cast(nn.Parameter, transform_t)
                        param_extensions.append(extension)
                        shared_param_memo[param] = (submodule, submodule_name, param_name)
                        is_padding_mask.append(False)
                        param_infos.append(ParamInfo(param_name, submodule, submodule_name))
                        numels.append(param.numel())
                        shapes.append(param.shape)
                        fqn = (
                            submodule_name + "." + param_name
                            if submodule_name
                            else param_name
                        )
                        fqns.append(fqn)
                    else:
                        params_to_flatten.append(param)
                        total_numel += param.numel()
                        total_numel_without_padding += param.numel()
        if len(params_to_flatten) == 0:
            raise ValueError(
                f"`params` were not found in `module`'s tree"
                f"params: {params}\nmodule: {module}"
            )
        if (
            self.rank == 0
            and aligned_numel > 0
            and total_numel != total_numel_without_padding
        ):
            log.info(
                "FSDP FlatParameter address alignment created "
                "%s numel of padding (%s vs. %s)",
                total_numel - total_numel_without_padding,
                total_numel,
                total_numel_without_padding,
            )
        if aligned_numel > 0:
            # Pad to be divisible by world size to avoid a copy for the
            # post-backward reduce-scatter
            numel_to_pad = self.world_size - (total_numel % self.world_size)
            if numel_to_pad > 0 and numel_to_pad < self.world_size:
                if self.rank == 0:
                    log.info(
                        "FSDP FlatParameter world size divisibility created "
                        "%s numel of padding",
                        numel_to_pad,
                    )
                padding_tensor = _construct_padding_tensor(
                    numel_to_pad, dtype, False, device
                )
                params_to_flatten.append(padding_tensor)
                is_padding_mask.append(True)
                numels.append(numel_to_pad)
                total_numel += numel_to_pad
        # Pass `aligned_numel=0` since we already included padding tensors
        self.flat_param: FlatParameter = self.flatten_tensors_into_flat_param(
            params_to_flatten,
            aligned_numel=0,
            requires_grad=flat_param_requires_grad,
        )

        FlatParameter._init_metadata(
            self.flat_param,
            param_infos,
            numels,
            shapes,
            fqns,
            shared_param_infos,
            param_extensions,
            _convert_to_params(params_to_flatten) if use_orig_params else None,
            _convert_to_params(shared_params) if use_orig_params else None,
            is_padding_mask,
        )

def new_init_flat_param_attributes(self) -> None:
    if not self.is_moe_layer:
        original_init_flat_param_attributes(self)
    else:
        flat_param = self.flat_param
        if flat_param.dtype != self._orig_param_dtype:
            # Entering this branch means that the user changed the parameter
            # dtype after FSDP initialization, in which case we may need to
            # refresh some saved dtype attributes (dtypes specified as a part
            # of mixed precision take precedence).
            if not self._low_prec_param_dtype_specified:
                self._fwd_bwd_param_dtype = flat_param.dtype
            # For `reduce_dtype`, require `param_dtype` was not specified since
            # then we infer the `reduce_dtype` from the specified `param_dtype`
            if (
                not self._low_prec_reduce_dtype_specified
                and not self._low_prec_param_dtype_specified
            ):
                self._reduce_dtype = flat_param.dtype
            self._orig_param_dtype = flat_param.dtype
        cpu_device = torch.device("cpu")
        if self._offload_params:
            _p_assert(
                flat_param.device == cpu_device,
                f"Expects the `FlatParameter` to be on CPU when parameter CPU "
                f"offloading is enabled, not {flat_param.device}",
            )
        else:
            self._check_on_compute_device(self.flat_param)
        flat_param._local_shard = flat_param.data
        if self._offload_params:
            # Pin the memory for faster H2D transfer
            flat_param._local_shard = flat_param._local_shard.pin_memory()
            # Pre-allocate the sharded gradient on CPU to enable non-blocking
            # D2H transfer during the backward pass
            flat_param._cpu_grad = torch.zeros_like(
                flat_param._local_shard, device=cpu_device
            ).pin_memory()
        if self._uses_param_mixed_precision:
            # For parameter mixed precision, we maintain a low precision
            # sharded tensor on the compute device to be all-gathered (for
            # sharded strategies) or directly used (for `NO_SHARD`) for
            # computation.
            flat_param._mp_shard = torch.empty_like(
                flat_param._local_shard,
                device=self.device,
                dtype=self._fwd_bwd_param_dtype,
            )
            _free_storage(flat_param._mp_shard)
        if self.uses_sharded_strategy:
            # We maintain a padded unsharded tensor that serves as the
            # all-gather destination and owns the original parameter storages.
            unsharded_param_dtype = (
                self._fwd_bwd_param_dtype
                if self._uses_param_mixed_precision
                else flat_param.dtype
            )  # use low precision if parameter mixed precision is enabled
            # FSEP UPD: Fix new unsharded numel
            padded_unsharded_numel = flat_param.numel() * self.world_size // self.global_expert_num * self.local_expert_num
            flat_param._full_param_padded = torch.empty(
                padded_unsharded_numel,
                device=self.device,
                dtype=unsharded_param_dtype,
            )
            flat_param._padded_unsharded_size = flat_param._full_param_padded.size()
            _free_storage(flat_param._full_param_padded)

            if self._uses_param_mixed_precision:
                # For parameter mixed precision, we maintain a full precision
                # padded unsharded tensor for when we force full precision.
                flat_param._full_prec_full_param_padded = torch.empty(
                    padded_unsharded_numel,
                    device=self.device,
                    dtype=flat_param.dtype,  # full precision
                )
                _free_storage(flat_param._full_prec_full_param_padded)

def new_flatten_tensors_into_flat_param(
    self,
    tensors: List[Tensor],
    aligned_numel: int,
    requires_grad: bool,
) -> FlatParameter:
    flat_param_data = self.flatten_tensors(tensors, aligned_numel)
    if self.is_moe_layer:
        _p_assert(
            flat_param_data.numel() % (self.global_expert_num * self.world_size) == 0,
            "The size of expert should be a multiple of world size"
        )
        with torch.no_grad():
            # TODO: Check if this operator effect meta data of flat parameter
            flat_param_data = flat_param_data.reshape(self.global_expert_num, self.world_size, -1).permute(1,0,2).reshape(-1).contiguous()
    return FlatParameter(flat_param_data, requires_grad=requires_grad)

def _all_to_all_flat_param_type1(
    self,
    padded_unsharded_flat_param: Tensor,
    sharded_flat_param: Tensor,
    process_group: dist.ProcessGroup,
):
    sharded_flat_param = sharded_flat_param.contiguous().reshape(self.global_expert_num, -1)  # (expert_num, expert_shard_size)
    expert_shard_size = sharded_flat_param.numel() // self.global_expert_num
    send_list = []
    recv_list = []

    # TODO: Reduce communication volume if experts are same
    for rank in range(self.world_size): 
        needed_experts = self.global_placement[rank]
        send_data = sharded_flat_param[needed_experts]  # (len(needed_experts), expert_shard_size)
        send_list.append(send_data.contiguous())

        recv_shape = (self.local_expert_num, expert_shard_size)
        recv_data = torch.empty(recv_shape, dtype=sharded_flat_param.dtype, device=sharded_flat_param.device)
        recv_list.append(recv_data)
    
    dist.all_to_all(recv_list, send_list, group=process_group)
    padded_unsharded_flat_param = padded_unsharded_flat_param.reshape(self.local_expert_num, -1)
    for rank, recv_data in enumerate(recv_list):
        start_col = rank * expert_shard_size
        end_col = (rank + 1) * expert_shard_size
        padded_unsharded_flat_param[:, start_col:end_col].copy_(recv_data)
    padded_unsharded_flat_param = padded_unsharded_flat_param.reshape(-1)
    sharded_flat_param = sharded_flat_param.reshape(-1)
    return padded_unsharded_flat_param, sharded_flat_param

@torch.no_grad()
def _all_to_all_grad_type1(
    handle: FlatParamHandle,
    padded_unsharded_grad: Tensor,
    new_sharded_grad: Tensor,
    process_group: dist.ProcessGroup,
):
    expert_shard_size = new_sharded_grad.numel() // handle.global_expert_num
    
    padded_unsharded_grad = padded_unsharded_grad.reshape(handle.local_expert_num, -1)
    
    send_list = []
    recv_list = []
    
    # TODO: Reduce communication volume if experts are same
    for rank in range(handle.world_size):
        start_col = rank * expert_shard_size
        end_col = (rank + 1) * expert_shard_size
        send_data = padded_unsharded_grad[:, start_col:end_col].contiguous()  # (local_expert_num, expert_shard_size)
        send_list.append(send_data)

        recv_shape = (handle.local_expert_num, expert_shard_size)
        recv_data = torch.empty(recv_shape, dtype=padded_unsharded_grad.dtype, device=padded_unsharded_grad.device)
        recv_list.append(recv_data)
    
    dist.all_to_all(recv_list, send_list, group=process_group)
    
    new_sharded_grad.zero_()
    new_sharded_grad = new_sharded_grad.reshape(handle.global_expert_num, -1)
    for rank, recv_data in enumerate(recv_list):
        needed_experts = handle.global_placement[rank]
        new_sharded_grad.index_add_(0, needed_experts.view(-1), recv_data)
    new_sharded_grad = new_sharded_grad.reshape(-1)

    padded_unsharded_grad = padded_unsharded_grad.reshape(-1)
    return padded_unsharded_grad, new_sharded_grad

def new_all_gather_flat_param(
    self,
    padded_unsharded_flat_param: Tensor,
) -> Tensor:
    if not self.is_moe_layer:
        return original_all_gather_flat_param(self, padded_unsharded_flat_param)
    _p_assert(
        hasattr(self, "process_group") and hasattr(self, "world_size"),
        "Expects a process group and world size to have been set via `shard()`",
    )
    sharded_flat_param = self.flat_param.data
    # FSEP UPD: Fix new unsharded numel
    expected_numel = sharded_flat_param.numel() * self.world_size // self.global_expert_num * self.local_expert_num
    _p_assert(
        padded_unsharded_flat_param.numel() == expected_numel,
        f"Expects {expected_numel} numel but got {padded_unsharded_flat_param.numel()}",
    )

    # HACK this should be handled by C10D
    if sharded_flat_param.is_cpu:  # type: ignore[attr-defined]
        tensor_list = list(
            torch.chunk(
                padded_unsharded_flat_param, dist.get_world_size(self.process_group)
            )
        )
        work = dist.all_gather(
            tensor_list, sharded_flat_param, group=self.process_group
        )
    else:
        # FSEP UPD: Flexiable all gather
        with torch.no_grad():
            padded_unsharded_flat_param, sharded_flat_param = self._all_to_all_flat_param(
                padded_unsharded_flat_param,
                sharded_flat_param,
                self.process_group,
            )
        # dist.all_gather_into_tensor(
        #     padded_unsharded_flat_param,
        #     sharded_flat_param,
        #     self.process_group,
        # )
    return padded_unsharded_flat_param

def _get_all_to_all_tensors(
    handle: FlatParamHandle, unsharded_grad: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    sharded_grad_size = unsharded_grad.numel() // handle.local_expert_num * handle.global_expert_num // handle.world_size
    new_sharded_grad = torch.empty(sharded_grad_size, dtype=unsharded_grad.dtype, device=unsharded_grad.device)
    return unsharded_grad, new_sharded_grad

def new_reduce_grad(state: _FSDPState, handle: FlatParamHandle) -> None:
    if not handle.is_moe_layer:
        original_reduce_grad(state, handle)
    else:
        flat_param = handle.flat_param
        uses_hybrid_sharded_strategy = handle._sharding_strategy in (
            HandleShardingStrategy.HYBRID_SHARD,
            HandleShardingStrategy._HYBRID_SHARD_ZERO2,
        )
        # We clear `.grad` to permit multiple backwards. This avoids a race where
        # the second backward pass computation precedes ahead of the first backward
        # pass reduction, which is possible since the reduction is issued in a
        # separate stream and is async and would result in reducing the wrong
        # gradient.
        unsharded_grad = flat_param.grad.data
        flat_param.grad = None
        # FSEP UPD: Get new tensor shape
        padded_unsharded_grad, new_sharded_grad = _get_all_to_all_tensors(
            handle, unsharded_grad
        )
        if state._comm_hook is None:  # default path
            _div_if_needed(padded_unsharded_grad, state._gradient_predivide_factor)
            # FSEP UPD: Update new all to all communication
            # dist.reduce_scatter_tensor(
            #     new_sharded_grad,
            #     padded_unsharded_grad,
            #     group=state.process_group,
            # )
            padded_unsharded_grad, new_sharded_grad = _all_to_all_grad(
                handle,
                padded_unsharded_grad,
                new_sharded_grad,
                state.process_group
            )
            if uses_hybrid_sharded_strategy:
                state._all_reduce_stream.wait_stream(state._post_backward_stream)
                with state._device_handle.stream(state._all_reduce_stream):
                    # Since the new sharded gradient is produced in the post-
                    # backward stream and consumed in the all-reduce stream,
                    # inform the caching allocator
                    _no_dispatch_record_stream(new_sharded_grad, state._all_reduce_stream)
                    dist.all_reduce(new_sharded_grad, group=state._inter_node_pg)
                    _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
                    grad_to_offload = _accumulate_sharded_grad(
                        state, handle, new_sharded_grad
                    )
                    _post_reduce_grad_callback(state, handle, grad_to_offload)
                    return
            _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
        else:
            state._comm_hook(
                state._comm_hook_state, padded_unsharded_grad, new_sharded_grad
            )
            # NOTE: HSDP variants do not support communication hook.
        grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
        _post_reduce_grad_callback(state, handle, grad_to_offload)

def new_prepare_gradient_for_backward(self):
    if not self.is_moe_layer:
        original_prepare_gradient_for_backward(self)
    else:
        _p_assert(
            self._training_state
            in (HandleTrainingState.BACKWARD_PRE, HandleTrainingState.IDLE),
            "Expects to be in `BACKWARD_PRE` or `IDLE` (if prefetching)",
        )
        flat_param = self.flat_param
        if flat_param.grad is not None:
        # FSEP UPD: Disable check of size()
        # TODO: It is hacky and need to find a better way(maybe only work when not use no sync)
            #and (
            # flat_param.grad.size() != flat_param._unpadded_unsharded_size
            # or flat_param.grad.device != flat_param.device  # grad on CPU
        #):
            self._check_on_compute_device(self.flat_param)
            grad_offloaded = flat_param.grad.device != self.device
            _p_assert(
                not grad_offloaded or self._offload_params,
                f"Expects the sharded gradient to be on {self.device} "
                f"but got {flat_param.grad.device}",
            )
            prev_iter_synced_gradients = (
                flat_param.grad.size()
                == flat_param._local_shard.size()  # type: ignore[attr-defined]
            )
            if prev_iter_synced_gradients:
                # TODO (awgu): Gradient accumulation outside `no_sync()`
                # does not work with CPU offloading. The issue should be
                # that, in the post-backward hook, we cannot do an addition
                # between a CPU tensor (the existing sharded gradient) and
                # a GPU tensor (the new sharded gradient).
                if not grad_offloaded:
                    flat_param._saved_grad_shard = flat_param.grad.data  # type: ignore[attr-defined]
                    sharded_grad = flat_param._saved_grad_shard  # type: ignore[attr-defined]
                else:
                    _p_assert(
                        hasattr(flat_param, "_cpu_grad"),
                        "`_cpu_grad` should be defined if the gradient is on CPU",
                    )
                    sharded_grad = flat_param._cpu_grad  # type: ignore[attr-defined]
                # If user specified to keep the gradient in low precision, then
                # the gradient may still be of the low precision dtype if the
                # user did not set the gradient to `None` after the previous
                # backward, in which case FSDP should cast back to the full
                # precision dtype so that FSDP can accumulate in that dtype in
                # the post-backward hook and assign to `.grad` in that dtype in
                # the post-backward callback.
                local_shard_dtype = flat_param._local_shard.dtype  # type: ignore[attr-defined]
                if (
                    self._keep_low_precision_grads
                    and sharded_grad.dtype != local_shard_dtype
                ):
                    sharded_grad.data = sharded_grad.to(local_shard_dtype)
            else:
                padded_unsharded_size = flat_param._padded_unsharded_size  # type: ignore[attr-defined]
                _p_assert(
                    flat_param.grad.size() == padded_unsharded_size,
                    "Expects `.grad` to be the unsharded gradient in "
                    f"`no_sync()` with size {padded_unsharded_size} "
                    f"but got size {flat_param.grad.size()}",
                )
            flat_param.grad = None

FlatParamHandle.__init__ = new_init
FlatParamHandle._init_flat_param_and_metadata = new_init_flat_param_and_metadata
FlatParamHandle.init_flat_param_attributes = new_init_flat_param_attributes
FlatParamHandle.flatten_tensors_into_flat_param = new_flatten_tensors_into_flat_param
FlatParamHandle._all_gather_flat_param = new_all_gather_flat_param
FlatParamHandle._all_to_all_flat_param = _all_to_all_flat_param_type1
FlatParamHandle.prepare_gradient_for_backward = new_prepare_gradient_for_backward

# TODO: maybe improve this function to save memory? (same experts)
# FlatParamHandle._use_unsharded_views = new_use_unsharded_views

_runtime_utils._reduce_grad = new_reduce_grad
_all_to_all_grad = _all_to_all_grad_type1