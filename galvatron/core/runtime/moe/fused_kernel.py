# modify from te 2.1


import torch
import triton
import triton.language as tl
from typing import Union, Tuple
import warnings
from megatron.core.tensor_parallel import all_to_all

def moe_unpermute(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: torch.Tensor = None,
    restore_shape: torch.Tensor = None,
    map_type: str = "mask",
    probs: torch.Tensor = None,
) -> torch.Tensor:
    """
    Unpermute a tensor with permuted tokens, and optionally merge the tokens with their
    corresponding probabilities.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor with permuted tokens of shape `[num_tokens, hidden_size]` to be unpermuted.
    row_id_map: torch.Tensor
        The tensor of a mapping table for sorted indices used to unpermute the tokens,
        which is the second output tensor of `Permute`.
    merging_probs: torch.Tensor, default = None
        The tensor of probabilities corresponding to the permuted tokens. If provided,
        the unpermuted tokens will be merged with their respective probabilities.
        By default, set to an empty tensor, which means that the tokens are directly merged by accumulation.
    restore_shape: torch.Tensor
        The output shape after the unpermute operation.
    map_type: str, default = 'mask'
        Type of the routing map tensor. Should be the same as the value passed to moe_permute.
        Options are: 'mask', 'index'.
    probs: torch.Tensor, default = None
        Renamed to merging_probs. Keep for backward compatibility.
    """
    if probs is not None:
        if merging_probs is not None:
            raise ValueError(
                "Both merging_probs and probs kwarg are provided. probs is deprecated."
            )
        warnings.warn("probs kwarg is deprecated. Use merging_probs kwarg instead.")
        merging_probs = probs
    if map_type == "index":
        assert False, "index type not support yet!"
        # return _moe_unpermute_index_map.apply(inp, row_id_map, merging_probs)
    if map_type == "mask":
        return _moe_unpermute_mask_map.apply(inp, row_id_map, merging_probs, restore_shape)
    raise ValueError("map_type should be one of 'mask' or 'index'")

class _moe_unpermute_mask_map(torch.autograd.Function):
    """functional Unpermute with mask router map"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        row_id_map: torch.Tensor,
        merging_probs: torch.Tensor,
        restore_shape: torch.Size,
    ) -> torch.Tensor:
        # pylint: disable=missing-function-docstring
        if not inp.numel():
            ctx.merging_probs = merging_probs
            return inp

        if restore_shape is None:
            restore_shape = inp.shape
        num_tokens, hidden_size = restore_shape
        num_experts = row_id_map.size(0)

        with_probs = merging_probs is not None
        if with_probs:
            assert merging_probs.is_cuda, "TransformerEngine needs CUDA."

        # Device check
        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert row_id_map.is_cuda, "TransformerEngine needs CUDA."

        unpermuted_output, _ = triton_unpermute_with_mask_map(
            inp,
            row_id_map,
            merging_probs,
            None,
            num_tokens,
            num_experts,
            hidden_size,
        )
        if with_probs:
            ctx.save_for_backward(inp, row_id_map, merging_probs)
        else:
            ctx.save_for_backward(row_id_map)
        ctx.num_experts = num_experts
        ctx.num_tokens = num_tokens
        ctx.num_permuted_tokens = inp.size(0)
        ctx.hidden_size = hidden_size
        ctx.with_probs = with_probs
        return unpermuted_output

    @staticmethod
    def backward(ctx, unpermuted_act_grad):
        # pylint: disable=missing-function-docstring
        if not unpermuted_act_grad.numel():
            return unpermuted_act_grad, None, ctx.merging_probs, None

        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            if ctx.with_probs:
                fwd_input, row_id_map, merging_probs = ctx.saved_tensors
            else:
                (row_id_map,) = ctx.saved_tensors

            if ctx.with_probs:
                act_grad, probs_grad = (
                    triton_unpermute_with_mask_map_bwd_with_merging_probs(
                        unpermuted_act_grad,
                        row_id_map,
                        fwd_input,
                        merging_probs,
                        ctx.num_tokens,
                        ctx.num_experts,
                        ctx.num_permuted_tokens,
                        ctx.hidden_size,
                    )
                )
            else:
                assert False, "no probs not support yet!"
                # act_grad, _ = triton_permute_with_mask_map(
                #     unpermuted_act_grad,
                #     row_id_map,
                #     None,
                #     ctx.num_tokens,
                #     ctx.num_experts,
                #     ctx.num_permuted_tokens,
                #     ctx.hidden_size,
                # )

        if not ctx.needs_input_grad[2]:
            probs_grad = None
        return act_grad, None, probs_grad, None

def triton_unpermute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    merging_probs: Union[torch.Tensor, None],
    permuted_probs: Union[torch.Tensor, None],
    num_tokens: int,
    num_experts: int,
    hidden_size: int,
):
    output = torch.empty((num_tokens, hidden_size), dtype=inp.dtype, device="cuda")
    if permuted_probs is not None:
        unpermuted_probs = torch.empty(
            (num_tokens, num_experts), dtype=permuted_probs.dtype, device="cuda"
        )
    else:
        unpermuted_probs = None
    grid = (num_tokens,)
    _unpermute_kernel[grid](
        inp,
        output,
        row_id_map,
        merging_probs,
        permuted_probs,
        unpermuted_probs,
        num_tokens,
        num_experts,
        hidden_size,
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        merging_probs.stride(0) if merging_probs is not None else None,
        merging_probs.stride(1) if merging_probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        unpermuted_probs.stride(0) if unpermuted_probs is not None else None,
        unpermuted_probs.stride(1) if unpermuted_probs is not None else None,
        WITH_MERGING_PROBS=merging_probs is not None,
        PERMUTE_PROBS=permuted_probs is not None,
    )
    return output, unpermuted_probs

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["hidden_size"],
)
@triton.jit
def _unpermute_kernel(
    # pointers
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    merging_probs_ptr,
    permuted_probs_ptr,
    unpermuted_probs_ptr,
    # sizes
    num_tokens,
    num_experts,
    hidden_size,
    # strides
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_merging_probs_token,
    stride_merging_probs_expert,
    stride_permuted_probs_token,
    stride_unpermuted_probs_token,
    stride_unpermuted_probs_expert,
    # metas
    WITH_MERGING_PROBS: tl.constexpr,
    PERMUTE_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    data_type = input_ptr.dtype.element_ty
    compute_type = tl.float32

    pid = tl.program_id(0)
    current_start = 0
    while current_start < hidden_size:
        current_offset = current_start + tl.arange(0, BLOCK_SIZE)
        mask = current_offset < hidden_size
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=compute_type)
        for expert_idx in range(num_experts):
            src_row = tl.load(row_id_map_ptr + expert_idx * num_tokens + pid)
            if src_row != -1:
                input_off = src_row * stride_input_token + current_offset * stride_input_hidden
                inp = tl.load(input_ptr + input_off, mask=mask)
                inp = inp.to(compute_type)
                if WITH_MERGING_PROBS:

                    merging_prob_off = (
                        pid * stride_merging_probs_token + expert_idx * stride_merging_probs_expert
                    )
                    merging_prob = tl.load(merging_probs_ptr + merging_prob_off).to(compute_type)
                    inp *= merging_prob
                accumulator += inp
            if PERMUTE_PROBS:
                if current_start == 0:
                    unpermuted_prob_off = (
                        pid * stride_unpermuted_probs_token
                        + expert_idx * stride_unpermuted_probs_expert
                    )
                    if src_row != -1:
                        permuted_prob_off = src_row * stride_permuted_probs_token
                        prob = tl.load(permuted_probs_ptr + permuted_prob_off)
                        tl.store(unpermuted_probs_ptr + unpermuted_prob_off, prob)
                    else:
                        tl.store(unpermuted_probs_ptr + unpermuted_prob_off, 0.0)
        accumulator = accumulator.to(data_type)
        output_off = pid * stride_output_token + current_offset * stride_output_hidden
        tl.store(output_ptr + output_off, accumulator, mask=mask)
        current_start += BLOCK_SIZE

def triton_unpermute_with_mask_map_bwd_with_merging_probs(
    fwd_output_grad: torch.Tensor,
    row_id_map: torch.Tensor,
    fwd_input: torch.Tensor,
    merging_probs: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
):
    act_grad = torch.empty(
        (num_out_tokens, hidden_size), dtype=fwd_output_grad.dtype, device="cuda"
    )
    merging_probs_grad = torch.empty(
        (num_tokens, num_experts), dtype=merging_probs.dtype, device="cuda"
    )
    grid = (num_tokens,)
    _unpermute_bwd_with_merging_probs_kernel[grid](
        fwd_output_grad,
        act_grad,
        fwd_input,
        merging_probs,
        merging_probs_grad,
        row_id_map,
        num_tokens,
        num_experts,
        hidden_size,
        fwd_output_grad.stride(0),
        fwd_output_grad.stride(1),
        act_grad.stride(0),
        act_grad.stride(1),
        fwd_input.stride(0),
        fwd_input.stride(1),
        merging_probs.stride(0),
        merging_probs.stride(1),
        merging_probs_grad.stride(0),
        merging_probs_grad.stride(1),
    )
    return act_grad, merging_probs_grad

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["hidden_size"],
)
@triton.jit
def _unpermute_bwd_with_merging_probs_kernel(
    # pointers
    fwd_output_grad_ptr,
    fwd_input_grad_ptr,
    fwd_input_ptr,
    merging_probs_ptr,
    merging_probs_grad_ptr,
    row_id_map_ptr,
    # sizes
    num_tokens,
    num_experts,
    hidden_size,
    # strides
    stride_fwd_output_grad_token,
    stride_fwd_output_grad_hidden,
    stride_fwd_input_grad_token,
    stride_fwd_input_grad_hidden,
    stride_fwd_input_token,
    stride_fwd_input_hidden,
    stride_merging_probs_token,
    stride_merging_probs_expert,
    stride_merging_probs_grad_token,
    stride_merging_probs_grad_expert,
    # metas
    BLOCK_SIZE: tl.constexpr,
):
    data_type = fwd_output_grad_ptr.dtype.element_ty
    compute_type = tl.float32

    pid = tl.program_id(0)

    # add zero tensor
    zero_tensor = tl.zeros((1,), dtype=merging_probs_grad_ptr.dtype.element_ty)
    zero_val = tl.sum(zero_tensor).to(merging_probs_grad_ptr.dtype.element_ty)

    for expert_idx in range(num_experts):
        dst_row = tl.load(row_id_map_ptr + expert_idx * num_tokens + pid)
        if dst_row != -1:
            prob_grad_accum = tl.zeros((BLOCK_SIZE,), dtype=compute_type)
            current_start = 0
            while current_start < hidden_size:
                current_offset = current_start + tl.arange(0, BLOCK_SIZE)
                mask = current_offset < hidden_size
                input_off = (
                    pid * stride_fwd_output_grad_token
                    + current_offset * stride_fwd_output_grad_hidden
                )
                inp = tl.load(fwd_output_grad_ptr + input_off, mask=mask)
                inp = inp.to(compute_type)
                merging_prob_off = (
                    pid * stride_merging_probs_token + expert_idx * stride_merging_probs_expert
                )
                merging_prob = tl.load(merging_probs_ptr + merging_prob_off).to(compute_type)
                output = inp * merging_prob
                output = output.to(data_type)
                output_off = (
                    dst_row * stride_fwd_input_grad_token
                    + current_offset * stride_fwd_input_grad_hidden
                )
                tl.store(fwd_input_grad_ptr + output_off, output, mask=mask)

                fwd_input_off = (
                    dst_row * stride_fwd_input_token + current_offset * stride_fwd_input_hidden
                )
                fwd_input = tl.load(fwd_input_ptr + fwd_input_off, mask=mask)
                prob_grad_accum += fwd_input.to(compute_type) * inp
                current_start += BLOCK_SIZE
            probs_grad = tl.sum(prob_grad_accum).to(merging_probs_grad_ptr.dtype.element_ty)
            probs_grad_off = (
                pid * stride_merging_probs_grad_token
                + expert_idx * stride_merging_probs_grad_expert
            )
            tl.store(merging_probs_grad_ptr + probs_grad_off, probs_grad)
        else:
            probs_grad_off = (
                pid * stride_merging_probs_grad_token
                + expert_idx * stride_merging_probs_grad_expert
            )
            # Modify 0.0 -> zero_val
            tl.store(merging_probs_grad_ptr + probs_grad_off, zero_val)

def moe_permute(
    inp: torch.Tensor,
    routing_map: torch.Tensor,
    num_out_tokens: int = -1,
    max_token_num: int = -1,
    map_type: str = "mask",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Permute the tokens based on the routing_map. Token with the same index will be grouped together.
    Tokens with the same designated expert will be grouped together.
    The routing_map indicates which experts were selected by each token.

    Parameters
    ----------
    inp: torch.Tensor
        Input tensor of shape `[num_tokens, hidden_size]`, on which permutation will be applied.
    routing_map: torch.Tensor
        The token to expert mapping tensor.
        If map_type is 'mask', routing_map is of shape [num_tokens, num_experts] and dtype 'int32'.
        The values in it: 1 means the token is routed to this expert and 0 means not.
        If map_type is 'index', routing_map is of shape [num_tokens, topK] and dtype 'int32'.
        The values in it are the routed expert indices.
    num_out_tokens: int, default = -1
        The effective output token count, representing the number of tokens not dropped.
        By default, set to '-1', meaning no tokens are dropped.
    max_token_num: int, default = -1
        The maximum number of tokens, used for workspace allocation.
        By default, set to '-1', meaning the calculation of the size of workspace is
        automatically taken over by the operator.
    map_type: str, default = 'mask'
        Type of the routing map tensor.
        Options are: 'mask', 'index'.
        Refer to `routing_map` for more details.
    """
    if map_type == "index":
        assert False, "index type not support yet!"
        # return _moe_permute_index_map.apply(inp, routing_map, num_out_tokens, max_token_num)
    if map_type == "mask":
        output, row_id_map, _ = _moe_permute_mask_map.apply(inp, routing_map, num_out_tokens, None)
        return output, row_id_map
    raise ValueError("map_type should be one of 'mask' or 'index'")

class _moe_permute_mask_map(torch.autograd.Function):
    """functional Permute with mask router map"""

    @staticmethod
    def forward(
        ctx,
        inp: torch.Tensor,
        routing_map: torch.Tensor,
        num_out_tokens: int,
        probs: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # pylint: disable=missing-function-docstring
        if not inp.numel():
            ctx.probs = probs
            return inp, torch.tensor([], device=inp.device), torch.tensor([], device=inp.device)

        assert inp.is_cuda, "TransformerEngine needs CUDA."
        assert routing_map.is_cuda, "TransformerEngine needs CUDA."
        if probs is not None:
            assert probs.is_cuda, "TransformerEngine needs CUDA."

        assert inp.size(0) == routing_map.size(0), "Permute not possible"
        num_tokens, hidden_size = inp.size()
        num_experts = routing_map.size(1)
        assert (
            num_out_tokens is not None
        ), "num_out_tokens must be provided to the fused permute function."

        row_id_map = triton_make_row_id_map(routing_map, num_tokens, num_experts)

        output, permuted_probs = triton_permute_with_mask_map(
            inp,
            row_id_map,
            probs,
            num_tokens,
            num_experts,
            num_out_tokens,
            hidden_size,
        )

        ctx.save_for_backward(row_id_map)
        ctx.num_experts = num_experts
        ctx.num_tokens = num_tokens
        ctx.hidden_size = hidden_size
        return output, row_id_map, permuted_probs

    @staticmethod
    def backward(
        ctx,
        permuted_act_grad: torch.Tensor,
        _,
        permuted_probs_grad: torch.Tensor,
    ) -> Tuple[torch.Tensor, ...]:
        # pylint: disable=missing-function-docstring
        if not permuted_act_grad.numel():
            return permuted_act_grad, None, None, ctx.probs

        act_grad = None
        probs_grad = None
        if ctx.needs_input_grad[0]:
            (row_id_map,) = ctx.saved_tensors
            act_grad, probs_grad = triton_unpermute_with_mask_map(
                permuted_act_grad,
                row_id_map,
                None,
                permuted_probs_grad,
                ctx.num_tokens,
                ctx.num_experts,
                ctx.hidden_size,
            )
        if not ctx.needs_input_grad[3]:
            probs_grad = None
        return act_grad, None, None, probs_grad

def triton_make_row_id_map(
    routing_map: torch.Tensor,
    num_tokens: int,
    num_experts: int,
):
    # pylint: disable=missing-function-docstring
    row_id_map = torch.empty((num_experts, num_tokens), dtype=torch.int64, device="cuda")
    block_size = 256
    grid = (num_experts, triton.cdiv(num_tokens, block_size))
    workspace_tensor = torch.empty(grid, dtype=torch.int64, device="cuda")
    # block cumsum
    _row_id_map_pass_1_kernel[grid](
        routing_map,
        row_id_map,
        workspace_tensor,
        num_tokens,
        routing_map.stride(0),
        routing_map.stride(1),
        block_size,
    )
    # cumsum all and process the mask
    _row_id_map_pass_2_kernel[grid](
        row_id_map,
        workspace_tensor,
        num_tokens,
        triton.next_power_of_2(num_experts * triton.cdiv(num_tokens, block_size)),
        block_size,
    )
    return row_id_map

@triton.jit
def _row_id_map_pass_1_kernel(
    # pointers
    routing_map_ptr,
    row_id_map_ptr,
    workspace_ptr,
    # sizes
    num_tokens,
    # strides
    stride_routing_map_token,
    stride_routing_map_expert,
    # metas
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offset = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    expert_token_mask = tl.load(
        routing_map_ptr + pid_m * stride_routing_map_expert + offset * stride_routing_map_token,
        mask=(offset < num_tokens),
        other=0,
    ).to(tl.int64)
    row_id_within_token_block = tl.cumsum(expert_token_mask) * expert_token_mask
    tl.store(
        row_id_map_ptr + pid_m * num_tokens + offset,
        row_id_within_token_block,
        mask=offset < num_tokens,
    )
    n_tokens_per_block = tl.sum(expert_token_mask)
    tl.store(workspace_ptr + pid_m * tl.cdiv(num_tokens, BLOCK_SIZE) + pid_n, n_tokens_per_block)

@triton.jit
def _row_id_map_pass_2_kernel(
    # pointers
    row_id_map_ptr,
    workspace_ptr,
    # sizes
    num_tokens,
    # metas
    WORKSPACE_LOAD_WIDTH: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    chunk_idx = pid_m * tl.cdiv(num_tokens, BLOCK_SIZE) + pid_n
    offset = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    row_id_within_token_block = tl.load(
        row_id_map_ptr + pid_m * num_tokens + offset, mask=(offset < num_tokens), other=0
    )

    workspace_off = tl.arange(0, WORKSPACE_LOAD_WIDTH)
    n_tokens_per_chunk = tl.load(workspace_ptr + workspace_off, mask=workspace_off < chunk_idx)
    row_id = tl.where(
        row_id_within_token_block == 0,
        -1,
        row_id_within_token_block + tl.sum(n_tokens_per_chunk) - 1,
    )
    tl.store(
        row_id_map_ptr + pid_m * num_tokens + offset,
        row_id,
        mask=(offset < num_tokens),
    )

def triton_permute_with_mask_map(
    inp: torch.Tensor,
    row_id_map: torch.Tensor,
    probs: torch.Tensor,
    num_tokens: int,
    num_experts: int,
    num_out_tokens: int,
    hidden_size: int,
):
    # pylint: disable=missing-function-docstring
    output = torch.empty((num_out_tokens, hidden_size), dtype=inp.dtype, device="cuda")
    if probs is not None:
        permuted_probs = torch.empty((num_out_tokens,), dtype=probs.dtype, device="cuda")
    else:
        permuted_probs = None
    grid = (num_tokens,)
    _permute_kernel[grid](
        inp,
        output,
        row_id_map,
        probs,
        permuted_probs,
        num_tokens,
        num_experts,
        hidden_size,
        inp.stride(0),
        inp.stride(1),
        output.stride(0),
        output.stride(1),
        probs.stride(0) if probs is not None else None,
        probs.stride(1) if probs is not None else None,
        permuted_probs.stride(0) if permuted_probs is not None else None,
        PERMUTE_PROBS=probs is not None,
    )
    return output, permuted_probs

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["hidden_size"],
)
@triton.jit
def _permute_kernel(
    # pointers
    input_ptr,
    output_ptr,
    row_id_map_ptr,
    probs_ptr,
    permuted_probs_ptr,
    # sizes
    num_tokens,
    num_experts,
    hidden_size,
    # strides
    stride_input_token,
    stride_input_hidden,
    stride_output_token,
    stride_output_hidden,
    stride_probs_token,
    stride_probs_expert,
    stride_permuted_probs_token,
    # metas
    PERMUTE_PROBS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    cur_pos = 0
    while cur_pos < hidden_size:
        cur_off = cur_pos + tl.arange(0, BLOCK_SIZE)
        mask = cur_off < hidden_size
        input_off = pid * stride_input_token + cur_off * stride_input_hidden
        inp = tl.load(input_ptr + input_off, mask=mask)
        for expert_idx in range(num_experts):
            dst_row = tl.load(row_id_map_ptr + expert_idx * num_tokens + pid)
            if dst_row != -1:
                output_off = dst_row * stride_output_token + cur_off * stride_output_hidden
                tl.store(output_ptr + output_off, inp, mask=mask)
                if PERMUTE_PROBS:
                    if cur_pos == 0:
                        prob_off = pid * stride_probs_token + expert_idx * stride_probs_expert
                        prob = tl.load(probs_ptr + prob_off)
                        permuted_prob_off = dst_row * stride_permuted_probs_token
                        tl.store(permuted_probs_ptr + permuted_prob_off, prob)
        cur_pos += BLOCK_SIZE

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["expert_shard_size"],
)
@triton.jit
def _prepare_all_to_all_send_kernel(
    # Input tensors
    sharded_param_ptr,
    global_placement_ptr,
    # Output tensors
    send_buffer_ptr,
    # Configuration
    world_size,
    global_expert_num,
    local_expert_num,
    expert_shard_size,
    # Stride information
    stride_sharded_expert,
    stride_sharded_shard,
    stride_send_rank,
    stride_send_expert,
    stride_send_shard,
    # Meta parameters
    BLOCK_SIZE: tl.constexpr,
):
    """Prepare send data for all_to_all by reorganizing sharded expert params"""
    rank = tl.program_id(0)
    expert_idx = tl.program_id(1)
    block_idx = tl.program_id(2)
    block_base_offset = block_idx * DATA_BLOCK_SIZE
    
    # Load needed expert ID for current rank and expert
    needed_expert_idx = rank * local_expert_num + expert_idx
    if needed_expert_idx < world_size * local_expert_num:
        needed_expert = tl.load(global_placement_ptr + needed_expert_idx)
        
        # Process entire shard in blocks
        shard_offset = 0
        while shard_offset < DATA_BLOCK_SIZE:
            # Calculate current block range
            block_offset = block_base_offset + shard_offset + tl.arange(0, BLOCK_SIZE)
            mask = block_offset < expert_shard_size
            
            # Read data from sharded params
            src_offset = (needed_expert * stride_sharded_expert + 
                         block_offset * stride_sharded_shard)
            data = tl.load(sharded_param_ptr + src_offset, mask=mask)
            
            # Write to send buffer
            dst_offset = (rank * stride_send_rank + 
                         expert_idx * stride_send_expert +
                         block_offset * stride_send_shard)
            tl.store(send_buffer_ptr + dst_offset, data, mask=mask)
            
            # Move to next block
            shard_offset += BLOCK_SIZE

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["expert_shard_size"],
)
@triton.jit
def _process_all_to_all_recv_kernel(
    # Input tensors
    recv_buffer_ptr,
    # Output tensors
    padded_unsharded_ptr,
    # Configuration
    world_size,
    local_expert_num,
    expert_shard_size,
    # Stride information
    stride_recv_rank,
    stride_recv_expert,
    stride_recv_shard,
    stride_output_expert,
    stride_output_shard,
    # Meta parameters
    BLOCK_SIZE: tl.constexpr,
):
    """Process all_to_all received data and reorganize to padded_unsharded tensor"""
    expert_idx = tl.program_id(0)
    rank = tl.program_id(1)
    block_idx = tl.program_id(2)
    block_base_offset = block_idx * DATA_BLOCK_SIZE
    
    # Process entire shard in blocks
    shard_offset = 0
    while shard_offset < DATA_BLOCK_SIZE:
        # Calculate current block range
        block_offset = block_base_offset + shard_offset + tl.arange(0, BLOCK_SIZE)
        mask = block_offset < expert_shard_size
        
        # Read from receive buffer
        src_offset = (rank * stride_recv_rank + 
                     expert_idx * stride_recv_expert +
                     block_offset * stride_recv_shard)
        data = tl.load(recv_buffer_ptr + src_offset, mask=mask)
        
        # Write to corresponding column in output tensor
        dst_col_start = rank * expert_shard_size
        dst_offset = (expert_idx * stride_output_expert + 
                     (dst_col_start + block_offset) * stride_output_shard)
        tl.store(padded_unsharded_ptr + dst_offset, data, mask=mask)
        
        # Move to next block
        shard_offset += BLOCK_SIZE

def triton_all_to_all_forward(
    padded_unsharded_flat_param: torch.Tensor,
    sharded_flat_param: torch.Tensor,
    global_placement: torch.Tensor,
    world_size: int,
    local_expert_num: int,
    process_group,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Triton-optimized all_to_all forward pass
    
    Args:
        sharded_flat_param: Sharded flat params [global_expert_num, expert_shard_size]
        global_placement: Global expert placement [world_size, local_expert_num]
        world_size: World size
        local_expert_num: Number of local experts
        process_group: Process group
    
    Returns:
        padded_unsharded_flat_param: Padded unsharded params
        sharded_flat_param: Reshaped sharded params
    """
    global_expert_num, expert_shard_size = sharded_flat_param.shape
    device = sharded_flat_param.device
    dtype = sharded_flat_param.dtype
    
    # 1. Prepare send data using Triton kernel
    send_shape = (world_size, local_expert_num, expert_shard_size)
    send_buffer = torch.empty(send_shape, dtype=dtype, device=device)
    
    # Calculate grid  
    num_blocks = triton.cdiv(expert_shard_size, DATA_BLOCK_SIZE)
    grid = (world_size, local_expert_num, num_blocks)
    
    _prepare_all_to_all_send_kernel[grid](
        sharded_flat_param,
        global_placement.view(-1),
        send_buffer,
        world_size,
        global_expert_num,
        local_expert_num,
        expert_shard_size,
        # Stride information
        expert_shard_size,  # stride_sharded_expert
        1,                   # stride_sharded_shard
        local_expert_num * expert_shard_size,  # stride_send_rank
        expert_shard_size,   # stride_send_expert
        1,                   # stride_send_shard
    )
    
    # 2. Process received data using Triton kernel
    # padded_unsharded_flat_param = padded_unsharded_flat_param.view(local_expert_num, world_size * expert_shard_size)
    
    # Prepare tensor for megatron all_to_all (expects single tensor)
    send_tensor = send_buffer.contiguous().view(-1)
    
    # 3. Execute all_to_all communication using megatron
    recv_tensor = all_to_all(process_group, send_tensor, 
                            [local_expert_num * expert_shard_size] * world_size,
                            [local_expert_num * expert_shard_size] * world_size)
    
    # Reshape back to buffer format
    recv_buffer = recv_tensor.view(world_size, local_expert_num, expert_shard_size)
    
    grid = (local_expert_num, world_size, num_blocks)
    
    _process_all_to_all_recv_kernel[grid](
        recv_buffer,
        padded_unsharded_flat_param,
        world_size,
        local_expert_num,
        expert_shard_size,
        # Stride information
        local_expert_num * expert_shard_size,  # stride_recv_rank
        expert_shard_size,                     # stride_recv_expert
        1,                                     # stride_recv_shard
        world_size * expert_shard_size,       # stride_output_expert
        1,                                     # stride_output_shard
    )
    
    # Reshape output
    # padded_unsharded_flat_param = padded_unsharded_flat_param.view(-1)
    # sharded_flat_param = sharded_flat_param.view(-1)
    
    return padded_unsharded_flat_param, sharded_flat_param

@torch.no_grad()
def triton_optimized_all_to_all_expert_weights(
    padded_unsharded_flat_param: torch.Tensor,
    sharded_flat_param: torch.Tensor,
    global_placement: torch.Tensor,
    world_size: int,
    local_expert_num: int,
    process_group,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Complete Triton-optimized expert weights all_to_all operation.
    Maintains same API as original but uses Triton acceleration internally."""
    return triton_all_to_all_forward(
        padded_unsharded_flat_param, sharded_flat_param, global_placement, world_size, local_expert_num, process_group
    )

DATA_BLOCK_SIZE = 4096

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 64}),
        triton.Config({"BLOCK_SIZE": 128}),
        triton.Config({"BLOCK_SIZE": 256}),
        triton.Config({"BLOCK_SIZE": 512}),
        triton.Config({"BLOCK_SIZE": 1024}),
    ],
    key=["expert_shard_size"],
)
@triton.jit
def _process_grad_all_to_all_recv_kernel(
    # Input tensors
    recv_buffer_ptr,
    global_placement_ptr,
    # Output tensors
    new_sharded_grad_ptr,
    # Configuration
    world_size,
    global_expert_num,
    local_expert_num,
    expert_shard_size,
    # Stride information
    stride_recv_rank,
    stride_recv_expert,
    stride_recv_shard,
    stride_output_expert,
    stride_output_shard,
    # Meta parameters
    BLOCK_SIZE: tl.constexpr,
):
    """Process gradient all_to_all received data and accumulate using index_add"""
    expert_global_idx = tl.program_id(0)
    block_idx = tl.program_id(1)
    block_base_offset = block_idx * DATA_BLOCK_SIZE
    # Process entire shard in blocks
    shard_offset = 0
    while shard_offset < DATA_BLOCK_SIZE:
        # Calculate current block range
        block_offset = block_base_offset + shard_offset + tl.arange(0, BLOCK_SIZE)
        mask = block_offset < expert_shard_size
        
        # Initialize accumulator for this block
        accumulator = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
        
        # Loop through all ranks to accumulate gradients
        for rank in range(world_size):
            # Check if current expert is in this rank's placement and accumulate
            rank_start = rank * local_expert_num
            for local_idx in range(local_expert_num):
                placement_idx = rank_start + local_idx
                if placement_idx < world_size * local_expert_num:
                    needed_expert = tl.load(global_placement_ptr + placement_idx)
                    
                    if needed_expert == expert_global_idx:
                        # Read gradient from receive buffer
                        src_offset = (rank * stride_recv_rank + 
                                     local_idx * stride_recv_expert +
                                     block_offset * stride_recv_shard)
                        grad_data = tl.load(recv_buffer_ptr + src_offset, mask=mask)
                        accumulator += grad_data.to(tl.float32)
        
        # Store final accumulated result
        dst_offset = expert_global_idx * stride_output_expert + block_offset * stride_output_shard
        result = accumulator.to(new_sharded_grad_ptr.dtype.element_ty)
        tl.store(new_sharded_grad_ptr + dst_offset, result, mask=mask)
        
        # Move to next block
        shard_offset += BLOCK_SIZE

@triton.jit
def _smart_routing_kernel(
    # Input tensors
    tokens_per_expert_ptr,  # [tp_size, ep_size, num_global_experts]
    expert_locations_ptr,  # [origin_expert_num, max_locations] Expert replica locations, -1 for invalid
    # Output tensors
    output_ptr,  # [tp_size, ep_size, ep_size * num_local_experts]
    # Configuration
    tp_size: tl.constexpr,
    ep_size: tl.constexpr,
    num_global_experts: tl.constexpr,
    num_local_experts: tl.constexpr,
    max_locations: tl.constexpr,
    gpus_per_node: tl.constexpr,
    # Meta parameters
    BLOCK_SIZE: tl.constexpr,
):
    """
    Efficient smart routing kernel with intra-node priority and inter-node load balancing.
    """
    tp_idx = tl.program_id(0)
    src_ep_rank = tl.program_id(1)
    
    src_node = src_ep_rank // gpus_per_node
    
    # Load token distribution for current source rank
    tokens_offset = tp_idx * ep_size * num_global_experts + src_ep_rank * num_global_experts
    tokens_base = tokens_per_expert_ptr + tokens_offset
    output_base = output_ptr + tp_idx * ep_size * ep_size * num_local_experts + src_ep_rank * ep_size * num_local_experts
    
    # Process token allocation for each expert
    for expert_id in range(num_global_experts):
        tokens_for_expert = tl.load(tokens_base + expert_id)
        if tokens_for_expert != 0:
            expert_locations_base = expert_locations_ptr + expert_id * max_locations
            intra_count = 0
            inter_count = 0

            for loc_idx in range(max_locations):
                location = tl.load(expert_locations_base + loc_idx)
                if location >= 0:
                    target_node = location // gpus_per_node // num_local_experts
                    
                    if target_node == src_node:
                        intra_count += 1
                    else:
                        inter_count += 1
            
            remaining_tokens = tokens_for_expert

            if intra_count > 0:
                tokens_per_location = remaining_tokens // intra_count
                extra_tokens = remaining_tokens % intra_count
                
                assigned = 0
                for loc_idx in range(max_locations):
                    location = tl.load(expert_locations_base + loc_idx)
                    if location >= 0:
                        target_node = location // gpus_per_node // num_local_experts
                        
                        if target_node == src_node:
                            tokens_to_assign = tokens_per_location
                            if assigned < extra_tokens:
                                tokens_to_assign += 1
                            tl.atomic_add(output_base + location, tokens_to_assign)
                            assigned += 1
                remaining_tokens = 0
            
            if remaining_tokens > 0 and inter_count > 0:
                tokens_per_location = remaining_tokens // inter_count
                extra_tokens = remaining_tokens % inter_count
                
                assigned = 0
                for loc_idx in range(max_locations):
                    location = tl.load(expert_locations_base + loc_idx)
                    if location >= 0:
                        # target_node = location // gpus_per_node // num_local_experts
                        tokens_to_assign = tokens_per_location
                        if assigned < extra_tokens:
                            tokens_to_assign += 1
                        tl.atomic_add(output_base + location, tokens_to_assign)
                        assigned += 1

@triton.jit
def _token_assignment_kernel(
    # Input tensors
    routing_map_ptr,  # [token_num, origin_expert_num]
    probs_ptr,  # [token_num, origin_expert_num] 
    expert_locations_ptr,  # [origin_expert_num, max_locations] Expert replica locations, -1 for invalid
    copy_num_ptr,  # [num_global_experts]
    # Output tensors
    new_routing_map_ptr,  # [token_num, num_global_experts]
    new_probs_ptr,  # [token_num, num_global_experts]
    # Dimension parameters
    token_num: tl.constexpr,
    origin_expert_num: tl.constexpr,
    num_global_experts: tl.constexpr,
    max_locations: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Token assignment kernel using expert_locations tensor.
    expert_locations[i, j] stores the location of the j-th replica of expert i, -1 for invalid.
    """
    token_id = tl.program_id(0)
    
    if token_id >= token_num:
        return
    
    routing_base = routing_map_ptr + token_id * origin_expert_num
    probs_base = probs_ptr + token_id * origin_expert_num
    new_routing_base = new_routing_map_ptr + token_id * num_global_experts
    new_probs_base = new_probs_ptr + token_id * num_global_experts
    
    # Process all expert assignments for current token
    for expert_idx in range(origin_expert_num):
        is_routed = tl.load(routing_base + expert_idx)
        if is_routed:
            
            prob_val = tl.load(probs_base + expert_idx)
            
            # Try to find available location for this token
            expert_locations_base = expert_locations_ptr + expert_idx * max_locations
            
            allocated = 0
            for loc_idx in range(max_locations):
                if allocated == 0:
                    location = tl.load(expert_locations_base + loc_idx)            
                    # Check if location is valid
                    if location >= 0:
                        # Atomic allocation attempt
                        old_count = tl.atomic_add(copy_num_ptr + location, -1)
                        if old_count > 0:
                            # Allocation successful
                            tl.store(new_routing_base + location, 1)
                            tl.store(new_probs_base + location, prob_val)
                            allocated = 1
                        else:
                            # Restore counter, location is full
                            tl.atomic_add(copy_num_ptr + location, 1)

@triton.jit
def _gradient_mapping_kernel(
    # Input tensors
    grad_new_probs_ptr,         # [token_num, num_global_experts]
    new_routing_map_ptr,        # [token_num, num_global_experts]
    inverse_expert_map_ptr,     # [num_global_experts] - location to expert mapping
    # Output tensors
    grad_probs_ptr,             # [token_num, origin_expert_num]
    # Dimensions
    token_num: tl.constexpr,
    origin_expert_num: tl.constexpr,
    num_global_experts: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Map gradients from new_probs back to original probs using inverse expert mapping
    Each token processed in parallel
    """
    token_idx = tl.program_id(0)
    
    if token_idx >= token_num:
        return
    
    # Process each global expert location for this token
    for location in range(num_global_experts):
        # Get gradient from new_probs
        map_value = tl.load(new_routing_map_ptr + token_idx * num_global_experts + location)
        grad_value = tl.load(grad_new_probs_ptr + token_idx * num_global_experts + location)
        
        if map_value != 0:
            # Find which original expert this location maps to
            expert_idx = tl.load(inverse_expert_map_ptr + location)
            tl.store(grad_probs_ptr + token_idx * origin_expert_num + expert_idx, grad_value)

@torch.no_grad()
def smart_routing_map_gpu(
    num_global_tokens_per_expert: torch.Tensor,  # [tp_size, ep_size, num_global_experts]
    expert_locations: torch.Tensor,  # [origin_expert_num, capacity * local_expert] expert replication location mapping, -1 means invalid
    num_local_experts: int,
    gpus_per_node: int = 8
) -> torch.Tensor:
    """
    get_smart_routing_map function
    """
    tp_size, ep_size, num_global_experts = num_global_tokens_per_expert.shape
    _, max_locations = expert_locations.shape
    
    device = num_global_tokens_per_expert.device
    dtype = num_global_tokens_per_expert.dtype
    
    # output tensor
    output = torch.zeros(
        (tp_size, ep_size, ep_size * num_local_experts),
        dtype=dtype,
        device=device
    )
    
    tokens_tensor = num_global_tokens_per_expert.contiguous()
    expert_locations = expert_locations.contiguous()
    grid = (tp_size, ep_size)
    _smart_routing_kernel[grid](
        tokens_tensor,
        expert_locations,
        output,
        tp_size=tp_size,
        ep_size=ep_size,
        num_global_experts=num_global_experts,
        num_local_experts=num_local_experts,
        max_locations=max_locations,
        gpus_per_node=gpus_per_node,
        BLOCK_SIZE=1,
    )
    # torch.set_printoptions(threshold=100000000)
    # print(f"{expert_locations},{tokens_tensor},{output}")
    return output

def new_routing_map_vectorized_gpu(
    new_num_global_tokens_per_expert: torch.Tensor,  # [tp_size, ep_size, num_global_experts]
    expert_locations: torch.Tensor,  # [origin_expert_num, capacity * local_expert] expert replication location mapping, -1 means invalid
    routing_map: torch.Tensor,  # [token_num, origin_expert_num]
    probs: torch.Tensor,  # [token_num, origin_expert_num]
    tp_rank: int,
    ep_rank: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    get_new_routing_map
    """
    tp_size, ep_size, num_global_experts = new_num_global_tokens_per_expert.shape
    origin_expert_num = routing_map.size(1)
    token_num = routing_map.size(0)
    max_locations_per_expert = expert_locations.size(1)
    
    device = routing_map.device
    
    # output tensor
    new_routing_map = torch.zeros(
        (token_num, num_global_experts),
        dtype=routing_map.dtype,
        device=device
    )
    new_probs = torch.zeros(
        (token_num, num_global_experts),
        dtype=probs.dtype,
        device=device
    )
    
    copy_num = new_num_global_tokens_per_expert[tp_rank, ep_rank, :].clone()
    
    grid = (token_num,)
    _token_assignment_kernel[grid](
        routing_map.contiguous(),
        probs.contiguous(),
        expert_locations.contiguous(),
        copy_num.contiguous(),
        new_routing_map,
        new_probs,
        token_num=token_num,
        origin_expert_num=origin_expert_num,
        num_global_experts=num_global_experts,
        max_locations=max_locations_per_expert,
        BLOCK_SIZE=1,
    )
    return new_routing_map, new_probs

class NewRoutingMapWithGradients(torch.autograd.Function):
    """
    Routing Map with Gradients, calc probs using inverse_expert_map
    """
    
    @staticmethod
    def forward(ctx, 
                new_num_global_tokens_per_expert: torch.Tensor,
                expert_locations: torch.Tensor,
                inverse_expert_map: torch.Tensor,
                routing_map: torch.Tensor,
                probs: torch.Tensor,
                tp_rank: int,
                ep_rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
        
        ctx.tp_rank = tp_rank
        ctx.ep_rank = ep_rank

        tp_size, ep_size, num_global_experts = new_num_global_tokens_per_expert.shape
        origin_expert_num = routing_map.size(1)
        token_num = routing_map.size(0)
        max_locations_per_expert = expert_locations.size(1)
        
        device = routing_map.device
        
        new_routing_map = torch.zeros(
            (token_num, num_global_experts),
            dtype=routing_map.dtype,
            device=device
        )
        new_probs = torch.zeros(
            (token_num, num_global_experts),
            dtype=probs.dtype,
            device=device
        )
        
        copy_num = new_num_global_tokens_per_expert[tp_rank, ep_rank, :].clone()
        
        grid = (token_num,)
        _token_assignment_kernel[grid](
            routing_map.contiguous(),
            probs.contiguous(),
            expert_locations.contiguous(),
            copy_num.contiguous(),
            new_routing_map,
            new_probs,
            token_num=token_num,
            origin_expert_num=origin_expert_num,
            num_global_experts=num_global_experts,
            max_locations=max_locations_per_expert,
            BLOCK_SIZE=1,
        )

        ctx.save_for_backward(new_routing_map, inverse_expert_map)
        ctx.shape = token_num, origin_expert_num
        
        return new_routing_map, new_probs
    
    @staticmethod
    def backward(ctx, grad_new_routing_map, grad_new_probs):
        """
        Backward pass: map gradients from new_probs back to probs using inverse expert mapping
        """
        new_routing_map, inverse_expert_map = ctx.saved_tensors
        token_num, origin_expert_num = ctx.shape

        assert grad_new_probs is not None, "grad_new_probs is None"
        # Get dimensions
        num_global_experts = grad_new_probs.size(1)

        # Initialize probs gradients
        grad_probs = torch.zeros((token_num, origin_expert_num), device=grad_new_probs.device, dtype=grad_new_probs.dtype)
        
        # Use Triton kernel for gradient mapping
        grid = (token_num,)
        _gradient_mapping_kernel[grid](
            grad_new_probs.contiguous(),
            new_routing_map.contiguous(),
            inverse_expert_map.contiguous(),
            grad_probs,
            token_num=token_num,
            origin_expert_num=origin_expert_num,
            num_global_experts=num_global_experts,
            BLOCK_SIZE=1,
        )
        
        return None, None, None, None, grad_probs, None, None

def new_routing_map_with_gradients(
    new_num_global_tokens_per_expert: torch.Tensor,
    expert_locations: torch.Tensor,
    inverse_expert_map: torch.Tensor,
    routing_map: torch.Tensor,
    probs: torch.Tensor,
    tp_rank: int,
    ep_rank: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    return NewRoutingMapWithGradients.apply(
        new_num_global_tokens_per_expert,
        expert_locations, 
        inverse_expert_map, 
        routing_map,
        probs,
        tp_rank,
        ep_rank
    )

import moe_all_to_all_kernels
from megatron.core.parallel_state import get_global_memory_buffer

def cuda_optimized_all_to_all_expert_weights(
    padded_unsharded_flat_param: torch.Tensor,
    sharded_flat_param: torch.Tensor,
    global_placement: torch.Tensor,
    world_size: int,
    local_expert_num: int,
    global_expert_num: int,
    process_group,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CUDA-optimized expert weights all_to_all operation using our custom kernels.
    
    Args:
        padded_unsharded_flat_param: Output tensor for padded unsharded params
        sharded_flat_param: Input sharded flat params [global_expert_num, expert_shard_size]
        global_placement: Global expert placement [world_size, local_expert_num]
        world_size: World size
        local_expert_num: Number of local experts
        process_group: Process group
    
    Returns:
        padded_unsharded_flat_param: Padded unsharded params
        sharded_flat_param: Reshaped sharded params
    """
    
    expert_shard_size = sharded_flat_param.numel() // global_expert_num
    device = sharded_flat_param.device
    dtype = sharded_flat_param.dtype

    stream = torch.cuda.current_stream(device).cuda_stream

    moe_all_to_all_kernels.moe_nccl_forward(
        sharded_flat_param,
        global_placement,
        padded_unsharded_flat_param,
        world_size,
        global_expert_num,
        local_expert_num,
        expert_shard_size,
    )
    
    return padded_unsharded_flat_param, sharded_flat_param

def cuda_optimized_all_to_all_expert_grads(
    padded_unsharded_grad: torch.Tensor,
    new_sharded_grad: torch.Tensor,
    global_placement_cpu: torch.Tensor,
    global_placement: torch.Tensor,
    world_size: int,
    global_expert_num: int,
    local_expert_num: int,
    process_group,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """CUDA-optimized expert gradients all_to_all operation using our custom kernels.
    
    Args:
        padded_unsharded_grad: Padded unsharded gradients
        new_sharded_grad: Output tensor for new sharded gradients
        global_placement: Global expert placement
        world_size: World size
        global_expert_num: Number of global experts
        local_expert_num: Number of local experts
        process_group: Process group
    
    Returns:
        padded_unsharded_grad: Reshaped gradients
        new_sharded_grad: New sharded gradients
    """
    device = padded_unsharded_grad.device
    dtype = padded_unsharded_grad.dtype
    expert_shard_size = padded_unsharded_grad.numel() // (local_expert_num * world_size)

    recv_buffer = get_global_memory_buffer().get_tensor([world_size * local_expert_num * expert_shard_size], padded_unsharded_grad.dtype, "p2p")

    # Get current CUDA stream
    stream = torch.cuda.current_stream(device).cuda_stream
    
    # Call CUDA kernel for backward pass
    moe_all_to_all_kernels.moe_nccl_backward(
        padded_unsharded_grad,
        global_placement_cpu,
        recv_buffer,
        world_size,
        global_expert_num,
        local_expert_num,
        expert_shard_size,
    )

    num_blocks = triton.cdiv(expert_shard_size, DATA_BLOCK_SIZE)
    grid = (global_expert_num, num_blocks)
    
    _process_grad_all_to_all_recv_kernel[grid](
        recv_buffer,
        global_placement.view(-1),
        new_sharded_grad,
        world_size,
        global_expert_num,
        local_expert_num,
        expert_shard_size,
        # Stride information
        local_expert_num * expert_shard_size,  # stride_recv_rank
        expert_shard_size,                     # stride_recv_expert
        1,                                     # stride_recv_shard
        expert_shard_size,                     # stride_output_expert
        1,                                     # stride_output_shard
    )
    
    return padded_unsharded_grad, new_sharded_grad