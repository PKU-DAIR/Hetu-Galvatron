"""Standalone FusedRoPE extracted from NVIDIA TransformerEngine.

Drop-in replacement for galvatron's `apply_rotary_pos_emb` fused path: the
caller passes the FULL (un-sliced) `freqs` table plus the CP/SP world sizes and
ranks, and the kernel applies the CP zigzag + SP narrow internally — exactly
mirroring `get_pos_emb_on_this_cp_sp_rank` (for sbhd/bshd) and
`_apply_rotary_pos_emb_thd` (for thd).
"""

from typing import Tuple, Union

import torch

from . import _fused_rope_C as _C  # compiled extension built from csrc/binding.cu

__all__ = ["FusedRoPEFunc", "fused_apply_rotary_pos_emb"]


class FusedRoPEFunc(torch.autograd.Function):
    """Function for FusedRoPE with built-in CP + SP support.

    `freqs` is always the FULL (pre-CP, pre-SP) positional table. The kernel
    remaps each local seq position to the correct row in this table using
    cp_size/cp_rank/sp_size/sp_rank.

    Tensor layouts:
        sbhd / bshd: input shape [s_local, b, h, d] / [b, s_local, h, d];
            `freqs` shape [s_full, 1, 1, d2] with s_full >= s_local*cp_size*sp_size.
        thd: input shape [T_local, h, d] where
                T_local = (sum(seqlens) / cp_size) / sp_size;
            `cu_seqlens` is the ORIGINAL (pre-CP, pre-SP) cumulative sequence
            lengths as int32 with shape [b + 1]; `freqs` shape [max_s, 1, 1, d2].
    """

    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
        tensor_format: str = "sbhd",
        cu_seqlens: Union[torch.Tensor, None] = None,
        cp_size: int = 1,
        cp_rank: int = 0,
        sp_size: int = 1,
        sp_rank: int = 0,
    ) -> torch.Tensor:
        if freqs.dtype != torch.float32:
            freqs = freqs.float()
        if tensor_format == "sbhd":
            output = _C.fused_rope_forward(t, freqs, False, cp_size, cp_rank, sp_size, sp_rank)
        elif tensor_format == "bshd":
            output = _C.fused_rope_forward(
                t.transpose(0, 1), freqs, True, cp_size, cp_rank, sp_size, sp_rank
            ).transpose(0, 1)
        elif tensor_format == "thd":
            output = _C.fused_rope_thd_forward(
                t, cu_seqlens, freqs, cp_size, cp_rank, sp_size, sp_rank
            )
        else:
            raise ValueError(f"Unsupported tensor_format: {tensor_format}.")
        ctx.save_for_backward(freqs, cu_seqlens)
        ctx.tensor_format = tensor_format
        ctx.cp_size = cp_size
        ctx.cp_rank = cp_rank
        ctx.sp_size = sp_size
        ctx.sp_rank = sp_rank
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple[Union[torch.Tensor, None], ...]:
        freqs, cu_seqlens = ctx.saved_tensors
        if ctx.tensor_format == "sbhd":
            grad_input = _C.fused_rope_backward(
                grad_output, freqs, False, ctx.cp_size, ctx.cp_rank, ctx.sp_size, ctx.sp_rank
            )
        elif ctx.tensor_format == "bshd":
            grad_input = _C.fused_rope_backward(
                grad_output.transpose(0, 1), freqs, True, ctx.cp_size, ctx.cp_rank, ctx.sp_size,
                ctx.sp_rank,
            ).transpose(0, 1)
        elif ctx.tensor_format == "thd":
            grad_input = _C.fused_rope_thd_backward(
                grad_output, cu_seqlens, freqs, ctx.cp_size, ctx.cp_rank, ctx.sp_size, ctx.sp_rank
            )
        else:
            raise ValueError(f"Unsupported tensor_format: {ctx.tensor_format}.")
        # one None per non-tensor / non-grad-requiring forward arg
        return grad_input, None, None, None, None, None, None, None


def fused_apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    cu_seqlens: Union[torch.Tensor, None] = None,
    cp_size: int = 1,
    cp_rank: int = 0,
    sp_size: int = 1,
    sp_rank: int = 0,
) -> torch.Tensor:
    """Convenience wrapper that calls into `FusedRoPEFunc.apply`."""
    return FusedRoPEFunc.apply(
        t, freqs, tensor_format, cu_seqlens, cp_size, cp_rank, sp_size, sp_rank
    )
