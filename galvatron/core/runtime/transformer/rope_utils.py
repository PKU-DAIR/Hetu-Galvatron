# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import logging

import torch
from torch import Tensor

from galvatron.core.runtime.args_schema import GalvatronModelArgs
from galvatron.core.runtime.utils.utils import is_te_min_version
from galvatron.core.runtime.parallel_state import get_parallel_world_size, get_parallel_rank

logger = logging.getLogger(__name__)

try:
    from flash_attn.layers.rotary import apply_rotary_emb as apply_rotary_emb_flash
except ImportError:
    apply_rotary_emb_flash = None


# Galvatron's in-tree fused RoPE CUDA kernel. Build on demand via:
#   cd galvatron/core/runtime/transformer/fused_rope && python setup.py build_ext --inplace
try:
    from galvatron.core.runtime.transformer.fused_rope import FusedRoPEFunc as _GalvatronFusedRoPEFunc
except ImportError:
    _GalvatronFusedRoPEFunc = None


__all__ = ['apply_rotary_emb_flash']

def get_pos_emb_on_this_cp_rank(
    pos_emb: Tensor, seq_dim: int, cp_group: Optional[torch.distributed.ProcessGroup] = None
) -> Tensor:
    """Get the position embedding on the current context parallel rank.

    Args:
        pos_emb (Tensor): Positional embedding tensor
        seq_dim (int): Sequence dimension
    """
    cp_size = 1 if cp_group is None else get_parallel_world_size(cp_group)
    cp_rank = 0 if cp_group is None else get_parallel_rank(cp_group)
    cp_idx = torch.tensor(
        [cp_rank, (2 * cp_size - cp_rank - 1)], device="cpu", pin_memory=True
    ).cuda(non_blocking=True)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * cp_size, -1, *pos_emb.shape[(seq_dim + 1) :]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2) :])
    return pos_emb


def get_pos_emb_on_this_cp_sp_rank(
    pos_emb: Tensor,
    seq_dim: int,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    sp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tensor:
    """Select this rank's CP zigzag shard and sequence-parallel shard."""
    cp_size = 1 if cp_group is None else get_parallel_world_size(cp_group)
    if cp_size > 1:
        pos_emb = get_pos_emb_on_this_cp_rank(pos_emb, seq_dim, cp_group)

    sp_size = 1 if sp_group is None else get_parallel_world_size(sp_group)
    if sp_size > 1:
        sp_rank = 0 if sp_group is None else get_parallel_rank(sp_group)
        seq_len = pos_emb.shape[seq_dim]
        sp_seq_len = seq_len // sp_size
        sp_start = sp_rank * sp_seq_len
        pos_emb = pos_emb.narrow(seq_dim, sp_start, sp_seq_len).contiguous()
    return pos_emb


def _rotate_half(x: Tensor, rotary_interleaved: bool) -> Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x (Tensor): Input tensor

    Returns:
        Tensor: Tensor rotated half
    """
    if not rotary_interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)
    else:
        x1 = x[:, :, :, ::2]
        x2 = x[:, :, :, 1::2]
        x_new = torch.stack((-x2, x1), dim=-1)
        return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_bshd(
    t: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
) -> Tensor:
    """Apply rotary positional embedding to input tensor T.

    check https://kexue.fm/archives/8265 for detailed formulas

    Args:
        t (Tensor): Input tensor T is of shape [seq_length, ... , dim]
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [seq_length, ..., dim]

    Returns:
        Tensor: The input tensor after applying RoPE
    """
    rot_dim = freqs.shape[-1]

    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    if multi_latent_attention:
        x1 = t[..., 0::2]
        x2 = t[..., 1::2]
        t = torch.cat((x1, x2), dim=-1)

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    cos_ = (torch.cos(freqs) * mscale).to(t.dtype)
    sin_ = (torch.sin(freqs) * mscale).to(t.dtype)

    t = (t * cos_) + (_rotate_half(t, rotary_interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def _get_thd_freqs_on_this_cp_rank(
    cp_rank: int,
    cp_size: int,
    cp_seq_len: int,
    freqs: Tensor,
    offset: int = 0,
) -> Tensor:
    if cp_size > 1:
        cp_seg = cp_seq_len // 2
        full_seqlen = cp_size * cp_seq_len
        return torch.cat(
            [
                freqs[offset + cp_rank * cp_seg : offset + (cp_rank + 1) * cp_seg],
                freqs[
                    offset
                    + full_seqlen
                    - (cp_rank + 1) * cp_seg : offset
                    + full_seqlen
                    - cp_rank * cp_seg
                ],
            ]
        )
    return freqs[offset : offset + cp_seq_len]


def _apply_rotary_pos_emb_thd(
    t: Tensor,
    cu_seqlens: Tensor,
    freqs: Tensor,
    rotary_interleaved: bool = False,
    multi_latent_attention: bool = False,
    mscale: float = 1.0,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    sp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tensor:
    """A baseline implementation of applying RoPE for `thd` format.

    Args:
        t (Tensor): Input tensor T is of shape [t, h, d]
        cu_seqlens(Tensor):  Cumulative sum of sequence lengths in a batch for `t`,
        with shape [b + 1] and dtype torch.int32.
        freqs (Tensor): Rotary Positional embedding tensor freq is of shape [max_s, 1, 1, d]

    Returns:
        Tensor: Shape [t, h, d]. The input tensor after applying RoPE.
    """

    cp_size = 1 if cp_group is None else get_parallel_world_size(cp_group)
    cp_rank = 0 if cp_group is None else get_parallel_rank(cp_group)
    sp_size = 1 if sp_group is None else get_parallel_world_size(sp_group)
    sp_rank = 0 if sp_group is None else get_parallel_rank(sp_group)
    cp_seqlens = ((cu_seqlens[1:] - cu_seqlens[:-1]) // cp_size).tolist()

    freqs_packed = torch.cat(
        [
            _get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, cp_seq_len, freqs)
            for cp_seq_len in cp_seqlens
        ],
        dim=0,
    )

    if sp_size > 1:
        sp_seq_len = freqs_packed.size(0) // sp_size
        freqs_packed = freqs_packed.narrow(0, sp_rank * sp_seq_len, sp_seq_len).contiguous()

    return _apply_rotary_pos_emb_bshd(
        t.unsqueeze(1), # [t, h, d] -> [t, 1, h, d]
        freqs_packed,
        rotary_interleaved=rotary_interleaved,
        multi_latent_attention=multi_latent_attention,
        mscale=mscale,
    ).squeeze(1) # [t, 1, h, d] -> [t, h, d]

# TODO: support fine grained CP group size
def apply_rotary_pos_emb(
    t: Tensor,
    freqs: Tensor,
    config: GalvatronModelArgs,
    cu_seqlens: Optional[Tensor] = None,
    mscale: float = 1.0,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    sp_group: Optional[torch.distributed.ProcessGroup] = None,
):
    """
    Reroute to the appropriate apply_rotary_pos_emb function depending on
    fused/unfused kernels, or bshd / thd tensor format
    """

    if config.apply_rope_fusion:
        assert _GalvatronFusedRoPEFunc is not None, (
            "config.apply_rope_fusion=True but the in-tree fused RoPE kernel is not built. "
            "Build it via: cd galvatron/core/runtime/transformer/fused_rope && "
            "python setup.py build_ext --inplace"
        )
        assert not config.rotary_interleaved, "fused RoPE: rotary_interleaved not supported"
        assert not config.multi_latent_attention, "fused RoPE: multi_latent_attention not supported"
        assert mscale == 1.0, "fused RoPE: mscale != 1.0 not supported"

        cp_size = 1 if cp_group is None else get_parallel_world_size(cp_group)
        cp_rank = 0 if cp_group is None else get_parallel_rank(cp_group)
        sp_size = 1 if sp_group is None else get_parallel_world_size(sp_group)
        sp_rank = 0 if sp_group is None else get_parallel_rank(sp_group)

        tensor_format = "thd" if cu_seqlens is not None else "sbhd"
        return _GalvatronFusedRoPEFunc.apply(
            t, freqs, tensor_format, cu_seqlens, cp_size, cp_rank, sp_size, sp_rank
        )
    else:
        if cu_seqlens is None:
            freqs = get_pos_emb_on_this_cp_sp_rank(freqs, 0, cp_group, sp_group)
            return _apply_rotary_pos_emb_bshd(
                t,
                freqs,
                rotary_interleaved=config.rotary_interleaved,
                multi_latent_attention=config.multi_latent_attention,
                mscale=mscale,
            )
        else:
            return _apply_rotary_pos_emb_thd(
                t,
                cu_seqlens,
                freqs,
                rotary_interleaved=config.rotary_interleaved,
                multi_latent_attention=config.multi_latent_attention,
                mscale=mscale,
                cp_group=cp_group,
                sp_group=sp_group,
            )


def apply_rotary_pos_emb_with_cos_sin(
    t: Tensor,
    cos: Tensor,
    sin: Tensor,
    rotary_interleaved: bool = False,
    cp_group: Optional[torch.distributed.ProcessGroup] = None,
    sp_group: Optional[torch.distributed.ProcessGroup] = None,
) -> Tensor:
    """
    This function applies rotary positional embedding to the target tensor t
    using precomputed cos and sin of size (seq_len, d_rot / 2)
    """
    cos = cos.to(t.dtype)
    sin = sin.to(t.dtype)

    if apply_rotary_emb_flash is None:
        # Combine cos and sin into freqs
        freqs = torch.stack([cos, sin], dim=-1).flatten(start_dim=-2)

        # Expand freqs to match t's shape
        while freqs.dim() < t.dim():
            freqs = freqs.unsqueeze(1)
        freqs = freqs.expand(t.shape[:-1] + (-1,))

        y = _apply_rotary_pos_emb_bshd(
            t,
            freqs,
            rotary_interleaved=rotary_interleaved,
            multi_latent_attention=False,
            mscale=1.0,
        )
    else:
        # Use Flash Attention's optimized kernel for rotary embedding
        t = t.permute(1, 0, 2, 3)
        y = apply_rotary_emb_flash(t, cos, sin, rotary_interleaved)
        y = y.permute(1, 0, 2, 3)

    return y
