"""Numerical sanity check for the standalone FusedRoPE extraction.

Compares the fused CUDA kernel against a pure-PyTorch reference matching the
TransformerEngine semantics: rotate the first `d2` channels of the last dim;
leave the remaining `d - d2` channels unchanged.
"""

import pytest
import torch

from galvatron.core.runtime.transformer.fused_rope import FusedRoPEFunc


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    # TE's convention: rotate the first half of the last dim against the second half.
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def reference_rope_sbhd(t: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    s = t.size(0)
    d2 = freqs.size(-1)
    cos = freqs[:s].cos().to(t.dtype)
    sin = freqs[:s].sin().to(t.dtype)
    t_rot, t_pass = t[..., :d2], t[..., d2:]
    out_rot = t_rot * cos + _rotate_half(t_rot) * sin
    return torch.cat((out_rot, t_pass), dim=-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("d,d2", [(64, 64), (128, 64)])
def test_sbhd_forward(dtype, d, d2):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    s, b, h = 32, 2, 4
    t = torch.randn(s, b, h, d, device="cuda", dtype=dtype, requires_grad=True)
    freqs = torch.randn(s, 1, 1, d2, device="cuda", dtype=torch.float32)

    out = FusedRoPEFunc.apply(t, freqs, "sbhd", None, 1, 0)
    ref = reference_rope_sbhd(t.detach(), freqs)

    atol = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(out, ref, atol=atol, rtol=atol)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_sbhd_backward(dtype):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    s, b, h, d, d2 = 16, 2, 4, 64, 64
    t = torch.randn(s, b, h, d, device="cuda", dtype=dtype, requires_grad=True)
    freqs = torch.randn(s, 1, 1, d2, device="cuda", dtype=torch.float32)
    t_ref = t.detach().clone().requires_grad_(True)

    out = FusedRoPEFunc.apply(t, freqs, "sbhd", None, 1, 0)
    out.sum().backward()

    ref = reference_rope_sbhd(t_ref, freqs)
    ref.sum().backward()

    atol = {torch.float32: 1e-5, torch.float16: 2e-3, torch.bfloat16: 1e-2}[dtype]
    torch.testing.assert_close(t.grad, t_ref.grad, atol=atol, rtol=atol)


def test_bshd_matches_sbhd():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(0)
    s, b, h, d = 16, 2, 4, 64
    t_sbhd = torch.randn(s, b, h, d, device="cuda", dtype=torch.float32)
    freqs = torch.randn(s, 1, 1, d, device="cuda", dtype=torch.float32)
    t_bshd = t_sbhd.transpose(0, 1).contiguous()

    out_sbhd = FusedRoPEFunc.apply(t_sbhd, freqs, "sbhd", None, 1, 0)
    out_bshd = FusedRoPEFunc.apply(t_bshd, freqs, "bshd", None, 1, 0)

    torch.testing.assert_close(out_sbhd, out_bshd.transpose(0, 1))


# ----------------------------------------------------------------------------
# SBHD/BSHD + CP + SP — mirrors galvatron's `get_pos_emb_on_this_cp_sp_rank`.
# ----------------------------------------------------------------------------


def _galvatron_get_pos_emb_on_this_cp_rank(pos_emb, seq_dim, cp_size, cp_rank):
    if cp_size == 1:
        return pos_emb
    cp_idx = torch.tensor([cp_rank, 2 * cp_size - cp_rank - 1], device=pos_emb.device)
    pos_emb = pos_emb.view(
        *pos_emb.shape[:seq_dim], 2 * cp_size, -1, *pos_emb.shape[(seq_dim + 1):]
    )
    pos_emb = pos_emb.index_select(seq_dim, cp_idx)
    pos_emb = pos_emb.view(*pos_emb.shape[:seq_dim], -1, *pos_emb.shape[(seq_dim + 2):])
    return pos_emb


def _galvatron_get_pos_emb_on_this_cp_sp_rank(
    pos_emb, seq_dim, cp_size, cp_rank, sp_size, sp_rank
):
    if cp_size > 1:
        pos_emb = _galvatron_get_pos_emb_on_this_cp_rank(pos_emb, seq_dim, cp_size, cp_rank)
    if sp_size > 1:
        seq_len = pos_emb.shape[seq_dim]
        sp_seq_len = seq_len // sp_size
        pos_emb = pos_emb.narrow(seq_dim, sp_rank * sp_seq_len, sp_seq_len).contiguous()
    return pos_emb


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("cp_size,sp_size", [(1, 1), (2, 1), (1, 2), (2, 2), (4, 2)])
def test_sbhd_cp_sp_forward(dtype, cp_size, sp_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    s_full, b, h, d = 32, 2, 4, 64
    assert s_full % (cp_size * sp_size) == 0
    s_local = s_full // cp_size // sp_size
    full_freqs = torch.randn(s_full, 1, 1, d, device="cuda", dtype=torch.float32)

    atol = {torch.float32: 1e-5, torch.bfloat16: 2e-2}[dtype]
    for cp_rank in range(cp_size):
        for sp_rank in range(sp_size):
            torch.manual_seed(cp_rank * 100 + sp_rank)
            t = torch.randn(s_local, b, h, d, device="cuda", dtype=dtype, requires_grad=True)

            out = FusedRoPEFunc.apply(
                t, full_freqs, "sbhd", None, cp_size, cp_rank, sp_size, sp_rank
            )
            sliced = _galvatron_get_pos_emb_on_this_cp_sp_rank(
                full_freqs, 0, cp_size, cp_rank, sp_size, sp_rank
            )
            ref = reference_rope_sbhd(t.detach(), sliced)
            torch.testing.assert_close(
                out, ref, atol=atol, rtol=atol,
                msg=f"sbhd cp={cp_size}/{cp_rank} sp={sp_size}/{sp_rank} dtype={dtype}",
            )


@pytest.mark.parametrize("cp_size,sp_size", [(1, 1), (2, 2)])
def test_sbhd_cp_sp_backward(cp_size, sp_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    s_full, b, h, d = 16, 2, 4, 64
    s_local = s_full // cp_size // sp_size
    full_freqs = torch.randn(s_full, 1, 1, d, device="cuda", dtype=torch.float32)

    for cp_rank in range(cp_size):
        for sp_rank in range(sp_size):
            torch.manual_seed(cp_rank * 100 + sp_rank)
            t_a = torch.randn(s_local, b, h, d, device="cuda", dtype=torch.float32,
                              requires_grad=True)
            t_b = t_a.detach().clone().requires_grad_(True)

            FusedRoPEFunc.apply(
                t_a, full_freqs, "sbhd", None, cp_size, cp_rank, sp_size, sp_rank
            ).sum().backward()
            sliced = _galvatron_get_pos_emb_on_this_cp_sp_rank(
                full_freqs, 0, cp_size, cp_rank, sp_size, sp_rank
            )
            reference_rope_sbhd(t_b, sliced).sum().backward()

            torch.testing.assert_close(t_a.grad, t_b.grad, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("cp_size,sp_size", [(1, 1), (2, 2)])
def test_bshd_cp_sp_forward(cp_size, sp_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    s_full, b, h, d = 16, 2, 4, 64
    s_local = s_full // cp_size // sp_size
    full_freqs = torch.randn(s_full, 1, 1, d, device="cuda", dtype=torch.float32)

    for cp_rank in range(cp_size):
        for sp_rank in range(sp_size):
            torch.manual_seed(cp_rank * 100 + sp_rank)
            t_sbhd = torch.randn(s_local, b, h, d, device="cuda", dtype=torch.float32)
            t_bshd = t_sbhd.transpose(0, 1).contiguous()

            out_sbhd = FusedRoPEFunc.apply(
                t_sbhd, full_freqs, "sbhd", None, cp_size, cp_rank, sp_size, sp_rank
            )
            out_bshd = FusedRoPEFunc.apply(
                t_bshd, full_freqs, "bshd", None, cp_size, cp_rank, sp_size, sp_rank
            )
            torch.testing.assert_close(out_sbhd, out_bshd.transpose(0, 1))


# ----------------------------------------------------------------------------
# THD + CP + SP — mirrors galvatron's `_apply_rotary_pos_emb_thd` reference.
# ----------------------------------------------------------------------------


def _get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, cp_seq_len, freqs):
    """Galvatron-style per-sample CP zigzag slice of freqs."""
    if cp_size > 1:
        cp_seg = cp_seq_len // 2
        full_seqlen = cp_size * cp_seq_len
        return torch.cat(
            [
                freqs[cp_rank * cp_seg : (cp_rank + 1) * cp_seg],
                freqs[full_seqlen - (cp_rank + 1) * cp_seg : full_seqlen - cp_rank * cp_seg],
            ]
        )
    return freqs[:cp_seq_len]


def reference_thd(t_local, cu_seqlens, freqs, cp_size, cp_rank, sp_size, sp_rank):
    """Pure-PyTorch reference matching galvatron `_apply_rotary_pos_emb_thd`."""
    cp_seqlens = ((cu_seqlens[1:] - cu_seqlens[:-1]) // cp_size).tolist()
    freqs_packed = torch.cat(
        [_get_thd_freqs_on_this_cp_rank(cp_rank, cp_size, L, freqs) for L in cp_seqlens], dim=0
    )
    if sp_size > 1:
        sp_seq_len = freqs_packed.size(0) // sp_size
        freqs_packed = freqs_packed.narrow(0, sp_rank * sp_seq_len, sp_seq_len).contiguous()

    # bshd-style RoPE on [T_local, 1, h, d] with freqs [T_local, 1, 1, d2].
    rot_dim = freqs_packed.size(-1)
    t = t_local.unsqueeze(1)
    t_rot, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    cos_ = freqs_packed.cos().to(t.dtype)
    sin_ = freqs_packed.sin().to(t.dtype)
    out_rot = t_rot * cos_ + _rotate_half(t_rot) * sin_
    return torch.cat((out_rot, t_pass), dim=-1).squeeze(1)


def _build_thd_inputs(sample_lens, cp_size, cp_rank, sp_size, sp_rank, h, d, dtype, device):
    """Construct the per-rank local input `t_local` plus the pre-CP/SP cu_seqlens.

    The full input is generated once; for the current (cp_rank, sp_rank) we
    extract the slice exactly the way the data path would feed the kernel.
    """
    torch.manual_seed(0)
    cu_full = torch.zeros(len(sample_lens) + 1, dtype=torch.int32, device=device)
    cu_full[1:] = torch.tensor(sample_lens, dtype=torch.int32, device=device).cumsum(0)

    # Full input across all CP/SP ranks: [T_total, h, d]
    T_total = int(cu_full[-1].item())
    t_full = torch.randn(T_total, h, d, device=device, dtype=dtype)

    # CP zigzag slice per sample → per-CP-rank packed stream
    cp_chunks = []
    for b, L_full in enumerate(sample_lens):
        sample = t_full[int(cu_full[b]) : int(cu_full[b + 1])]  # [L_full, h, d]
        L_local = L_full // cp_size
        if cp_size > 1:
            cp_seg = L_local // 2
            front = sample[cp_rank * cp_seg : (cp_rank + 1) * cp_seg]
            back = sample[L_full - (cp_rank + 1) * cp_seg : L_full - cp_rank * cp_seg]
            cp_chunks.append(torch.cat([front, back], dim=0))
        else:
            cp_chunks.append(sample)
    t_cp = torch.cat(cp_chunks, dim=0)  # [T_cp, h, d]

    # SP narrow on the packed stream
    if sp_size > 1:
        sp_len = t_cp.size(0) // sp_size
        t_local = t_cp.narrow(0, sp_rank * sp_len, sp_len).contiguous()
    else:
        t_local = t_cp.contiguous()
    return t_local, cu_full


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize(
    "cp_size,sp_size",
    [(1, 1), (2, 1), (1, 2), (2, 2), (4, 2)],
)
def test_thd_cp_sp_forward(dtype, cp_size, sp_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    # Sample lengths must be divisible by 2 * cp_size (CP zigzag) and the resulting
    # T_cp must be divisible by sp_size.
    sample_lens = [16, 32, 16]  # T_total = 64
    h, d, d2 = 4, 64, 64
    max_s = max(sample_lens)
    freqs = torch.randn(max_s, 1, 1, d2, device="cuda", dtype=torch.float32)

    atol = {torch.float32: 1e-5, torch.bfloat16: 2e-2}[dtype]
    for cp_rank in range(cp_size):
        for sp_rank in range(sp_size):
            t_local, cu_full = _build_thd_inputs(
                sample_lens, cp_size, cp_rank, sp_size, sp_rank, h, d, dtype, "cuda"
            )
            out = FusedRoPEFunc.apply(
                t_local, freqs, "thd", cu_full, cp_size, cp_rank, sp_size, sp_rank
            )
            ref = reference_thd(t_local, cu_full, freqs, cp_size, cp_rank, sp_size, sp_rank)
            torch.testing.assert_close(
                out, ref, atol=atol, rtol=atol,
                msg=f"cp={cp_size}/{cp_rank} sp={sp_size}/{sp_rank} dtype={dtype}",
            )


@pytest.mark.parametrize("cp_size,sp_size", [(1, 1), (2, 2)])
def test_thd_cp_sp_backward(cp_size, sp_size):
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    sample_lens = [16, 32]
    h, d, d2 = 4, 64, 64
    max_s = max(sample_lens)
    freqs = torch.randn(max_s, 1, 1, d2, device="cuda", dtype=torch.float32)

    for cp_rank in range(cp_size):
        for sp_rank in range(sp_size):
            t_local, cu_full = _build_thd_inputs(
                sample_lens, cp_size, cp_rank, sp_size, sp_rank, h, d, torch.float32, "cuda"
            )
            t_a = t_local.detach().clone().requires_grad_(True)
            t_b = t_local.detach().clone().requires_grad_(True)

            FusedRoPEFunc.apply(
                t_a, freqs, "thd", cu_full, cp_size, cp_rank, sp_size, sp_rank
            ).sum().backward()
            reference_thd(t_b, cu_full, freqs, cp_size, cp_rank, sp_size, sp_rank).sum().backward()

            torch.testing.assert_close(t_a.grad, t_b.grad, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
