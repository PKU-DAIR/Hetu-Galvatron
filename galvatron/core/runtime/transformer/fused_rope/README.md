# Fused RoPE (in-tree CUDA kernel)

Galvatron's in-tree fused Rotary Positional Embedding (RoPE) CUDA op,
extracted from
[NVIDIA TransformerEngine](https://github.com/NVIDIA/TransformerEngine)
(`transformer_engine/common/fused_rope/fused_rope.cu` and
`transformer_engine/pytorch/csrc/extensions/apply_rope.cu`).

TE-internal abstractions (`NVTETensor` opaque handles, the internal `Tensor`
struct, `TRANSFORMER_ENGINE_TYPE_SWITCH_INPUT`, the `NVTE_*` logging macros)
have been stripped out. The kernel is now driven via raw pointers and
PyTorch's `AT_DISPATCH_FLOATING_TYPES_AND2`.

Capabilities added on top of upstream TE:

- **CP zigzag + SP narrow are inlined into the kernel.** The caller passes
  the **full, un-sliced** `freqs` table along with `cp_size / cp_rank /
  sp_size / sp_rank`; the kernel performs the index remap that
  `get_pos_emb_on_this_cp_sp_rank` (sbhd/bshd) and `_apply_rotary_pos_emb_thd`
  (thd) do in Python, removing the host-side `index_select` / `narrow`.

## Layout

C/CUDA sources live under the repo-level `csrc/`. The Python wrapper, the
build script and the compiled artifact all live in this directory:

```
csrc/fused_rope/
  binding.cu            # PyTorch glue, AT_DISPATCH, PYBIND11_MODULE
  fused_rope.cuh        # device kernels + templated launchers (header-only)

galvatron/core/runtime/transformer/fused_rope/
  __init__.py           # FusedRoPEFunc / fused_apply_rotary_pos_emb
  setup.py              # standalone build script (NOT wired into the top-level setup.py)
  _fused_rope_C*.so     # build artifact (produced on demand; gitignored)
  README.md             # this file
```

## On-demand build

The top-level `setup.py` does **not** compile this extension. Build it by
hand when you need it:

```bash
cd galvatron/core/runtime/transformer/fused_rope
python setup.py build_ext --inplace
```

`--inplace` drops `_fused_rope_C*.so` next to `__init__.py`, where the
relative `from . import _fused_rope_C` will pick it up.

Override the GPU architecture list if needed (otherwise PyTorch's default
list is used):

```bash
TORCH_CUDA_ARCH_LIST="8.0;9.0" python setup.py build_ext --inplace
```

> **Importing without building is safe.** The `try / except ImportError` at
> the top of `rope_utils.py` sets `_GalvatronFusedRoPEFunc = None` when the
> extension is missing. You only hit an assert when
> `config.apply_rope_fusion=True` actually exercises the fused path.

## Testing

```bash
pytest tests/kernels/test_fused_rope.py -v
```

Coverage:

- `sbhd` / `bshd` / `thd` forward and backward;
- CP zigzag + SP narrow across `(cp_size, sp_size)` combinations, validated
  against a pure-PyTorch reference (`reference_rope_sbhd` / `reference_thd`)
  that mirrors `_apply_rotary_pos_emb_bshd` and `_apply_rotary_pos_emb_thd`
  from `rope_utils.py` line for line.

The bfloat16 tolerance is `atol=rtol=2e-2`. The kernel accumulates in fp32
and only rounds once at the final store, while the Python reference rounds
on each bf16 op, so the two paths can differ by one bf16 ULP (≈ 1/64 ≈
1.5625e-2 in the `[2, 4)` range) — a tolerance strictly greater than one
ULP is required.

## Usage

```python
import torch
from galvatron.core.runtime.transformer.fused_rope import FusedRoPEFunc

s, b, h, d = 2048, 4, 16, 128
t = torch.randn(s, b, h, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
freqs = torch.randn(s, 1, 1, d, device="cuda", dtype=torch.float32)

# tensor_format: "sbhd" | "bshd" | "thd"
# The trailing four args are CP and SP world sizes / ranks.
out = FusedRoPEFunc.apply(t, freqs, "sbhd", None, 1, 0, 1, 0)
out.sum().backward()
```

For `thd` (packed sequences): pass `t` with shape `(T, h, d)` and an int32
`cu_seqlens` tensor of length `batch + 1` holding the **original**
(pre-CP/SP) cumulative sequence lengths.

## Integration with `apply_rotary_pos_emb`

`rope_utils.py` routes `config.apply_rope_fusion=True` into this op (see
`rope_utils.py:233` onward). The fused path currently does **not** support
the following — they assert at call time:

| Feature | Status |
|---|---|
| `config.rotary_interleaved=True` (interleaved rotation) | assert |
| `config.multi_latent_attention=True` (MLA deinterleave) | assert |
| `mscale != 1.0` (YaRN-style scaling) | assert |

To add support, extend the kernel parameters in
`csrc/fused_rope/fused_rope.cuh` / `binding.cu`, plumb the new knobs through
`FusedRoPEFunc.forward` in `__init__.py`, and add coverage in
`tests/kernels/test_fused_rope.py`.
