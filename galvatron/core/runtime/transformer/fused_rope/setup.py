"""On-demand build for the FusedRoPE CUDA extension.

Not invoked by the top-level galvatron `setup.py`. Build only when needed:

    cd galvatron/core/runtime/transformer/fused_rope
    python setup.py build_ext --inplace

The resulting `_fused_rope_C*.so` lands next to this file, so
`from galvatron.core.runtime.transformer.fused_rope import FusedRoPEFunc`
picks it up via the relative `from . import _fused_rope_C` in __init__.py.

Override architecture targets via TORCH_CUDA_ARCH_LIST if needed:
    TORCH_CUDA_ARCH_LIST="8.0;9.0" python setup.py build_ext --inplace
"""

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
# this file lives at galvatron/core/runtime/transformer/fused_rope/ → ../../../../../ is repo root
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, *[".."] * 5))
CSRC_DIR = os.path.join(PROJECT_ROOT, "csrc", "fused_rope")

nvcc_flags = [
    "-O3",
    "--use_fast_math",
    "-std=c++17",
    "-U__CUDA_NO_HALF_OPERATORS__",
    "-U__CUDA_NO_HALF_CONVERSIONS__",
    "-U__CUDA_NO_BFLOAT16_OPERATORS__",
    "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
    "-U__CUDA_NO_HALF2_OPERATORS__",
]

cxx_flags = ["-O3", "-std=c++17"]

ext_modules = [
    CUDAExtension(
        # bare name — `--inplace` drops the .so next to this setup.py, where
        # __init__.py's `from . import _fused_rope_C` will find it.
        name="_fused_rope_C",
        sources=[os.path.join(CSRC_DIR, "binding.cu")],
        include_dirs=[CSRC_DIR],
        extra_compile_args={"cxx": cxx_flags, "nvcc": nvcc_flags},
    )
]

setup(
    name="galvatron_fused_rope",
    version="0.1.0",
    description="Galvatron FusedRoPE CUDA kernel (extracted from NVIDIA TransformerEngine)",
    ext_modules=ext_modules,
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
)
