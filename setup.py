from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import pathlib
import os

try:
    import fused_dense_lib, dropout_layer_norm, rotary_emb, xentropy_cuda_lib
except ImportError:
    fused_dense_lib, dropout_layer_norm, rotary_emb, xentropy_cuda_lib = None, None, None, None
    

FLASH_ATTN_INSTALL = os.getenv("GALVATRON_FLASH_ATTN_INSTALL", "FALSE") == "TRUE"
MOE_KERNELS_INSTALL = os.getenv("GALVATRON_MOE_KERNELS_INSTALL", "TRUE") == "TRUE"

here = pathlib.Path(__file__).parent.resolve()

class CustomInstall(install):
    def run(self):
        install.run(self)

        # custom install flash-attention cuda ops by running shell scripts
        if FLASH_ATTN_INSTALL:
            cwd = pathlib.Path.cwd()
            
            if fused_dense_lib is None or dropout_layer_norm is None or rotary_emb is None or xentropy_cuda_lib is None:
                self.spawn(["bash", str(cwd / "galvatron" / "scripts" / "flash_attn_ops_install.sh")])

class CustomDevelop(develop):
    def run(self):
        develop.run(self)

        # custom install flash-attention cuda ops by running shell scripts
        if FLASH_ATTN_INSTALL:
            cwd = pathlib.Path.cwd()
            
            if fused_dense_lib is None or dropout_layer_norm is None or rotary_emb is None or xentropy_cuda_lib is None:
                self.spawn(["bash", str(cwd / "galvatron" / "scripts" / "flash_attn_ops_install.sh")])


class CustomBuildExt(build_ext):
    def run(self):
        import pybind11

        # Add pybind11 include directory to all extensions
        for ext in self.extensions:
            if hasattr(ext, 'include_dirs'):
                ext.include_dirs.append(pybind11.get_include())
            else:
                ext.include_dirs = [pybind11.get_include()]

        build_ext.run(self)

class CustomBuildExtension(BuildExtension):
    """Custom BuildExtension that adds pybind11 include directories"""
    
    def run(self):
        import pybind11
        
        # Add pybind11 include directory to all extensions
        for ext in self.extensions:
            if hasattr(ext, 'include_dirs'):
                ext.include_dirs.append(pybind11.get_include())
            else:
                ext.include_dirs = [pybind11.get_include()]
        
        # Call parent class run method
        super().run()


# Define the extension modules
dp_core_ext = Extension(
    'galvatron_dp_core',
    sources=['csrc/dp_core.cpp'],
    extra_compile_args=['-O3', '-Wall', '-shared', '-std=c++11', '-fPIC'],
    language='c++'
)

# MoE all_to_all kernels extension
moe_kernels_ext = None
if MOE_KERNELS_INSTALL:
    try:
        import torch
        if torch.cuda.is_available():
            # CUDA extension for MoE kernels
            moe_kernels_ext = CUDAExtension(
                name='moe_all_to_all_kernels',
                sources=[
                    'csrc/moe_all_to_all_binding.cpp',
                    'csrc/moe_all_to_all_kernels.cu'
                ],
                extra_compile_args={
                    'cxx': ['-O3', '-std=c++17', '-DUSE_C10D_NCCL'],
                    'nvcc': ['-O3', '-std=c++17', '-DUSE_C10D_NCCL']
                },
                include_dirs=torch.utils.cpp_extension.include_paths(),
                libraries=['nccl']
            )
    except ImportError:
        print("Warning: PyTorch not available, skipping MoE kernels compilation")
        moe_kernels_ext = None

_deps = [
    "torch>=2.0.1",
    "torchvision>=0.15.2",
    "transformers>=4.31.0",
    "h5py>=3.6.0",
    "attrs>=21.4.0",
    "yacs>=0.1.8",
    "six>=1.15.0",
    "sentencepiece>=0.1.95",
    "pybind11>=2.9.1",
    "scipy>=1.10.1",

]

if FLASH_ATTN_INSTALL:
    _deps.append("packaging")
    _deps.append("flash-attn>=2.0.8")

data_files = [
    (os.path.join('galvatron', 'site_package', 'megatron', 'core', 'datasets'),
     [os.path.join('galvatron', 'site_package', 'megatron', 'core', 'datasets', 'helpers.cpp'),
      os.path.join('galvatron', 'site_package', 'megatron', 'core', 'datasets', 'Makefile')])
]

setup(
    name="hetu-galvatron",
    version="1.0.0",
    description="Galvatron, a Efficient Transformer Training Framework for Multiple GPUs Using Automatic Parallelism",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="Yujie Wang, Shenhan Zhu, Xinyi Liu",
    author_email="alfredwang@pku.edu.cn, shenhan.zhu@pku.edu.cn, xy.liu@stu.pku.edu.cn",
    packages=find_packages(
        exclude=(
            "build",
            "csrc",
            "figs",
            "*egg-info"
        )
    ),
    package_data={"": ["*.json"]},
    include_package_data=True,
    scripts=["galvatron/scripts/flash_attn_ops_install.sh"],
    python_requires=">=3.8",
    cmdclass={
        "install": CustomInstall,
        "develop": CustomDevelop,
        "build_ext": CustomBuildExtension
    },
    install_requires=_deps,
    setup_requires=["pybind11>=2.9.1"],
    ext_modules=[dp_core_ext] + ([moe_kernels_ext] if moe_kernels_ext else []),
    data_files=data_files
)

