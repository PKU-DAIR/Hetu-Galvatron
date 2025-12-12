from setuptools import setup, find_packages, Extension
from setuptools.command.install import install
from setuptools.command.develop import develop
from setuptools.command.build_ext import build_ext
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import pathlib
import os

MOE_KERNELS_INSTALL = os.getenv("GALVATRON_MOE_KERNELS_INSTALL", "TRUE") == "TRUE"

here = pathlib.Path(__file__).parent.resolve()

class CustomInstall(install):
    def run(self):
        install.run(self)

class CustomDevelop(develop):
    def run(self):
        develop.run(self)

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

# Greedy balancer extension
greedy_balancer_ext = Extension(
    'greedy_balancer',
    sources=['csrc/greedy_balancer.cpp'],
    extra_compile_args=['-O3', '-Wall', '-shared', '-std=c++17', '-fPIC'],
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
    "torch==2.1.0+cu121",
    "torchvision>=0.15.2",
    "transformers==4.49.0",
    "numpy==1.26.4",
    "h5py>=3.6.0",
    "attrs>=21.4.0",
    "yacs>=0.1.8",
    "six>=1.15.0",
    "sentencepiece>=0.1.95",
    "pybind11>=2.9.1",
    "scipy>=1.10.1",

]

setup(
    name="hetu-galvatron",
    version="1.0.0",
    description="Galvatron, a Efficient Transformer Training Framework for Multiple GPUs Using Automatic Parallelism",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
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
    python_requires=">=3.8",
    cmdclass={
        "install": CustomInstall,
        "develop": CustomDevelop,
        "build_ext": CustomBuildExtension
    },
    install_requires=_deps,
    setup_requires=["pybind11>=2.9.1"],
    ext_modules=[greedy_balancer_ext] + ([moe_kernels_ext] if moe_kernels_ext else []),
)

