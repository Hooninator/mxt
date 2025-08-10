from setuptools import setup
import torch
from torch.utils import cpp_extension

ext_modules = [
    cpp_extension.CUDAExtension(
        'recover_kron_norm',
        [
            'cpp/recover_launcher.cu',
        ],
        include_dirs=[
            'cpp/',
        ],
        extra_compile_args={
            'cxx': ['-O3'],
            'nvcc': ['-O3', '--use_fast_math', '-arch=sm_80']
        }
    ),
]

setup(
    name='recover_kron_norm',
    ext_modules=ext_modules,
    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }
)