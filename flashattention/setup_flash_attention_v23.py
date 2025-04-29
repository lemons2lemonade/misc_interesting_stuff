from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='flash_attention_v3',
    ext_modules=[
        CUDAExtension(
            name='flash_attention_v3',
            sources=[
                'flash_attention_partial_v23.cpp',
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc': [
                    '-O3',
                    '-arch=sm_70',  # Adjust for your GPU architecture (Volta or newer recommended)
                    '--use_fast_math',
                    '--ptxas-options=-v',
                    '-lineinfo',
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
) 