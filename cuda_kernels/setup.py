from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='detect_and_classify_2d_slices_cuda',
    ext_modules=[
        CUDAExtension('detect_and_classify_2d_slices_cuda', [
            'detect_and_classify_2d_slices_cuda.cpp',
            'detect_and_classify_2d_slices_cuda_kernels.cu',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })