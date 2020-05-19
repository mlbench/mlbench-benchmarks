#!/usr/bin/env python3
# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


strided_batched_gemm = CUDAExtension(
                        name='strided_batched_gemm',
                        sources=['/codes/strided_batched_gemm/strided_batched_gemm.cpp', '/codes/strided_batched_gemm/strided_batched_gemm_cuda.cu'],
                        extra_compile_args={
                                'cxx': ['-O2',],
                                'nvcc':['--gpu-architecture=compute_70','--gpu-code=sm_70','-O3','-I./cutlass/','-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__']
                        }
)

padded_softmax = CUDAExtension(
                        name='padded_softmax',
                        sources=['/codes/padded_softmax/softmax.cu', '/codes/padded_softmax/padded_softmax.cpp'],
                        extra_compile_args={
                                'cxx': ['-O2',],
                                'nvcc':['--gpu-architecture=compute_70','--gpu-code=sm_70','-O3','-U__CUDA_NO_HALF_OPERATORS__', '-U__CUDA_NO_HALF_CONVERSIONS__', "--expt-relaxed-constexpr"]
                        }
)

setup(
    ext_modules=[strided_batched_gemm, padded_softmax],
    cmdclass={
                'build_ext': BuildExtension
    },
)
