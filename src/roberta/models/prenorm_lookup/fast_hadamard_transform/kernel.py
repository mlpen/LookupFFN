
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import math

curr_path = os.path.dirname(os.path.realpath(__file__))
src_files = ['cuda_kernel.cu', 'cuda_launch.cu', 'torch_extension.cpp']
src_files = [os.path.join(curr_path, 'extension', file) for file in src_files]
fast_hadamard_transform_kernel = load('fast_hadamard_transform_kernel', src_files, verbose = True)

import fast_hadamard_transform_kernel

def fast_hadamard_transform_dim512(X):
    assert X.is_contiguous()
    assert len(X.shape) == 2
    assert X.dtype == torch.float16
    assert X.shape[-1] == 512
    return fast_hadamard_transform_kernel.fast_hadamard_transform_dim512_fp16(X)

def fast_hadamard_transform_dim1024(X):
    assert X.is_contiguous()
    assert len(X.shape) == 2
    assert X.dtype == torch.float16
    assert X.shape[-1] == 1024
    return fast_hadamard_transform_kernel.fast_hadamard_transform_dim1024_fp16(X)

def hadamard_transform(X):
    assert X.is_contiguous()
    assert len(X.shape) == 2
    assert int(2 ** int(math.log2(X.shape[-1]))) == X.shape[-1]
    from scipy.linalg import hadamard
    H = torch.tensor(hadamard(X.shape[-1]), device = X.device, dtype = X.dtype)
    return torch.matmul(X, H)

def fast_hadamard_transform(X):
    assert X.is_contiguous()
    assert len(X.shape) == 2
    assert X.dtype == torch.float32
    assert int(2 ** int(math.log2(X.shape[-1]))) == X.shape[-1]
    return fast_hadamard_transform_kernel.fast_hadamard_transform(X)
