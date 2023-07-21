
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math
from .kernel import hadamard_transform, fast_hadamard_transform, fast_hadamard_transform_dim512, fast_hadamard_transform_dim1024

def measure(func):
    func()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        func()
    torch.cuda.synchronize()
    t1 = time.time()
    return t1 - t0

batch_size = 80000
vector_dim = 512

print(f"batch_size = {batch_size}")
print(f"vector_dim = {vector_dim}")

X = torch.randn(batch_size, vector_dim).cuda().to(torch.float16)
standard = measure(lambda :hadamard_transform(X))
fast_fp16 = measure(lambda :fast_hadamard_transform_dim512(X))
X = X.to(torch.float32)
fast = measure(lambda :fast_hadamard_transform(X))
print(f"standard={standard:.5f}, fast_fp16={fast_fp16:.5f}, fast_fp32={fast:.5f}")


batch_size = 80000
vector_dim = 1024

print(f"batch_size = {batch_size}")
print(f"vector_dim = {vector_dim}")

X = torch.randn(batch_size, vector_dim).cuda().to(torch.float16)
standard = measure(lambda :hadamard_transform(X))
fast_fp16 = measure(lambda :fast_hadamard_transform_dim1024(X))
X = X.to(torch.float32)
fast = measure(lambda :fast_hadamard_transform(X))
print(f"standard={standard:.5f}, fast_fp16={fast_fp16:.5f}, fast_fp32={fast:.5f}")
