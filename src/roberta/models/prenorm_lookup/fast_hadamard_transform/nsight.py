
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math
from .kernel import hadamard_transform, fast_hadamard_transform, fast_hadamard_transform_dim512, fast_hadamard_transform_dim1024

batch_size = 20000
vector_dim = 512

print(f"batch_size = {batch_size}")
print(f"vector_dim = {vector_dim}")

X = torch.randn(batch_size, vector_dim).cuda()
ref = hadamard_transform(X)
out_fp16 = fast_hadamard_transform_dim512(X.to(torch.float16))
out_fp32 = fast_hadamard_transform(X)

batch_size = 20000
vector_dim = 1024

print(f"batch_size = {batch_size}")
print(f"vector_dim = {vector_dim}")

X = torch.randn(batch_size, vector_dim).cuda()
ref = hadamard_transform(X)
out_fp16 = fast_hadamard_transform_dim1024(X.to(torch.float16))
out_fp32 = fast_hadamard_transform(X)
