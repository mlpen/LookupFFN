
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math
from kernel import weighted_vector_gather_add
from kernel import indexed_inner_product
from kernel import weighted_vector_scatter_add

batch_size = 32 * 512
index_size = 16
source_size = index_size * int(2 ** 16)
vector_dim = 256

print(f"batch_size = {batch_size}")
print(f"index_size = {index_size}")
print(f"source_size = {source_size}")
print(f"vector_dim = {vector_dim}")

def measure(func):
    func()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        func()
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0)

print("################### weighted_vector_gather_add check ##################")
indexes = torch.randint(0, source_size, size = (batch_size, index_size)).int().cuda()
weights = torch.randn(batch_size, index_size).cuda()
source = 0.02 * torch.randn(source_size, vector_dim).cuda()

custom_fp32 = measure(lambda :weighted_vector_gather_add(indexes, source, weights, compute_type = "custom_fp32"))
source, weights = source.to(torch.float16), weights.to(torch.float16)
custom_fp16 = measure(lambda :weighted_vector_gather_add(indexes, source, weights, compute_type = "custom_fp16"))

print(f"custom_fp32={custom_fp32:.5f}, custom_fp16={custom_fp16:.5f}")

A = torch.randn(batch_size, vector_dim * 4).cuda().to(torch.float16)
B = torch.randn(vector_dim * 4, vector_dim).cuda().to(torch.float16)

mm_fp16 = measure(lambda :torch.matmul(A, B))
print(f"mm_fp16={mm_fp16:.5f}")

A = torch.randn(batch_size, vector_dim * 4).cuda()
B = torch.randn(vector_dim * 4, vector_dim).cuda()

mm_fp32 = measure(lambda :torch.matmul(A, B))
print(f"mm_fp32={mm_fp32:.5f}")