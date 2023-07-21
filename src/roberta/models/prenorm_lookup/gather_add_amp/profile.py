
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

batch_size = random.randrange(1, 10000)
index_size = random.randrange(1, 500) * 2
source_size = random.randrange(1, 100000)
vector_dim = random.randrange(1, 30) * 64

batch_size = 16384 * 4
index_size = 330
source_size = 16384 * 4
vector_dim = 2048

print(f"batch_size = {batch_size}")
print(f"index_size = {index_size}")
print(f"source_size = {source_size}")
print(f"vector_dim = {vector_dim}")

def measure(func):
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        func()
    torch.cuda.synchronize()
    t1 = time.time()
    return t1 - t0

print("################### weighted_vector_gather_add check ##################")
indexes = torch.randint(0, source_size, size = (batch_size, index_size)).int().cuda()
weights = torch.randn(batch_size, index_size).cuda()
source = 0.02 * torch.randn(source_size, vector_dim).cuda()

custom_fp32 = measure(lambda :weighted_vector_gather_add(indexes, source, weights, compute_type = "custom_fp32"))
source, weights = source.to(torch.float16), weights.to(torch.float16)
custom_fp16 = measure(lambda :weighted_vector_gather_add(indexes, source, weights, compute_type = "custom_fp16"))

print(f"custom_fp32={custom_fp32:.5f}, custom_fp16={custom_fp16:.5f}")

print("################### weighted_vector_scatter_add check ##################")
indexes = torch.randint(0, source_size, size = (batch_size, index_size)).int().cuda()
source = torch.randn(batch_size, vector_dim).cuda()
weights = torch.randn(batch_size, index_size).cuda()

custom_fp32 = measure(lambda :weighted_vector_scatter_add(indexes, source, weights, source_size, compute_type = "custom_fp32"))
source, weights = source.to(torch.float16), weights.to(torch.float16)
custom_fp16 = measure(lambda :weighted_vector_scatter_add(indexes, source, weights, source_size, compute_type = "custom_fp16"))

print(f"custom_fp32={custom_fp32:.5f}, custom_fp16={custom_fp16:.5f}")

print("################### indexed_inner_product check ##################")
indexes = torch.randint(0, source_size, size = (batch_size, index_size)).int().cuda()
source_1 = torch.randn(batch_size, vector_dim).cuda()
source_2 = 0.02 * torch.randn(source_size, vector_dim).cuda()

custom_fp32 = measure(lambda :indexed_inner_product(indexes, source_1, source_2, compute_type = "custom_fp32"))
source_1, source_2 = source_1.to(torch.float16), source_2.to(torch.float16)
custom_fp16 = measure(lambda :indexed_inner_product(indexes, source_1, source_2, compute_type = "custom_fp16"))

print(f"custom_fp32={custom_fp32:.5f}, custom_fp16={custom_fp16:.5f}")
