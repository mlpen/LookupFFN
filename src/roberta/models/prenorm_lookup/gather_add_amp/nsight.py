
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math
from kernel import weighted_vector_gather_add
from kernel import weighted_vector_scatter_add
from kernel import indexed_inner_product

batch_size = 64 * 512
index_size = 256
source_size = 4096 * 4
vector_dim = 512

print(f"batch_size = {batch_size}")
print(f"index_size = {index_size}")
print(f"source_size = {source_size}")
print(f"vector_dim = {vector_dim}")

print("################### weighted_vector_gather_add ##################")
indexes = torch.randint(0, source_size, size = (batch_size, index_size), dtype = torch.int32, device = "cuda")
weights = torch.randn(batch_size, index_size, device = "cuda")
source = 0.02 * torch.randn(source_size, vector_dim, device = "cuda")

weighted_vector_gather_add(indexes, source.to(torch.float16), weights.to(torch.float16), compute_type = "custom_fp16")
weighted_vector_gather_add(indexes, source, weights, compute_type = "custom_fp32")

print("################### weighted_vector_scatter_add ##################")
indexes = torch.randint(0, source_size, size = (batch_size, index_size), dtype = torch.int32, device = "cuda")
source = 0.02 * torch.randn(batch_size, vector_dim, device = "cuda")
weights = torch.randn(batch_size, index_size, device = "cuda")

weighted_vector_scatter_add(indexes, source.to(torch.float16), weights.to(torch.float16), source_size, compute_type = "custom_fp16")
weighted_vector_scatter_add(indexes, source, weights, source_size, compute_type = "custom_fp32")

print("################### indexed_inner_product ##################")
indexes = torch.randint(0, source_size, size = (batch_size, index_size), dtype = torch.int32, device = "cuda")
source_1 = torch.randn(batch_size, vector_dim, device = "cuda")
source_2 = 0.02 * torch.randn(source_size, vector_dim, device = "cuda")

indexed_inner_product(indexes, source_1.to(torch.float16), source_2.to(torch.float16), compute_type = "custom_fp16")
indexed_inner_product(indexes, source_1, source_2, compute_type = "custom_fp32")
