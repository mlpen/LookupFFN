
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from kernel import weighted_vector_gather_add
from kernel import weighted_vector_scatter_add
from kernel import indexed_inner_product
import kernel

def plot(x, y, title):
    x = x.reshape(-1)
    y = y.reshape(-1)
    diff = (x - y).abs()
    topk_diff = torch.topk(diff, k = 1000, largest = True)
    print(f"{title}, top difference", topk_diff.values[:4].tolist())
    x = x[topk_diff.indices].cpu().detach().numpy()
    y = y[topk_diff.indices].cpu().detach().numpy()
    plt.scatter(x, y, alpha = 1)
    plt.plot([np.min(x), np.max(x)], [np.min(x), np.max(x)], color = "red")
    plt.title(title)
    plt.show(block = False)
    plt.pause(1)
    plt.close()

def corrcoef(x, y):
    return torch.corrcoef(torch.stack([x.reshape(-1).float(), y.reshape(-1).float()], dim = 0))[0, 1]

def print_results(x, y, title):
    diff = (x - y).abs()
    corr = corrcoef(x, y)
    print(f"{title}, max diff", diff.max())
    print(f"{title}, corr coef", corr)
    if corr.item() < 0.999:
        print(f"Warning: corr={corr}, possible error")
        print("ref", ref[0, :8])
        print("out", out[0, :8])
    else:
        plot(x, y, title)

batch_size = random.randrange(1, 20000)
index_size = random.randrange(1, 200) * 2
source_size = random.randrange(1, 20000)
vector_dim = random.randrange(1, 30) * 64

print(f"batch_size = {batch_size}")
print(f"index_size = {index_size}")
print(f"source_size = {source_size}")
print(f"vector_dim = {vector_dim}")

print("################### weighted_vector_gather_add check ##################")
indexes = torch.randint(0, source_size, size = (batch_size, index_size)).int().cuda()
weights = torch.randn(batch_size, index_size).cuda()
source = 0.02 * torch.randn(source_size, vector_dim).cuda()
ref = weighted_vector_gather_add(indexes, source, weights, compute_type = "torch")

print("*****************************")
out = weighted_vector_gather_add(indexes, source.to(torch.float16), weights.to(torch.float16), compute_type = "custom_fp16")
print_results(ref, out, "custom_fp16")

print("*****************************")
out = weighted_vector_gather_add(indexes, source, weights, compute_type = "custom_fp32")
print_results(ref, out, "custom_fp32")


print("################### weighted_vector_scatter_add check ##################")
indexes = torch.randint(0, source_size, size = (batch_size, index_size)).int().cuda()
source = 0.02 * torch.randn(batch_size, vector_dim).cuda()
weights = torch.randn(batch_size, index_size).cuda()

ref = weighted_vector_scatter_add(indexes, source, weights, source_size, compute_type = "torch")

print("*****************************")
out = weighted_vector_scatter_add(indexes, source.to(torch.float16), weights.to(torch.float16), source_size, compute_type = "custom_fp16")
print_results(ref, out, "custom_fp16")

print("*****************************")
out = weighted_vector_scatter_add(indexes, source, weights, source_size, compute_type = "custom_fp32")
print_results(ref, out, "custom_fp32")

print("################### indexed_inner_product check ##################")
indexes = torch.randint(0, source_size, size = (batch_size, index_size)).int().cuda()
source_1 = torch.randn(batch_size, vector_dim).cuda()
source_2 = 0.02 * torch.randn(source_size, vector_dim).cuda()

ref = indexed_inner_product(indexes, source_1, source_2, compute_type = "torch")

print("*****************************")
out = indexed_inner_product(indexes, source_1.to(torch.float16), source_2.to(torch.float16), compute_type = "custom_fp16")
print_results(ref, out, "custom_fp16")

print("*****************************")
out = indexed_inner_product(indexes, source_1, source_2, compute_type = "custom_fp32")
print_results(ref, out, "custom_fp32")
