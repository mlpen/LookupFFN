
import torch
import torch.nn as nn
import os
import time
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from autograd import weighted_gather_add
from autograd import weighted_scatter_add

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

batch_size = random.randrange(1, 1000)
index_size = random.randrange(1, 200) * 2
source_size = random.randrange(1, 2000)
vector_dim = random.randrange(1, 30) * 64

print(f"batch_size = {batch_size}")
print(f"index_size = {index_size}")
print(f"source_size = {source_size}")
print(f"vector_dim = {vector_dim}")

print("################### weighted_vector_gather_add check ##################")
def run_func(func):
    torch.manual_seed(0)
    indexes = torch.randint(0, source_size, size = (batch_size, index_size)).int().cuda()
    source = 0.02 * torch.randn(source_size, vector_dim, requires_grad = True).cuda()
    weights = torch.randn(batch_size, index_size, requires_grad = True).cuda()
    target = torch.randn(batch_size, vector_dim).cuda()

    outputs = func(indexes, source, weights)
    loss = ((outputs - target) ** 2).sum(dim = -1).mean()
    source.retain_grad()
    weights.retain_grad()
    loss.backward()

    return outputs, source.grad, weights.grad


ref_A, ref_B, ref_C = run_func(lambda x, y, z: weighted_gather_add(x, y, z, compute_type = "torch"))

out_A, out_B, out_C = run_func(lambda x, y, z: weighted_gather_add(x, y.to(torch.float16), z.to(torch.float16), compute_type = "custom_fp16"))
print_results(ref_A, out_A, "custom_fp16, outputs")
print_results(ref_B, out_B, "custom_fp16, source.grad")
print_results(ref_C, out_C, "custom_fp16, weights.grad")

out_A, out_B, out_C = run_func(lambda x, y, z: weighted_gather_add(x, y, z, compute_type = "custom_fp32"))
print_results(ref_A, out_A, "custom_fp32, outputs")
print_results(ref_B, out_B, "custom_fp32, source.grad")
print_results(ref_C, out_C, "custom_fp32, weights.grad")


print("################### weighted_vector_scatter_add check ##################")
def run_func(func):
    torch.manual_seed(0)
    indexes = torch.randint(0, source_size, size = (batch_size, index_size)).int().cuda()
    source = 0.02 * torch.randn(batch_size, vector_dim, requires_grad = True).cuda()
    weights = torch.randn(batch_size, index_size, requires_grad = True).cuda()
    target = torch.randn(source_size, vector_dim).cuda()

    outputs = func(indexes, source, weights)
    loss = ((outputs - target) ** 2).sum(dim = -1).mean()
    source.retain_grad()
    weights.retain_grad()
    loss.backward()

    return outputs, source.grad, weights.grad


ref_A, ref_B, ref_C = run_func(lambda x, y, z: weighted_scatter_add(x, y, z, source_size, compute_type = "torch"))

out_A, out_B, out_C = run_func(lambda x, y, z: weighted_scatter_add(x, y.to(torch.float16), z.to(torch.float16), source_size, compute_type = "custom_fp16"))
print_results(ref_A, out_A, "custom_fp16, outputs")
print_results(ref_B, out_B, "custom_fp16, source.grad")
print_results(ref_C, out_C, "custom_fp16, weights.grad")

out_A, out_B, out_C = run_func(lambda x, y, z: weighted_scatter_add(x, y, z, source_size, compute_type = "custom_fp32"))
print_results(ref_A, out_A, "custom_fp32, outputs")
print_results(ref_B, out_B, "custom_fp32, source.grad")
print_results(ref_C, out_C, "custom_fp32, weights.grad")
