
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from .kernel import hadamard_transform, fast_hadamard_transform, fast_hadamard_transform_dim512, fast_hadamard_transform_dim1024

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

batch_size = random.randrange(1, 80000)
vector_dim = 512

print(f"batch_size = {batch_size}")
print(f"vector_dim = {vector_dim}")

X = torch.randn(batch_size, vector_dim).cuda()

ref = hadamard_transform(X)
out_fp16 = fast_hadamard_transform_dim512(X.to(torch.float16))
out_fp32 = fast_hadamard_transform(X)
print_results(ref, out_fp16, "fp16")
print_results(ref, out_fp32, "fp32")


batch_size = random.randrange(1, 80000)
vector_dim = 1024

print(f"batch_size = {batch_size}")
print(f"vector_dim = {vector_dim}")

X = torch.randn(batch_size, vector_dim).cuda()

ref = hadamard_transform(X)
out_fp16 = fast_hadamard_transform_dim1024(X.to(torch.float16))
out_fp32 = fast_hadamard_transform(X)
print_results(ref, out_fp16, "fp16")
print_results(ref, out_fp32, "fp32")
