
import torch
import torch.nn as nn
import os
import time
import random
import math
import matplotlib.pyplot as plt
import numpy as np
from .autograd import hadamard

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
        print("ref", x[0, :8])
        print("out", y[0, :8])
    else:
        plot(x, y, title)

def run_func(func):
    torch.manual_seed(0)
    inp = 0.02 * torch.randn(batch_size, vector_dim, requires_grad = True).cuda().to(torch.float16)
    target = 0.02 * torch.randn(batch_size, vector_dim).cuda().to(torch.float16)
    print(inp.shape)
    outputs = func(inp)
    loss = ((outputs - target) ** 2).sum(dim = -1).sum()
    inp.retain_grad()
    loss.backward()

    return outputs, inp.grad


batch_size = random.randrange(1, 20000)
vector_dim = 512

print(f"batch_size = {batch_size}")
print(f"vector_dim = {vector_dim}")

ref_A, ref_B = run_func(lambda x: hadamard(x, "torch"))
out_A, out_B = run_func(lambda x: hadamard(x, "custom"))
print_results(ref_A, out_A, "custom, outputs")
print_results(ref_B, out_B, "custom, inp.grad")

batch_size = random.randrange(1, 20000)
vector_dim = 1024

print(f"batch_size = {batch_size}")
print(f"vector_dim = {vector_dim}")

ref_A, ref_B = run_func(lambda x: hadamard(x, "torch"))
out_A, out_B = run_func(lambda x: hadamard(x, "custom"))
print_results(ref_A, out_A, "custom, outputs")
print_results(ref_B, out_B, "custom, inp.grad")
