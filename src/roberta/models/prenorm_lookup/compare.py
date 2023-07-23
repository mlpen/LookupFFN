
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import math
from argparse import NameSpace
from .lookup import LookupFFN

B = 64 * 512
D = 512

block = nn.Sequential(
    nn.LayerNorm(512, eps = 1e-5),
    nn.Linear(512, 2048),
    nn.GELU(),
    nn.Linear(2048, 512)
)
    
num_iters = 20

x = torch.randn(B, D)
block(x)

t0 = time.perf_counter()
for _ in range(num_iters):
    block(x)
t1 = time.perf_counter()

print('time:', (t1 - t0) / num_iters)
print('Throughput:', num_iters / (t1 - t0))


config = NameSpace(
    hidden_size = 512, 
    ffn_num_table = 128, 
    ffn_table_size = 256, 
    layer_norm_eps = 1e-5,
    projection_type = "bh4",
    block_size = 64,
)

block = LookupFFN(config, output_size = 512)

num_iters = 20

x = torch.randn(B, D)
block(x)

t0 = time.perf_counter()
for _ in range(num_iters):
    block(x)
t1 = time.perf_counter()

print('time:', (t1 - t0) / num_iters)
print('Throughput:', num_iters / (t1 - t0))