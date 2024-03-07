
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import math
from torch.autograd import Function

curr_path = os.path.dirname(os.path.realpath(__file__))
src_files = ['torch_extension.cc']
src_files = [os.path.join(curr_path, file) for file in src_files]
ccs_cpu = load(
    'ccs_cpu',
    src_files,
    extra_cflags=['-fopenmp', '-O3', '-march=native'],
    extra_ldflags=['-lgomp', '-O3', '-march=native'],
    verbose=True)

import ccs_cpu

def compute_code_score(hash_scores, training):
    if training:
        assert hash_scores.is_cuda
        return compute_code_score_torch(hash_scores)
    else:
        if not hash_scores.is_cuda:
            return ComputeCodeScore.apply(hash_scores)
        else:
            return compute_code_score_torch(hash_scores)

def bin2dec(b, bits):
    mask = 2 ** torch.arange(bits - 1, -1, -1, dtype = b.dtype, device = b.device)
    return torch.sum(mask * b, -1)

def compute_code_score_cpu(hash_codes):
    B, T, D = hash_codes.shape
    codes = torch.empty(B, T, dtype = torch.int, device = hash_codes.device)
    scores = torch.empty(B, T, dtype = torch.float, device = hash_codes.device)
    ccs_cpu.compute_code_score(hash_codes, codes, scores)
    return codes, scores

def compute_code_score_torch(hash_scores):
    code_length = hash_scores.shape[-1]

    # code = [batch_size, num_table]
    code = 2 * (hash_scores > 0).to(hash_scores.dtype) - 1
    code = bin2dec((code + 1) / 2, code_length).int()

    # score = [batch_size, num_table]
    max_hash_scores = hash_scores.abs()
    denominator = torch.prod(1 + torch.exp(- 2 * max_hash_scores), dim = -1)
    score = max_hash_scores.sum(dim = -1) / denominator
    
    return code, score

class ComputeCodeScore(Function):
    @staticmethod
    def forward(ctx, hash_scores):
        hash_scores = hash_scores.contiguous()
        return compute_code_score_cpu(hash_scores)

    @staticmethod
    def backward(*args):
        raise NotImplementedError
    
def corrcoef(x, y):
    return torch.corrcoef(torch.stack([x.reshape(-1).float(), y.reshape(-1).float()], dim = 0))[0, 1]

def unit_test():
    B = 128 * 512
    H = 128
    D = 8
    hash_scores = torch.randn(B, H, D)
    ref = compute_code_score_torch(hash_scores)
    check = compute_code_score_cpu(hash_scores)
    print(corrcoef(ref, check))

def profile():
    B = 128 * 512
    H = 128
    D = 8
    hash_scores = torch.randn(B, H, D)

    t0 = time.time()
    for _ in range(20):
        ref = compute_code_score_torch(hash_scores)
    t1 = time.time()
    print("ref", (t1 - t0) / 20)

    t0 = time.time()
    for _ in range(20):
        check = compute_code_score_cpu(hash_scores)
    t1 = time.time()
    print("check", (t1 - t0) / 20)
    

if __name__ == "__main__":
    unit_test()
    profile()