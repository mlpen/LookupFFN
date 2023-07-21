
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import math

curr_path = os.path.dirname(os.path.realpath(__file__))
src_files = ['cuda_kernel.cu', 'cuda_launch.cu', 'torch_extension.cpp']
src_files = [os.path.join(curr_path, file) for file in src_files]
fast_hash_kernel = load('fast_hash_kernel', src_files, verbose = True)

import fast_hash_kernel

def fast_hash_input_check(mask, vector, Dmat, num_hash_f, hashcode_len):
    assert mask.is_contiguous()
    assert vector.is_contiguous()
    assert Dmat.is_contiguous()

def generate_Dmat(batch_size, vector_dim, num_hash_f, hashcode_len, device):
    num_part = int(math.ceil(num_hash_f / (vector_dim // hashcode_len)))
    Dmat = 2 * torch.randint(2, (batch_size, 3, num_part, vector_dim), dtype = torch.int32, device = device) - 1
    return Dmat

def fast_hash(mask, vector, Dmat, num_hash_f, hashcode_len):
    fast_hash_input_check(mask, vector, Dmat, num_hash_f, hashcode_len)
    return fast_hash_kernel.fast_hash(mask, vector, Dmat, num_hash_f, hashcode_len)

def regular_hash(mask, vector, rmat, num_hash_f, hashcode_len):
    batch_size, num_vector, vector_dim = vector.size()
    X = (torch.matmul(vector, rmat) > 0).int().reshape(batch_size, num_vector, num_hash_f, hashcode_len)
    pow = 2 ** torch.arange(0, hashcode_len, device = X.device, dtype = torch.int32)
    hashcode = torch.sum(X * pow, dim = -1)
    return hashcode

def hadamard_transform(X):
    from scipy.linalg import hadamard
    H = torch.tensor(hadamard(X.size(2)), device = X.device, dtype = X.dtype)
    return torch.matmul(X, H)

def profile():
    import time

    batch_size = 1000
    num_vector = 512
    vector_dim = 64
    num_hash_f = 32
    hashcode_len = 9

    vector = torch.randn(batch_size, num_vector, vector_dim).cuda()
    mask = torch.ones(batch_size, num_vector, dtype = torch.int32).cuda()
    Dmat = generate_Dmat(batch_size, vector_dim, num_hash_f, hashcode_len, device = vector.device)
    rmat = torch.randn(batch_size, vector_dim, num_hash_f * hashcode_len, device = vector.device)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        hashcode1 = fast_hash(mask, vector, Dmat, num_hash_f, hashcode_len)
    torch.cuda.synchronize()
    t1 = time.time()
    fast_hash_t = t1 - t0

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        hashcode2 = regular_hash(mask, vector, rmat, num_hash_f, hashcode_len)
    torch.cuda.synchronize()
    t1 = time.time()
    regular_hash_t = t1 - t0

    print(f"fast_hash_t={fast_hash_t:.5f}, regular_hash_t={regular_hash_t:.5f}")

def unit_test():
    batch_size = 32
    num_vector = 512
    vector_dim = 64
    num_hash_f = 32
    hashcode_len = 9

    vector = torch.rand(batch_size, num_vector, vector_dim).cuda() * 2 - 1
    mask = torch.ones(batch_size, num_vector, dtype = torch.int32).cuda()
    Dmat = generate_Dmat(batch_size, vector_dim, num_hash_f, hashcode_len, device = vector.device)

    result1 = fast_hash(mask, vector, Dmat, num_hash_f, hashcode_len)

    result2 = []
    for part_idx in range(Dmat.size(2)):
        X = hadamard_transform(vector * Dmat[:, 0, part_idx, :][:, None, :])
        X = hadamard_transform(X * Dmat[:, 1, part_idx, :][:, None, :])
        X = hadamard_transform(X * Dmat[:, 2, part_idx, :][:, None, :])
        result2.append(X)
    result2 = torch.stack(result2, dim = 2)
    result2 = result2[:, :, :, :(hashcode_len * (vector_dim // hashcode_len))]
    print(torch.min(torch.abs(result2)))
    result2 = (result2 > 0).int().reshape(batch_size, num_vector, -1, hashcode_len)
    pow = 2 ** torch.arange(0, hashcode_len, device = result2.device, dtype = torch.int32)
    result2 = torch.sum(result2 * pow.int(), dim = -1)[:, :, :num_hash_f]
    print(f"{torch.sum((result2 != result1).int())}/{batch_size * num_vector * num_hash_f}")
    assert torch.sum((result2 != result1).int()) < 10
    print("Passed Test")

if __name__ == "__main__":
    profile()
    unit_test()
