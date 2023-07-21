
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import math

curr_path = os.path.dirname(os.path.realpath(__file__))
src_files = ['cuda_kernel.cu', 'cuda_launch.cu', 'torch_extension.cpp']
src_files = [os.path.join(curr_path, file) for file in src_files]
lsh_cumulation_kernel = load('lsh_cumulation_kernel', src_files, verbose = True)

import lsh_cumulation_kernel

def lsh_cumulation_input_check(key_mask, key_hashcode, value, hashcode_len):
    pass

def lsh_query_input_check(query_mask, query_hashcode, hashtable):
    pass

def lsh_cumulation_query_input_check(query_mask, query_hashcode, key_mask, key_hashcode, value, hashcode_len):
    pass

def lsh_cumulation(key_mask, key_hashcode, value, hashcode_len):
    lsh_cumulation_input_check(key_mask, key_hashcode, value, hashcode_len)
    return lsh_cumulation_kernel.lsh_cumulation(key_mask, key_hashcode, value, int(2 ** hashcode_len))

def lsh_query(query_mask, query_hashcode, hashtable):
    lsh_query_input_check(query_mask, query_hashcode, hashtable)
    return lsh_cumulation_kernel.lsh_query(query_mask, query_hashcode, hashtable)

def lsh_cumulation_query(query_mask, query_hashcode, key_mask, key_hashcode, value, hashcode_len):
    lsh_cumulation_query_input_check(query_mask, query_hashcode, key_mask, key_hashcode, value, hashcode_len)
    return lsh_cumulation_kernel.lsh_cumulation_query(query_mask, query_hashcode, key_mask, key_hashcode, value, int(2 ** hashcode_len))

def profile():
    import math

    batch_size = 800
    num_query = 512
    num_key = 512
    vector_dim = 64
    num_hash_f = 32
    hashcode_len = 9

    query = torch.rand(batch_size, num_query, vector_dim).cuda() * 2 - 1
    key = torch.rand(batch_size, num_key, vector_dim).cuda() * 2 - 1
    query_hashcode = torch.randint(0, int(2 ** hashcode_len), size = (batch_size, num_query, num_hash_f), dtype = torch.int32).cuda()
    key_hashcode = torch.randint(0, int(2 ** hashcode_len), size = (batch_size, num_key, num_hash_f), dtype = torch.int32).cuda()
    value = torch.rand(batch_size, num_key, vector_dim).cuda() * 2 - 1
    query_mask = torch.ones(batch_size, num_query, dtype = torch.int32).cuda()
    key_mask = torch.ones(batch_size, num_key, dtype = torch.int32).cuda()

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        hashtable1 = lsh_cumulation(key_mask, key_hashcode, value, hashcode_len)
    torch.cuda.synchronize()
    t1 = time.time()
    lsh_cumulation_t = t1 - t0

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        result1 = lsh_query(query_mask, query_hashcode, hashtable1)
    torch.cuda.synchronize()
    t1 = time.time()
    lsh_query_t = t1 - t0

    print(f"lsh_cumulation_t={lsh_cumulation_t:.5f}, lsh_query_t={lsh_query_t:.5f}")

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        result2 = torch.matmul((torch.matmul(query, key.transpose(-1, -2)) / math.sqrt(vector_dim)).softmax(dim = -1), value)
    torch.cuda.synchronize()
    t1 = time.time()
    softmax_t = t1 - t0

    print(f"softmax_t={softmax_t:.5f}")

def unit_test():
    import numpy as np

    batch_size = 12
    num_vector = 4096
    dim = 64
    num_hash_f = 16
    hashcode_len = 8

    query_hashcode = torch.randint(0, int(2 ** hashcode_len), size = (batch_size, num_vector, num_hash_f), dtype = torch.int32).cuda()
    key_hashcode = torch.randint(0, int(2 ** hashcode_len), size = (batch_size, num_vector, num_hash_f), dtype = torch.int32).cuda()
    value = torch.rand(batch_size, num_vector, dim).cuda() * 2 - 1
    query_mask = torch.ones(batch_size, num_vector, dtype = torch.int32).cuda()
    key_mask = torch.ones(batch_size, num_vector, dtype = torch.int32).cuda()

    result0 = lsh_cumulation_query(query_mask, query_hashcode, key_mask, key_hashcode, value, hashcode_len)

    hashtable1 = lsh_cumulation(key_mask, key_hashcode, value, hashcode_len)

    hashtable2_np = np.zeros((batch_size, num_hash_f, int(2 ** hashcode_len), dim), dtype = np.float32)
    key_hashcode_np = key_hashcode.cpu().numpy()
    value_np = value.cpu().numpy()
    key_mask_np = key_mask.cpu().numpy()
    for b in range(batch_size):
        for k in range(num_vector):
            if key_mask_np[b, k].item() == 0:
                continue
            for h in range(num_hash_f):
                hc = key_hashcode_np[b, k, h].item()
                hashtable2_np[b, h, hc, :] += value_np[b, k, :]
        print(b, end = "\r")
    hashtable2 = torch.tensor(hashtable2_np).cuda()

    print(hashtable1[0, 0, 0, :8])
    print(torch.max(torch.abs(hashtable1 - hashtable2)))
    assert torch.max(torch.abs(hashtable1 - hashtable2)) < 1e-5
    print("Passed Test 1")

    result1 = lsh_query(query_mask, query_hashcode, hashtable1)

    result2_np = np.zeros((batch_size, num_vector, dim), dtype = np.float32)
    query_hashcode_np = query_hashcode.cpu().numpy()
    query_mask_np = query_mask.cpu().numpy()
    for b in range(batch_size):
        for k in range(num_vector):
            if query_mask_np[b, k].item() == 0:
                continue
            for h in range(num_hash_f):
                hc = query_hashcode_np[b, k, h].item()
                result2_np[b, k, :] += hashtable2_np[b, h, hc, :]
        print(b, end = "\r")
    result2 = torch.tensor(result2_np / num_hash_f).cuda()

    print(result1[0, 0, :8])
    print(torch.max(torch.abs(result1 - result2)))
    assert torch.max(torch.abs(result1 - result2)) < 1e-5

    print(torch.max(torch.abs(result0 - result2)))
    assert torch.max(torch.abs(result0 - result2)) < 1e-5

    print("Passed Test 2")

if __name__ == "__main__":
    profile()
    unit_test()
