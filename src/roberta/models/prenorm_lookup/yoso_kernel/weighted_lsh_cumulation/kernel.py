
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import math

curr_path = os.path.dirname(os.path.realpath(__file__))
src_files = ['cuda_kernel.cu', 'cuda_launch.cu', 'torch_extension.cpp']
src_files = [os.path.join(curr_path, file) for file in src_files]
weighted_lsh_cumulation_kernel = load('weighted_lsh_cumulation_kernel', src_files, verbose = True)

import weighted_lsh_cumulation_kernel

def weighted_lsh_cumulation_sorted_query_input_check(
    query_sort_info, query_sorted_idxes, key_mask, key_hashcode,
    query_weight_0, key_weight_0, query_weight_1, key_weight_1, value, workspace_size, tau
):
    pass

def weighted_lsh_cumulation_sorted_query(
    query_sort_info, query_sorted_idxes, key_mask, key_hashcode,
    query_weight_0, key_weight_0, query_weight_1, key_weight_1, value, workspace_size, tau
):
    weighted_lsh_cumulation_sorted_query_input_check(
        query_sort_info, query_sorted_idxes, key_mask, key_hashcode,
        query_weight_0, key_weight_0, query_weight_1, key_weight_1, value, workspace_size, tau
    )
    return weighted_lsh_cumulation_kernel.weighted_lsh_cumulation_sorted_query(
        query_sort_info, query_sorted_idxes, key_mask, key_hashcode,
        query_weight_0, key_weight_0, query_weight_1, key_weight_1, value, workspace_size, tau
    )

def weighted_lsh_cumulation_sorted_key_input_check(
    query_mask, query_hashcode, key_sort_info, key_sorted_idxes,
    query_weight_0, key_weight_0, query_weight_1, key_weight_1, value, workspace_size, tau
):
    pass

def weighted_lsh_cumulation_sorted_key(
    query_mask, query_hashcode, key_sort_info, key_sorted_idxes,
    query_weight_0, key_weight_0, query_weight_1, key_weight_1, value, workspace_size, tau
):
    weighted_lsh_cumulation_sorted_key_input_check(
        query_mask, query_hashcode, key_sort_info, key_sorted_idxes,
        query_weight_0, key_weight_0, query_weight_1, key_weight_1, value, workspace_size, tau
    )
    return weighted_lsh_cumulation_kernel.weighted_lsh_cumulation_sorted_key(
        query_mask, query_hashcode, key_sort_info, key_sorted_idxes,
        query_weight_0, key_weight_0, query_weight_1, key_weight_1, value, workspace_size, tau
    )

def weighted_lsh_cumulation_merged_shared_qk(
    sort_info, sorted_idxes, key_mask, key_hashcode,
    weight_0, weight_1, value, workspace_size, tau
):
    return weighted_lsh_cumulation_kernel.weighted_lsh_cumulation_merged_shared_qk(
        sort_info, sorted_idxes, key_mask, key_hashcode,
        weight_0, weight_1, value, workspace_size, tau
    )

def profile():
    import count_sort.kernel
    import torch.nn.functional as F
    import time

    batch_size = 430
    num_query = 521
    num_key = 521
    vector_dim = 64
    num_hash_f = 32
    hashcode_len = 9
    workspace_size = 1024
    tau = 8

    query_hashcode = torch.randint(0, int(2 ** hashcode_len), size = (batch_size, num_query, num_hash_f), dtype = torch.int32).cuda()
    key_hashcode = torch.randint(0, int(2 ** hashcode_len), size = (batch_size, num_key, num_hash_f), dtype = torch.int32).cuda()
    query_weight_0 = torch.rand(batch_size, num_query, vector_dim).cuda() * 2 - 1
    key_weight_0 = torch.rand(batch_size, num_key, vector_dim).cuda() * 2 - 1
    query_weight_1 = torch.rand(batch_size, num_query, vector_dim).cuda() * 2 - 1
    key_weight_1 = torch.rand(batch_size, num_key, vector_dim).cuda() * 2 - 1
    value = torch.rand(batch_size, num_key, vector_dim).cuda() * 2 - 1
    query_mask = torch.ones(batch_size, num_query, dtype = torch.int32).cuda()
    key_mask = torch.ones(batch_size, num_key, dtype = torch.int32).cuda()

    query_weight_1 = F.normalize(query_weight_1, p = 2, dim = -1)
    key_weight_1 = F.normalize(key_weight_1, p = 2, dim = -1)

    torch.cuda.synchronize()
    t0 = time.time()
    query_sort_info, query_sorted_idxes = count_sort.kernel.count_sort(query_mask, query_hashcode, int(2 ** hashcode_len))
    torch.cuda.synchronize()
    t1 = time.time()
    sort_t_1 = t1 - t0

    torch.cuda.synchronize()
    t0 = time.time()
    result1 = weighted_lsh_cumulation_sorted_query(
        query_sort_info, query_sorted_idxes, key_mask, key_hashcode,
        query_weight_0, key_weight_0, query_weight_1, key_weight_1, value, workspace_size, tau
    )
    torch.cuda.synchronize()
    t1 = time.time()
    weighted_lsh_1 = t1 - t0

    torch.cuda.synchronize()
    t0 = time.time()
    key_sort_info, key_sorted_idxes = count_sort.kernel.count_sort(key_mask, key_hashcode, int(2 ** hashcode_len))
    torch.cuda.synchronize()
    t1 = time.time()
    sort_t_2 = t1 - t0

    torch.cuda.synchronize()
    t0 = time.time()
    result2 = weighted_lsh_cumulation_sorted_key(
        query_mask, query_hashcode, key_sort_info, key_sorted_idxes,
        query_weight_0, key_weight_0, query_weight_1, key_weight_1, value, workspace_size, tau
    )
    torch.cuda.synchronize()
    t1 = time.time()
    weighted_lsh_2 = t1 - t0

    print(f"sort_t_1={sort_t_1:.5f}, weighted_lsh_1={weighted_lsh_1:.5f}")
    print(f"sort_t_2={sort_t_2:.5f}, weighted_lsh_2={weighted_lsh_2:.5f}")


def unit_test():
    import count_sort.kernel
    import torch.nn.functional as F
    import numpy as np
    import random
    import math
    import pickle

    batch_size = 3
    num_query = 200
    num_key = 234
    vector_dim = 64
    num_hash_f = 16
    hashcode_len = 8
    workspace_size = 512
    tau = 9

    query_hashcode = torch.randint(0, int(2 ** hashcode_len), size = (batch_size, num_query, num_hash_f), dtype = torch.int32).cuda()
    key_hashcode = torch.randint(0, int(2 ** hashcode_len), size = (batch_size, num_key, num_hash_f), dtype = torch.int32).cuda()
    query_weight_0 = torch.rand(batch_size, num_query, vector_dim).cuda() * 2 - 1
    key_weight_0 = torch.rand(batch_size, num_key, vector_dim).cuda() * 2 - 1
    query_weight_1 = torch.rand(batch_size, num_query, vector_dim).cuda() * 2 - 1
    key_weight_1 = torch.rand(batch_size, num_key, vector_dim).cuda() * 2 - 1
    value = torch.rand(batch_size, num_key, vector_dim).cuda() * 2 - 1
    query_mask = torch.ones(batch_size, num_query, dtype = torch.int32).cuda()
    key_mask = torch.ones(batch_size, num_key, dtype = torch.int32).cuda()

    # with open("temp.pickle", "rb") as f:
    #     query_hashcode, key_hashcode, query_weight_0, key_weight_0, query_weight_1, key_weight_1, value = [item.cuda() for item in pickle.load(f)]
    # with open("temp.pickle", "wb") as f:
    #     pickle.dump([query_hashcode.cpu(), key_hashcode.cpu(), query_weight_0.cpu(), key_weight_0.cpu(), query_weight_1.cpu(), key_weight_1.cpu(), value.cpu()], f)

    query_weight_1 = F.normalize(query_weight_1, p = 2, dim = -1)
    key_weight_1 = F.normalize(key_weight_1, p = 2, dim = -1)

    query_sort_info, query_sorted_idxes = count_sort.kernel.count_sort(query_mask, query_hashcode, int(2 ** hashcode_len))
    result1 = weighted_lsh_cumulation_sorted_query(
        query_sort_info, query_sorted_idxes, key_mask, key_hashcode,
        query_weight_0, key_weight_0, query_weight_1, key_weight_1, value, workspace_size, tau
    )

    key_sort_info, key_sorted_idxes = count_sort.kernel.count_sort(key_mask, key_hashcode, int(2 ** hashcode_len))
    result2 = weighted_lsh_cumulation_sorted_key(
        query_mask, query_hashcode, key_sort_info, key_sorted_idxes,
        query_weight_0, key_weight_0, query_weight_1, key_weight_1, value, workspace_size, tau
    )

    result3_np = np.zeros((batch_size, num_query, vector_dim), dtype = np.float32)
    query_hashcode_np = query_hashcode.cpu().numpy()
    key_hashcode_np = key_hashcode.cpu().numpy()
    query_weight_0_np = query_weight_0.cpu().numpy()
    key_weight_0_np = key_weight_0.cpu().numpy()
    query_weight_1_np = query_weight_1.cpu().numpy()
    key_weight_1_np = key_weight_1.cpu().numpy()
    value_np = value.cpu().numpy()
    query_mask_np = query_mask.cpu().numpy()
    key_mask_np = key_mask.cpu().numpy()
    for b in range(batch_size):
        for k in range(num_key):
            if key_mask_np[b, k].item() == 0:
                continue
            collided_item = set()
            for q in range(num_query):
                if query_mask_np[b, q].item() == 0:
                    continue
                for h in range(num_hash_f):
                    if query_hashcode_np[b, q, h].item() == key_hashcode_np[b, k, h].item():
                        collided_item.add(q)
            collided_item = list(collided_item)
            for q in collided_item:
                weight_0 = np.sum(query_weight_0_np[b, q, :] * key_weight_0_np[b, k, :])
                weight_1 = 0.99 * np.sum(query_weight_1_np[b, q, :] * key_weight_1_np[b, k, :])
                weight = weight_0 * tau * ((1 - np.arccos(weight_1) / math.pi) ** (tau - 1)) / (math.pi * np.sqrt(1 - weight_1 ** 2))
                result3_np[b, q, :] += weight * value_np[b, k, :]
            print(k, end = "\r")
        print(b, end = "\r")
    result3 = torch.tensor(result3_np, dtype = torch.float32).cuda()

    print(torch.max(torch.abs(result3 - result1)))
    print(torch.max(torch.abs(result3 - result2)))

    assert torch.max(torch.abs(result3 - result1)) < 1e-1
    assert torch.max(torch.abs(result3 - result2)) < 1e-1

    print("Passed Test")

if __name__ == "__main__":
    unit_test()
    profile()
