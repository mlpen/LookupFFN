
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import math

curr_path = os.path.dirname(os.path.realpath(__file__))
src_files = ['cuda_kernel.cu', 'cuda_launch.cu', 'torch_extension.cpp']
src_files = [os.path.join(curr_path, file) for file in src_files]
count_sort_kernel = load('count_sort_kernel', src_files, verbose = True)

import count_sort_kernel

def count_sort_input_check(key_mask, key_hashcode, hashtable_capacity):
    pass

def count_sort(key_mask, key_hashcode, hashtable_capacity):
    key_hashcode = key_hashcode.transpose(-1, -2).contiguous()
    count_sort_input_check(key_mask, key_hashcode, hashtable_capacity)
    return count_sort_kernel.count_sort(key_mask, key_hashcode, hashtable_capacity)

def unit_test():
    import numpy as np

    batch_size = 43
    num_vector = 521
    num_hash_f = 32
    hashcode_len = 9

    query_hashcode = torch.randint(0, int(2 ** hashcode_len), size = (batch_size, num_vector, num_hash_f), dtype = torch.int32).cuda()
    key_hashcode = torch.randint(0, int(2 ** hashcode_len), size = (batch_size, num_vector, num_hash_f), dtype = torch.int32).cuda()
    query_mask = torch.ones(batch_size, num_vector, dtype = torch.int32).cuda()
    key_mask = torch.ones(batch_size, num_vector, dtype = torch.int32).cuda()

    sort_info_1, sorted_idxes_1 = count_sort(key_mask, key_hashcode, int(2 ** hashcode_len))

    sort_info_2_np = np.zeros((batch_size, num_hash_f, int(2 ** hashcode_len)), dtype = np.int32)
    key_hashcode_np = key_hashcode.transpose(-1, -2).contiguous().cpu().numpy()
    key_mask_np = key_mask.cpu().numpy()
    for b in range(batch_size):
        for h in range(num_hash_f):
            for k in range(num_vector):
                if key_mask_np[b, k].item() != 0:
                    sort_info_2_np[b, h, key_hashcode_np[b, h, k]] += 1
        print(b, end = "\r")
    sort_info_2_np = np.cumsum(sort_info_2_np, axis = -1)
    sort_info_2 = torch.tensor(sort_info_2_np, dtype = torch.int32).cuda()
    assert torch.sum((sort_info_1 != sort_info_2).int()) == 0

    print("Passed Test 1")

if __name__ == "__main__":
    unit_test()
