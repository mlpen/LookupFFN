
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time

import os
import sys
curr_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(curr_path)

try:
    from fast_hash.kernel import generate_Dmat, fast_hash
    from lsh_cumulation.kernel import lsh_cumulation, lsh_query, lsh_cumulation_query
    from weighted_lsh_cumulation.count_sort.kernel import count_sort
    from weighted_lsh_cumulation.kernel import weighted_lsh_cumulation_sorted_key, weighted_lsh_cumulation_sorted_query
except Exception as e:
    print(e)

def to_contiguous(inp):
    if type(inp) is list:
        out = []
        for tensor in inp:
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            out.append(tensor)
        return out
    else:
        if not inp.is_contiguous():
            inp = inp.contiguous()
        return inp

class YOSO(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, Q_mask, K_mask, num_hash_f, hashcode_len):

        Q, K, V, Q_mask, K_mask = to_contiguous([Q, K, V, Q_mask, K_mask])
        batch_size, _, dim = Q.size()

        Dmat = generate_Dmat(batch_size, dim, num_hash_f, hashcode_len, device = Q.device)
        Q_hashcode = fast_hash(Q_mask, Q, Dmat, num_hash_f, hashcode_len)
        K_hashcode = fast_hash(K_mask, K, Dmat, num_hash_f, hashcode_len)

        # hashtable = lsh_cumulation(K_mask, K_hashcode, V, hashcode_len)
        # cumulation_V = lsh_query(Q_mask, Q_hashcode, hashtable)
        cumulation_V = lsh_cumulation_query(Q_mask, Q_hashcode, K_mask, K_hashcode, V, hashcode_len)

        ctx.save_for_backward(Q_mask, K_mask, Q_hashcode, K_hashcode, Q, K, V)
        ctx.num_hash_f = num_hash_f
        ctx.hashcode_len = hashcode_len

        return cumulation_V

    @staticmethod
    def backward(ctx, grad):
        
        grad = to_contiguous(grad)
        Q_mask, K_mask, Q_hashcode, K_hashcode, Q, K, V = ctx.saved_tensors
        num_hash_f = ctx.num_hash_f
        hashcode_len = ctx.hashcode_len

        # hashtable = lsh_cumulation(Q_mask, Q_hashcode, grad, hashcode_len)
        # grad_V = lsh_query(K_mask, K_hashcode, hashtable)
        grad_V = lsh_cumulation_query(K_mask, K_hashcode, Q_mask, Q_hashcode, grad, hashcode_len)

        K_sort_info, K_sorted_idxes = count_sort(K_mask, K_hashcode, int(2 ** hashcode_len))

        grad_Q = weighted_lsh_cumulation_sorted_key(
            Q_mask, Q_hashcode, K_sort_info, K_sorted_idxes,
            grad, V, Q, K, K, min(1024, K.size(1)), hashcode_len
        )

        grad_K = weighted_lsh_cumulation_sorted_query(
            K_sort_info, K_sorted_idxes, Q_mask, Q_hashcode,
            V, grad, K, Q, Q, min(1024, Q.size(1)), hashcode_len
        )

        return grad_Q, grad_K, grad_V, None, None, None, None

def normalize(X):
    return F.normalize(X, p = 2, dim = -1)

def yoso(Q, K, V, Q_mask, K_mask, num_hash_f, hashcode_len):

    assert Q.size(0) == K.size(0)
    assert Q.size(0) == V.size(0)
    assert K.size(1) == V.size(1)
    assert Q.size(2) == K.size(2)

    return YOSO.apply(Q, K, V, Q_mask, K_mask, num_hash_f, hashcode_len)

def softmax_attn(Q, K, V, Q_mask, K_mask):
    dot = torch.matmul(Q, K.transpose(-1, -2)) / math.sqrt(K.size(-1))
    attn = F.softmax(dot, dim = -1) * K_mask[:, None, :]
    return torch.matmul(attn, V)

def yoso_e(Q, K, V, Q_mask, K_mask, hashcode_len):
    dot = torch.matmul(Q, K.transpose(-1, -2)) * 0.98
    attn = (1 - torch.acos(dot) / math.pi) ** hashcode_len * Q_mask[:, :, None] * K_mask[:, None, :]
    return torch.matmul(attn, V)

def profile_func(func, inputs, backward = False):
    torch.cuda.synchronize()
    t0 = time.time()
    if backward:
        output = func(*inputs)
        loss = torch.mean(output)
        loss.backward()
    else:
        output = func(*inputs)
    torch.cuda.synchronize()
    t1 = time.time()
    return t1 - t0

def generate_random_inputs(batch_size, num_query, num_key, vector_dim, requires_grad = False):
    query = torch.rand(batch_size, num_query, vector_dim, requires_grad = requires_grad).cuda() * 2 - 1
    key = torch.rand(batch_size, num_key, vector_dim, requires_grad = requires_grad).cuda() * 2 - 1
    value = torch.rand(batch_size, num_key, vector_dim, requires_grad = requires_grad).cuda() * 2 - 1
    query_mask = torch.ones(batch_size, num_query, dtype = torch.int32).cuda()
    key_mask = torch.ones(batch_size, num_key, dtype = torch.int32).cuda()
    query = F.normalize(query, p = 2, dim = -1)
    key = F.normalize(key, p = 2, dim = -1)
    return query, key, value, query_mask, key_mask

def profile():

    batch_size = 512
    num_query = 512
    num_key = 512
    vector_dim = 64
    num_hash_f = 32
    hashcode_len = 9

    for _ in range(10):
        inputs = generate_random_inputs(batch_size, num_query, num_key, vector_dim, requires_grad = True)
        yoso_f_t = profile_func(yoso, inputs + (num_hash_f, hashcode_len))
        yoso_b_t = profile_func(yoso, inputs + (num_hash_f, hashcode_len), backward = True) - yoso_f_t

        inputs = generate_random_inputs(batch_size, num_query, num_key, vector_dim, requires_grad = True)
        yoso_e_f_t = profile_func(yoso_e, inputs + (hashcode_len,))
        yoso_e_b_t = profile_func(yoso_e, inputs + (hashcode_len,), backward = True) - yoso_e_f_t

        inputs = generate_random_inputs(batch_size, num_query, num_key, vector_dim, requires_grad = True)
        softmax_f_t = profile_func(softmax_attn, inputs)
        softmax_b_t = profile_func(softmax_attn, inputs, backward = True) - softmax_f_t

        print(f"yoso_f_t={yoso_f_t:.5f}, yoso_b_t={yoso_b_t:.5f}")
        print(f"yoso_e_f_t={yoso_e_f_t:.5f}, yoso_e_b_t={yoso_e_b_t:.5f}")
        print(f"softmax_f_t={softmax_f_t:.5f}, softmax_b_t={softmax_b_t:.5f}")

def unit_test_1():
    import pickle
    import random
    import numpy as np
    import matplotlib.pyplot as plt

    def get_output_grad(func, Q, K, V, mask):
        output = func(Q, K, V, mask)
        loss = (output ** 2).mean()
        Q.retain_grad()
        K.retain_grad()
        V.retain_grad()
        loss.backward()
        return (
            output.cpu().detach().numpy(),
            Q.grad.cpu().detach().numpy(),
            K.grad.cpu().detach().numpy(),
            V.grad.cpu().detach().numpy()
        )

    def plot(pts1, pts2, ax):
        ax.scatter(pts1, pts2, alpha = 0.5)
        x = [np.min(pts1).item(), np.max(pts1).item()]
        ax.plot(x, x, color = "red")

    hashcode_len = 8
    num_hash_f = 128
    curr_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = os.path.dirname(curr_path)
    data_path = os.path.join(parent_path, "test_data")
    data_files = [os.path.join(data_path, file) for file in os.listdir(data_path) if file.endswith(".pickle")]
    data_files = sorted(data_files)
    for idx, file in enumerate(data_files[:16]):
        with open(file, "rb") as f:
            qkv = pickle.load(f)
            for key in qkv:
                qkv[key] = qkv[key][0]

        ones = torch.ones(qkv["Q"].shape[0], random.randrange(128, qkv["Q"].shape[1])).int().cuda()
        zeros = torch.zeros(qkv["Q"].shape[0], qkv["Q"].shape[1] - ones.size(1)).int().cuda()
        mask = torch.cat([ones, zeros], dim = -1).contiguous()

        Q, K, V = [torch.tensor(qkv[key], requires_grad = True).float().cuda() for key in ["Q", "K", "V"]]
        Q, K = normalize(Q), normalize(K)
        output_e, Q_grad_e, K_grad_e, V_grad_e = get_output_grad(
            lambda inp0, inp1, inp2, inp3:yoso_e(inp0, inp1, inp2, inp3, inp3, hashcode_len), Q, K, V, mask)

        size = output_e.reshape(-1).shape[0]
        ridxes = np.random.choice(size, size = 4096, replace = False)

        Q, K, V = [torch.tensor(qkv[key], requires_grad = True).float().cuda() for key in ["Q", "K", "V"]]
        Q, K = normalize(Q), normalize(K)
        output_a, Q_grad_a, K_grad_a, V_grad_a = get_output_grad(
            lambda inp0, inp1, inp2, inp3:yoso(inp0, inp1, inp2, inp3, inp3, num_hash_f, hashcode_len), Q, K, V, mask)

        output_e, output_a = output_e.reshape(-1)[ridxes], output_a.reshape(-1)[ridxes]
        Q_grad_e, Q_grad_a = Q_grad_e.reshape(-1)[ridxes], Q_grad_a.reshape(-1)[ridxes]
        K_grad_e, K_grad_a = K_grad_e.reshape(-1)[ridxes], K_grad_a.reshape(-1)[ridxes]
        V_grad_e, V_grad_a = V_grad_e.reshape(-1)[ridxes], V_grad_a.reshape(-1)[ridxes]

        fig, axes = plt.subplots(1, 4, figsize = (16, 4))
        plot(output_e, output_a, axes[0])
        plot(Q_grad_e, Q_grad_a, axes[1])
        plot(K_grad_e, K_grad_a, axes[2])
        plot(V_grad_e, V_grad_a, axes[3])

        fig.savefig(os.path.join(curr_path, "kernel_test", f"{idx}.png"), dpi = 400)
        print(f"completed {file}")

if __name__ == "__main__":
    unit_test_1()
    profile()