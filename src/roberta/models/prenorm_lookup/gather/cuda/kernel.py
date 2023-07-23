
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math
from torch.utils.checkpoint import checkpoint
from torch.autograd import Function

curr_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "extension")
src_files = ['gather_kernel_fp16.cu', 'gather_fp16.cu', 'gather_kernel_fp32.cu', 'gather_fp32.cu', 'torch_extension.cpp']
src_files = [os.path.join(curr_path, file) for file in src_files]
gather_add = load('gather_add', src_files, verbose = True)

import gather_add

def power2(val):
    return (2 ** int(round(math.log2(val)))) == val

def weighted_vector_gather_add(indexes, source, weights, use_torch = False):
    # indexes = [batch_size, index_size]
    # source = [source_size, vector_dim]
    # weights = [batch_size, index_size]
    # outputs = [batch_size, vector_dim]

    assert indexes.is_contiguous()
    assert source.is_contiguous()
    assert weights.is_contiguous()
    assert indexes.dtype == torch.int32, indexes.dtype
    assert source.dtype in [torch.float32, torch.float16], source.dtype
    assert weights.dtype in [torch.float32, torch.float16], weights.dtype

    assert len(indexes.shape) == 2, indexes.shape
    assert len(source.shape) == 2, source.shape
    assert len(weights.shape) == 2, weights.shape
    batch_size, index_size = indexes.shape
    source_size, vector_dim = source.shape
    assert weights.shape[0] == batch_size and weights.shape[1] == index_size

    assert index_size % 2 == 0, index_size
    assert vector_dim % 64 == 0, vector_dim

    if use_torch:
        indexes = indexes.long()

        def mini_batching(source, indexes, weights, batch_idx):
            outputs = source[indexes[batch_idx, :], :].reshape(index_size, vector_dim)
            outputs = (outputs * weights[batch_idx, :, None]).sum(dim = 0)
            return outputs

        output_list = []
        for batch_idx in range(batch_size):
            output_list.append(checkpoint(mini_batching, source, indexes, weights, batch_idx))
        outputs = torch.stack(output_list, dim = 0)

    else:
        if source.dtype == torch.float32:
            assert weights.dtype == torch.float32
            outputs = torch.zeros(batch_size, vector_dim, dtype = torch.float32, device = source.device)
            gather_add.weighted_vector_gather_add_fp32(indexes, source, weights, outputs)
        elif source.dtype == torch.float16:
            assert weights.dtype == torch.float16
            outputs = torch.zeros(batch_size, vector_dim, dtype = torch.float16, device = source.device)
            gather_add.weighted_vector_gather_add_fp16(indexes, source, weights, outputs)
        else:
            raise NotImplementedError

    return outputs

def indexed_inner_product(indexes, source_1, source_2, use_torch = False):
    # indexes = [batch_size, index_size]
    # source_1 = [batch_size, vector_dim]
    # source_2 = [source_size, vector_dim]
    # outputs = [batch_size, index_size]

    assert indexes.is_contiguous()
    assert source_1.is_contiguous()
    assert source_2.is_contiguous()
    assert indexes.dtype == torch.int32, indexes.dtype
    assert source_1.dtype in [torch.float32, torch.float16], source_1.dtype
    assert source_2.dtype in [torch.float32, torch.float16], source_2.dtype

    assert len(indexes.shape) == 2, indexes.shape
    assert len(source_1.shape) == 2, source_1.shape
    assert len(source_2.shape) == 2, source_2.shape
    batch_size, index_size = indexes.shape
    source_size, vector_dim = source_2.shape
    assert source_1.shape[0] == batch_size and source_1.shape[1] == vector_dim

    assert index_size % 2 == 0, index_size
    assert vector_dim % 64 == 0, vector_dim

    if use_torch:
        indexes = indexes.long()

        def mini_batching(indexes, source_1, source_2, batch_idx):
            source_1 = source_1[batch_idx, None, :]
            source_2 = source_2[indexes[batch_idx, :], :].reshape(index_size, vector_dim)
            outputs = (source_1 * source_2).sum(dim = -1)
            return outputs

        output_list = []
        for batch_idx in range(batch_size):
            output_list.append(checkpoint(mini_batching, indexes, source_1, source_2, batch_idx))
        outputs = torch.stack(output_list, dim = 0)

    else:
        if source_1.dtype == torch.float32:
            assert source_2.dtype == torch.float32
            outputs = torch.zeros(batch_size, index_size, dtype = torch.float32, device = source_2.device)
            gather_add.indexed_inner_product_fp32(indexes, source_1, source_2, outputs)
        elif source_1.dtype == torch.float16:
            assert source_2.dtype == torch.float16
            outputs = torch.zeros(batch_size, index_size, dtype = torch.float16, device = source_2.device)
            gather_add.indexed_inner_product_fp16(indexes, source_1, source_2, outputs)
        else:
            raise NotImplementedError
        

    return outputs

def weighted_vector_scatter_add(indexes, source, weights, output_size, use_torch = False):
    # indexes = [batch_size, index_size]
    # source = [batch_size, vector_dim]
    # weights = [batch_size, index_size]
    # outputs = [output_size, vector_dim]

    assert indexes.is_contiguous()
    assert source.is_contiguous()
    assert weights.is_contiguous()
    assert indexes.dtype == torch.int32, indexes.dtype
    assert source.dtype in [torch.float32, torch.float16], source.dtype
    assert weights.dtype in [torch.float32, torch.float16], weights.dtype

    assert len(indexes.shape) == 2, indexes.shape
    assert len(source.shape) == 2, source.shape
    assert len(weights.shape) == 2, weights.shape
    batch_size, index_size = indexes.shape
    _, vector_dim = source.shape

    assert weights.shape[0] == batch_size and weights.shape[1] == index_size
    assert source.shape[0] == batch_size

    assert index_size % 2 == 0, index_size
    assert vector_dim % 64 == 0, vector_dim

    if use_torch:
        indexes = indexes.long()

        outputs = torch.zeros(output_size, vector_dim, dtype = source.dtype, device = source.device)
        for batch_idx in range(batch_size):
            inst_source = (source[batch_idx, None, :] * weights[batch_idx, :, None]).reshape(index_size, -1)
            inst_indexes = indexes[batch_idx, :, None].repeat(1, vector_dim).reshape(index_size, -1)
            outputs.scatter_add_(0, inst_indexes, inst_source)

    else:
        if source.dtype == torch.float32:
            assert weights.dtype == torch.float32
            source = source.to(torch.float32)
            outputs = torch.zeros(output_size, vector_dim, dtype = torch.float32, device = source.device)
            gather_add.weighted_vector_scatter_add_fp32(indexes, source, weights, outputs)

        elif source.dtype == torch.float16:
            assert weights.dtype == torch.float16
            outputs = torch.zeros(output_size, vector_dim, dtype = torch.float16, device = source.device)
            gather_add.weighted_vector_scatter_add_fp16(indexes, source, weights, outputs)

        else:
            raise NotImplementedError

    return outputs

class WeightedVectorGatherAdd(Function):
    @staticmethod
    def forward(ctx, indexes, source, weights):
        # indexes = [batch_size, index_size]
        # source = [source_size, vector_dim]
        # weights = [batch_size, index_size]
        # outputs = [batch_size, vector_dim]

        indexes, source, weights = indexes.contiguous(), source.contiguous(), weights.contiguous()
        ctx.save_for_backward(indexes, source, weights)

        outputs = weighted_vector_gather_add(indexes, source, weights)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        # grad_outputs = [batch_size, vector_dim]
        # grad_source = [source_size, vector_dim]
        # grad_weights = [batch_size, index_size]

        grad_outputs = grad_outputs.contiguous()
        indexes, source, weights = ctx.saved_tensors

        grad_weights = indexed_inner_product(indexes, grad_outputs, source)
        grad_source = weighted_vector_scatter_add(indexes, grad_outputs, weights, source.shape[0])

        return None, grad_source, grad_weights

def weighted_gather_add(indexes, source, weights, use_torch = False):
    if use_torch:
        return weighted_vector_gather_add(indexes, source, weights, use_torch = use_torch)
    else:
        return WeightedVectorGatherAdd.apply(indexes, source, weights)

def corrcoef(x, y):
    return torch.corrcoef(torch.stack([x.reshape(-1).float(), y.reshape(-1).float()], dim = 0))[0, 1]

def unit_test():
    batch_size = random.randrange(1, 20000)
    index_size = random.randrange(1, 200) * 2
    source_size = random.randrange(1, 20000)
    vector_dim = random.randrange(1, 30) * 64

    print(f"batch_size = {batch_size}")
    print(f"index_size = {index_size}")
    print(f"source_size = {source_size}")
    print(f"vector_dim = {vector_dim}")

    print("################### weighted_vector_gather_add check ##################")
    indexes = torch.randint(0, source_size, size = (batch_size, index_size)).int().cuda()
    weights = torch.randn(batch_size, index_size).cuda()
    source = 0.02 * torch.randn(source_size, vector_dim).cuda()
    ref = weighted_vector_gather_add(indexes, source, weights, use_torch = True)

    print("*****************************")
    out = weighted_vector_gather_add(indexes, source.to(torch.float16), weights.to(torch.float16))
    print(corrcoef(ref, out))

    print("*****************************")
    out = weighted_vector_gather_add(indexes, source, weights)
    print(corrcoef(ref, out))


    print("################### weighted_vector_scatter_add check ##################")
    indexes = torch.randint(0, source_size, size = (batch_size, index_size)).int().cuda()
    source = 0.02 * torch.randn(batch_size, vector_dim).cuda()
    weights = torch.randn(batch_size, index_size).cuda()

    ref = weighted_vector_scatter_add(indexes, source, weights, source_size, use_torch = True)

    print("*****************************")
    out = weighted_vector_scatter_add(indexes, source.to(torch.float16), weights.to(torch.float16), source_size)
    print(corrcoef(ref, out))

    print("*****************************")
    out = weighted_vector_scatter_add(indexes, source, weights, source_size)
    print(corrcoef(ref, out))

    print("################### indexed_inner_product check ##################")
    indexes = torch.randint(0, source_size, size = (batch_size, index_size)).int().cuda()
    source_1 = torch.randn(batch_size, vector_dim).cuda()
    source_2 = 0.02 * torch.randn(source_size, vector_dim).cuda()

    ref = indexed_inner_product(indexes, source_1, source_2, use_torch = True)

    print("*****************************")
    out = indexed_inner_product(indexes, source_1.to(torch.float16), source_2.to(torch.float16))
    print(corrcoef(ref, out))

    print("*****************************")
    out = indexed_inner_product(indexes, source_1, source_2)
    print(corrcoef(ref, out))

def autograd_unit_test():
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


    ref_A, ref_B, ref_C = run_func(lambda x, y, z: weighted_gather_add(x, y, z, use_torch = True))

    out_A, out_B, out_C = run_func(lambda x, y, z: weighted_gather_add(x, y.to(torch.float16), z.to(torch.float16), compute_type = "custom_fp16"))
    print(corrcoef(ref_A, out_A), "custom_fp16, outputs")
    print(corrcoef(ref_B, out_B), "custom_fp16, source.grad")
    print(corrcoef(ref_C, out_C), "custom_fp16, weights.grad")

    out_A, out_B, out_C = run_func(lambda x, y, z: weighted_gather_add(x, y, z))
    print(corrcoef(ref_A, out_A), "custom_fp32, outputs")
    print(corrcoef(ref_B, out_B), "custom_fp32, source.grad")
    print(corrcoef(ref_C, out_C), "custom_fp32, weights.grad")


if __name__ == "__main__":
    unit_test()
    autograd_unit_test()
