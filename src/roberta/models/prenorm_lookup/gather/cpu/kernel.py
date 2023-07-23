
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import random
import math
from torch.utils.checkpoint import checkpoint
from torch.autograd import Function

curr_path = os.path.dirname(os.path.realpath(__file__))
src_files = ['torch_extension.cpp']
src_files = [os.path.join(curr_path, file) for file in src_files]
gather_add_cpu = load(
    'gather_add_cpu',
    src_files,
    extra_cflags=['-fopenmp', '-O3', '-march=native'],
    extra_ldflags=['-lgomp', '-O3', '-march=native'],
    verbose=True)

import gather_add_cpu
from torch.autograd import Function

def corrcoef(x, y):
    return torch.corrcoef(torch.stack([x.reshape(-1), y.reshape(-1)], dim = 0))[0, 1]

def weighted_vector_gather_add(indexes, source, weights, use_torch = False):
    # indexes = [batch_size, index_size]
    # source = [source_size, vector_dim]
    # weights = [batch_size, index_size]
    # outputs = [batch_size, vector_dim]

    assert len(indexes.shape) == 2
    assert len(source.shape) == 2
    assert len(weights.shape) == 2

    batch_size, index_size = indexes.shape
    source_size, vector_dim = source.shape
    assert weights.shape[0] == batch_size and weights.shape[1] == index_size

    if not use_torch:
        if not indexes.is_contiguous(): indexes = indexes.contiguous()
        if not source.is_contiguous(): source = source.contiguous()
        if not weights.is_contiguous(): weights = weights.contiguous()
        if indexes.dtype != torch.int32: indexes = indexes.int()

        # with perftools.pinperf.perf_roi(3, 'weighted_vector_gather_add', 'weighted_vector_gather_add'):
        return gather_add_cpu.weighted_vector_gather_add_avx2_par(indexes, source, weights)


    mini_batch = 8
    assert batch_size % mini_batch == 0
    def mini_batching(source, indexes, weights, start, mini_batch):
        outputs = source[indexes[start:(start + mini_batch), :].reshape(-1), :].reshape(mini_batch, index_size, vector_dim)
        outputs = (outputs * weights[start:(start + mini_batch), :, None]).sum(dim = 1)
        return outputs

    output_list = []
    for start in range(0, batch_size, mini_batch):
        output_list.append(checkpoint(mini_batching, source, indexes, weights, start, mini_batch))
    outputs = torch.cat(output_list, dim = 0)

    return outputs

class WeightedVectorGatherAdd(Function):
    @staticmethod
    def forward(ctx, indexes, source, weights):
        # indexes = [batch_size, index_size]
        # source = [source_size, vector_dim]
        # weights = [batch_size, index_size]
        # outputs = [batch_size, vector_dim]
        assert indexes.shape[0] == 64 * 512
        assert indexes.shape[1] == 128
        assert source.shape[0] == 128 * 256
        assert source.shape[1] == 512

        indexes, source, weights = indexes.contiguous(), source.contiguous(), weights.contiguous()
        ctx.save_for_backward(indexes, source, weights)

        outputs = weighted_vector_gather_add(indexes, source, weights)

        return outputs

    @staticmethod
    def backward(*args):
        raise NotImplementedError

def weighted_gather_add(indexes, source, weights, use_torch = False):
    if use_torch:
        return weighted_vector_gather_add(indexes, source, weights, use_torch = use_torch)
    else:
        return WeightedVectorGatherAdd.apply(indexes, source, weights)

def unit_test():
    batch_size = 64 * 512
    index_size = 128
    source_size = 256 * 128
    vector_dim = 512

    print("batch_size", batch_size)
    print("index_size", index_size)
    print("source_size", source_size)
    print("vector_dim", vector_dim)

    indexes = torch.randint(0, 256, size = (batch_size, index_size))
    indexes = indexes + torch.arange(index_size) * 256
    print(indexes)
    source = torch.randn(source_size, vector_dim)
    weights = torch.randn(batch_size, index_size)

    ref = weighted_vector_gather_add(indexes, source, weights, use_torch = True)
    out = weighted_vector_gather_add(indexes, source, weights, use_torch = False)
    print(out)
    print('Checking...')
    print('ref', ref.shape)
    print('out', out.shape)
    print("max diff", (out - ref).abs().max())
    print("corrcoef", corrcoef(out, ref))



if __name__ == '__main__':
    unit_test()