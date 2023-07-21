
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

def weighted_vector_gather_add(indexes, source, weights, compute_type):
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

    if compute_type == "torch":
        indexes = indexes.long()

        def mini_batching(source, indexes, weights, batch_idx):
            outputs = source[indexes[batch_idx, :], :].reshape(index_size, vector_dim)
            outputs = (outputs * weights[batch_idx, :, None]).sum(dim = 0)
            return outputs

        output_list = []
        for batch_idx in range(batch_size):
            output_list.append(checkpoint(mini_batching, source, indexes, weights, batch_idx))
        outputs = torch.stack(output_list, dim = 0)

    elif compute_type == "custom_fp32":
        if source.dtype != torch.float32:
            source = source.to(torch.float32)
        if weights.dtype != torch.float32:
            weights = weights.to(torch.float32)
        outputs = torch.zeros(batch_size, vector_dim, dtype = torch.float32, device = source.device)
        gather_add.weighted_vector_gather_add_fp32(indexes, source, weights, outputs)

    elif compute_type == "custom_fp16":
        if source.dtype != torch.float16:
            source = source.to(torch.float16)
        if weights.dtype != torch.float16:
            weights = weights.to(torch.float16)
        outputs = torch.zeros(batch_size, vector_dim, dtype = torch.float16, device = source.device)
        gather_add.weighted_vector_gather_add_fp16(indexes, source, weights, outputs)

    else:
        raise Exception()

    return outputs

def indexed_inner_product(indexes, source_1, source_2, compute_type):
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

    if compute_type == "torch":
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

    elif compute_type == "custom_fp32":
        if source_1.dtype != torch.float32:
            source_1 = source_1.to(torch.float32)
        if source_2.dtype != torch.float32:
            source_2 = source_2.to(torch.float32)
        outputs = torch.zeros(batch_size, index_size, dtype = torch.float32, device = source_2.device)
        gather_add.indexed_inner_product_fp32(indexes, source_1, source_2, outputs)

    elif compute_type == "custom_fp16":
        if source_1.dtype != torch.float16:
            source_1 = source_1.to(torch.float16)
        if source_2.dtype != torch.float16:
            source_2 = source_2.to(torch.float16)
        outputs = torch.zeros(batch_size, index_size, dtype = torch.float16, device = source_2.device)
        gather_add.indexed_inner_product_fp16(indexes, source_1, source_2, outputs)

    else:
        raise Exception()

    return outputs

def weighted_vector_scatter_add(indexes, source, weights, output_size, compute_type):
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

    if compute_type == "torch":
        indexes = indexes.long()

        outputs = torch.zeros(output_size, vector_dim, dtype = source.dtype, device = source.device)
        for batch_idx in range(batch_size):
            inst_source = (source[batch_idx, None, :] * weights[batch_idx, :, None]).reshape(index_size, -1)
            inst_indexes = indexes[batch_idx, :, None].repeat(1, vector_dim).reshape(index_size, -1)
            outputs.scatter_add_(0, inst_indexes, inst_source)

    elif compute_type == "custom_fp32":
        if source.dtype != torch.float32:
            source = source.to(torch.float32)
        if weights.dtype != torch.float32:
            weights = weights.to(torch.float32)
        outputs = torch.zeros(output_size, vector_dim, dtype = torch.float32, device = source.device)
        gather_add.weighted_vector_scatter_add_fp32(indexes, source, weights, outputs)

    elif compute_type == "custom_fp16":
        if source.dtype != torch.float16:
            source = source.to(torch.float16)
        if weights.dtype != torch.float16:
            weights = weights.to(torch.float16)
        outputs = torch.zeros(output_size, vector_dim, dtype = torch.float16, device = source.device)
        gather_add.weighted_vector_scatter_add_fp16(indexes, source, weights, outputs)

    else:
        raise Exception()

    return outputs
