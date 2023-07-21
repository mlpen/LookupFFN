
import torch
import torch.nn as nn
import os
import time
import random
import math
from torch.utils.checkpoint import checkpoint
from torch.autograd import Function
from .kernel import weighted_vector_gather_add
from .kernel import indexed_inner_product
from .kernel import weighted_vector_scatter_add

class WeightedVectorGatherAdd(Function):
    @staticmethod
    def forward(ctx, indexes, source, weights, compute_type):
        # indexes = [batch_size, index_size]
        # source = [source_size, vector_dim]
        # weights = [batch_size, index_size]
        # outputs = [batch_size, vector_dim]

        indexes, source, weights = indexes.contiguous(), source.contiguous(), weights.contiguous()
        ctx.save_for_backward(indexes, source, weights)
        ctx._compute_type = compute_type

        outputs = weighted_vector_gather_add(indexes, source, weights, compute_type = ctx._compute_type)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        # grad_outputs = [batch_size, vector_dim]
        # grad_source = [source_size, vector_dim]
        # grad_weights = [batch_size, index_size]

        grad_outputs = grad_outputs.contiguous()
        indexes, source, weights = ctx.saved_tensors

        grad_weights = indexed_inner_product(indexes, grad_outputs, source, compute_type = ctx._compute_type)
        grad_source = weighted_vector_scatter_add(indexes, grad_outputs, weights, source.shape[0], compute_type = ctx._compute_type)

        return None, grad_source, grad_weights, None

class WeightedVectorScatterAdd(Function):
    @staticmethod
    def forward(ctx, indexes, source, weights, output_size, compute_type):
        # indexes = [batch_size, index_size]
        # source = [batch_size, vector_dim]
        # weights = [batch_size, index_size]
        # outputs = [output_size, vector_dim]

        indexes, source, weights = indexes.contiguous(), source.contiguous(), weights.contiguous()
        ctx.save_for_backward(indexes, source, weights)
        ctx._compute_type = compute_type

        outputs = weighted_vector_scatter_add(indexes, source, weights, output_size, compute_type = ctx._compute_type)

        return outputs

    @staticmethod
    def backward(ctx, grad_outputs):
        # grad_outputs = [output_size, vector_dim]
        # grad_source = [batch_size, vector_dim]
        # grad_weights = [batch_size, index_size]

        grad_outputs = grad_outputs.contiguous()
        indexes, source, weights = ctx.saved_tensors

        grad_weights = indexed_inner_product(indexes, source, grad_outputs, compute_type = ctx._compute_type)
        grad_source = weighted_vector_gather_add(indexes, grad_outputs, weights, compute_type = ctx._compute_type)

        return None, grad_source, grad_weights, None, None

def weighted_gather_add(indexes, source, weights, compute_type):
    if compute_type == "torch":
        return weighted_vector_gather_add(indexes, source, weights, "torch")
    elif compute_type.startswith("custom"):
        return WeightedVectorGatherAdd.apply(indexes, source, weights, compute_type)
    else:
        raise Exception()

def weighted_scatter_add(indexes, source, weights, output_size, compute_type):
    if compute_type == "torch":
        return weighted_vector_scatter_add(indexes, source, weights, output_size, "torch")
    elif compute_type.startswith("custom"):
        return WeightedVectorScatterAdd.apply(indexes, source, weights, output_size, compute_type)
    else:
        raise Exception()
