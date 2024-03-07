
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import math
from .cuda.kernel import weighted_gather_add as weighted_gather_add_cuda
from .cpu.kernel import weighted_gather_add as weighted_gather_add_cpu

def gather(indices, weights, tables, training):
    if training:
        assert indices.is_cuda
        assert weights.is_cuda
        assert tables.is_cuda
        return weighted_gather_add_cuda(indices, tables, weights)
    else:
        if not indices.is_cuda:
            assert not weights.is_cuda
            assert not tables.is_cuda
            return weighted_gather_add_cpu(indices, tables, weights)
        else:
            assert weights.is_cuda
            assert tables.is_cuda
            return weighted_gather_add_cuda(indices, tables, weights)
        
    