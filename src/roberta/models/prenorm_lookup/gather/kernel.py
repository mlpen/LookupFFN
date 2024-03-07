
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
        assert indices.device == torch.device("cuda")
        assert weights.device == torch.device("cuda")
        assert tables.device == torch.device("cuda")
        return weighted_gather_add_cuda(indices, tables, weights)
    else:
        if indices.device == torch.device("cpu"):
            assert weights.device == torch.device("cpu")
            assert tables.device == torch.device("cpu")
            return weighted_gather_add_cpu(indices, tables, weights)
        else:
            assert weights.device == torch.device("cuda")
            assert tables.device == torch.device("cuda")
            return weighted_gather_add_cuda(indices, tables, weights)
        
    