
import torch
import torch.nn as nn
import os
import time
import random
import math
from torch.utils.checkpoint import checkpoint
from torch.autograd import Function
from .kernel import fast_hadamard_transform, hadamard_transform, fast_hadamard_transform_dim512, fast_hadamard_transform_dim1024

class FastHadamardTransform(Function):
    @staticmethod
    def forward(ctx, inp):
        return fast_hadamard_transform(inp)

    @staticmethod
    def backward(ctx, grad_outputs):
        return fast_hadamard_transform(grad_outputs)

class FastHadamardTransformDim512(Function):
    @staticmethod
    def forward(ctx, inp):
        return fast_hadamard_transform_dim512(inp)

    @staticmethod
    def backward(ctx, grad_outputs):
        return fast_hadamard_transform_dim512(grad_outputs)

class FastHadamardTransformDim1024(Function):
    @staticmethod
    def forward(ctx, inp):
        return fast_hadamard_transform_dim1024(inp)

    @staticmethod
    def backward(ctx, grad_outputs):
        return fast_hadamard_transform_dim1024(grad_outputs)

def hadamard(inp, compute_type = "custom"):
    if compute_type == "torch":
        print("Warning: using very slow hadamard transform implementation")
        return hadamard_transform(inp)
    elif compute_type.startswith("custom"):
        if inp.shape[-1] == 512:
            return FastHadamardTransformDim512.apply(inp)
        elif inp.shape[-1] == 1024:
            return FastHadamardTransformDim1024.apply(inp)
        else:
            print("Warning: using slower hadamard transform implementation")
            return FastHadamardTransform.apply(inp)
    else:
        raise Exception()
