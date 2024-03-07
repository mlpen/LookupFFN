
import torch
import torch.nn as nn
from .cpu.kernel import BH4Function
from .fast_hadamard_transform.kernel import hadamard
import os
import time
import math

def bh4(x, w, training):
    if training:
        assert x.is_cuda
        assert w.is_cuda
        return bh4_cuda(x, w)
    else:
        if x.device == torch.device("cpu"):
            assert not w.is_cuda
            return BH4Function.apply(x, w)
        else:
            assert w.is_cuda
            return bh4_cuda(x, w)

def bh4_cuda(x, w):
    BS = w.shape[-1]
    B, D = x.shape
    NB = D // BS
    out = []
    for i in range(w.shape[0]):
        y = x
        for j in range(w.shape[1]):
            y = y.reshape(B, NB, BS)
            y = torch.einsum("bni,noi->bno", y, w[i, j])
            y = y.reshape(B, D)
            y = hadamard(y)
        out.append(y)
    return torch.cat(out, dim = -1)

# if __name__ == "__main__":
#     unit_test()
#     autograd_unit_test()
#     profile()