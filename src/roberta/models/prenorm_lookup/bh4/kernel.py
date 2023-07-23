
import torch
import torch.nn as nn
from .cpu.kernel import BH4Function
from .fast_hadamard_transform.kernel import hadamard
import os
import time
import math

def bh4(x, w, training):
    if training:
        assert x.device == torch.device("cuda")
        assert w.device == torch.device("cuda")
        return bh4_cuda(x, w)
    else:
        if x.device == torch.device("cuda"):
            assert w.device == torch.device("cuda")
            return bh4_cuda(x, w)
        elif x.device == torch.device("cpu"):
            assert w.device == torch.device("cpu")
            return BH4Function.apply(x, w)
        else:
            raise NotImplementedError

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
    return torch.stack(out, dim = -1)

# if __name__ == "__main__":
#     unit_test()
#     autograd_unit_test()
#     profile()