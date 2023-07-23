
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import math
from scipy.linalg import hadamard
from torch.autograd import Function

try:
    libxsmm_path = '/workspace/libxsmm/install'
    ldpath = os.environ.get("LD_LIBRARY_PATH")
    if ldpath is not None: 
        os.environ["LD_LIBRARY_PATH"] = ldpath + ":" + f'{libxsmm_path}/lib'
    else: 
        os.environ["LD_LIBRARY_PATH"] = f'{libxsmm_path}/lib'

    curr_path = os.path.dirname(os.path.realpath(__file__))
    src_files = ['torch_extension.cpp']
    src_files = [os.path.join(curr_path, file) for file in src_files]

    sbdht_cpu = load(
        'sbdht_cpu',
        src_files,
        extra_include_paths=[f'{libxsmm_path}/include'],
        extra_cflags=['-fopenmp', '-O3', '-march=native'],
        extra_ldflags=['-lgomp', '-O3', '-march=native', f'-L{libxsmm_path}/lib', '-lxsmm'],
        verbose=True)


    import sbdht_cpu

    D = 512
    BS = 64

    sbdht_cpu.setup_xsmm_kernel_float_avx2(BS, BS, BS, D, BS, D)
except Exception as e:
    print(e)

def fwht_(x):
    sbdht_cpu.fwht_cpu_fp32_avx2_(x)

def bh4(x, w):
    expand_ratio = w.shape[0]
    out = []
    for i in range(expand_ratio):
        x_copy = x.clone()
        sbdht_cpu.sbdht(x_copy, w[i])
        out.append(x_copy)
    return torch.cat(out, dim = -1)

class BH4Function(Function):
    @staticmethod
    def forward(ctx, x, w):
        assert x.shape[0] == 64 * 512, x.shape
        assert x.shape[1] == 512, x.shape
        assert w.shape[1] == 4, w.shape
        assert w.shape[2] == 64, w.shape
        assert w.shape[3] == 64, w.shape
        return bh4(x, w)

    @staticmethod
    def backward(*args):
        raise NotImplementedError

def corrcoef(x, y):
    return torch.corrcoef(torch.stack([x.reshape(-1).float(), y.reshape(-1).float()], dim = 0))[0, 1]

def measure(func):
    func()
    t0 = time.time()
    for _ in range(20):
        func()
    t1 = time.time()
    return (t1 - t0) / 20

def unit_test():
    B = 64 * 512
    D = 512
    BS = 64
    NB = D // BS

    x = torch.randn(B, D)
    w = torch.randn(1, 4, NB, BS, BS)

    ref = x.clone()
    for i in range(w.shape[1]):
        ref = torch.einsum("bni,nio->bno", ref.reshape(B, NB, BS), w[0, i]).reshape(B, D)
        fwht_(ref)
    out = bh4(x, w)
    print(corrcoef(ref, out))

def profile():
    B = 64 * 512
    D = 512
    BS = 64
    NB = D // BS

    x = torch.randn(B, D)
    w = torch.randn(1, 4, NB, BS, BS) # [4, 8, 512]
    
    print('BH4 time:', measure(lambda : bh4(x, w)))

    x = torch.randn(B, D)
    w = torch.randn(D, D)

    print('Dense time:', measure(lambda : torch.matmul(x, w)))


if __name__ == '__main__':
    unit_test()
    profile()



