
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load
import os
import time
import math

curr_path = os.path.dirname(os.path.realpath(__file__))
src_files = ['cuda_kernel.cu', 'cuda_launch.cu', 'torch_extension.cpp']
src_files = [os.path.join(curr_path, 'extension', file) for file in src_files]
fast_hadamard_transform_kernel = load('fast_hadamard_transform_kernel', src_files, verbose = True)

import fast_hadamard_transform_kernel
from torch.autograd import Function

def fast_hadamard_transform_dim512(X):
    assert X.is_contiguous()
    assert len(X.shape) == 2
    assert X.dtype == torch.float16
    assert X.shape[-1] == 512
    return fast_hadamard_transform_kernel.fast_hadamard_transform_dim512_fp16(X)

def fast_hadamard_transform_dim1024(X):
    assert X.is_contiguous()
    assert len(X.shape) == 2
    assert X.dtype == torch.float16
    assert X.shape[-1] == 1024
    return fast_hadamard_transform_kernel.fast_hadamard_transform_dim1024_fp16(X)

def hadamard_transform(X):
    assert X.is_contiguous()
    assert len(X.shape) == 2
    assert int(2 ** int(math.log2(X.shape[-1]))) == X.shape[-1]
    from scipy.linalg import hadamard
    H = torch.tensor(hadamard(X.shape[-1]), device = X.device, dtype = X.dtype)
    return torch.matmul(X, H)

def fast_hadamard_transform(X):
    assert X.is_contiguous()
    assert len(X.shape) == 2
    assert X.dtype == torch.float32
    assert int(2 ** int(math.log2(X.shape[-1]))) == X.shape[-1]
    return fast_hadamard_transform_kernel.fast_hadamard_transform(X)


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
        inp = inp.contiguous()
        return fast_hadamard_transform_dim512(inp)

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.contiguous()
        return fast_hadamard_transform_dim512(grad_outputs)

class FastHadamardTransformDim1024(Function):
    @staticmethod
    def forward(ctx, inp):
        inp = inp.contiguous()
        return fast_hadamard_transform_dim1024(inp)

    @staticmethod
    def backward(ctx, grad_outputs):
        grad_outputs = grad_outputs.contiguous()
        return fast_hadamard_transform_dim1024(grad_outputs)

def hadamard(inp, use_torch = False):
    if use_torch:
        print("Warning: using very slow hadamard transform implementation")
        return hadamard_transform(inp)
    else:
        if inp.shape[-1] == 512 and inp.dtype == torch.float16:
            return FastHadamardTransformDim512.apply(inp)
        elif inp.shape[-1] == 1024 and inp.dtype == torch.float16:
            return FastHadamardTransformDim1024.apply(inp)
        else:
            print("Warning: using slower hadamard transform implementation")
            return FastHadamardTransform.apply(inp)
        
def corrcoef(x, y):
    return torch.corrcoef(torch.stack([x.reshape(-1).float(), y.reshape(-1).float()], dim = 0))[0, 1]

def measure(func):
    func()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(100):
        func()
    torch.cuda.synchronize()
    t1 = time.time()
    return (t1 - t0) / 100

def autograd_unit_test():
    def run_func(func):
        torch.manual_seed(0)
        inp = 0.02 * torch.randn(batch_size, vector_dim, requires_grad = True).cuda().to(torch.float16)
        target = 0.02 * torch.randn(batch_size, vector_dim).cuda().to(torch.float16)
        print(inp.shape)
        outputs = func(inp)
        loss = ((outputs - target) ** 2).sum(dim = -1).sum()
        inp.retain_grad()
        loss.backward()

        return outputs, inp.grad


    batch_size = 128 * 512
    vector_dim = 512

    ref_A, ref_B = run_func(lambda x: hadamard(x, use_torch = True))
    out_A, out_B = run_func(lambda x: hadamard(x, use_torch = False))
    print(corrcoef(ref_A, out_A))
    print(corrcoef(ref_B, out_B))

    batch_size = 128 * 512
    vector_dim = 1024

    ref_A, ref_B = run_func(lambda x: hadamard(x, use_torch = True))
    out_A, out_B = run_func(lambda x: hadamard(x, use_torch = False))
    print(corrcoef(ref_A, out_A))
    print(corrcoef(ref_B, out_B))

def unit_test():
    batch_size = 128 * 512
    vector_dim = 512

    X = torch.randn(batch_size, vector_dim).cuda()

    ref = hadamard_transform(X)
    out_fp16 = fast_hadamard_transform_dim512(X.to(torch.float16))
    out_fp32 = fast_hadamard_transform(X)
    print(corrcoef(ref, out_fp16))
    print(corrcoef(ref, out_fp32))


    batch_size = 128 * 512
    vector_dim = 1024

    X = torch.randn(batch_size, vector_dim).cuda()

    ref = hadamard_transform(X)
    out_fp16 = fast_hadamard_transform_dim1024(X.to(torch.float16))
    out_fp32 = fast_hadamard_transform(X)
    print(corrcoef(ref, out_fp16))
    print(corrcoef(ref, out_fp32))

def profile():
    batch_size = 128 * 512
    vector_dim = 512

    X = torch.randn(batch_size, vector_dim).cuda().to(torch.float16)
    standard = measure(lambda :hadamard_transform(X))
    fast_fp16 = measure(lambda :fast_hadamard_transform_dim512(X))
    X = X.to(torch.float32)
    fast = measure(lambda :fast_hadamard_transform(X))
    print(f"standard={standard:.5f}, fast_fp16={fast_fp16:.5f}, fast_fp32={fast:.5f}")

    batch_size = 128 * 512
    vector_dim = 1024

    X = torch.randn(batch_size, vector_dim).cuda().to(torch.float16)
    standard = measure(lambda :hadamard_transform(X))
    fast_fp16 = measure(lambda :fast_hadamard_transform_dim1024(X))
    X = X.to(torch.float32)
    fast = measure(lambda :fast_hadamard_transform(X))
    print(f"standard={standard:.5f}, fast_fp16={fast_fp16:.5f}, fast_fp32={fast:.5f}")


if __name__ == "__main__":
    unit_test()
    autograd_unit_test()
    profile()