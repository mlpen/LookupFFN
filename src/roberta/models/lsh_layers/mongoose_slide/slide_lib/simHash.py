
import torch
import torch.nn as nn
# from .cupy_kernel import cupyKernel
import numpy as np

kernel = '''

extern "C"
__global__ void fingerprint(const float* src, const int k, const int L, long* fp)
{
        // product (N x kL bits) -> Column-Major Order
        // const int L = gridDim.y;
        // const int k = blockDim.x;
        int offset = (k * L * blockIdx.x) + (k * blockIdx.y + threadIdx.x);
	long value = (threadIdx.x >= k || src[offset] <= 0) ? 0 : 1;
        value <<= threadIdx.x;

        for (int offset = warpSize/2; offset > 0; offset /= 2)
        {
                value |= __shfl_down_sync(0xFFFFFFFF, value, offset, 32);
        }

        if(!threadIdx.x)
        {
                int fp_offset = L * blockIdx.x + blockIdx.y;
                fp[fp_offset] = value;
        }
}
'''

class SimHash(nn.Module):
    def __init__(self, d_, k_, L_, weights=None,seed_=8191, proj_is_param = False):
        super().__init__()
        self.d = d_
        self.k = k_
        self.L = L_
        # self.fp = cupyKernel(kernel, "fingerprint")

        if weights is None:
            self.rp = SimHash.generate(d_, k_, L_, seed_)
        else:
            self.rp = SimHash.generate_from_weight(weights)   

        if proj_is_param:
            self.rp = nn.Parameter(self.rp.clone().detach())
        
    def generate_from_weight(weights):
        matrix = weights
        positive = torch.gt(matrix, 0).int()
        negative = (matrix < 0.0).int()
        result = (positive - negative).float()

        return result
    
    def generate(d, k, L, seed):
        print("random generate hash table weight")
        rand_gen = np.random.RandomState(seed)
        matrix = rand_gen.randn(d, k*L)
        positive = np.greater_equal(matrix, 0.0)
        negative = np.less(matrix, 0.0)
        result = positive.astype(np.float32) - negative.astype(np.float32)
        return torch.from_numpy(result)

    def hash(self, data, transpose=False):
        N, D = data.size()
        srp = torch.matmul(data, self.rp.to(data.device))

        srp_ = srp.reshape(N, self.L, self.k)
        srp_ = (srp_ > 0).int()
        mask = 2 ** torch.arange(self.k).to(data.device)
        result = torch.sum(mask * srp_, -1)

        if transpose:
            result = torch.t(result) 

        return result

    # def fingerprint(self, srp, N):
    #     result = torch.zeros(N, self.L).long().to(srp.device)
    #     self.fp(grid=(N,self.L,1),
    #             block=(32,1,1),
    #             args=[srp.data_ptr(), self.k, self.L, result.data_ptr()],
    #             strm=torch.cuda.current_stream().cuda_stream)
    #     return result.int()

