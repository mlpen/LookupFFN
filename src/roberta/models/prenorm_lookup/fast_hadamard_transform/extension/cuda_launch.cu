#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_launch.h"
#include "cuda_kernel.h"
#include <cuda.h>
#include <cuda_fp16.h>

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

at::Tensor fast_hadamard_transform_kernel_dim512_fp16(
  at::Tensor inp
) {

  int batch_size = inp.size(0);
  int vector_dim = inp.size(1);

  at::Tensor out = at::zeros({batch_size, vector_dim}, inp.options());

  dim3 threads(WARP_SIZE, 8);
  dim3 blocks(batch_size / 8 + 1);

  fast_hadamard_transform_cuda_kernel_dim512_fp16<<<blocks, threads>>>(
    (__half2*)(inp.data_ptr<at::Half>()),
    (__half2*)(out.data_ptr<at::Half>()),
    batch_size
  );

  return out;

}

at::Tensor fast_hadamard_transform_kernel_dim1024_fp16(
  at::Tensor inp
) {

  int batch_size = inp.size(0);
  int vector_dim = inp.size(1);

  at::Tensor out = at::zeros({batch_size, vector_dim}, inp.options());

  dim3 threads(WARP_SIZE, 8);
  dim3 blocks(batch_size / 8 + 1);

  fast_hadamard_transform_cuda_kernel_dim1024_fp16<<<blocks, threads>>>(
    (__half2*)(inp.data_ptr<at::Half>()),
    (__half2*)(out.data_ptr<at::Half>()),
    batch_size
  );

  return out;

}


at::Tensor fast_hadamard_transform_kernel(
  at::Tensor inp
) {

  int batch_size = inp.size(0);
  int vector_dim = inp.size(1);

  at::Tensor out = at::zeros({batch_size, vector_dim}, inp.options());

  dim3 threads(vector_dim);
  dim3 blocks(batch_size);
  int shared_mem = vector_dim * sizeof(float);

  fast_hadamard_transform_cuda_kernel<<<blocks, threads, shared_mem>>>(
    inp.data_ptr<float>(),
    out.data_ptr<float>(),
    batch_size,
    vector_dim
  );

  return out;

}
