#include <torch/extension.h>
#include <ATen/ATen.h>

at::Tensor fast_hadamard_transform_kernel_dim512_fp16(
  at::Tensor inp
);

at::Tensor fast_hadamard_transform_kernel_dim1024_fp16(
  at::Tensor inp
);

at::Tensor fast_hadamard_transform_kernel(
  at::Tensor inp
);
