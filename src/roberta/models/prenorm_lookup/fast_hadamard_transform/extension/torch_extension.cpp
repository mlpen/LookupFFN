#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_launch.h"

at::Tensor fast_hadamard_transform_dim512_fp16(
  at::Tensor inp
) {
  return fast_hadamard_transform_kernel_dim512_fp16(
    inp
  );
}

at::Tensor fast_hadamard_transform_dim1024_fp16(
  at::Tensor inp
) {
  return fast_hadamard_transform_kernel_dim1024_fp16(
    inp
  );
}

at::Tensor fast_hadamard_transform(
  at::Tensor inp
) {
  return fast_hadamard_transform_kernel(
    inp
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fast_hadamard_transform_dim512_fp16", &fast_hadamard_transform_dim512_fp16, "fast_hadamard_transform_dim512_fp16 (CUDA)");
  m.def("fast_hadamard_transform_dim1024_fp16", &fast_hadamard_transform_dim1024_fp16, "fast_hadamard_transform_dim1024_fp16 (CUDA)");
  m.def("fast_hadamard_transform", &fast_hadamard_transform, "fast_hadamard_transform (CUDA)");
}
