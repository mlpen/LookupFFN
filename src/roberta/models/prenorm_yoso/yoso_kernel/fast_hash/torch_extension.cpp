#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_launch.h"

at::Tensor fast_hash(
  at::Tensor mask,
  at::Tensor vector,
  at::Tensor Dmat,
  int num_hash_f,
  int hashcode_len
) {
  return fast_hash_kernel(
    mask,
    vector,
    Dmat,
    num_hash_f,
    hashcode_len
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("fast_hash", &fast_hash, "Fast Hash (CUDA)");
}
