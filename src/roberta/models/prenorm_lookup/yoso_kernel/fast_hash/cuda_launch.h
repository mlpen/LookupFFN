#include <torch/extension.h>
#include <ATen/ATen.h>

at::Tensor fast_hash_kernel(
  at::Tensor mask,
  at::Tensor vector,
  at::Tensor Dmat,
  int num_hash_f,
  int hashcode_len
);
