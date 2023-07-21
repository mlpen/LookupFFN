#include <torch/extension.h>
#include <ATen/ATen.h>
#include <vector>

std::vector<at::Tensor> count_sort_kernel(
  at::Tensor key_mask,
  at::Tensor key_hashcode,
  int hashtable_capacity
);
