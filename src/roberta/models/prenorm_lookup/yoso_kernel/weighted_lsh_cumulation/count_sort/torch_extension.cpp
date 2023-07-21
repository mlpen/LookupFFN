#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_launch.h"
#include <vector>

std::vector<at::Tensor> count_sort(
  at::Tensor key_mask,
  at::Tensor key_hashcode,
  int hashtable_capacity
) {
  return count_sort_kernel(
    key_mask,
    key_hashcode,
    hashtable_capacity
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("count_sort", &count_sort, "Count Sort (CUDA)");
}
