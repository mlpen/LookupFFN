#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_launch.h"

at::Tensor weighted_lsh_cumulation_sorted_query(
  at::Tensor query_sort_info,
  at::Tensor query_sorted_idxes,
  at::Tensor key_mask,
  at::Tensor key_hashcode,
  at::Tensor query_weight_0,
  at::Tensor key_weight_0,
  at::Tensor query_weight_1,
  at::Tensor key_weight_1,
  at::Tensor value,
  int workspace_size,
  float tau
) {
  return weighted_lsh_cumulation_sorted_query_kernel(
    query_sort_info,
    query_sorted_idxes,
    key_mask,
    key_hashcode,
    query_weight_0,
    key_weight_0,
    query_weight_1,
    key_weight_1,
    value,
    workspace_size,
    tau
  );
}

at::Tensor weighted_lsh_cumulation_sorted_key(
  at::Tensor query_mask,
  at::Tensor query_hashcode,
  at::Tensor key_sort_info,
  at::Tensor key_sorted_idxes,
  at::Tensor query_weight_0,
  at::Tensor key_weight_0,
  at::Tensor query_weight_1,
  at::Tensor key_weight_1,
  at::Tensor value,
  int workspace_size,
  float tau
) {
  return weighted_lsh_cumulation_sorted_key_kernel(
    query_mask,
    query_hashcode,
    key_sort_info,
    key_sorted_idxes,
    query_weight_0,
    key_weight_0,
    query_weight_1,
    key_weight_1,
    value,
    workspace_size,
    tau
  );
}

at::Tensor weighted_lsh_cumulation_merged_shared_qk(
  at::Tensor sort_info,
  at::Tensor sorted_idxes,
  at::Tensor mask,
  at::Tensor hashcode,
  at::Tensor weight_0,
  at::Tensor weight_1,
  at::Tensor value,
  int workspace_size,
  float tau
) {
  return weighted_lsh_cumulation_merged_shared_qk_kernel(
    sort_info,
    sorted_idxes,
    mask,
    hashcode,
    weight_0,
    weight_1,
    value,
    workspace_size,
    tau
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("weighted_lsh_cumulation_sorted_query", &weighted_lsh_cumulation_sorted_query, "Weighted LSH Cumulation Sorted Query (CUDA)");
  m.def("weighted_lsh_cumulation_sorted_key", &weighted_lsh_cumulation_sorted_key, "Weighted LSH Cumulation Sorted Key (CUDA)");
  m.def("weighted_lsh_cumulation_merged_shared_qk", &weighted_lsh_cumulation_merged_shared_qk, "Weighted LSH Cumulation Shared QK (CUDA)");
}
