#include <torch/extension.h>
#include <ATen/ATen.h>

at::Tensor weighted_lsh_cumulation_sorted_query_kernel(
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
);

at::Tensor weighted_lsh_cumulation_sorted_key_kernel(
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
);

at::Tensor weighted_lsh_cumulation_merged_shared_qk_kernel(
  at::Tensor sort_info,
  at::Tensor sorted_idxes,
  at::Tensor mask,
  at::Tensor hashcode,
  at::Tensor weight_0,
  at::Tensor weight_1,
  at::Tensor value,
  int workspace_size,
  float tau
);
