#include <torch/extension.h>
#include <ATen/ATen.h>

at::Tensor lsh_cumulation_kernel(
  at::Tensor key_mask,
  at::Tensor key_hashcode,
  at::Tensor value,
  int hashtable_capacity
);

at::Tensor lsh_query_kernel(
  at::Tensor query_mask,
  at::Tensor query_hashcode,
  at::Tensor hashtable
);

at::Tensor lsh_cumulation_query_kernel(
  at::Tensor query_mask,
  at::Tensor query_hashcode,
  at::Tensor key_mask,
  at::Tensor key_hashcode,
  at::Tensor value,
  int hashtable_capacity
);
