#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_launch.h"

at::Tensor lsh_cumulation(
  at::Tensor key_mask,
  at::Tensor key_hashcode,
  at::Tensor value,
  int hashtable_capacity
) {
  return lsh_cumulation_kernel(
    key_mask,
    key_hashcode,
    value,
    hashtable_capacity
  );
}

at::Tensor lsh_query(
  at::Tensor query_mask,
  at::Tensor query_hashcode,
  at::Tensor hashtable
) {
  return lsh_query_kernel(
    query_mask,
    query_hashcode,
    hashtable
  );
}

at::Tensor lsh_cumulation_query(
  at::Tensor query_mask,         // [batch_size, num_query]
  at::Tensor query_hashcode,     // [batch_size, num_query, num_hash_f]
  at::Tensor key_mask,           // [batch_size, num_key]
  at::Tensor key_hashcode,       // [batch_size, num_key, num_hash_f]
  at::Tensor value,              // [batch_size, num_key, value_dim]
  int hashtable_capacity
) {
  return lsh_cumulation_query_kernel(
    query_mask,
    query_hashcode,
    key_mask,
    key_hashcode,
    value,
    hashtable_capacity
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("lsh_cumulation", &lsh_cumulation, "LSH Cumulation (CUDA)");
  m.def("lsh_query", &lsh_query, "LSH Query (CUDA)");
  m.def("lsh_cumulation_query", &lsh_cumulation_query, "LSH Cumulation and Query (CUDA)");
}
