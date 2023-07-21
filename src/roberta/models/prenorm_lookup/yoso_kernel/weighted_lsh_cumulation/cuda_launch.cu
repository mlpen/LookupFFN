#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_launch.h"
#include "cuda_kernel.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

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
) {

  int batch_size = key_hashcode.size(0);
  int num_key = key_hashcode.size(1);
  int num_hash_f = key_hashcode.size(2);
  int num_query = query_sorted_idxes.size(2);
  int value_dim = value.size(2);
  int weight_0_dim = query_weight_0.size(2);
  int weight_1_dim = query_weight_1.size(2);
  int hashtable_capacity = query_sort_info.size(2);
  int num_worker = OPTIMAL_THREADS / WARP_SIZE;

  at::Tensor cumulation_value = at::zeros({batch_size, num_query, value_dim}, value.options());

  dim3 threads(WARP_SIZE, num_worker);
  dim3 blocks(num_key, batch_size);
  int shared_mem = max(3 * workspace_size, workspace_size + weight_0_dim + weight_1_dim + value_dim + 3 * WARP_SIZE) * sizeof(float);

  weighted_lsh_cumulation_sorted_query_cuda_kernel<<<blocks, threads, shared_mem>>>(
    query_sort_info.data_ptr<int>(),
    query_sorted_idxes.data_ptr<int>(),
    key_mask.data_ptr<int>(),
    key_hashcode.data_ptr<int>(),
    query_weight_0.data_ptr<float>(),
    key_weight_0.data_ptr<float>(),
    query_weight_1.data_ptr<float>(),
    key_weight_1.data_ptr<float>(),
    value.data_ptr<float>(),
    cumulation_value.data_ptr<float>(),
    batch_size,
    num_hash_f,
    num_query,
    num_key,
    value_dim,
    weight_0_dim,
    weight_1_dim,
    hashtable_capacity,
    workspace_size,
    tau
  );

  return cumulation_value;

}

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
) {

  int batch_size = query_hashcode.size(0);
  int num_query = query_hashcode.size(1);
  int num_hash_f = query_hashcode.size(2);
  int num_key = key_sorted_idxes.size(2);
  int value_dim = value.size(2);
  int weight_0_dim = query_weight_0.size(2);
  int weight_1_dim = query_weight_1.size(2);
  int hashtable_capacity = key_sort_info.size(2);
  int num_worker = OPTIMAL_THREADS / WARP_SIZE;

  at::Tensor cumulation_value = at::zeros({batch_size, num_query, value_dim}, value.options());

  dim3 threads(WARP_SIZE, num_worker);
  dim3 blocks(num_query, batch_size);
  int shared_mem = max(3 * workspace_size, workspace_size + weight_0_dim + weight_1_dim + num_worker * value_dim + 3 * WARP_SIZE) * sizeof(float);

  weighted_lsh_cumulation_sorted_key_cuda_kernel<<<blocks, threads, shared_mem>>>(
    query_mask.data_ptr<int>(),
    query_hashcode.data_ptr<int>(),
    key_sort_info.data_ptr<int>(),
    key_sorted_idxes.data_ptr<int>(),
    query_weight_0.data_ptr<float>(),
    key_weight_0.data_ptr<float>(),
    query_weight_1.data_ptr<float>(),
    key_weight_1.data_ptr<float>(),
    value.data_ptr<float>(),
    cumulation_value.data_ptr<float>(),
    batch_size,
    num_hash_f,
    num_query,
    num_key,
    value_dim,
    weight_0_dim,
    weight_1_dim,
    hashtable_capacity,
    workspace_size,
    tau
  );

  return cumulation_value;

}

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
) {

  int batch_size = hashcode.size(0);
  int num_vector = hashcode.size(1);
  int num_hash_f = hashcode.size(2);
  int value_dim = value.size(2);
  int weight_dim = weight_0.size(2);
  int hashtable_capacity = sort_info.size(2);
  int num_worker = OPTIMAL_THREADS / WARP_SIZE;

  at::Tensor cumulation_value = at::zeros({batch_size, num_vector, value_dim}, value.options());

  dim3 threads(WARP_SIZE, num_worker);
  dim3 blocks(num_vector, batch_size);
  int shared_mem = max(3 * workspace_size, workspace_size + 2 * weight_dim + value_dim + 3 * WARP_SIZE) * sizeof(float);

  weighted_lsh_cumulation_merged_shared_qk_cuda_kernel<<<blocks, threads, shared_mem>>>(
    sort_info.data_ptr<int>(),
    sorted_idxes.data_ptr<int>(),
    mask.data_ptr<int>(),
    hashcode.data_ptr<int>(),
    weight_0.data_ptr<float>(),
    weight_1.data_ptr<float>(),
    value.data_ptr<float>(),
    cumulation_value.data_ptr<float>(),
    batch_size,
    num_hash_f,
    num_vector,
    value_dim,
    weight_dim,
    hashtable_capacity,
    workspace_size,
    tau
  );

  return cumulation_value;

}
