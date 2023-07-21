#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_launch.h"
#include "cuda_kernel.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

at::Tensor lsh_cumulation_kernel(
  at::Tensor key_mask,
  at::Tensor key_hashcode,
  at::Tensor value,
  int hashtable_capacity
) {

  int batch_size = key_hashcode.size(0);
  int num_key = key_hashcode.size(1);
  int num_hash_f = key_hashcode.size(2);
  int value_dim = value.size(2);
  int num_worker = OPTIMAL_THREADS / value_dim;

  at::Tensor hashtable = at::zeros({batch_size, num_hash_f, hashtable_capacity, value_dim}, value.options());

  dim3 threads(value_dim, num_worker);
  dim3 blocks(num_key, batch_size);
  int shared_mem = num_hash_f * sizeof(float);

  lsh_cumulation_cuda_kernel<<<blocks, threads, shared_mem>>>(
    key_mask.data_ptr<int>(),
    key_hashcode.data_ptr<int>(),
    value.data_ptr<float>(),
    hashtable.data_ptr<float>(),
    batch_size,
    num_hash_f,
    hashtable_capacity,
    num_key,
    value_dim
  );

  return hashtable;

}

at::Tensor lsh_query_kernel(
  at::Tensor query_mask,
  at::Tensor query_hashcode,
  at::Tensor hashtable
) {

  int batch_size = query_hashcode.size(0);
  int num_query = query_hashcode.size(1);
  int num_hash_f = query_hashcode.size(2);
  int hashtable_capacity = hashtable.size(2);
  int value_dim = hashtable.size(3);
  int num_worker = OPTIMAL_THREADS / value_dim;

  at::Tensor cumulation_value = at::zeros({batch_size, num_query, value_dim}, hashtable.options());

  dim3 threads(value_dim, num_worker);
  dim3 blocks(num_query, batch_size);
  int shared_mem = num_hash_f * sizeof(float);

  lsh_query_cuda_kernel<<<blocks, threads, shared_mem>>>(
    query_mask.data_ptr<int>(),
    query_hashcode.data_ptr<int>(),
    hashtable.data_ptr<float>(),
    cumulation_value.data_ptr<float>(),
    batch_size,
    num_hash_f,
    hashtable_capacity,
    num_query,
    value_dim
  );

  return cumulation_value;

}

at::Tensor lsh_cumulation_query_kernel(
  at::Tensor query_mask,
  at::Tensor query_hashcode,
  at::Tensor key_mask,
  at::Tensor key_hashcode,
  at::Tensor value,
  int hashtable_capacity
) {

  int batch_size = query_hashcode.size(0);
  int num_hash_f = query_hashcode.size(2);

  int num_query = query_hashcode.size(1);
  int num_key = key_hashcode.size(1);
  int value_dim = value.size(2);

  at::Tensor hashtable_value = at::empty({batch_size, num_hash_f, hashtable_capacity, WARP_SIZE}, value.options());
  at::Tensor cumulation_value = at::zeros({batch_size, num_query, value_dim}, value.options());

  int threads_x = WARP_SIZE;
  int threads_y = OPTIMAL_THREADS / WARP_SIZE;
  int block_x_step1 = num_key / threads_y;
  int block_x_step2 = num_query / threads_y;
  int block_y = batch_size;

  dim3 threads(threads_x, threads_y);
  dim3 blocks_step1(block_x_step1, block_y);
  dim3 blocks_step2(block_x_step2, block_y);

  for (int value_offset = 0; value_offset < value_dim; value_offset = value_offset + WARP_SIZE) {

    cudaMemset(hashtable_value.data_ptr<float>(), 0, (batch_size * num_hash_f * hashtable_capacity * WARP_SIZE) * sizeof(float));

    lsh_cumulation_query_step1_cuda_kernel<<<blocks_step1, threads>>>(
      key_mask.data_ptr<int>(),
      key_hashcode.data_ptr<int>(),
      value.data_ptr<float>(),
      hashtable_value.data_ptr<float>(),
      batch_size,
      num_hash_f,
      hashtable_capacity,
      num_key,
      value_dim,
      value_offset
    );

    lsh_cumulation_query_step2_cuda_kernel<<<blocks_step2, threads>>>(
      query_mask.data_ptr<int>(),
      query_hashcode.data_ptr<int>(),
      hashtable_value.data_ptr<float>(),
      cumulation_value.data_ptr<float>(),
      batch_size,
      num_hash_f,
      hashtable_capacity,
      num_query,
      value_dim,
      value_offset
    );
  }

  return cumulation_value;

}
