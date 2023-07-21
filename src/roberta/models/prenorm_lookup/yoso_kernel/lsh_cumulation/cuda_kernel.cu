#include "cuda_kernel.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void lsh_cumulation_cuda_kernel(
  int *key_mask,           // [batch_size, num_key]
  int *key_hashcode,       // [batch_size, num_key, num_hash_f]
  float *value,            // [batch_size, num_key, value_dim]
  float *hashtable,        // [batch_size, num_hash_f, hashtable_capacity, value_dim]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_key,
  int value_dim
) {

  int batch_idx = blockIdx.y;
  int key_idx = blockIdx.x;

  int num_worker = blockDim.y;
  int worker_id = threadIdx.y;

  int dim_idx = threadIdx.x;

  int num_thread = blockDim.x * blockDim.y;
  int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  int batch_idx__key_idx = batch_idx * num_key + key_idx;

  if (key_mask[batch_idx__key_idx] == 0) {
    return;
  }

  extern __shared__ float buffer[];
  int *key_hashcode_buffer = (int*)buffer;

  for (int hash_f_start = 0; hash_f_start < num_hash_f; hash_f_start = hash_f_start + num_thread) {
    int hash_f_idx = hash_f_start + thread_id;
    if (hash_f_idx < num_hash_f) {
      key_hashcode_buffer[hash_f_idx] = key_hashcode[batch_idx__key_idx * num_hash_f + hash_f_idx];
    }
  }
  __syncthreads();

  float value_scalar = value[batch_idx__key_idx * value_dim + dim_idx];
  for (int hash_f_start = 0; hash_f_start < num_hash_f; hash_f_start = hash_f_start + num_worker) {
    int hash_f_idx = hash_f_start + worker_id;
    int hashcode = key_hashcode_buffer[hash_f_idx];
    int hashtable_idx = (batch_idx * num_hash_f + hash_f_idx) * hashtable_capacity + hashcode;
    atomicAdd(&hashtable[hashtable_idx * value_dim + dim_idx], value_scalar);
  }
}

__global__ void lsh_query_cuda_kernel(
  int *query_mask,         // [batch_size, num_query]
  int *query_hashcode,     // [batch_size, num_query, num_hash_f]
  float *hashtable,        // [batch_size, num_hash_f, hashtable_capacity, value_dim]
  float *cumulation_value, // [batch_size, num_query, value_dim]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_query,
  int value_dim
) {

  int batch_idx = blockIdx.y;
  int query_idx = blockIdx.x;

  int num_worker = blockDim.y;
  int worker_id = threadIdx.y;

  int dim_idx = threadIdx.x;

  int num_thread = blockDim.x * blockDim.y;
  int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  int batch_idx__query_idx = batch_idx * num_query + query_idx;

  if (query_mask[batch_idx__query_idx] == 0) {
    return;
  }

  extern __shared__ float buffer[];
  int *query_hashcode_buffer = (int*)buffer;

  for (int hash_f_start = 0; hash_f_start < num_hash_f; hash_f_start = hash_f_start + num_thread) {
    int hash_f_idx = hash_f_start + thread_id;
    if (hash_f_idx < num_hash_f) {
      query_hashcode_buffer[hash_f_idx] = query_hashcode[batch_idx__query_idx * num_hash_f + hash_f_idx];
    }
  }
  __syncthreads();

  float cumulation_value_scalar = 0;
  for (int hash_f_start = 0; hash_f_start < num_hash_f; hash_f_start = hash_f_start + num_worker) {
    int hash_f_idx = hash_f_start + worker_id;
    int hashcode = query_hashcode_buffer[hash_f_idx];
    int hashtable_idx = (batch_idx * num_hash_f + hash_f_idx) * hashtable_capacity + hashcode;
    cumulation_value_scalar += hashtable[hashtable_idx * value_dim + dim_idx];
  }
  atomicAdd(&cumulation_value[batch_idx__query_idx * value_dim + dim_idx], cumulation_value_scalar / (float)num_hash_f);
}


__global__ void lsh_cumulation_query_step1_cuda_kernel(
  int *key_mask,           // [batch_size, num_key]
  int *key_hashcode,      // [batch_size, num_key, num_hash_f]
  float *value,            // [batch_size, num_key, value_dim]
  float *hashtable_value,  // [batch_size, num_hash_f, hashtable_capacity, WARP_SIZE]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_key,
  int value_dim,
  int offset_warp
) {

  int warp_thread_idx = threadIdx.x;

  int batch_idx = blockIdx.y;
  int key_idx = blockIdx.x * blockDim.y + threadIdx.y;

  int batch_idx__key_idx = batch_idx * num_key + key_idx;
  if (key_mask[batch_idx__key_idx] == 0) {
    return;
  }

  if (num_hash_f > WARP_SIZE) {
    float warp_value = value[batch_idx__key_idx * value_dim + offset_warp + warp_thread_idx];
    for (int hash_f_start = 0; hash_f_start < num_hash_f; hash_f_start = hash_f_start + WARP_SIZE) {
      int warp_hashcode = key_hashcode[batch_idx__key_idx * num_hash_f + hash_f_start + warp_thread_idx];
      #pragma unroll
      for (int hash_f_offset = 0; hash_f_offset < WARP_SIZE; hash_f_offset++) {
        int current_hashcode = warp_hashcode;
        current_hashcode = __shfl_sync(FULL_MASK, current_hashcode, hash_f_offset);
        int hashtable_idx = (batch_idx * num_hash_f + (hash_f_start + hash_f_offset)) * hashtable_capacity + current_hashcode;
        atomicAdd(&hashtable_value[hashtable_idx * WARP_SIZE + warp_thread_idx], warp_value);
      }
    }
  } else {
    float warp_value = value[batch_idx__key_idx * value_dim + offset_warp + warp_thread_idx];
    int warp_hashcode = 0;
    if (warp_thread_idx < num_hash_f) {
      warp_hashcode = key_hashcode[batch_idx__key_idx * num_hash_f + warp_thread_idx];
    }
    for (int hash_f_idx = 0; hash_f_idx < num_hash_f; hash_f_idx++) {
      int current_hashcode = warp_hashcode;
      current_hashcode = __shfl_sync(FULL_MASK, current_hashcode, hash_f_idx);
      int hashtable_idx = (batch_idx * num_hash_f + hash_f_idx) * hashtable_capacity + current_hashcode;
      atomicAdd(&hashtable_value[hashtable_idx * WARP_SIZE + warp_thread_idx], warp_value);
    }
  }

}

__global__ void lsh_cumulation_query_step2_cuda_kernel(
  int *query_mask,         // [batch_size, num_query]
  int *query_hashcode,    // [batch_size, num_query, num_hash_f]
  float *hashtable_value,  // [batch_size, num_hash_f, hashtable_capacity, WARP_SIZE]
  float *cumulation_value, // [batch_size, num_query, value_dim]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_query,
  int value_dim,
  int offset_warp
) {

  int warp_thread_idx = threadIdx.x;

  int batch_idx = blockIdx.y;
  int query_idx = blockIdx.x * blockDim.y + threadIdx.y;

  int batch_idx__query_idx = batch_idx * num_query + query_idx;
  if (query_mask[batch_idx__query_idx] == 0) {
    return;
  }

  if (num_hash_f > WARP_SIZE) {
    float warp_value = 0;
    for (int hash_f_start = 0; hash_f_start < num_hash_f; hash_f_start = hash_f_start + WARP_SIZE) {
      int warp_hashcode = query_hashcode[batch_idx__query_idx * num_hash_f + hash_f_start + warp_thread_idx];
      #pragma unroll
      for (int hash_f_offset = 0; hash_f_offset < WARP_SIZE; hash_f_offset++) {
        int current_hashcode = warp_hashcode;
        current_hashcode = __shfl_sync(FULL_MASK, current_hashcode, hash_f_offset);
        int hashtable_idx = (batch_idx * num_hash_f + (hash_f_start + hash_f_offset)) * hashtable_capacity + current_hashcode;
        warp_value = warp_value + hashtable_value[hashtable_idx * WARP_SIZE + warp_thread_idx];
      }
    }
    cumulation_value[batch_idx__query_idx * value_dim + offset_warp + warp_thread_idx] = warp_value / float(num_hash_f);
  } else {
    float warp_value = 0;
    int warp_hashcode = 0;
    if (warp_thread_idx < num_hash_f) {
      warp_hashcode = query_hashcode[batch_idx__query_idx * num_hash_f + warp_thread_idx];
    }
    for (int hash_f_idx = 0; hash_f_idx < num_hash_f; hash_f_idx++) {
      int current_hashcode = warp_hashcode;
      current_hashcode = __shfl_sync(FULL_MASK, current_hashcode, hash_f_idx);
      int hashtable_idx = (batch_idx * num_hash_f + hash_f_idx) * hashtable_capacity + current_hashcode;
      warp_value = warp_value + hashtable_value[hashtable_idx * WARP_SIZE + warp_thread_idx];
    }
    cumulation_value[batch_idx__query_idx * value_dim + offset_warp + warp_thread_idx] = warp_value / float(num_hash_f);
  }

}
