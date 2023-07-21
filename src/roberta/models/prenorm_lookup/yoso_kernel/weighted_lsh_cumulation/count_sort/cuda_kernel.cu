#include "cuda_kernel.h"
#include <stdio.h>

__device__ inline void cumulative_sum(int *buffer, int buffer_size, int thread_id, int num_thread) {
  for (int slot_start = 0; slot_start < buffer_size; slot_start = slot_start + num_thread) {
    int slot = slot_start + thread_id;
    int thread_value = buffer[slot];
    int next_thread_value = 0;
    for (int offset = 1; offset < WARP_SIZE; offset = offset << 1) {
      next_thread_value = __shfl_up_sync(FULL_MASK, thread_value, offset);
      if (thread_id % WARP_SIZE >= offset) {
        thread_value = thread_value + next_thread_value;
      }
    }
    buffer[slot] = thread_value;
  }
  __syncthreads();
  if (thread_id < WARP_SIZE) {
    for (int slot_start = WARP_SIZE; slot_start < buffer_size; slot_start = slot_start + WARP_SIZE) {
      buffer[slot_start + thread_id] += buffer[slot_start - 1];
    }
  }
  __syncthreads();
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void count_sort_cuda_kernel(
  int *key_mask,      // [batch_size, num_key]
  int *key_hashcode,  // [batch_size, num_hash_f, num_key]
  int *sort_info,     // [batch_size, num_hash_f, hashtable_capacity]
  int *sorted_idxes,  // [batch_size, num_hash_f, num_key]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_key
) {

  int batch_idx = blockIdx.y;
  int hash_f_idx = blockIdx.x;

  int num_thread = blockDim.x;
  int thread_id = threadIdx.x;

  extern __shared__ float buffer[];
  int *table_buffer = (int*)buffer;
  int *key_buffer = (int*)&buffer[hashtable_capacity];
  int *result_buffer = (int*)&buffer[hashtable_capacity + num_key];

  int *key_hashcode_pt = &key_hashcode[(batch_idx * num_hash_f + hash_f_idx) * num_key];
  int *sorted_idxes_pt = &sorted_idxes[(batch_idx * num_hash_f + hash_f_idx) * num_key];
  int *sort_info_pt = &sort_info[(batch_idx * num_hash_f + hash_f_idx) * hashtable_capacity];

  // load keys to shared memory
  for (int key_idx_start = 0; key_idx_start < num_key; key_idx_start = key_idx_start + num_thread) {
    int key_idx = key_idx_start + thread_id;
    if (key_idx < num_key) {
      if (key_mask[batch_idx * num_key + key_idx] == 0) {
        key_buffer[key_idx] = EMPTY_VALUE;
      } else {
        key_buffer[key_idx] = key_hashcode_pt[key_idx];
      }
    }
  }

  // initialize table to zeros
  for (int slot_start = 0; slot_start < hashtable_capacity; slot_start = slot_start + num_thread) {
    int slot = slot_start + thread_id;
    table_buffer[slot] = 0;
  }
  __syncthreads();

  // counting the number of keys for each code
  for (int key_idx_start = 0; key_idx_start < num_key; key_idx_start = key_idx_start + num_thread) {
    int key_idx = key_idx_start + thread_id;
    if (key_idx < num_key) {
      int key = key_buffer[key_idx];
      if (key != EMPTY_VALUE) {
        atomicAdd(&table_buffer[key], 1);
      }
    }
  }
  __syncthreads();

  cumulative_sum(table_buffer, hashtable_capacity, thread_id, num_thread);

  // write cumulative count to sort_info result
  for (int slot_start = 0; slot_start < hashtable_capacity; slot_start = slot_start + num_thread) {
    int slot = slot_start + thread_id;
    sort_info_pt[slot] = table_buffer[slot];
  }
  __syncthreads();

  // compute the index of each sorted key and place the key in key_buffer shared memory
  for (int key_idx_start = 0; key_idx_start < num_key; key_idx_start = key_idx_start + num_thread) {
    int key_idx = key_idx_start + thread_id;
    if (key_idx < num_key) {
      int key = key_buffer[key_idx];
      if (key != EMPTY_VALUE) {
        int insert_idx = atomicAdd(&table_buffer[key], -1) - 1;
        result_buffer[insert_idx] = key_idx;
      }
    }
  }
  __syncthreads();

  // write key_buffer to key_sorted_idxes result
  for (int key_idx_start = 0; key_idx_start < num_key; key_idx_start = key_idx_start + num_thread) {
    int key_idx = key_idx_start + thread_id;
    if (key_idx < num_key) {
      sorted_idxes_pt[key_idx] = result_buffer[key_idx];
    }
  }
}
