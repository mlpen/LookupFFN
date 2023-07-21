#include "cuda_kernel.h"
#include <math.h>
#include <stdio.h>

__device__ inline float warp_vector_dot(float *vec_0, float *vec_1, int dim, int warp_idx) {
  float dot = 0;
  for (int idx_start = 0; idx_start < dim; idx_start = idx_start + WARP_SIZE) {
    int idx = idx_start + warp_idx;
    float val = vec_0[idx] * vec_1[idx];
    #pragma unroll
    for (int offset = 1; offset < WARP_SIZE; offset = offset << 1) {
      val += __shfl_xor_sync(FULL_MASK, val, offset);
    }
    dot = dot + val;
  }
  return dot;
}

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

__device__ inline float interpolate(float *buffer, float position) {
  int floor = (int)position;
  float dist = position - (float)floor;
  return (1 - dist) * buffer[floor] + dist * buffer[floor + 1];
}

template<typename T>
__device__ inline void init_buffer_nonblocking(T init_value, T *buffer, int buffer_size, int num_thread, int thread_id) {
  for (int i = 0; i < buffer_size; i = i + num_thread) {
    int offset_idx = i + thread_id;
    if (offset_idx < buffer_size) {
      buffer[offset_idx] = init_value;
    }
  }
}

template<typename T>
__device__ inline void copy_data_nonblocking(T *src_pt, T *dist_pt, int data_length, int num_thread, int thread_id) {
  for (int i = 0; i < data_length; i = i + num_thread) {
    int offset_idx = i + thread_id;
    if (offset_idx < data_length) {
      dist_pt[offset_idx] = src_pt[offset_idx];
    }
  }
}

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

__global__ void weighted_lsh_cumulation_sorted_query_cuda_kernel(
  int *query_sort_info,       // [batch_size, num_hash_f, hashtable_capacity]
  int *query_sorted_idxes,    // [batch_size, num_hash_f, num_query]
  int *key_mask,              // [batch_size, num_key]
  int *key_hashcode,          // [batch_size, num_key, num_hash_f]
  float *query_weight_0,      // [batch_size, num_query, weight_0_dim]
  float *key_weight_0,        // [batch_size, num_key, weight_0_dim]
  float *query_weight_1,      // [batch_size, num_query, weight_1_dim]
  float *key_weight_1,        // [batch_size, num_key, weight_1_dim]
  float *value,               // [batch_size, num_key, value_dim]
  float *cumulation_value,    // [batch_size, num_query, value_dim]
  int batch_size,
  int num_hash_f,
  int num_query,
  int num_key,
  int value_dim,
  int weight_0_dim,
  int weight_1_dim,
  int hashtable_capacity,
  int workspace_size,
  float tau
) {

  int batch_idx = blockIdx.y;
  int key_idx = blockIdx.x;

  int num_thread = blockDim.y * blockDim.x;
  int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  int num_worker = blockDim.y;
  int worker_id = threadIdx.y;

  int warp_idx = threadIdx.x;

  int batch_idx__key_idx = batch_idx * num_key + key_idx;
  if (key_mask[batch_idx__key_idx] == 0) {
    return;
  }

  extern __shared__ float buffer[];

  // need space 3 * workspace_size
  int *hashcode_buffer = (int*)buffer;
  int *workspace_value = (int*)&buffer[workspace_size];
  int *workspace_count = (int*)&buffer[2 * workspace_size];

  copy_data_nonblocking<int>(&key_hashcode[batch_idx__key_idx * num_hash_f], hashcode_buffer, num_hash_f, num_thread, thread_id);
  init_buffer_nonblocking<int>(EMPTY_VALUE, workspace_value, workspace_size, num_thread, thread_id);
  init_buffer_nonblocking<int>(0, workspace_count, workspace_size, num_thread, thread_id);

  __syncthreads();

  int rnumber = (int)(key_weight_0[batch_idx__key_idx * weight_0_dim + thread_id % weight_0_dim] * 1000000);
  rnumber = rnumber + hashcode_buffer[thread_id % num_hash_f] + batch_idx__key_idx * num_thread + thread_id;
  rnumber = next_rint(rnumber);

  for (int hash_f_idx_start = 0; hash_f_idx_start < num_hash_f; hash_f_idx_start = hash_f_idx_start + num_worker) {
    int hash_f_idx = hash_f_idx_start + worker_id;
    int hashcode = hashcode_buffer[hash_f_idx];
    int batch_idx__hash_f_idx = batch_idx * num_hash_f + hash_f_idx;
    int batch_idx__hash_f_idx__hashcode = batch_idx__hash_f_idx * hashtable_capacity + hashcode;

    int offset = 0;
    int count = 0;
    if (hashcode != 0) {
      int temp_value = query_sort_info[batch_idx__hash_f_idx__hashcode - warp_idx % 2];
      offset = __shfl_sync(FULL_MASK, temp_value, 1);
      count = __shfl_sync(FULL_MASK, temp_value, 0) - offset;
    } else {
      offset = 0;
      count = query_sort_info[batch_idx__hash_f_idx__hashcode];
    }

    while (count > 0) {
      int work_size = min(count, WARP_SIZE);
      rnumber = next_rint(rnumber);
      if (warp_idx < work_size) {
        int query_idx = query_sorted_idxes[batch_idx__hash_f_idx * num_query + offset + warp_idx];
        int slot = query_idx % workspace_size;
        if (((float)rnumber / (float)LCG_M) * (float)atomicAdd(&workspace_count[slot], 1) < 1) {
          workspace_value[slot] = query_idx;
        }
      }
      offset += work_size;
      count -= work_size;
    }
  }

  __syncthreads();

  for (int slot_start = 0; slot_start < workspace_size; slot_start = slot_start + num_thread) {
    int slot = slot_start + thread_id;
    workspace_count[slot] = (int)(workspace_count[slot] > 0);
  }
  __syncthreads();
  cumulative_sum(workspace_count, workspace_size, thread_id, num_thread);

  int *nonzero_idxes = (int*)buffer;
  for (int slot_start = 0; slot_start < workspace_size; slot_start = slot_start + num_thread) {
    int slot = slot_start + thread_id;
    if (workspace_value[slot] != EMPTY_VALUE) {
      nonzero_idxes[workspace_count[slot] - 1] = workspace_value[slot];
    }
  }
  int num_nonzero_idxes = workspace_count[workspace_size - 1];
  __syncthreads();

  // need space workspace_size + weight_0_dim + weight_1_dim + value_dim + 3 * WARP_SIZE
  float *weight_0_buffer = &buffer[workspace_size];
  float *weight_1_buffer = &buffer[workspace_size + weight_0_dim];
  float *value_buffer = &buffer[workspace_size + weight_0_dim + weight_1_dim];
  float *scaled_acos_buffer = &buffer[workspace_size + weight_0_dim + weight_1_dim + value_dim];
  float *inv_sqrt_buffer = &buffer[workspace_size + weight_0_dim + weight_1_dim + value_dim + WARP_SIZE];
  float *pow_buffer = &buffer[workspace_size + weight_0_dim + weight_1_dim + value_dim + 2 * WARP_SIZE];

  copy_data_nonblocking<float>(&key_weight_0[batch_idx__key_idx * weight_0_dim], weight_0_buffer, weight_0_dim, num_thread, thread_id);
  copy_data_nonblocking<float>(&key_weight_1[batch_idx__key_idx * weight_1_dim], weight_1_buffer, weight_1_dim, num_thread, thread_id);
  copy_data_nonblocking<float>(&value[batch_idx__key_idx * value_dim], value_buffer, value_dim, num_thread, thread_id);
  if (worker_id == 0) {
    scaled_acos_buffer[warp_idx] = acosf(2 * (float)warp_idx / WARP_SIZE_MINUS_ONE - 1) * INV_PI;
  } else if (worker_id == 1) {
    inv_sqrt_buffer[warp_idx] = min(10, 1 / sqrtf((float)warp_idx / WARP_SIZE_MINUS_ONE + 0.000001));
  } else if (worker_id == 2) {
    pow_buffer[warp_idx] = min(10, powf((float)warp_idx / WARP_SIZE_MINUS_ONE + 0.000001, tau - 1));
  }
  __syncthreads();

  for (int idx_start = 0; idx_start < num_nonzero_idxes; idx_start = idx_start + num_worker) {
    int idx = idx_start + worker_id;
    if (idx < num_nonzero_idxes) {
      int query_idx = nonzero_idxes[idx];
      int batch_idx__query_idx = batch_idx * num_query + query_idx;

      float weight_0 = warp_vector_dot(weight_0_buffer, &query_weight_0[batch_idx__query_idx * weight_0_dim], weight_0_dim, warp_idx);
      float weight_1 = NORMALIZER * warp_vector_dot(weight_1_buffer, &query_weight_1[batch_idx__query_idx * weight_1_dim], weight_1_dim, warp_idx);
      float inv_sqrt = interpolate(inv_sqrt_buffer, (1 - weight_1 * weight_1) * WARP_SIZE_MINUS_ONE);
      float scaled_acos = 1 - interpolate(scaled_acos_buffer, (1 + weight_1) * WARP_SIZE_MINUS_ONE / 2);
      float pow_scaled_acos = interpolate(pow_buffer, scaled_acos * WARP_SIZE_MINUS_ONE);
      float weight = weight_0 * tau * pow_scaled_acos * INV_PI * inv_sqrt;

      for (int value_dim_idx_start = 0; value_dim_idx_start < value_dim; value_dim_idx_start = value_dim_idx_start + WARP_SIZE) {
        int value_dim_idx = value_dim_idx_start + warp_idx;
        atomicAdd(&cumulation_value[batch_idx__query_idx * value_dim + value_dim_idx], weight * value_buffer[value_dim_idx]);
      }
    }
  }

}

__global__ void weighted_lsh_cumulation_sorted_key_cuda_kernel(
  int *query_mask,          // [batch_size, num_query]
  int *query_hashcode,      // [batch_size, num_query, num_hash_f]
  int *key_sort_info,       // [batch_size, num_hash_f, hashtable_capacity]
  int *key_sorted_idxes,    // [batch_size, num_hash_f, num_key]
  float *query_weight_0,    // [batch_size, num_query, weight_0_dim]
  float *key_weight_0,      // [batch_size, num_key, weight_0_dim]
  float *query_weight_1,    // [batch_size, num_query, weight_1_dim]
  float *key_weight_1,      // [batch_size, num_key, weight_1_dim]
  float *value,             // [batch_size, num_key, value_dim]
  float *cumulation_value,  // [batch_size, num_query, value_dim]
  int batch_size,
  int num_hash_f,
  int num_query,
  int num_key,
  int value_dim,
  int weight_0_dim,
  int weight_1_dim,
  int hashtable_capacity,
  int workspace_size,
  float tau
) {

  int batch_idx = blockIdx.y;
  int query_idx = blockIdx.x;

  int num_thread = blockDim.y * blockDim.x;
  int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  int num_worker = blockDim.y;
  int worker_id = threadIdx.y;

  int warp_idx = threadIdx.x;

  int batch_idx__query_idx = batch_idx * num_query + query_idx;
  if (query_mask[batch_idx__query_idx] == 0) {
    return;
  }

  extern __shared__ float buffer[];

  // need space 3 * workspace_size
  int *hashcode_buffer = (int*)buffer;
  int *workspace_value = (int*)&buffer[workspace_size];
  int *workspace_count = (int*)&buffer[2 * workspace_size];

  copy_data_nonblocking<int>(&query_hashcode[batch_idx__query_idx * num_hash_f], hashcode_buffer, num_hash_f, num_thread, thread_id);
  init_buffer_nonblocking<int>(EMPTY_VALUE, workspace_value, workspace_size, num_thread, thread_id);
  init_buffer_nonblocking<int>(0, workspace_count, workspace_size, num_thread, thread_id);

  __syncthreads();

  int rnumber = (int)(query_weight_0[batch_idx__query_idx * weight_0_dim + thread_id % weight_0_dim] * 1000000);
  rnumber = rnumber + hashcode_buffer[thread_id % num_hash_f] + batch_idx__query_idx * num_thread + thread_id;
  rnumber = next_rint(rnumber);

  for (int hash_f_idx_start = 0; hash_f_idx_start < num_hash_f; hash_f_idx_start = hash_f_idx_start + num_worker) {
    int hash_f_idx = hash_f_idx_start + worker_id;
    int hashcode = hashcode_buffer[hash_f_idx];
    int batch_idx__hash_f_idx = batch_idx * num_hash_f + hash_f_idx;
    int batch_idx__hash_f_idx__hashcode = batch_idx__hash_f_idx * hashtable_capacity + hashcode;

    int offset = 0;
    int count = 0;
    if (hashcode != 0) {
      int temp_value = key_sort_info[batch_idx__hash_f_idx__hashcode - warp_idx % 2];
      offset = __shfl_sync(FULL_MASK, temp_value, 1);
      count = __shfl_sync(FULL_MASK, temp_value, 0) - offset;
    } else {
      offset = 0;
      count = key_sort_info[batch_idx__hash_f_idx__hashcode];
    }

    while (count > 0) {
      int work_size = min(count, WARP_SIZE);
      rnumber = next_rint(rnumber);
      if (warp_idx < work_size) {
        int key_idx = key_sorted_idxes[batch_idx__hash_f_idx * num_key + offset + warp_idx];
        int slot = key_idx % workspace_size;
        if (((float)rnumber / (float)LCG_M) * (float)atomicAdd(&workspace_count[slot], 1) < 1) {
          workspace_value[slot] = key_idx;
        }
      }
      offset += work_size;
      count -= work_size;
    }
  }

  __syncthreads();

  for (int slot_start = 0; slot_start < workspace_size; slot_start = slot_start + num_thread) {
    int slot = slot_start + thread_id;
    workspace_count[slot] = (int)(workspace_count[slot] > 0);
  }
  __syncthreads();
  cumulative_sum(workspace_count, workspace_size, thread_id, num_thread);

  int *nonzero_idxes = (int*)buffer;
  for (int slot_start = 0; slot_start < workspace_size; slot_start = slot_start + num_thread) {
    int slot = slot_start + thread_id;
    if (workspace_value[slot] != EMPTY_VALUE) {
      nonzero_idxes[workspace_count[slot] - 1] = workspace_value[slot];
    }
  }
  int num_nonzero_idxes = workspace_count[workspace_size - 1];
  __syncthreads();

  // need space workspace_size + weight_0_dim + weight_1_dim + num_worker * value_dim + 3 * WARP_SIZE
  float *weight_0_buffer = &buffer[workspace_size];
  float *weight_1_buffer = &buffer[workspace_size + weight_0_dim];
  float *result_buffer = &buffer[workspace_size + weight_0_dim + weight_1_dim + worker_id * value_dim];
  float *scaled_acos_buffer = &buffer[workspace_size + weight_0_dim + weight_1_dim + num_worker * value_dim];
  float *inv_sqrt_buffer = &buffer[workspace_size + weight_0_dim + weight_1_dim + num_worker * value_dim + WARP_SIZE];
  float *pow_buffer = &buffer[workspace_size + weight_0_dim + weight_1_dim + num_worker * value_dim + 2 * WARP_SIZE];

  copy_data_nonblocking<float>(&query_weight_0[batch_idx__query_idx * weight_0_dim], weight_0_buffer, weight_0_dim, num_thread, thread_id);
  copy_data_nonblocking<float>(&query_weight_1[batch_idx__query_idx * weight_1_dim], weight_1_buffer, weight_1_dim, num_thread, thread_id);
  for (int value_dim_idx_start = 0; value_dim_idx_start < value_dim; value_dim_idx_start = value_dim_idx_start + WARP_SIZE) {
    result_buffer[value_dim_idx_start + warp_idx] = 0;
  }
  if (worker_id == 0) {
    scaled_acos_buffer[warp_idx] = acosf(2 * (float)warp_idx / WARP_SIZE_MINUS_ONE - 1) * INV_PI;
  } else if (worker_id == 1) {
    inv_sqrt_buffer[warp_idx] = min(10, 1 / sqrtf((float)warp_idx / WARP_SIZE_MINUS_ONE + 0.000001));
  } else if (worker_id == 2) {
    pow_buffer[warp_idx] = min(10, powf((float)warp_idx / WARP_SIZE_MINUS_ONE + 0.000001, tau - 1));
  }
  __syncthreads();

  for (int idx_start = 0; idx_start < num_nonzero_idxes; idx_start = idx_start + num_worker) {
    int idx = idx_start + worker_id;
    if (idx < num_nonzero_idxes) {
      int key_idx = nonzero_idxes[idx];
      int batch_idx__key_idx = batch_idx * num_key + key_idx;

      float weight_0 = warp_vector_dot(weight_0_buffer, &key_weight_0[batch_idx__key_idx * weight_0_dim], weight_0_dim, warp_idx);
      float weight_1 = NORMALIZER * warp_vector_dot(weight_1_buffer, &key_weight_1[batch_idx__key_idx * weight_1_dim], weight_1_dim, warp_idx);
      float inv_sqrt = interpolate(inv_sqrt_buffer, (1 - weight_1 * weight_1) * WARP_SIZE_MINUS_ONE);
      float scaled_acos = 1 - interpolate(scaled_acos_buffer, (1 + weight_1) * WARP_SIZE_MINUS_ONE / 2);
      float pow_scaled_acos = interpolate(pow_buffer, scaled_acos * WARP_SIZE_MINUS_ONE);
      float weight = weight_0 * tau * pow_scaled_acos * INV_PI * inv_sqrt;

      for (int value_dim_idx_start = 0; value_dim_idx_start < value_dim; value_dim_idx_start = value_dim_idx_start + WARP_SIZE) {
        int value_dim_idx = value_dim_idx_start + warp_idx;
        result_buffer[value_dim_idx] += weight * value[batch_idx__key_idx * value_dim + value_dim_idx];
      }
    }
  }

  for (int value_dim_idx_start = 0; value_dim_idx_start < value_dim; value_dim_idx_start = value_dim_idx_start + WARP_SIZE) {
    int value_dim_idx = value_dim_idx_start + warp_idx;
    atomicAdd(&cumulation_value[batch_idx__query_idx * value_dim + value_dim_idx], result_buffer[value_dim_idx]);
  }

}


__global__ void weighted_lsh_cumulation_merged_shared_qk_cuda_kernel(
  int *sort_info,             // [batch_size, num_hash_f, hashtable_capacity]
  int *sorted_idxes,          // [batch_size, num_hash_f, num_vector]
  int *mask,                  // [batch_size, num_vector]
  int *hashcode,              // [batch_size, num_vector, num_hash_f]
  float *weight_0,            // [batch_size, num_vector, weight_dim]
  float *weight_1,            // [batch_size, num_vector, weight_dim]
  float *value,               // [batch_size, num_vector, value_dim]
  float *cumulation_value,    // [batch_size, num_vector, value_dim]
  int batch_size,
  int num_hash_f,
  int num_vector,
  int value_dim,
  int weight_dim,
  int hashtable_capacity,
  int workspace_size,
  float tau
) {

  int batch_idx = blockIdx.y;
  int vec_idx = blockIdx.x;

  int num_thread = blockDim.y * blockDim.x;
  int thread_id = threadIdx.y * blockDim.x + threadIdx.x;

  int num_worker = blockDim.y;
  int worker_id = threadIdx.y;

  int warp_idx = threadIdx.x;

  int batch_idx__vec_idx = batch_idx * num_vector + vec_idx;
  if (mask[batch_idx__vec_idx] == 0) {
    return;
  }

  extern __shared__ float buffer[];

  // need space 3 * workspace_size
  int *hashcode_buffer = (int*)buffer;
  int *workspace_value = (int*)&buffer[workspace_size];
  int *workspace_count = (int*)&buffer[2 * workspace_size];

  copy_data_nonblocking<int>(&hashcode[batch_idx__vec_idx * num_hash_f], hashcode_buffer, num_hash_f, num_thread, thread_id);
  init_buffer_nonblocking<int>(EMPTY_VALUE, workspace_value, workspace_size, num_thread, thread_id);
  init_buffer_nonblocking<int>(0, workspace_count, workspace_size, num_thread, thread_id);

  __syncthreads();

  int rnumber = (int)(weight_0[batch_idx__vec_idx * weight_dim + thread_id % weight_dim] * 1000000);
  rnumber = rnumber + hashcode_buffer[thread_id % num_hash_f] + batch_idx__vec_idx * num_thread + thread_id;
  rnumber = next_rint(rnumber);

  for (int hash_f_idx_start = 0; hash_f_idx_start < num_hash_f; hash_f_idx_start = hash_f_idx_start + num_worker) {
    int hash_f_idx = hash_f_idx_start + worker_id;
    int hashcode = hashcode_buffer[hash_f_idx];
    int batch_idx__hash_f_idx = batch_idx * num_hash_f + hash_f_idx;
    int batch_idx__hash_f_idx__hashcode = batch_idx__hash_f_idx * hashtable_capacity + hashcode;

    int offset = 0;
    int count = 0;
    if (hashcode != 0) {
      int temp_value = sort_info[batch_idx__hash_f_idx__hashcode - warp_idx % 2];
      offset = __shfl_sync(FULL_MASK, temp_value, 1);
      count = __shfl_sync(FULL_MASK, temp_value, 0) - offset;
    } else {
      offset = 0;
      count = sort_info[batch_idx__hash_f_idx__hashcode];
    }

    while (count > 0) {
      int work_size = min(count, WARP_SIZE);
      rnumber = next_rint(rnumber);
      if (warp_idx < work_size) {
        int vec2_idx = sorted_idxes[batch_idx__hash_f_idx * num_vector + offset + warp_idx];
        if (vec2_idx != vec_idx) {
          int slot = vec2_idx % workspace_size;
          if (((float)rnumber / (float)LCG_M) * (float)atomicAdd(&workspace_count[slot], 1) < 1) {
            workspace_value[slot] = vec2_idx;
          }
        }
      }
      offset += work_size;
      count -= work_size;
    }
  }

  __syncthreads();

  for (int slot_start = 0; slot_start < workspace_size; slot_start = slot_start + num_thread) {
    int slot = slot_start + thread_id;
    workspace_count[slot] = (int)(workspace_count[slot] > 0);
  }
  __syncthreads();
  cumulative_sum(workspace_count, workspace_size, thread_id, num_thread);

  int *nonzero_idxes = (int*)buffer;
  for (int slot_start = 0; slot_start < workspace_size; slot_start = slot_start + num_thread) {
    int slot = slot_start + thread_id;
    if (workspace_value[slot] != EMPTY_VALUE) {
      nonzero_idxes[workspace_count[slot] - 1] = workspace_value[slot];
    }
  }
  int num_nonzero_idxes = workspace_count[workspace_size - 1];
  __syncthreads();

  // need space workspace_size + 2 * weight_dim + value_dim + 3 * WARP_SIZE
  float *weight_0_buffer = &buffer[workspace_size];
  float *weight_1_buffer = &buffer[workspace_size + weight_dim];
  float *value_buffer = &buffer[workspace_size + 2 * weight_dim];
  float *scaled_acos_buffer = &buffer[workspace_size + 2 * weight_dim + value_dim];
  float *inv_sqrt_buffer = &buffer[workspace_size + 2 * weight_dim + value_dim + WARP_SIZE];
  float *pow_buffer = &buffer[workspace_size + 2 * weight_dim + value_dim + 2 * WARP_SIZE];

  copy_data_nonblocking<float>(&weight_0[batch_idx__vec_idx * weight_dim], weight_0_buffer, weight_dim, num_thread, thread_id);
  copy_data_nonblocking<float>(&weight_1[batch_idx__vec_idx * weight_dim], weight_1_buffer, weight_dim, num_thread, thread_id);
  copy_data_nonblocking<float>(&value[batch_idx__vec_idx * value_dim], value_buffer, value_dim, num_thread, thread_id);
  if (worker_id == 0) {
    scaled_acos_buffer[warp_idx] = acosf(2 * (float)warp_idx / WARP_SIZE_MINUS_ONE - 1) * INV_PI;
  } else if (worker_id == 1) {
    inv_sqrt_buffer[warp_idx] = min(10, 1 / sqrtf((float)warp_idx / WARP_SIZE_MINUS_ONE + 0.000001));
  } else if (worker_id == 2) {
    pow_buffer[warp_idx] = min(10, powf((float)warp_idx / WARP_SIZE_MINUS_ONE + 0.000001, tau - 1));
  }
  __syncthreads();

  for (int idx_start = 0; idx_start < num_nonzero_idxes; idx_start = idx_start + num_worker) {
    int idx = idx_start + worker_id;
    if (idx < num_nonzero_idxes) {
      int vec2_idx = nonzero_idxes[idx];
      int batch_idx__vec2_idx = batch_idx * num_vector + vec2_idx;

      float w0 = warp_vector_dot(weight_0_buffer, &weight_1[batch_idx__vec2_idx * weight_dim], weight_dim, warp_idx);
      float w1 = warp_vector_dot(weight_1_buffer, &weight_0[batch_idx__vec2_idx * weight_dim], weight_dim, warp_idx);
      float w2 = NORMALIZER * warp_vector_dot(value_buffer, &value[batch_idx__vec2_idx * value_dim], value_dim, warp_idx);

      float inv_sqrt = interpolate(inv_sqrt_buffer, (1 - w2 * w2) * WARP_SIZE_MINUS_ONE);
      float scaled_acos = 1 - interpolate(scaled_acos_buffer, (1 + w2) * WARP_SIZE_MINUS_ONE / 2);
      float pow_scaled_acos = interpolate(pow_buffer, scaled_acos * WARP_SIZE_MINUS_ONE);
      float weight = (w0 + w1) * tau * pow_scaled_acos * INV_PI * inv_sqrt;

      for (int value_dim_idx_start = 0; value_dim_idx_start < value_dim; value_dim_idx_start = value_dim_idx_start + WARP_SIZE) {
        int value_dim_idx = value_dim_idx_start + warp_idx;
        atomicAdd(&cumulation_value[batch_idx__vec2_idx * value_dim + value_dim_idx], weight * value_buffer[value_dim_idx]);
      }
    }
  }

}
