#include "cuda_kernel.h"
#include <stdio.h>

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

__device__ inline void fast_hadamard_transform(float *vector_buffer, int vector_dim, int dim_idx) {
  int stride = vector_dim / 2;
  while (stride > (WARP_SIZE / 2)) {
    __syncthreads();
    int sign = 1 - ((dim_idx / stride) % 2) * 2;
    float val1 = vector_buffer[dim_idx];
    float val2 = vector_buffer[dim_idx + sign * stride];
    __syncthreads();
    vector_buffer[dim_idx] = float(sign) * val1 + val2;
    stride = stride / 2;
  }

  float val = vector_buffer[dim_idx];
  #pragma unroll
  for (stride = (WARP_SIZE / 2); stride > 0; stride = stride / 2) {
    int sign = 1 - ((dim_idx / stride) % 2) * 2;
    val = float(sign) * val + __shfl_xor_sync(FULL_MASK, val, stride);
  }
  vector_buffer[dim_idx] = val;
}

__global__ void fast_hash_cuda_kernel(
  int *mask,        // [batch_size, num_vector]
  float *vector,    // [batch_size, num_vector, vector_dim]
  int *Dmat,        // [batch_size, 3, num_part, vector_dim]
  int *hashcode,    // [batch_size, num_vector, num_hash_f]
  int batch_size,
  int num_vector,
  int vector_dim,
  int num_part,
  int num_hash_f,
  int hashcode_len
) {

  int batch_idx = blockIdx.y;
  int vector_idx = blockIdx.x;

  int num_worker = blockDim.y;
  int worker_id = threadIdx.y;

  int dim_idx = threadIdx.x;

  int batch_idx__vector_idx = batch_idx * num_vector + vector_idx;
  if (mask[batch_idx__vector_idx] == 0) {
    return;
  }

  extern __shared__ float buffer[];
  float *vector_buffer = buffer;
  float *result_buffer = &buffer[vector_dim + worker_id * vector_dim];

  int *Dmat_pt = &Dmat[batch_idx * 3 * num_part * vector_dim];

  if (worker_id == 0) {
    vector_buffer[dim_idx] = vector[batch_idx__vector_idx * vector_dim + dim_idx];
  }
  __syncthreads();

  int num_hash_per_part = vector_dim / hashcode_len;

  for (int part_idx_start = 0; part_idx_start < num_part; part_idx_start = part_idx_start + num_worker) {
    int part_idx = part_idx_start + worker_id;
    if (part_idx < num_part) {
      result_buffer[dim_idx] = vector_buffer[dim_idx] * (float)Dmat_pt[(0 * num_part + part_idx) * vector_dim + dim_idx];
      fast_hadamard_transform(result_buffer, vector_dim, dim_idx);
      result_buffer[dim_idx] = result_buffer[dim_idx] * (float)Dmat_pt[(1 * num_part + part_idx) * vector_dim + dim_idx];
      fast_hadamard_transform(result_buffer, vector_dim, dim_idx);
      result_buffer[dim_idx] = result_buffer[dim_idx] * (float)Dmat_pt[(2 * num_part + part_idx) * vector_dim + dim_idx];
      fast_hadamard_transform(result_buffer, vector_dim, dim_idx);
      result_buffer[dim_idx] = (int)(result_buffer[dim_idx] > 0) << (dim_idx % hashcode_len);
    }
    __syncthreads();
    if (part_idx < num_part && dim_idx < num_hash_per_part) {
      int code = 0;
      for (int i = 0; i < hashcode_len; i++) {
        code += result_buffer[dim_idx * hashcode_len + i];
      }
      int hash_f_idx = part_idx * num_hash_per_part + dim_idx;
      if (hash_f_idx < num_hash_f) {
        hashcode[batch_idx__vector_idx * num_hash_f + hash_f_idx] = code;
      }
    }
    __syncthreads();
  }

}
