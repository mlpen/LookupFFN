#include "gather_kernel_fp32.h"

__global__ void weighted_vector_gather_add_cuda_kernel_fp32(
  int   *indexes,      // [batch_size, index_size]
  float *source,       // [source_size, vector_dim]
  float *weights,      // [batch_size, index_size]
  float *outputs,      // [batch_size, vector_dim]
  int batch_size,
  int index_size,
  int source_size,
  int vector_dim
) {

  int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int thread_idx = threadIdx.x;
  int num_thread = blockDim.x;
  if (batch_idx >= batch_size) {
    return;
  }

  extern __shared__ float buffer[];
  float *work_buffer = &buffer[threadIdx.y * 2 * index_size];
  int *index_buffer = (int*)work_buffer;
  float *weights_buffer = &work_buffer[index_size];

  int *indexes_pt = &indexes[(long)batch_idx * (long)index_size];
  float *weights_pt = &weights[(long)batch_idx * (long)index_size];
  for (int idx_start = 0; idx_start < index_size; idx_start = idx_start + num_thread) {
    int idx = idx_start + thread_idx;
    if (idx < index_size) {
      index_buffer[idx] = indexes_pt[idx];
      weights_buffer[idx] = weights_pt[idx];
    }
  }
  __syncthreads();

  for (int dim_idx_start = 0; dim_idx_start < vector_dim; dim_idx_start = dim_idx_start + num_thread) {
    int dim_idx = dim_idx_start + thread_idx;
    if (dim_idx < vector_dim) {
      float output_scalar = 0;
      for (int idx = 0; idx < index_size; idx++) {
        output_scalar = output_scalar + weights_buffer[idx] * source[(long)index_buffer[idx] * (long)vector_dim + (long)dim_idx];
      }
      outputs[(long)batch_idx * (long)vector_dim + (long)dim_idx] = output_scalar;
    }
  }
}

__global__ void weighted_vector_scatter_add_cuda_kernel_fp32(
  int   *indexes,      // [batch_size, index_size]
  float *source,       // [batch_size, vector_dim]
  float *weights,      // [batch_size, index_size]
  float *outputs,      // [output_size, vector_dim]
  int batch_size,
  int index_size,
  int output_size,
  int vector_dim
) {

  int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int thread_idx = threadIdx.x;
  int num_thread = blockDim.x;
  if (batch_idx >= batch_size) {
    return;
  }

  extern __shared__ float buffer[];
  float *work_buffer = &buffer[threadIdx.y * 2 * index_size];
  int *index_buffer = (int*)work_buffer;
  float *weights_buffer = &work_buffer[index_size];

  int *indexes_pt = &indexes[(long)batch_idx * (long)index_size];
  float *weights_pt = &weights[(long)batch_idx * (long)index_size];
  for (int idx_start = 0; idx_start < index_size; idx_start = idx_start + num_thread) {
    int idx = idx_start + thread_idx;
    if (idx < index_size) {
      index_buffer[idx] = (int)indexes_pt[idx];
      weights_buffer[idx] = weights_pt[idx];
    }
  }
  __syncthreads();

  for (int dim_idx_start = 0; dim_idx_start < vector_dim; dim_idx_start = dim_idx_start + num_thread) {
    int dim_idx = dim_idx_start + thread_idx;
    if (dim_idx < vector_dim) {
      float source_scalar = source[(long)batch_idx * (long)vector_dim + (long)dim_idx];
      for (int idx = 0; idx < index_size; idx++) {
        atomicAdd(&outputs[(long)index_buffer[idx] * (long)vector_dim + (long)dim_idx], weights_buffer[idx] * source_scalar);
      }
    }
  }
}

__global__ void indexed_inner_product_cuda_kernel_fp32(
  int   *indexes,      // [batch_size, index_size]
  float *source_1,     // [batch_size, vector_dim]
  float *source_2,     // [source_size, vector_dim]
  float *outputs,      // [batch_size, index_size]
  int batch_size,
  int index_size,
  int source_size,
  int vector_dim
) {

  int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int thread_idx = threadIdx.x;
  int num_thread = blockDim.x;
  if (batch_idx >= batch_size) {
    return;
  }

  extern __shared__ float buffer[];
  float *work_buffer = &buffer[threadIdx.y * 2 * index_size];
  float *outputs_buffer = work_buffer;
  int *index_buffer = (int*)&work_buffer[index_size];

  int *indexes_pt = &indexes[(long)batch_idx * (long)index_size];

  for (int idx_start = 0; idx_start < index_size; idx_start = idx_start + num_thread) {
    int idx = idx_start + thread_idx;
    if (idx < index_size) {
      index_buffer[idx] = indexes_pt[idx];
      outputs_buffer[idx] = 0;
    }
  }

  __syncthreads();

  for (int dim_idx_start = 0; dim_idx_start < vector_dim; dim_idx_start = dim_idx_start + num_thread) {
    int dim_idx = dim_idx_start + thread_idx;
    if (dim_idx < vector_dim) {
      float source_1_scalar = source_1[(long)batch_idx * (long)vector_dim + (long)dim_idx];
      for (int idx = 0; idx < index_size; idx++) {
        float source_2_scalar = source_2[(long)index_buffer[idx] * (long)vector_dim + (long)dim_idx];
        float val = source_1_scalar * source_2_scalar;
        #pragma unroll
        for (int offset = 1; offset < WARP_SIZE; offset = offset << 1) {
          val += __shfl_xor_sync(FULL_MASK, val, offset);
        }
        if (thread_idx % WARP_SIZE == 0) {
          atomicAdd(&outputs_buffer[idx], val);
        }
      }
    }
  }

  __syncthreads();


  float *outputs_pt = &outputs[(long)batch_idx * (long)index_size];
  for (int idx_start = 0; idx_start < index_size; idx_start = idx_start + num_thread) {
    int idx = idx_start + thread_idx;
    if (idx < index_size) {
      outputs_pt[idx] = outputs_buffer[idx];
    }
  }
}
