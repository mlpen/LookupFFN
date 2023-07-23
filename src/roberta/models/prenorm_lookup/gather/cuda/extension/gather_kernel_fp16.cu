#include <cuda.h>
#include <cuda_fp16.h>
#include <stdio.h>
#include "gather_kernel_fp16.h"

__global__ void weighted_vector_gather_add_cuda_kernel_fp16(
  int     *indexes,    // [batch_size, index_size]
  __half2 *source,     // [source_size, vector_dim_div_2]
  __half2 *weights,    // [batch_size, index_size_div_2]
  __half2 *outputs,    // [batch_size, vector_dim_div_2]
  int batch_size,
  int index_size_div_2,
  int source_size,
  int vector_dim_div_2
) {

  int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int thread_idx = threadIdx.x;
  int num_thread = blockDim.x;
  if (batch_idx >= batch_size) {
    return;
  }

  extern __shared__ float buffer[];
  float *work_buffer = &buffer[threadIdx.y * 3 * index_size_div_2];
  __half2 *weights_buffer = (__half2*)work_buffer;
  int *index_buffer = (int*)&work_buffer[index_size_div_2];

  int *indexes_pt = &indexes[(long)batch_idx * (long)index_size_div_2 * 2];
  __half2 *weights_pt = &weights[(long)batch_idx * (long)index_size_div_2];

  for (int idx_start = 0; idx_start < index_size_div_2; idx_start = idx_start + num_thread) {
    int idx = idx_start + thread_idx;
    if (idx < index_size_div_2) {
      weights_buffer[idx] = weights_pt[idx];
    }
  }

  for (int idx_start = 0; idx_start < index_size_div_2 * 2; idx_start = idx_start + num_thread) {
    int idx = idx_start + thread_idx;
    if (idx < index_size_div_2 * 2) {
      index_buffer[idx] = indexes_pt[idx];
    }
  }

  __syncthreads();

  for (int dim_idx_start = 0; dim_idx_start < vector_dim_div_2; dim_idx_start = dim_idx_start + num_thread) {
    int dim_idx = dim_idx_start + thread_idx;
    if (dim_idx < vector_dim_div_2) {
      __half2 output_scalar2 = __float2half2_rn(0.0);
      for (int idx = 0; idx < index_size_div_2; idx++) {
        __half2 weight_scalar2 = weights_buffer[idx];
        __half2 source0_scalar2 = source[(long)index_buffer[idx * 2] * (long)vector_dim_div_2 + (long)dim_idx];
        __half2 source1_scalar2 = source[(long)index_buffer[idx * 2 + 1] * (long)vector_dim_div_2 + (long)dim_idx];
        output_scalar2 = __hfma2(__half2half2(weight_scalar2.x), source0_scalar2, output_scalar2);
        output_scalar2 = __hfma2(__half2half2(weight_scalar2.y), source1_scalar2, output_scalar2);
      }
      outputs[(long)batch_idx * (long)vector_dim_div_2 + (long)dim_idx] = output_scalar2;
    }
  }
}

__global__ void weighted_vector_scatter_add_cuda_kernel_fp16(
  int     *indexes,      // [batch_size, index_size]
  __half2 *source,       // [batch_size, vector_dim_div_2]
  __half2 *weights,      // [batch_size, index_size_div_2]
  __half2 *outputs,      // [output_size, vector_dim_div_2]
  int batch_size,
  int index_size_div_2,
  int output_size,
  int vector_dim_div_2
) {

  int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int thread_idx = threadIdx.x;
  int num_thread = blockDim.x;
  if (batch_idx >= batch_size) {
    return;
  }

  extern __shared__ float buffer[];
  float *work_buffer = &buffer[threadIdx.y * 3 * index_size_div_2];
  __half2 *weights_buffer = (__half2*)work_buffer;
  int *index_buffer = (int*)&work_buffer[index_size_div_2];

  int *indexes_pt = &indexes[(long)batch_idx * (long)index_size_div_2 * 2];
  __half2 *weights_pt = &weights[(long)batch_idx * (long)index_size_div_2];

  for (int idx_start = 0; idx_start < index_size_div_2; idx_start = idx_start + num_thread) {
    int idx = idx_start + thread_idx;
    if (idx < index_size_div_2) {
      weights_buffer[idx] = weights_pt[idx];
    }
  }

  for (int idx_start = 0; idx_start < index_size_div_2 * 2; idx_start = idx_start + num_thread) {
    int idx = idx_start + thread_idx;
    if (idx < index_size_div_2 * 2) {
      index_buffer[idx] = indexes_pt[idx];
    }
  }

  __syncthreads();

  for (int dim_idx_start = 0; dim_idx_start < vector_dim_div_2; dim_idx_start = dim_idx_start + num_thread) {
    int dim_idx = dim_idx_start + thread_idx;
    if (dim_idx < vector_dim_div_2) {
      __half2 source_scalar2 = source[(long)batch_idx * (long)vector_dim_div_2 + (long)dim_idx];
      for (int idx = 0; idx < index_size_div_2; idx++) {
        __half2 weight_scalar2 = weights_buffer[idx];
        __half2 output0_scalar2 = __hmul2(__half2half2(weight_scalar2.x), source_scalar2);
        __half2 output1_scalar2 = __hmul2(__half2half2(weight_scalar2.y), source_scalar2);
        atomicAdd(&outputs[(long)index_buffer[idx * 2] * (long)vector_dim_div_2 + (long)dim_idx], output0_scalar2);
        atomicAdd(&outputs[(long)index_buffer[idx * 2 + 1] * (long)vector_dim_div_2 + (long)dim_idx], output1_scalar2);
      }
    }
  }
}

__global__ void indexed_inner_product_cuda_kernel_fp16(
  int     *indexes,      // [batch_size, index_size]
  __half2 *source_1,     // [batch_size, vector_dim_div_2]
  __half2 *source_2,     // [source_size, vector_dim_div_2]
  __half2 *outputs,      // [batch_size, index_size_div_2]
  int batch_size,
  int index_size_div_2,
  int source_size,
  int vector_dim_div_2
) {

  int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int thread_idx = threadIdx.x;
  int num_thread = blockDim.x;
  if (batch_idx >= batch_size) {
    return;
  }

  extern __shared__ float buffer[];
  float *work_buffer = &buffer[threadIdx.y * 4 * index_size_div_2];
  float *outputs_buffer = work_buffer;
  int *index_buffer = (int*)&work_buffer[index_size_div_2 * 2];

  int *indexes_pt = &indexes[(long)batch_idx * (long)index_size_div_2 * 2];

  for (int idx_start = 0; idx_start < index_size_div_2 * 2; idx_start = idx_start + num_thread) {
    int idx = idx_start + thread_idx;
    if (idx < index_size_div_2 * 2) {
      index_buffer[idx] = indexes_pt[idx];
      outputs_buffer[idx] = 0;
    }
  }

  __syncthreads();

  for (int dim_idx_start = 0; dim_idx_start < vector_dim_div_2; dim_idx_start = dim_idx_start + num_thread) {
    int dim_idx = dim_idx_start + thread_idx;
    if (dim_idx < vector_dim_div_2) {
      __half2 source_1_scalar2 = source_1[(long)batch_idx * (long)vector_dim_div_2 + (long)dim_idx];
      for (int idx = 0; idx < index_size_div_2 * 2; idx++) {
        __half2 source_2_scalar2 = source_2[(long)index_buffer[idx] * (long)vector_dim_div_2 + (long)dim_idx];
        float val = __half2float(__hmul(source_1_scalar2.x, source_2_scalar2.x)) + __half2float(__hmul(source_1_scalar2.y, source_2_scalar2.y));
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


  __half2 *outputs_pt = &outputs[(long)batch_idx * (long)index_size_div_2];
  for (int idx_start = 0; idx_start < index_size_div_2; idx_start = idx_start + num_thread) {
    int idx = idx_start + thread_idx;
    if (idx < index_size_div_2) {
      outputs_pt[idx] = __floats2half2_rn(outputs_buffer[2 * idx], outputs_buffer[2 * idx + 1]);
    }
  }
}
