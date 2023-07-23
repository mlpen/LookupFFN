#include "cuda_kernel.h"
#include <stdio.h>
#include <cuda.h>
#include <cuda_fp16.h>

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

typedef struct
{
    __half2 array[8];
} __half16;

typedef struct
{
    __half2 array[16];
} __half32;

__global__ void fast_hadamard_transform_cuda_kernel_dim512_fp16(
  __half2 *inp,    // [batch_size, 256]
  __half2 *out,    // [batch_size, 256]
  int batch_size
) {

  int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int warp_idx = threadIdx.x;

  if (batch_idx >= batch_size) {
    return;
  }

  __shared__ __half2 buffer[2048];
  __half2 *warp_buffer = &buffer[threadIdx.y * 256];
  __half2 *inp_pt = &inp[batch_idx * 256];

  #pragma unroll
  for (int i = 0; i < 8; i++) {
    warp_buffer[i * WARP_SIZE + warp_idx] = inp_pt[i * WARP_SIZE + warp_idx];
  }
  __half16 reg1 = ((__half16*)warp_buffer)[warp_idx];

  #pragma unroll
  for (int stride = (WARP_SIZE / 2); stride > 0; stride = stride / 2) {
    __half2 sign = __float2half2_rn(1 - ((warp_idx / stride) % 2) * 2);
    #pragma unroll
    for (int i = 0; i < 8; i++) {
      reg1.array[i] = __hadd2_rn(__hmul2_rn(sign, reg1.array[i]), __shfl_xor_sync(FULL_MASK, reg1.array[i], stride));
    }
  }

  __half16 reg2;

  reg2.array[0] = __hadd2_rn(reg1.array[0], reg1.array[4]);
  reg2.array[1] = __hadd2_rn(reg1.array[1], reg1.array[5]);
  reg2.array[2] = __hadd2_rn(reg1.array[2], reg1.array[6]);
  reg2.array[3] = __hadd2_rn(reg1.array[3], reg1.array[7]);
  reg2.array[4] = __hadd2_rn(__hneg2(reg1.array[4]), reg1.array[0]);
  reg2.array[5] = __hadd2_rn(__hneg2(reg1.array[5]), reg1.array[1]);
  reg2.array[6] = __hadd2_rn(__hneg2(reg1.array[6]), reg1.array[2]);
  reg2.array[7] = __hadd2_rn(__hneg2(reg1.array[7]), reg1.array[3]);

  reg1.array[0] = __hadd2_rn(reg2.array[0], reg2.array[2]);
  reg1.array[1] = __hadd2_rn(reg2.array[1], reg2.array[3]);
  reg1.array[2] = __hadd2_rn(__hneg2(reg2.array[2]), reg2.array[0]);
  reg1.array[3] = __hadd2_rn(__hneg2(reg2.array[3]), reg2.array[1]);
  reg1.array[4] = __hadd2_rn(reg2.array[4], reg2.array[6]);
  reg1.array[5] = __hadd2_rn(reg2.array[5], reg2.array[7]);
  reg1.array[6] = __hadd2_rn(__hneg2(reg2.array[6]), reg2.array[4]);
  reg1.array[7] = __hadd2_rn(__hneg2(reg2.array[7]), reg2.array[5]);

  reg2.array[0] = __hadd2_rn(reg1.array[0], reg1.array[1]);
  reg2.array[1] = __hadd2_rn(__hneg2(reg1.array[1]), reg1.array[0]);
  reg2.array[2] = __hadd2_rn(reg1.array[2], reg1.array[3]);
  reg2.array[3] = __hadd2_rn(__hneg2(reg1.array[3]), reg1.array[2]);
  reg2.array[4] = __hadd2_rn(reg1.array[4], reg1.array[5]);
  reg2.array[5] = __hadd2_rn(__hneg2(reg1.array[5]), reg1.array[4]);
  reg2.array[6] = __hadd2_rn(reg1.array[6], reg1.array[7]);
  reg2.array[7] = __hadd2_rn(__hneg2(reg1.array[7]), reg1.array[6]);

  #pragma unroll
  for (int i = 0; i < 8; i++) {
    reg1.array[i].x = __hadd_rn(reg2.array[i].x, reg2.array[i].y);
    reg1.array[i].y = __hadd_rn(__hneg(reg2.array[i].y), reg2.array[i].x);
  }

  // #pragma unroll
  // for (int i = 0; i < 8; i++) {
  //   reg1.array[i] = __hmul2_rn(reg1.array[i], __float2half2_rn(0.0441941738));
  // }

  ((__half16*)warp_buffer)[warp_idx] = reg1;
  __half2 *out_pt = &out[batch_idx * 256];
  #pragma unroll
  for (int i = 0; i < 8; i++) {
    out_pt[i * WARP_SIZE + warp_idx] = warp_buffer[i * WARP_SIZE + warp_idx];
  }

}


__global__ void fast_hadamard_transform_cuda_kernel_dim1024_fp16(
  __half2 *inp,    // [batch_size, 512]
  __half2 *out,    // [batch_size, 512]
  int batch_size
) {

  int batch_idx = blockIdx.x * blockDim.y + threadIdx.y;
  int warp_idx = threadIdx.x;

  if (batch_idx >= batch_size) {
    return;
  }

  __shared__ __half2 buffer[4096];
  __half2 *warp_buffer = &buffer[threadIdx.y * 512];
  __half2 *inp_pt = &inp[batch_idx * 512];

  #pragma unroll
  for (int i = 0; i < 16; i++) {
    warp_buffer[i * WARP_SIZE + warp_idx] = inp_pt[i * WARP_SIZE + warp_idx];
  }
  __half32 reg2 = ((__half32*)warp_buffer)[warp_idx];

  #pragma unroll
  for (int stride = (WARP_SIZE / 2); stride > 0; stride = stride / 2) {
    __half2 sign = __float2half2_rn(1 - ((warp_idx / stride) % 2) * 2);
    #pragma unroll
    for (int i = 0; i < 16; i++) {
      reg2.array[i] = __hadd2_rn(__hmul2_rn(sign, reg2.array[i]), __shfl_xor_sync(FULL_MASK, reg2.array[i], stride));
    }
  }

  __half32 reg1;

  reg1.array[0] = __hadd2_rn(reg2.array[0], reg2.array[8]);
  reg1.array[1] = __hadd2_rn(reg2.array[1], reg2.array[9]);
  reg1.array[2] = __hadd2_rn(reg2.array[2], reg2.array[10]);
  reg1.array[3] = __hadd2_rn(reg2.array[3], reg2.array[11]);
  reg1.array[4] = __hadd2_rn(reg2.array[4], reg2.array[12]);
  reg1.array[5] = __hadd2_rn(reg2.array[5], reg2.array[13]);
  reg1.array[6] = __hadd2_rn(reg2.array[6], reg2.array[14]);
  reg1.array[7] = __hadd2_rn(reg2.array[7], reg2.array[15]);
  reg1.array[8] = __hadd2_rn(__hneg2(reg2.array[8]), reg2.array[0]);
  reg1.array[9] = __hadd2_rn(__hneg2(reg2.array[9]), reg2.array[1]);
  reg1.array[10] = __hadd2_rn(__hneg2(reg2.array[10]), reg2.array[2]);
  reg1.array[11] = __hadd2_rn(__hneg2(reg2.array[11]), reg2.array[3]);
  reg1.array[12] = __hadd2_rn(__hneg2(reg2.array[12]), reg2.array[4]);
  reg1.array[13] = __hadd2_rn(__hneg2(reg2.array[13]), reg2.array[5]);
  reg1.array[14] = __hadd2_rn(__hneg2(reg2.array[14]), reg2.array[6]);
  reg1.array[15] = __hadd2_rn(__hneg2(reg2.array[15]), reg2.array[7]);

  reg2.array[0] = __hadd2_rn(reg1.array[0], reg1.array[4]);
  reg2.array[1] = __hadd2_rn(reg1.array[1], reg1.array[5]);
  reg2.array[2] = __hadd2_rn(reg1.array[2], reg1.array[6]);
  reg2.array[3] = __hadd2_rn(reg1.array[3], reg1.array[7]);
  reg2.array[4] = __hadd2_rn(__hneg2(reg1.array[4]), reg1.array[0]);
  reg2.array[5] = __hadd2_rn(__hneg2(reg1.array[5]), reg1.array[1]);
  reg2.array[6] = __hadd2_rn(__hneg2(reg1.array[6]), reg1.array[2]);
  reg2.array[7] = __hadd2_rn(__hneg2(reg1.array[7]), reg1.array[3]);
  reg2.array[8] = __hadd2_rn(reg1.array[8], reg1.array[12]);
  reg2.array[9] = __hadd2_rn(reg1.array[9], reg1.array[13]);
  reg2.array[10] = __hadd2_rn(reg1.array[10], reg1.array[14]);
  reg2.array[11] = __hadd2_rn(reg1.array[11], reg1.array[15]);
  reg2.array[12] = __hadd2_rn(__hneg2(reg1.array[12]), reg1.array[8]);
  reg2.array[13] = __hadd2_rn(__hneg2(reg1.array[13]), reg1.array[9]);
  reg2.array[14] = __hadd2_rn(__hneg2(reg1.array[14]), reg1.array[10]);
  reg2.array[15] = __hadd2_rn(__hneg2(reg1.array[15]), reg1.array[11]);

  reg1.array[0] = __hadd2_rn(reg2.array[0], reg2.array[2]);
  reg1.array[1] = __hadd2_rn(reg2.array[1], reg2.array[3]);
  reg1.array[2] = __hadd2_rn(__hneg2(reg2.array[2]), reg2.array[0]);
  reg1.array[3] = __hadd2_rn(__hneg2(reg2.array[3]), reg2.array[1]);
  reg1.array[4] = __hadd2_rn(reg2.array[4], reg2.array[6]);
  reg1.array[5] = __hadd2_rn(reg2.array[5], reg2.array[7]);
  reg1.array[6] = __hadd2_rn(__hneg2(reg2.array[6]), reg2.array[4]);
  reg1.array[7] = __hadd2_rn(__hneg2(reg2.array[7]), reg2.array[5]);
  reg1.array[8] = __hadd2_rn(reg2.array[8], reg2.array[10]);
  reg1.array[9] = __hadd2_rn(reg2.array[9], reg2.array[11]);
  reg1.array[10] = __hadd2_rn(__hneg2(reg2.array[10]), reg2.array[8]);
  reg1.array[11] = __hadd2_rn(__hneg2(reg2.array[11]), reg2.array[9]);
  reg1.array[12] = __hadd2_rn(reg2.array[12], reg2.array[14]);
  reg1.array[13] = __hadd2_rn(reg2.array[13], reg2.array[15]);
  reg1.array[14] = __hadd2_rn(__hneg2(reg2.array[14]), reg2.array[12]);
  reg1.array[15] = __hadd2_rn(__hneg2(reg2.array[15]), reg2.array[13]);

  reg2.array[0] = __hadd2_rn(reg1.array[0], reg1.array[1]);
  reg2.array[1] = __hadd2_rn(__hneg2(reg1.array[1]), reg1.array[0]);
  reg2.array[2] = __hadd2_rn(reg1.array[2], reg1.array[3]);
  reg2.array[3] = __hadd2_rn(__hneg2(reg1.array[3]), reg1.array[2]);
  reg2.array[4] = __hadd2_rn(reg1.array[4], reg1.array[5]);
  reg2.array[5] = __hadd2_rn(__hneg2(reg1.array[5]), reg1.array[4]);
  reg2.array[6] = __hadd2_rn(reg1.array[6], reg1.array[7]);
  reg2.array[7] = __hadd2_rn(__hneg2(reg1.array[7]), reg1.array[6]);
  reg2.array[8] = __hadd2_rn(reg1.array[8], reg1.array[9]);
  reg2.array[9] = __hadd2_rn(__hneg2(reg1.array[9]), reg1.array[8]);
  reg2.array[10] = __hadd2_rn(reg1.array[10], reg1.array[11]);
  reg2.array[11] = __hadd2_rn(__hneg2(reg1.array[11]), reg1.array[10]);
  reg2.array[12] = __hadd2_rn(reg1.array[12], reg1.array[13]);
  reg2.array[13] = __hadd2_rn(__hneg2(reg1.array[13]), reg1.array[12]);
  reg2.array[14] = __hadd2_rn(reg1.array[14], reg1.array[15]);
  reg2.array[15] = __hadd2_rn(__hneg2(reg1.array[15]), reg1.array[14]);

  #pragma unroll
  for (int i = 0; i < 16; i++) {
    reg1.array[i].x = __hadd_rn(reg2.array[i].x, reg2.array[i].y);
    reg1.array[i].y = __hadd_rn(__hneg(reg2.array[i].y), reg2.array[i].x);
  }

  ((__half32*)warp_buffer)[warp_idx] = reg1;
  __half2 *out_pt = &out[batch_idx * 512];
  #pragma unroll
  for (int i = 0; i < 16; i++) {
    out_pt[i * WARP_SIZE + warp_idx] = warp_buffer[i * WARP_SIZE + warp_idx];
  }

}


__global__ void fast_hadamard_transform_cuda_kernel(
  float *inp,    // [batch_size, vector_dim]
  float *out,    // [batch_size, vector_dim]
  int batch_size,
  int vector_dim
) {

  int batch_idx = blockIdx.x;
  int dim_idx = threadIdx.x;

  extern __shared__ float buffer[];
  float *vector_buffer = buffer;

  vector_buffer[dim_idx] = inp[batch_idx * vector_dim + dim_idx];
  __syncthreads();

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
  __syncthreads();

  // out[batch_idx * vector_dim + dim_idx] = vector_buffer[dim_idx] / sqrt(vector_dim);
  out[batch_idx * vector_dim + dim_idx] = vector_buffer[dim_idx];
}
