#include <cuda.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

#define min(a, b) ((a)<(b)?(a):(b))
#define max(a, b) ((a)>(b)?(a):(b))
#define ceil_divide(a, b) ((a)/(b)+((a)%(b)!=0))

__global__ void fast_hadamard_transform_cuda_kernel_dim512_fp16(
  __half2 *inp,    // [batch_size, 256]
  __half2 *out,    // [batch_size, 256]
  int batch_size
);

__global__ void fast_hadamard_transform_cuda_kernel_dim1024_fp16(
  __half2 *inp,    // [batch_size, 512]
  __half2 *out,    // [batch_size, 512]
  int batch_size
);


__global__ void fast_hadamard_transform_cuda_kernel(
  float *inp,    // [batch_size, vector_dim]
  float *out,    // [batch_size, vector_dim]
  int batch_size,
  int vector_dim
);
