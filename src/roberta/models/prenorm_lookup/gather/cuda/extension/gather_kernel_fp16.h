#include <cuda.h>
#include <cuda_fp16.h>

#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 128
#define FULL_MASK 0xffffffff
#define MIN(a, b) ((a)<(b)?(a):(b))
#define MAX(a, b) ((a)>(b)?(a):(b))

__global__ void weighted_vector_gather_add_cuda_kernel_fp16(
  int     *indexes,    // [batch_size, index_size]
  __half2 *source,     // [source_size, vector_dim_div_2]
  __half2 *weights,    // [batch_size, index_size_div_2]
  __half2 *outputs,    // [batch_size, vector_dim_div_2]
  int batch_size,
  int index_size,
  int source_size,
  int vector_dim
);

__global__ void weighted_vector_scatter_add_cuda_kernel_fp16(
  int     *indexes,      // [batch_size, index_size]
  __half2 *source,       // [batch_size, vector_dim_div_2]
  __half2 *weights,      // [batch_size, index_size_div_2]
  __half2 *outputs,      // [output_size, vector_dim_div_2]
  int batch_size,
  int index_size_div_2,
  int output_size,
  int vector_dim_div_2
);

__global__ void indexed_inner_product_cuda_kernel_fp16(
  int     *indexes,      // [batch_size, index_size]
  __half2 *source_1,     // [batch_size, vector_dim_div_2]
  __half2 *source_2,     // [source_size, vector_dim_div_2]
  __half2 *outputs,      // [batch_size, index_size_div_2]
  int batch_size,
  int index_size_div_2,
  int source_size,
  int vector_dim_div_2
);
