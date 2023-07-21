
#define WARP_SIZE 32
#define MAX_THREADS_PER_BLOCK 128
#define FULL_MASK 0xffffffff
#define MIN(a, b) ((a)<(b)?(a):(b))
#define MAX(a, b) ((a)>(b)?(a):(b))

__global__ void weighted_vector_gather_add_cuda_kernel_fp32(
  int  *indexes,      // [batch_size, index_size]
  float *source,       // [source_size, vector_dim]
  float *weights,      // [batch_size, index_size]
  float *outputs,      // [batch_size, vector_dim]
  int batch_size,
  int index_size,
  int source_size,
  int vector_dim
);

__global__ void weighted_vector_scatter_add_cuda_kernel_fp32(
  int  *indexes,      // [batch_size, index_size]
  float *source,       // [batch_size, vector_dim]
  float *weights,      // [batch_size, index_size]
  float *outputs,      // [output_size, vector_dim]
  int batch_size,
  int index_size,
  int output_size,
  int vector_dim
);

__global__ void indexed_inner_product_cuda_kernel_fp32(
  int  *indexes,      // [batch_size, index_size]
  float *source_1,     // [batch_size, vector_dim]
  float *source_2,     // [source_size, vector_dim]
  float *outputs,      // [batch_size, index_size]
  int batch_size,
  int index_size,
  int source_size,
  int vector_dim
);
