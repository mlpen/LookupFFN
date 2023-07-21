
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff
#define OPTIMAL_THREADS 128

#define min(a, b) ((a)<(b)?(a):(b))
#define max(a, b) ((a)>(b)?(a):(b))
#define ceil_divide(a, b) ((a)/(b)+((a)%(b)!=0))

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
);
