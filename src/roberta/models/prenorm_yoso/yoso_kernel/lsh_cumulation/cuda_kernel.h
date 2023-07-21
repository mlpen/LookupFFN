
#define OPTIMAL_THREADS 128
#define WARP_SIZE 32
#define FULL_MASK 0xffffffff

__global__ void lsh_cumulation_cuda_kernel(
  int *key_mask,           // [batch_size, num_key]
  int *key_hashcode,       // [batch_size, num_key, num_hash_f]
  float *value,            // [batch_size, num_key, value_dim]
  float *hashtable,        // [batch_size, num_hash_f, hashtable_capacity, value_dim]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_key,
  int value_dim
);

__global__ void lsh_query_cuda_kernel(
  int *query_mask,         // [batch_size, num_query]
  int *query_hashcode,     // [batch_size, num_query, num_hash_f]
  float *hashtable,        // [batch_size, num_hash_f, hashtable_capacity, value_dim]
  float *cumulation_value, // [batch_size, num_query, value_dim]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_query,
  int value_dim
);

__global__ void lsh_cumulation_query_step1_cuda_kernel(
  int *key_mask,           // [batch_size, num_key]
  int *key_hashcode,      // [batch_size, num_key, num_hash_f]
  float *value,            // [batch_size, num_key, value_dim]
  float *hashtable_value,  // [batch_size, num_hash_f, hashtable_capacity, WARP_SIZE]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_key,
  int value_dim,
  int offset_warp
);

__global__ void lsh_cumulation_query_step2_cuda_kernel(
  int *query_mask,         // [batch_size, num_query]
  int *query_hashcode,    // [batch_size, num_query, num_hash_f]
  float *hashtable_value,  // [batch_size, num_hash_f, hashtable_capacity, WARP_SIZE]
  float *cumulation_value, // [batch_size, num_query, value_dim]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_query,
  int value_dim,
  int offset_warp
);
