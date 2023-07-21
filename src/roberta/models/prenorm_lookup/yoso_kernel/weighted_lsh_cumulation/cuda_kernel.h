#define WARP_SIZE 32
#define WARP_SIZE_MINUS_ONE 31
#define OPTIMAL_THREADS 128
#define FULL_MASK 0xffffffff
#define EMPTY_VALUE -1
#define INV_PI 0.31830989
#define NORMALIZER 0.99
#define LCG_A 8525
#define LCG_C 132241
#define LCG_M 1048576

#define min(a, b) ((a)<(b)?(a):(b))
#define max(a, b) ((a)>(b)?(a):(b))
#define next_rint(x) ((LCG_A * (x) + LCG_C) % LCG_M)

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
);

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
);

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
);
