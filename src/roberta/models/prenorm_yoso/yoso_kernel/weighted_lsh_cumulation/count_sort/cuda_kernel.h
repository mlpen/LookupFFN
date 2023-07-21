
#define WARP_SIZE 32
#define OPTIMAL_THREADS 128
#define EMPTY_VALUE -1
#define FULL_MASK 0xffffffff

__global__ void count_sort_cuda_kernel(
  int *key_mask,      // [batch_size, num_key]
  int *key_hashcode,  // [batch_size, num_hash_f, num_key]
  int *sort_info,     // [batch_size, num_hash_f, hashtable_capacity]
  int *sorted_idxes,  // [batch_size, num_hash_f, num_key]
  int batch_size,
  int num_hash_f,
  int hashtable_capacity,
  int num_key
);
