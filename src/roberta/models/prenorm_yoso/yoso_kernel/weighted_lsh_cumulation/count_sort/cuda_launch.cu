#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_launch.h"
#include "cuda_kernel.h"
#include <vector>

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

std::vector<at::Tensor> count_sort_kernel(
  at::Tensor key_mask,
  at::Tensor key_hashcode,
  int hashtable_capacity
) {

  int batch_size = key_hashcode.size(0);
  int num_hash_f = key_hashcode.size(1);
  int num_key = key_hashcode.size(2);

  at::Tensor sort_info = at::zeros({batch_size, num_hash_f, hashtable_capacity}, key_hashcode.options());
  at::Tensor sorted_idxes = at::zeros({batch_size, num_hash_f, num_key}, key_hashcode.options());

  dim3 threads(OPTIMAL_THREADS);
  dim3 blocks(num_hash_f, batch_size);
  int shared_mem = (hashtable_capacity + 2 * num_key) * sizeof(float);

  count_sort_cuda_kernel<<<blocks, threads, shared_mem>>>(
    key_mask.data_ptr<int>(),
    key_hashcode.data_ptr<int>(),
    sort_info.data_ptr<int>(),
    sorted_idxes.data_ptr<int>(),
    batch_size,
    num_hash_f,
    hashtable_capacity,
    num_key
  );

  return {sort_info, sorted_idxes};
}
