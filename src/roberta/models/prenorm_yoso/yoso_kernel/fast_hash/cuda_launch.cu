#include <torch/extension.h>
#include <ATen/ATen.h>
#include "cuda_launch.h"
#include "cuda_kernel.h"

//////////////////////////////////////////////////////////////////////////////////////////////////
//////////////////////////////////////////////////////////////////////////////////////////////////

at::Tensor fast_hash_kernel(
  at::Tensor mask,
  at::Tensor vector,
  at::Tensor Dmat,
  int num_hash_f,
  int hashcode_len
) {

  int batch_size = vector.size(0);
  int num_vector = vector.size(1);
  int vector_dim = vector.size(2);
  int num_part = Dmat.size(2);
  int num_worker = max(1, OPTIMAL_THREADS / vector_dim);

  at::Tensor hashcode = at::zeros({batch_size, num_vector, num_hash_f}, mask.options());

  dim3 threads(vector_dim, num_worker);
  dim3 blocks(num_vector, batch_size);
  int shared_mem = (1 + num_worker) * vector_dim * sizeof(float);

  fast_hash_cuda_kernel<<<blocks, threads, shared_mem>>>(
    mask.data_ptr<int>(),
    vector.data_ptr<float>(),
    Dmat.data_ptr<int>(),
    hashcode.data_ptr<int>(),
    batch_size,
    num_vector,
    vector_dim,
    num_part,
    num_hash_f,
    hashcode_len
  );

  return hashcode;

}
