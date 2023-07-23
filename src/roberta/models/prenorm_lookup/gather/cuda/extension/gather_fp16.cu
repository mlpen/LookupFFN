#include <torch/extension.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_fp16.h>

#include "gather_fp16.h"
#include "gather_kernel_fp16.h"

void weighted_vector_gather_add_kernel_fp16(
  at::Tensor indexes,
  at::Tensor source,
  at::Tensor weights,
  at::Tensor outputs
) {

  int batch_size = indexes.size(0);
  int index_size = indexes.size(1);
  int source_size = source.size(0);
  int vector_dim = source.size(1);

  int thread_x = MIN(vector_dim / 2, MAX_THREADS_PER_BLOCK);
  int thread_y = MAX_THREADS_PER_BLOCK / thread_x;
  int block_x = batch_size / thread_y + 1;
  dim3 threads(thread_x, thread_y);
  dim3 blocks(block_x);
  int shared_mem = thread_y * index_size * 6;

  weighted_vector_gather_add_cuda_kernel_fp16<<<blocks, threads, shared_mem>>>(
    indexes.data_ptr<int>(),
    (__half2*)(source.data_ptr<at::Half>()),
    (__half2*)(weights.data_ptr<at::Half>()),
    (__half2*)(outputs.data_ptr<at::Half>()),
    batch_size,
    index_size / 2,
    source_size,
    vector_dim / 2
  );
}

void weighted_vector_scatter_add_kernel_fp16(
  at::Tensor indexes,
  at::Tensor source,
  at::Tensor weights,
  at::Tensor outputs
) {

  int batch_size = indexes.size(0);
  int index_size = indexes.size(1);
  int output_size = outputs.size(0);
  int vector_dim = source.size(1);

  int thread_x = MIN(vector_dim / 2, MAX_THREADS_PER_BLOCK);
  int thread_y = MAX_THREADS_PER_BLOCK / thread_x;
  int block_x = batch_size / thread_y + 1;
  dim3 threads(thread_x, thread_y);
  dim3 blocks(block_x);
  int shared_mem = thread_y * index_size * 6;

  weighted_vector_scatter_add_cuda_kernel_fp16<<<blocks, threads, shared_mem>>>(
    indexes.data_ptr<int>(),
    (__half2*)(source.data_ptr<at::Half>()),
    (__half2*)(weights.data_ptr<at::Half>()),
    (__half2*)(outputs.data_ptr<at::Half>()),
    batch_size,
    index_size / 2,
    output_size,
    vector_dim / 2
  );
}

void indexed_inner_product_kernel_fp16(
  at::Tensor indexes,
  at::Tensor source_1,
  at::Tensor source_2,
  at::Tensor outputs
) {

  int batch_size = indexes.size(0);
  int index_size = indexes.size(1);
  int source_size = source_2.size(0);
  int vector_dim = source_2.size(1);

  int thread_x = MIN(vector_dim / 2, MAX_THREADS_PER_BLOCK);
  int thread_y = MAX_THREADS_PER_BLOCK / thread_x;
  int block_x = batch_size / thread_y + 1;
  dim3 threads(thread_x, thread_y);
  dim3 blocks(block_x);
  int shared_mem = thread_y * 2 * index_size * sizeof(float);

  indexed_inner_product_cuda_kernel_fp16<<<blocks, threads, shared_mem>>>(
    indexes.data_ptr<int>(),
    (__half2*)(source_1.data_ptr<at::Half>()),
    (__half2*)(source_2.data_ptr<at::Half>()),
    (__half2*)(outputs.data_ptr<at::Half>()),
    batch_size,
    index_size / 2,
    source_size,
    vector_dim / 2
  );
}
