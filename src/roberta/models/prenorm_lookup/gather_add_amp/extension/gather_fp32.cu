#include <torch/extension.h>
#include <ATen/ATen.h>
#include "gather_fp32.h"
#include "gather_kernel_fp32.h"

void weighted_vector_gather_add_kernel_fp32(
  at::Tensor indexes,
  at::Tensor source,
  at::Tensor weights,
  at::Tensor outputs
) {

  int batch_size = indexes.size(0);
  int index_size = indexes.size(1);
  int source_size = source.size(0);
  int vector_dim = source.size(1);

  int thread_x = MIN(vector_dim, MAX_THREADS_PER_BLOCK);
  int thread_y = MAX_THREADS_PER_BLOCK / thread_x;
  int block_x = batch_size / thread_y + 1;
  dim3 threads(thread_x, thread_y);
  dim3 blocks(block_x);
  int shared_mem = thread_y * 2 * index_size * sizeof(float);

  weighted_vector_gather_add_cuda_kernel_fp32<<<blocks, threads, shared_mem>>>(
    indexes.data_ptr<int>(),
    source.data_ptr<float>(),
    weights.data_ptr<float>(),
    outputs.data_ptr<float>(),
    batch_size,
    index_size,
    source_size,
    vector_dim
  );

}

void weighted_vector_scatter_add_kernel_fp32(
  at::Tensor indexes,
  at::Tensor source,
  at::Tensor weights,
  at::Tensor outputs
) {

  int batch_size = indexes.size(0);
  int index_size = indexes.size(1);
  int output_size = outputs.size(0);
  int vector_dim = source.size(1);

  int thread_x = MIN(vector_dim, MAX_THREADS_PER_BLOCK);
  int thread_y = MAX_THREADS_PER_BLOCK / thread_x;
  int block_x = batch_size / thread_y + 1;
  dim3 threads(thread_x, thread_y);
  dim3 blocks(block_x);
  int shared_mem = thread_y * 2 * index_size * sizeof(float);

  weighted_vector_scatter_add_cuda_kernel_fp32<<<blocks, threads, shared_mem>>>(
    indexes.data_ptr<int>(),
    source.data_ptr<float>(),
    weights.data_ptr<float>(),
    outputs.data_ptr<float>(),
    batch_size,
    index_size,
    output_size,
    vector_dim
  );
}

void indexed_inner_product_kernel_fp32(
  at::Tensor indexes,
  at::Tensor source_1,
  at::Tensor source_2,
  at::Tensor outputs
) {

  int batch_size = indexes.size(0);
  int index_size = indexes.size(1);
  int source_size = source_2.size(0);
  int vector_dim = source_2.size(1);

  int thread_x = MIN(vector_dim, MAX_THREADS_PER_BLOCK);
  int thread_y = MAX_THREADS_PER_BLOCK / thread_x;
  int block_x = batch_size / thread_y + 1;
  dim3 threads(thread_x, thread_y);
  dim3 blocks(block_x);
  int shared_mem = thread_y * 2 * index_size * sizeof(float);

  indexed_inner_product_cuda_kernel_fp32<<<blocks, threads, shared_mem>>>(
    indexes.data_ptr<int>(),
    source_1.data_ptr<float>(),
    source_2.data_ptr<float>(),
    outputs.data_ptr<float>(),
    batch_size,
    index_size,
    source_size,
    vector_dim
  );
}
