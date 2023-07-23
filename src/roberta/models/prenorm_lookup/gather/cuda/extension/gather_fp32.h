#include <torch/extension.h>
#include <ATen/ATen.h>

void weighted_vector_gather_add_kernel_fp32(
  at::Tensor indexes,
  at::Tensor source,
  at::Tensor weights,
  at::Tensor outputs
);

void weighted_vector_scatter_add_kernel_fp32(
  at::Tensor indexes,
  at::Tensor source,
  at::Tensor weights,
  at::Tensor outputs
);

void indexed_inner_product_kernel_fp32(
  at::Tensor indexes,
  at::Tensor source_1,
  at::Tensor source_2,
  at::Tensor outputs
);
