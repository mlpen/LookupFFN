#include <torch/extension.h>
#include <ATen/ATen.h>
#include "gather_fp32.h"
#include "gather_fp16.h"

void weighted_vector_gather_add_fp32(
  at::Tensor indexes,
  at::Tensor source,
  at::Tensor weights,
  at::Tensor outputs
) {
  weighted_vector_gather_add_kernel_fp32(
    indexes,
    source,
    weights,
    outputs
  );
}

void indexed_inner_product_fp32(
  at::Tensor indexes,
  at::Tensor source_1,
  at::Tensor source_2,
  at::Tensor outputs
) {
  indexed_inner_product_kernel_fp32(
    indexes,
    source_1,
    source_2,
    outputs
  );
}

void weighted_vector_scatter_add_fp32(
  at::Tensor indexes,
  at::Tensor source,
  at::Tensor weights,
  at::Tensor outputs
) {
  weighted_vector_scatter_add_kernel_fp32(
    indexes,
    source,
    weights,
    outputs
  );
}

void weighted_vector_gather_add_fp16(
  at::Tensor indexes,
  at::Tensor source,
  at::Tensor weights,
  at::Tensor outputs
) {
  weighted_vector_gather_add_kernel_fp16(
    indexes,
    source,
    weights,
    outputs
  );
}

void indexed_inner_product_fp16(
  at::Tensor indexes,
  at::Tensor source_1,
  at::Tensor source_2,
  at::Tensor outputs
) {
  indexed_inner_product_kernel_fp16(
    indexes,
    source_1,
    source_2,
    outputs
  );
};

void weighted_vector_scatter_add_fp16(
  at::Tensor indexes,
  at::Tensor source,
  at::Tensor weights,
  at::Tensor outputs
) {
  weighted_vector_scatter_add_kernel_fp16(
    indexes,
    source,
    weights,
    outputs
  );
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("weighted_vector_gather_add_fp32", &weighted_vector_gather_add_fp32, "weighted_vector_gather_add_fp32 (CUDA)");
  m.def("indexed_inner_product_fp32", &indexed_inner_product_fp32, "indexed_inner_product_fp32 (CUDA)");
  m.def("weighted_vector_scatter_add_fp32", &weighted_vector_scatter_add_fp32, "weighted_vector_scatter_add_fp32 (CUDA)");
  m.def("weighted_vector_gather_add_fp16", &weighted_vector_gather_add_fp16, "weighted_vector_gather_add_fp16 (CUDA)");
  m.def("indexed_inner_product_fp16", &indexed_inner_product_fp16, "indexed_inner_product_fp16 (CUDA)");
  m.def("weighted_vector_scatter_add_fp16", &weighted_vector_scatter_add_fp16, "weighted_vector_scatter_add_fp16 (CUDA)");
}
