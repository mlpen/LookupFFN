#include <iostream>
#include <torch/extension.h>
#include <ATen/ATen.h>
#define _OPENMP
#include <ATen/ParallelOpenMP.h>

#include <sys/mman.h>
#include <cstring>

#ifdef __AVX__
    #include <immintrin.h>
#else
    #warning No AVX support - will not compile
#endif


at::Tensor weighted_vector_gather_add_scalar(
    at::Tensor indexes,      // [batch_size, index_size]
    at::Tensor source,       // [source_size, vector_dim]
    at::Tensor weights       // [batch_size, index_size]
) {
    const ssize_t B = indexes.size(0); // batch_size
    const ssize_t D = source.size(1);  // vector_dim
    const ssize_t I = indexes.size(1); // index_size
    const ssize_t S = source.size(0);  // source_size

    at::Tensor out = at::zeros({B, D});

    int * indexes_d = indexes.data_ptr<int>();
    float * source_d = source.data_ptr<float>();
    float * weights_d = weights.data_ptr<float>();
    float * out_d = out.data_ptr<float>();

    for (ssize_t b = 0; b < B; b++) {
        for (ssize_t i = 0; i < I; i++) {
            const ssize_t idx = indexes_d[b * I + i];
            const float w = weights_d[b * I + i];
            for (ssize_t d = 0; d < D; d++) {
                out_d[b * D + d] += source_d[idx * D + d] * w;
            }
        }
    }

    return out;
}

at::Tensor weighted_vector_gather_add_avx2_par(
    at::Tensor indexes,      // [batch_size, index_size]  // [batch_size, num_tables]
    at::Tensor source,       // [source_size, vector_dim] // [num_tables, table_size, vector_dim]
    at::Tensor weights       // [batch_size, index_size]
) {

    const ssize_t B = indexes.size(0); // batch_size
    const ssize_t D = source.size(1);  // vector_dim
    const ssize_t I = indexes.size(1); // index_size
    const ssize_t S = source.size(0);  // source_size
    at::Tensor out = at::zeros({B, D});

    const ssize_t source_num_elem = S * D;
    const ssize_t GS = 1;

    int * indexes_d = indexes.data_ptr<int>();
    float * source_d = source.data_ptr<float>();
    float * weights_d = weights.data_ptr<float>();
    float * out_d = out.data_ptr<float>();

    const ssize_t TS = S / I;
    const ssize_t NT = omp_get_num_threads();
    const ssize_t TT = 64; // Tiling factor for tokens
    const ssize_t NBATCH = B / TT;

    const ssize_t TQ = 256; // Tiling factor for table accesses
    const ssize_t TD = 512; // Tiling factor for D

    #pragma omp parallel
    {
        for (ssize_t qi = 0; qi < S; qi += TQ) {
            const ssize_t i = (qi / TS);
            #pragma omp for schedule(static)
            for (ssize_t b = 0; b < B; b += TT) {
                for (ssize_t ti = b; ti < b + TT; ti++) {
                    const ssize_t idx = indexes_d[(ti * I + i)];
                    if (idx < qi || idx >= qi + TQ) continue;
                    const float w = weights_d[(ti * I + i)];

                    #pragma GCC ivdep
                    for (ssize_t d = 0; d < D; d++) {
                        out_d[(ti * D + d)] += source_d[(idx) * D + d] * w;
                    }
                }
            }
        }
    }

    return out;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("weighted_vector_gather_add_scalar", &weighted_vector_gather_add_scalar, "weighted_vector_gather_add (x86)");
    m.def("weighted_vector_gather_add_avx2_par", &weighted_vector_gather_add_avx2_par, "weighted_vector_gather_add (AVX parallel)");
}
