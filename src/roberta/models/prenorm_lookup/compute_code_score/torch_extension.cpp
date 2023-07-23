#include <iostream>
#include <torch/extension.h>
#include <ATen/ATen.h>
#define _OPENMP
#include <ATen/ParallelOpenMP.h>


#ifdef __AVX__
    #include <immintrin.h>
#else
    #warning No AVX support - will not compile
#endif

void compute_code_score(
    at::Tensor hash_scores,      // [batch_size, num_table, code_length]
    at::Tensor codes,            // [batch_size, num_table]
    at::Tensor scores            // [batch_size, num_table]
) {
    const ssize_t B = hash_scores.size(0);  // batch_size
    const ssize_t T = hash_scores.size(1);  // num_table
    const ssize_t D = hash_scores.size(2);  // code_length

    float *hash_scores_pt = hash_scores.data_ptr<float>();
    int *codes_pt = codes.data_ptr<int>();
    float *scores_pt = scores.data_ptr<float>();

    #pragma omp parallel for schedule(static, 1)
    for (ssize_t b = 0; b < B * T; b++) {
        int code = 0;
        float numerator = 0;
        float denominator = 1;
        for (ssize_t d = 0; d < D; d++) {
            float hs = hash_scores_pt[b * D + d];
            float abs_hs = abs(hs);
            code = code + (hs > 0) * (1 << (D - d - 1));
            numerator = numerator + abs_hs;
            denominator = denominator * (1 + exp(- 2 * abs_hs));
        }
        codes_pt[b] = code;
        scores_pt[b] = numerator / denominator;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("compute_code_score", &compute_code_score, "compute_code_score (x86)");
}