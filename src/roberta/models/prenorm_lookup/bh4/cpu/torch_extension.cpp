#include <iostream>
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <libxsmm.h>


#define _OPENMP
#include <ATen/ParallelOpenMP.h>

#ifdef __AVX__
    #include <immintrin.h>
#else
    #warning No AVX support - will not compile
#endif

typedef libxsmm_mmfunction<float> float_kernel;

float_kernel kernel_fp32;

void setup_xsmm_kernel_float_avx2(int m, int n, int k, int ldb, int lda, int ldc) {
    // Libxsmm has a "just in time" compile step which generates machine code
    // for a GEMM given M, N, K values.

    std::cout << "Compiling libxsmm kernel for FP32 / AVX2..." << std::endl;
    kernel_fp32 = float_kernel(
        LIBXSMM_GEMM_FLAG_NONE,
        m, n, k,
        lda, ldb, ldc);
}

inline void fwht_cpu_fp32_avx2_subvec_(
    // [batch_size, num_query]
    float * td,
    const ssize_t i,
    const ssize_t B,
    const ssize_t N
) {
#ifdef __AVX__
    static const __m256i swap1 = _mm256_setr_epi32(1, 0, 3, 2, 5, 4, 7, 6);
    static const __m256i swap2 = _mm256_setr_epi32(2, 3, 0, 1, 6, 7, 4, 5);
    static const __m256i swap3 = _mm256_setr_epi32(4, 5, 6, 7, 0, 1, 2, 3);
    static const __m256 flip1 = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000));
    static const __m256 flip2 = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0x80000000, 0x80000000, 0, 0, 0x80000000, 0x80000000));
    static const __m256 flip3 = _mm256_castsi256_ps(_mm256_setr_epi32(0, 0, 0, 0, 0x80000000, 0x80000000, 0x80000000, 0x80000000));
    static const __m256 norm = _mm256_set1_ps(0.125f);

    __m256 va = _mm256_loadu_ps(td + i);

    __m256 vb = _mm256_permutevar8x32_ps(va, swap1);
    va = _mm256_add_ps(vb, _mm256_xor_ps(va, flip1));

    vb = _mm256_permutevar8x32_ps(va, swap2);
    va = _mm256_add_ps(vb, _mm256_xor_ps(va, flip2));

    vb = _mm256_permutevar8x32_ps(va, swap3);
    va = _mm256_add_ps(vb, _mm256_xor_ps(va, flip3));

    va = _mm256_mul_ps(va, norm);

    _mm256_storeu_ps(td + i, va);
#else
    #warning No AVX support - will not compile
#endif
}

inline void internal_fwht_cpu_fp32_avx2_(
    float * td,
    const ssize_t B,
    const ssize_t N
) {
#ifdef __AVX__
    static const __m256 norm = _mm256_set1_ps(0.5f);

    #pragma omp parallel for schedule(static, 4)
    for (ssize_t b = 0; b < B; b++) {
        float * td_b = td + b * N;

        // #pragma omp parallel for schedule(static, 4)
        for (ssize_t i = 0; i < N; i += 8) {
            fwht_cpu_fp32_avx2_subvec_(td_b, i, B, N);
        }

        // #pragma omp barrier

        for (ssize_t h = 8; h < N; h *= 2) {
            // #pragma omp parallel for collapse(2)
            for (ssize_t blk = 0; blk < N; blk += h * 2) {
                for (ssize_t j = blk; j < blk + h; j += 8) {

                    __m256 x = _mm256_loadu_ps(td_b + j);
                    __m256 y = _mm256_loadu_ps(td_b + j + h);

                    _mm256_storeu_ps(
                        td_b + j,
                        _mm256_mul_ps(_mm256_add_ps(x, y), norm));

                    _mm256_storeu_ps(
                        td_b + j + h,
                        _mm256_mul_ps(_mm256_sub_ps(x, y), norm));
                }
            }

            // #pragma omp barrier
        }
    }


#else
    #warning No AVX support - will not compile
#endif
}

void fwht_cpu_fp32_avx2_(
    at::Tensor t
) {
    float * td = t.data_ptr<float>();
    const ssize_t B = t.size(0);
    const ssize_t N = t.size(1);

    internal_fwht_cpu_fp32_avx2_(td, B, N);
}

void sbdht(
    at::Tensor& x,
    at::Tensor& w
) {
    const ssize_t batch_size = x.size(0);
    const ssize_t dim = x.size(1);

    const ssize_t num_iters = w.size(0);
    const ssize_t num_blocks = w.size(1); // Number of Blocks
    const ssize_t block_size = w.size(2); // Block Size

    const ssize_t num_iters_stride = w.stride(0);
    const ssize_t num_blocks_stride = w.stride(1);

    at::Tensor cache = at::zeros({block_size, dim}, x.options());

    float *w_pt = w.data_ptr<float>();
    float *x_pt = x.data_ptr<float>();
    float *y_pt = cache.data_ptr<float>();

    assert(dim == num_blocks * block_size);
    assert(batch_size % block_size == 0);

    for (size_t batch_idx = 0; batch_idx < batch_size; batch_idx += block_size) {
        #pragma omp parallel for schedule(static, 1)
        for (size_t dim_offset = 0; dim_offset < dim; dim_offset += block_size) {
            kernel_fp32(
                w_pt + 0 * num_iters_stride + (dim_offset / block_size) * num_blocks_stride,
                x_pt + batch_idx * dim + dim_offset,
                y_pt + dim_offset
            );
        }
        internal_fwht_cpu_fp32_avx2_(y_pt, block_size, dim);

        memset(x_pt + batch_idx * dim, 0.0, block_size * dim * sizeof(float));

        #pragma omp parallel for schedule(static, 1)
        for (size_t dim_offset = 0; dim_offset < dim; dim_offset += block_size) {
            kernel_fp32(
                w_pt + 1 * num_iters_stride + (dim_offset / block_size) * num_blocks_stride,
                y_pt + dim_offset,
                x_pt + batch_idx * dim + dim_offset
            );
        }
        internal_fwht_cpu_fp32_avx2_(x_pt + batch_idx * dim, block_size, dim);

        memset(y_pt, 0.0, block_size * dim * sizeof(float));

        #pragma omp parallel for schedule(static, 1)
        for (size_t dim_offset = 0; dim_offset < dim; dim_offset += block_size) {
            kernel_fp32(
                w_pt + 2 * num_iters_stride + (dim_offset / block_size) * num_blocks_stride,
                x_pt + batch_idx * dim + dim_offset,
                y_pt + dim_offset
            );
        }
        internal_fwht_cpu_fp32_avx2_(y_pt, block_size, dim);

        memset(x_pt + batch_idx * dim, 0.0, block_size * dim * sizeof(float));

        #pragma omp parallel for schedule(static, 1)
        for (size_t dim_offset = 0; dim_offset < dim; dim_offset += block_size) {
            kernel_fp32(
                w_pt + 3 * num_iters_stride + (dim_offset / block_size) * num_blocks_stride,
                y_pt + dim_offset,
                x_pt + batch_idx * dim + dim_offset
            );
        }
        internal_fwht_cpu_fp32_avx2_(x_pt + batch_idx * dim, block_size, dim);

        memset(y_pt, 0.0, block_size * dim * sizeof(float));
    }

}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("setup_xsmm_kernel_float_avx2", &setup_xsmm_kernel_float_avx2, "Setup XSMM kernel");
    m.def("fwht_cpu_fp32_avx2_", &fwht_cpu_fp32_avx2_, "LSH Cumulation (AVX2)");
    m.def("sbdht", &sbdht, "SBDHT");
}
