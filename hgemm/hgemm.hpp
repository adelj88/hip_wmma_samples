#ifndef HIP_HGEMM_HPP
#define HIP_HGEMM_HPP

#include <common/matrix.hpp>
#include <hgemm/kernels/shared.hpp>
#include <hgemm/kernels/wmma.hpp>
#include <hgemm/kernels/wmma_shared.hpp>
#include <hgemm/kernels/wmma_shared_warp.hpp>
#include <hgemm/kernels/wmma_shared_warp_buf.hpp>
#include <hgemm/kernels/wmma_shared_warp_buf_vec.hpp>
#include <hgemm/kernels/wmma_prefetch.hpp>
#include <hgemm/kernels/rocblas.hpp>
#ifdef HAS_ROCWMMA
#include <hgemm/kernels/rocwmma.hpp>
#endif

/**
 * @brief CPU reference implementation
 */
template<matrix_layout L1, matrix_layout L2, matrix_layout L3>
void hgemm_cpu(matrix<half, L1>& C, const matrix<half, L2>& A, const matrix<half, L3>& B)
{
    for(size_t i = 0; i < C.rows(); ++i)
    {
        for(size_t j = 0; j < C.cols(); ++j)
        {
            float acc = 0.0f;
            for(size_t k = 0; k < A.cols(); ++k)
            {
                acc += static_cast<float>(A(i, k)) * static_cast<float>(B(k, j));
            }
            C(i, j) = static_cast<half>(acc);
        }
    }
}

/**
 * @brief Verify results against CPU reference
 */
template<matrix_layout L>
bool verify_results(const matrix<half, L>& gpu_result,
                    const matrix<half, L>& cpu_result,
                    float                  tolerance = 5e-2f)
{
    for(size_t i = 0; i < gpu_result.rows(); ++i)
    {
        for(size_t j = 0; j < gpu_result.cols(); ++j)
        {
            float gpu_val  = static_cast<float>(gpu_result(i, j));
            float cpu_val  = static_cast<float>(cpu_result(i, j));
            float abs_diff = std::abs(gpu_val - cpu_val);
            float rel_diff = abs_diff / std::max(std::abs(cpu_val), 1e-5f);

            if(rel_diff > tolerance)
            {
                std::cerr << "Verification failed at (" << i << "," << j
                          << "): " << "GPU=" << gpu_val << " CPU=" << cpu_val
                          << " rel_diff=" << rel_diff << std::endl;
                return false;
            }
        }
    }
    std::cout << "Verification passed" << std::endl;
    return true;
}

#endif // HIP_HGEMM_HPP
