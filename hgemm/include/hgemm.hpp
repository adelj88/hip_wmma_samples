/*
 * MIT License
 *
 * Copyright (c) 2024 Adel Johar
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef HIP_HGEMM_HPP
#define HIP_HGEMM_HPP

#include <common/matrix.hpp>
#include <kernels/rocblas.hpp>
#include <kernels/shared.hpp>
#include <kernels/wmma.hpp>
#include <kernels/wmma_opt_1.hpp>
#include <kernels/wmma_opt_2.hpp>
#include <kernels/wmma_opt_3.hpp>
#include <kernels/wmma_prefetch.hpp>
#include <kernels/wmma_shared.hpp>
#include <kernels/wmma_shared_warp.hpp>
#include <kernels/wmma_shared_warp_buf.hpp>
#include <kernels/wmma_shared_warp_buf_vec.hpp>
#include <kernels/wmma_shared_warp_vec.hpp>

/**
 * @brief CPU reference implementation
 */
template<matrix_layout L1, matrix_layout L2, matrix_layout L3>
void hgemm_cpu(matrix<half, L1>& C, const matrix<half, L2>& A, const matrix<half, L3>& B)
{
    for(size_t i = 0; i < C.m(); ++i)
    {
        for(size_t j = 0; j < C.n(); ++j)
        {
            float acc = 0.0f;
            for(size_t k = 0; k < A.n(); ++k)
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
bool verify_results(const matrix<half, L>& gpu_result, const matrix<half, L>& cpu_result)
{
    // Calculate matrix sizes and properties
    size_t m              = gpu_result.m();
    size_t n              = gpu_result.n();
    size_t total_elements = m * n;

    // Calculate matrix norms for objective comparison
    float gpu_frob_norm  = 0.0f;
    float cpu_frob_norm  = 0.0f;
    float diff_frob_norm = 0.0f;

    // For pattern analysis
    float dot_product = 0.0f; // For cosine similarity
    float sum_gpu = 0.0f, sum_cpu = 0.0f; // For structure similarity

    // Error statistics
    float  max_rel_diff          = 0.0f;
    float  sum_rel_diff          = 0.0f;
    size_t valid_rel_comparisons = 0;
    size_t max_rel_i = 0, max_rel_j = 0;
    float  max_rel_gpu_val = 0.0f, max_rel_cpu_val = 0.0f;

    // Scale tolerance based on matrix size - logarithmic scaling with more lenient approach
    float size_factor = std::log2(std::max(m, n)) / 8.0f;
    float tolerance   = 0.02f + 0.02f * size_factor; // Base: 2% + more aggressive scaling

    std::cout << "Using tolerance: " << tolerance << " for matrix size " << m << "x" << n
              << std::endl;

    // Calculate norms and element-wise differences
    for(size_t i = 0; i < m; ++i)
    {
        for(size_t j = 0; j < n; ++j)
        {
            float gpu_val = static_cast<float>(gpu_result(i, j));
            float cpu_val = static_cast<float>(cpu_result(i, j));
            float diff    = gpu_val - cpu_val;

            // Update Frobenius norms
            gpu_frob_norm += gpu_val * gpu_val;
            cpu_frob_norm += cpu_val * cpu_val;
            diff_frob_norm += diff * diff;

            // For pattern analysis
            dot_product += gpu_val * cpu_val;
            sum_gpu += gpu_val;
            sum_cpu += cpu_val;

            // Calculate relative difference for non-zero values
            if(std::abs(cpu_val) > 1e-5f)
            {
                float rel_diff = std::abs(diff) / std::abs(cpu_val);
                sum_rel_diff += rel_diff;
                valid_rel_comparisons++;

                // Track maximum relative difference
                if(rel_diff > max_rel_diff)
                {
                    max_rel_diff    = rel_diff;
                    max_rel_i       = i;
                    max_rel_j       = j;
                    max_rel_gpu_val = gpu_val;
                    max_rel_cpu_val = cpu_val;
                }
            }
        }
    }

    // Finalize norm calculations
    gpu_frob_norm  = std::sqrt(gpu_frob_norm);
    cpu_frob_norm  = std::sqrt(cpu_frob_norm);
    diff_frob_norm = std::sqrt(diff_frob_norm);

    // Calculate relative Frobenius norm error
    float rel_frob_error = diff_frob_norm / cpu_frob_norm;

    // Calculate average relative error
    float avg_rel_diff = valid_rel_comparisons > 0 ? sum_rel_diff / valid_rel_comparisons : 0.0f;

    // Pattern similarity measures
    // 1. Structural Similarity Index (simplified version)
    float mean_gpu = sum_gpu / total_elements;
    float mean_cpu = sum_cpu / total_elements;

    // Calculate variance and covariance for SSIM
    float var_gpu = 0.0f, var_cpu = 0.0f, covar = 0.0f;
    for(size_t i = 0; i < m; ++i)
    {
        for(size_t j = 0; j < n; ++j)
        {
            float gpu_val = static_cast<float>(gpu_result(i, j));
            float cpu_val = static_cast<float>(cpu_result(i, j));

            float gpu_diff = gpu_val - mean_gpu;
            float cpu_diff = cpu_val - mean_cpu;

            var_gpu += gpu_diff * gpu_diff;
            var_cpu += cpu_diff * cpu_diff;
            covar += gpu_diff * cpu_diff;
        }
    }

    var_gpu /= total_elements;
    var_cpu /= total_elements;
    covar /= total_elements;

    // Constants for SSIM
    const float C1 = 0.01f * mean_cpu * mean_cpu;
    const float C2 = 0.03f * var_cpu;

    // Calculate SSIM
    float ssim = ((2 * mean_gpu * mean_cpu + C1) * (2 * covar + C2))
                 / ((mean_gpu * mean_gpu + mean_cpu * mean_cpu + C1) * (var_gpu + var_cpu + C2));

    // Output validation statistics
    std::cout << "Maximum relative error: " << max_rel_diff << " at (" << max_rel_i << ","
              << max_rel_j << ") GPU=" << max_rel_gpu_val << " CPU=" << max_rel_cpu_val
              << std::endl;
    std::cout << "Average relative error: " << avg_rel_diff << " (over " << valid_rel_comparisons
              << " valid comparisons)" << std::endl;
    std::cout << "Relative Frobenius norm error: " << rel_frob_error << std::endl;

    // Pattern similarity measures
    std::cout << "Structural similarity (SSIM): " << ssim << std::endl;

    // Define objective pass criteria with consistent standards
    bool element_wise_pass = max_rel_diff <= tolerance;
    bool matrix_norm_pass  = rel_frob_error <= 0.05f; // 5% error in Frobenius norm

    // Use consistent SSIM threshold regardless of matrix size
    float ssim_threshold = 0.95f;

    bool pattern_pass = ssim > ssim_threshold; // Pattern similarity threshold

    std::cout << "Element-wise validation: " << (element_wise_pass ? "PASSED" : "FAILED")
              << std::endl;
    std::cout << "Matrix norm validation: " << (matrix_norm_pass ? "PASSED" : "FAILED")
              << std::endl;
    std::cout << "Pattern validation: " << (pattern_pass ? "PASSED" : "FAILED")
              << " (threshold: " << ssim_threshold << ")" << std::endl;

    // Overall pass requires all criteria to be met
    bool passed = element_wise_pass && matrix_norm_pass && pattern_pass;
    std::cout << "Overall validation: " << (passed ? "PASSED" : "FAILED") << std::endl;

    return passed;
}

#endif // HIP_HGEMM_HPP
