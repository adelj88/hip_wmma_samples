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

#ifndef HIP_WMMA_OPT_3_HPP
#define HIP_WMMA_OPT_3_HPP

#include <common/matrix.hpp>
#include <kernels/common.hpp>

template<>
struct wmma_config<kernel_type::wmma_opt_3>
{
    static constexpr int warps_m     = 4;
    static constexpr int warps_n     = 4;
    static constexpr int total_warps = warps_m * warps_n;

    static constexpr int warp_tile_m = 4;
    static constexpr int warp_tile_n = 4;

    static constexpr int block_m = warps_m * warp_tile_m * wmma_tile; // 4*4*16 = 256
    static constexpr int block_n = warps_n * warp_tile_n * wmma_tile; // 4*4*16 = 256
    static constexpr int block_k = 16;

    // For A (stored column-major), each column has block_m elements.
    static constexpr int lds_stride_A = block_m;
    // For B (stored row-major), each row has block_n elements.
    static constexpr int lds_stride_B = block_n;
    // Total shared memory size: region for A plus region for B.
    static constexpr int lds_size = (block_m * block_k) + (block_k * block_n);

    // Vector loading configuration (512-bits = 4 128-bit loads)
    using vector_type                 = float16;
    static constexpr int vector_width = (sizeof(float16) / sizeof(half));
};

using config_o3 = wmma_config<kernel_type::wmma_opt_3>;

/**
    * @brief Half-precision GEMM using WMMA with a 3-pipeline using register prefetching
    *
    * This kernel combines the best aspects of wmma_opt_2 and wmma_prefetch to create a 3-
    * pipeline without increasing shared memory usage. It maintains the double-buffered shared memory
    * and fragment structures from wmma_opt_2 while adding register prefetching as a third .
    * The three s are:
    * 1. Global memory -> Register prefetch
    * 2. Register -> Shared memory
    * 3. Shared memory -> Fragment compute
    *
    * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_opt_3'
    * @param[out] C  Output matrix of size M × N
    * @param[in]  A  Input matrix A of size M × K (stored in column-major format)
    * @param[in]  B  Input matrix B of size K × N (stored in row-major format)
    * @param[in]  M  Number of rows in matrices A and C
    * @param[in]  N  Number of columns in matrices B and C
    * @param[in]  K  Number of columns in matrix A/rows in matrix B
    *
    * @note Implements double-buffering at global->shared and shared->fragment
    * @note Adds register prefetching as a third pipeline
    * @note Each warp processes a 4×4 grid of 16×16 WMMA tiles
    */
template<>
__global__ void
    __launch_bounds__(warp_size* config_o3::total_warps) kernel_hgemm<kernel_type::wmma_opt_3>(
        half* C, const half* A, const half* B, int M, int N, int K);

/**
    * Function Definition for calling WMMA Optimized V3 GEMM kernel
    *
    * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_opt_3'
    * @param C       Output matrix
    * @param A       Input matrix A (stored in column-major format)
    * @param B       Input matrix B (stored in row-major format)
    * @param M       Number of rows in matrices A and C
    * @param N       Number of columns in matrices B and C
    * @param K       Number of columns in matrix A/rows in matrix B
    * @param stream  HIP stream to execute kernel
    */
template<>
__host__ void hgemm_gpu<kernel_type::wmma_opt_3>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

#endif // HIP_WMMA_OPT_3_HPP
