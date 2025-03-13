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

#ifndef HIP_KERNEL_HPP
#define HIP_KERNEL_HPP

#include <hip/hip_fp16.h>
#include <type_traits>

// Enum to choose between shared memory and WMMA-based kernel implementation
enum class kernel_type
{
    shared,
    wmma_naive,
    wmma_shared,
    wmma_shared_warp,
    wmma_shared_warp_buf,
    wmma_shared_warp_vec,
    wmma_shared_warp_buf_vec,
    wmma_prefetch,
    wmma_opt_1,
    wmma_opt_2,
    rocblas
};

// Tile size used for wmma kernel
constexpr int wmma_tile = 16;

constexpr int warp_size = 32;

typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef _Float16 half16 __attribute__((ext_vector_type(16)));

typedef float float8 __attribute__((ext_vector_type(8)));

template<kernel_type KT>
struct wmma_config;

/**
 * Kernel Definition for half-precision GEMM.
 *
 * @tparam K_TYPE The type of kernel
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 */
template<kernel_type K_TYPE>
__global__ void kernel_hgemm(half* C, const half* A, const half* B, int M, int N, int K);

/**
 * @brief Calculate ceiling division of two integers
 * @param a Dividend
 * @param b Divisor
 * @return Ceiling of a/b
 */
int ceil_div(int a, int b)
{
    return (a + b - 1) / b;
}

/**
 * Function Definition for calling GEMM kernel
 *
 * @tparam K_TYPE The type of kernel
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 * @param stream  HIP stream to execute kernel
 */
template<kernel_type K_TYPE>
__host__ void
    hgemm_gpu(half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream);

#endif // HIP_KERNEL_HPP
