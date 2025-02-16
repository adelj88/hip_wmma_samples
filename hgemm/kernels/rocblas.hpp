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

#ifndef HIP_ROCBLAS_HPP
#define HIP_ROCBLAS_HPP

#include <hgemm/kernels/common.hpp>
#include <rocblas/rocblas.h>

// Global rocBLAS handle
static rocblas_handle handle = nullptr;

/**
 * @brief Initialize rocBLAS library and create handle
 * @return true if initialization successful, false otherwise
 */
bool init_rocblas()
{
    if(handle != nullptr)
    {
        return true; // Already initialized
    }

    rocblas_status status = rocblas_create_handle(&handle);
    return (status == rocblas_status_success);
}

/**
 * @brief Clean up rocBLAS resources
 */
void cleanup_rocblas()
{
    if(handle != nullptr)
    {
        rocblas_destroy_handle(handle);
        handle = nullptr;
    }
}

/**
 * Function Definition for calling rocBLAS
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::rocblas'
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 * @param stream  HIP stream to execute kernel
 */
template<>
__host__ void hgemm_gpu<kernel_type::rocblas>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    if(handle == nullptr)
    {
        throw std::runtime_error("rocBLAS not initialized. Call init_rocblas() first.");
    }

    // Set stream
    rocblas_status status = rocblas_set_stream(handle, stream);
    if(status != rocblas_status_success)
    {
        throw std::runtime_error("Failed to set rocBLAS stream");
    }

    const _Float16 tmp_alpha = 1.0f;
    const _Float16 tmp_beta = 0.0f;
    const rocblas_half alpha = *reinterpret_cast<const rocblas_half*>(&tmp_alpha);
    const rocblas_half beta = *reinterpret_cast<const rocblas_half*>(&tmp_beta);

    //transpose_matrix(A, A, M, K, stream);

    const rocblas_half* rocblas_B = reinterpret_cast<const rocblas_half*>(B);
    const rocblas_half* rocblas_A = reinterpret_cast<const rocblas_half*>(A);
    rocblas_half* rocblas_C = reinterpret_cast<rocblas_half*>(C);

    // Perform matrix multiplication (result in column-major)
    status = rocblas_hgemm(handle,
                           rocblas_operation_none, // op(A)
                           rocblas_operation_none, // op(B)
                           M, // M
                           N, // N
                           K, // K
                           &alpha,
                           rocblas_A, // A (row-major input)
                           K, // lda
                           rocblas_B, // B (col-major input)
                           K, // ldb
                           &beta,
                           rocblas_C, // C (col-major output)
                           M); // ldc

    if(status != rocblas_status_success)
    {
        throw std::runtime_error("rocBLAS HGEMM failed");
    }
}

#endif // HIP_ROCBLAS_HPP
