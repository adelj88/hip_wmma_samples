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

#ifndef HIP_WMMA_RDNA3_HPP
#define HIP_WMMA_RDNA3_HPP

#include <common/matrix.hpp>
#include <hgemm/kernels/common.hpp>

/**
 * @brief Half-precision GEMM implementation using WMMA (Wave Matrix Multiply-Accumulate) instructions
 *
 * This kernel implements matrix multiplication C = A × B using AMD's WMMA instructions for
 * hardware-accelerated matrix operations. It processes 16×16 matrix tiles using wave-wide
 * operations.
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_naive'
 * @param[out] C  Output matrix of size M × N
 * @param[in]  A  Input matrix A of size M × K
 * @param[in]  B  Input matrix B of size K × N (stored in column-major format)
 * @param[in]  M  Number of rows in matrices A and C
 * @param[in]  N  Number of columns in matrices B and C
 * @param[in]  K  Number of columns in matrix A/rows in matrix B
 *
 * @note Uses WMMA intrinsics specific to AMD RDNA 3 architecture
 * @note Each warp processes a 16×16 tile of the output matrix
 * @note Matrix B should be in column-major format for optimal memory access
 */
template<matrix_input matrix, matrix_layout access>
__device__ inline auto
    load_matrix(half16& frag, const half* data, int row, int col, int M, int N) ->
    typename std::enable_if<(matrix == matrix_input::matrix_a
                             && access == matrix_layout::row_major),
                            void>::type
{
    constexpr int half_warp = warpSize / 2;
    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA 3
    int lane   = threadIdx.x % half_warp; // Lane index within the half-wave
    int offset = row + lane;

    const half* tmp = reinterpret_cast<const half*>(data + offset * N + col);
#pragma unroll
    for(int i = 0; i < wmma_tile; ++i)
    {
        if(offset < M && col + i < N)
        {
            frag[i] = tmp[i];
        }
    }
}

template<matrix_input matrix, matrix_layout access>
__device__ inline auto
    load_matrix(half16& frag, const half* data, int row, int col, int M, int N) ->
    typename std::enable_if<(matrix == matrix_input::matrix_a
                             && access == matrix_layout::col_major),
                            void>::type
{
    constexpr int half_warp = warpSize / 2;
    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA 3
    int lane   = threadIdx.x % half_warp; // Lane index within the half-wave
    int offset = row + lane;

    const half* tmp = reinterpret_cast<const half*>(data + col * M + offset);
#pragma unroll
    for(int i = 0; i < wmma_tile; ++i)
    {
        if(col + i < N && offset < M)
        {
            frag[i] = *tmp;
            tmp += M;
        }
    }
}

template<matrix_input matrix, matrix_layout access>
__device__ inline auto
    load_matrix(half16& frag, const half* data, int row, int col, int M, int N) ->
    typename std::enable_if<(matrix == matrix_input::matrix_b
                             && access == matrix_layout::row_major),
                            void>::type
{
    constexpr int half_warp = warpSize / 2;
    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA 3
    int lane   = threadIdx.x % half_warp; // Lane index within the half-wave
    int offset = col + lane;

    const half* tmp = reinterpret_cast<const half*>(data + row * N + offset);
#pragma unroll
    for(int i = 0; i < wmma_tile; ++i)
    {
        if(row + i < M && offset < N)
        {
            frag[i] = *tmp;
            tmp += N;
        }
    }
}

template<matrix_input matrix, matrix_layout access>
__device__ inline auto
    load_matrix(half16& frag, const half* data, int row, int col, int M, int N) ->
    typename std::enable_if<(matrix == matrix_input::matrix_b
                             && access == matrix_layout::col_major),
                            void>::type
{
    constexpr int half_warp = warpSize / 2;
    // lane is (0-31) mod 16 instead of 0-31 due to matrix replication in RDNA 3
    int lane   = threadIdx.x % half_warp; // Lane index within the half-wave
    int offset = col + lane;

    const half* tmp = reinterpret_cast<const half*>(data + offset * M + row);
#pragma unroll
    for(int i = 0; i < wmma_tile; ++i)
    {
        if(offset < N && (row + i) < M)
        {
            frag[i] = tmp[i];
        }
    }
}

/**
 * Device function to store a tile of matrix C in row-major order.
 *
 * @param data Pointer to output matrix data
 * @param frag Fragment containing the computed results
 * @param row Starting row index in the matrix
 * @param col Starting column index in the matrix
 * @param M Number of rows in the matrix
 * @param N Number of columns in the matrix
 */
__device__ inline void store_matrix(half* data, half16& frag, int row, int col, int M, int N)
{
    constexpr int half_warp    = warpSize / 2;
    int           lane         = threadIdx.x % half_warp; // Lane index within the half-wave
    int           half_warp_id = (threadIdx.x % warpSize) / half_warp; // Index for half-warp
    int           offset       = col + lane;

    half* tmp = reinterpret_cast<half*>(offset + row * N);

#pragma unroll
    for(int i = 0; i < wmma_tile / 2; ++i)
    {
        const int r                  = i * 2 + half_warp_id;
        data[(row + r) * N + offset] = frag[i * 2]; // Store results from unpacked c_frag output
    }
}

/**
 * Kernel for half-precision GEMM using WMMA intrinsics.
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_naive'
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 */
template<kernel_type K_TYPE>
__global__ auto __launch_bounds__(warpSize * 16)
    kernel_hgemm(half* C, const half* A, const half* B, int M, int N, int K) ->
    typename std::enable_if<(K_TYPE == kernel_type::wmma_naive), void>::type
{
    int ix = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize; // Row of tile in C/A
    int iy = blockIdx.y * blockDim.y + threadIdx.y; // Column of tile in C/B

    int c_row = ix * wmma_tile; // Starting row index for tile in A/C
    int c_col = iy * wmma_tile; // Starting column index for tile in B/C
    int steps = (K + wmma_tile - 1) / wmma_tile; // Number of K tiles to process

    half16 c_frag = {}; // Fragment to store results of WMMA operation

    for(int m = 0; m < steps; ++m)
    {
        int k = m * wmma_tile; // Current K block index

        half16 a_frag = {};
        half16 b_frag = {};

        load_matrix<matrix_input::matrix_a, matrix_layout::row_major>(a_frag, A, c_row, k, M, K);
        load_matrix<matrix_input::matrix_b, matrix_layout::col_major>(b_frag, B, k, c_col, K, N);

        // Compute matrix multiplication using WMMA intrinsic
        c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);
    }

    store_matrix(C, c_frag, c_row, c_col, M, N); // Store results in row-major order
}

/**
 * Function Definition for calling WMMA Naive GEMM kernel
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_naive'
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 * @param stream  HIP stream to execute kernel
 */
template<>
__host__ void hgemm_gpu<kernel_type::wmma_naive>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    constexpr int warp_size = 32; // bug on my system where warp size on host side is not 32
    dim3          block_dim(warp_size * 4, 4);
    dim3          grid_dim(ceil_div(M, wmma_tile * block_dim.x / warp_size),
                  ceil_div(N, wmma_tile * block_dim.y));
    kernel_hgemm<kernel_type::wmma_naive><<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}

#endif // HIP_WMMA_RDNA3_HPP
