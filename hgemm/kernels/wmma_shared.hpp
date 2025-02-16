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

#ifndef HIP_WMMA_SHARED_HPP
#define HIP_WMMA_SHARED_HPP

#include <common/matrix.hpp>
#include <hgemm/kernels/common.hpp>

template<>
struct wmma_config<kernel_type::wmma_shared>
{
    static constexpr int warps_m     = 4;
    static constexpr int warps_n     = 4;
    static constexpr int total_warps = warps_m * warps_n;

    static constexpr int block_m = 64;
    static constexpr int block_n = 64;
    static constexpr int block_k = 64;

    // Shared memory layout
    static constexpr int lds_width  = block_k;
    static constexpr int lds_height = block_m + block_n;
    static constexpr int lds_size   = lds_height * lds_width;
    static constexpr int lds_stride = lds_width;
};

using config_s = wmma_config<kernel_type::wmma_shared>;

/**
 * @brief Half-precision GEMM using WMMA with shared memory tiling
 *
 * This kernel combines WMMA instructions with shared memory tiling to optimize matrix
 * multiplication. It loads larger tiles into shared memory and then processes them using
 * WMMA operations, reducing global memory bandwidth requirements while maintaining
 * the efficiency of hardware matrix operations.
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_shared'
 * @param[out] C  Output matrix of size M × N
 * @param[in]  A  Input matrix A of size M × K
 * @param[in]  B  Input matrix B of size K × N (stored in column-major format)
 * @param[in]  M  Number of rows in matrices A and C
 * @param[in]  N  Number of columns in matrices B and C
 * @param[in]  K  Number of columns in matrix A/rows in matrix B
 *
 * @note Uses 64×64 shared memory tiles with 16×16 WMMA operations
 * @note Employs a 4×4 warp grid configuration for better occupancy
 * @note Each block processes a 64×64 tile of the output matrix
 */
template<kernel_type K_TYPE>
__global__ auto __launch_bounds__(warpSize* config_s::total_warps)
    kernel_hgemm(half* C, const half* A, const half* B, int M, int N, int K) ->
    typename std::enable_if<(K_TYPE == kernel_type::wmma_shared), void>::type
{
    // Single unified shared memory buffer
    __shared__ half lds_mem[config_s::lds_size];

    // Create pointers for A and B regions
    half* a_tile = &lds_mem[0];
    half* b_tile = &lds_mem[config_s::block_m * config_s::lds_stride];

    const int tid         = threadIdx.y * blockDim.y + threadIdx.x;
    const int num_threads = blockDim.x * blockDim.y;

    const int block_row = blockIdx.x * config_s::block_m;
    const int block_col = blockIdx.y * config_s::block_n;

    // Calculate base pointers for this block
    const half* A_base = A + block_row; // Column-major A
    const half* B_base = B + block_col; // Row-major B

    const int warp_row      = threadIdx.x / warpSize;
    const int warp_col      = threadIdx.y;
    const int warp_m_offset = warp_row * wmma_tile;
    const int warp_n_offset = warp_col * wmma_tile;

    constexpr int half_warp    = warpSize / 2;
    const int     half_warp_id = (threadIdx.x % warpSize) / half_warp;
    const int     half_lane    = threadIdx.x % half_warp;

    if(warp_n_offset < config_s::block_n)
    {
        half16 c_frag = {};

        // Main K-dimension loop
        for(int k_tile = 0; k_tile < K; k_tile += config_s::block_k)
        {
            const half* A_next = A_base + k_tile * M;
            const half* B_next = B_base + k_tile * N;

            // Load A tile (column-major)
            for(int i = tid; i < (config_s::block_m * config_s::block_k); i += num_threads)
            {
                const int col = i / config_s::block_m;
                const int row = i % config_s::block_m;

                if(block_row + row < M && k_tile + col < K)
                {
                    *(a_tile + col * config_s::lds_stride + row) = *(A_next + col * M + row);
                }
                else
                {
                    *(a_tile + col * config_s::lds_stride + row) = static_cast<half>(0.0f);
                }
            }

            // Load B tile (row-major)
            for(int i = tid; i < (config_s::block_k * config_s::block_n); i += num_threads)
            {
                const int row = i / config_s::block_n;
                const int col = i % config_s::block_n;

                if(k_tile + row < K && block_col + col < N)
                {
                    *(b_tile + row * config_s::lds_stride + col) = *(B_next + row * N + col);
                }
                else
                {
                    *(b_tile + row * config_s::lds_stride + col) = static_cast<half>(0.0f);
                }
            }

            __syncthreads();

            // Process K-dimension in wmma_tile chunks
            for(int k = 0; k < config_s::block_k; k += wmma_tile)
            {
                half16 a_frag = {};
                half16 b_frag = {};

                // Load A fragment (column-major)
                if(warp_m_offset + half_lane < config_s::block_m)
                {
                    const half* src
                        = a_tile + k * config_s::lds_stride + (warp_m_offset + half_lane);
                    half* dest = reinterpret_cast<half*>(&a_frag);

                    for(int i = 0; i < wmma_tile; ++i)
                    {
                        *dest++ = *src;
                        src += config_s::lds_stride;
                    }
                }

                // Load B fragment (row-major)
                if(warp_n_offset + half_lane < config_s::block_n)
                {
                    const half* src = b_tile + k * config_s::lds_stride + warp_n_offset + half_lane;
                    half*       dest = reinterpret_cast<half*>(&b_frag);

                    for(int i = 0; i < wmma_tile; ++i)
                    {
                        *dest++ = *src;
                        src += config_s::lds_stride;
                    }
                }

                // Perform WMMA operation
                c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);
            }
            __syncthreads();

            A_base += config_s::block_k * M; // Column-major stride for A
            B_base += config_s::block_k * N; // Row-major stride for B
        }

        // Store results
        for(int i = 0; i < wmma_tile / 2; ++i)
        {
            const int row = i * 2 + half_warp_id;
            if(block_row + warp_m_offset + row < M && block_col + warp_n_offset + half_lane < N)
            {
                C[(block_row + warp_m_offset + row) * N + (block_col + warp_n_offset + half_lane)]
                    = c_frag[i * 2];
            }
        }
    }
}

/**
 * Function Definition for calling WMMA + Shared GEMM kernel
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_shared'
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 * @param stream  HIP stream to execute kernel
 */
template<>
__host__ void hgemm_gpu<kernel_type::wmma_shared>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    constexpr int warp_size = 32;
    dim3          block_dim(warp_size * config_s::warps_m, config_s::warps_n);
    dim3          grid_dim(ceil_div(M, config_s::block_m), ceil_div(N, config_s::block_n));

    kernel_hgemm<kernel_type::wmma_shared><<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}

#endif // HIP_WMMA_SHARED_HPP
