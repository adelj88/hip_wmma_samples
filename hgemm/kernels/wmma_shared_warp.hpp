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

#ifndef HIP_WMMA_SHARED_WARP_HPP
#define HIP_WMMA_SHARED_WARP_HPP

#include <common/matrix.hpp>
#include <hgemm/kernels/common.hpp>

template<>
struct wmma_config<kernel_type::wmma_shared_warp>
{
    static constexpr int warps_m     = 2;
    static constexpr int warps_n     = 4;
    static constexpr int total_warps = warps_m * warps_n;

    // Warp tile dimensions (number of WMMA tiles per warp)
    static constexpr int warp_tile_m = 4;
    static constexpr int warp_tile_n = 2;

    static constexpr int block_m = warps_m * warp_tile_m * wmma_tile;
    static constexpr int block_n = warps_n * warp_tile_n * wmma_tile;
    static constexpr int block_k = 64;

    // Shared memory layout
    static constexpr int lds_width  = block_k;
    static constexpr int lds_height = block_m + block_n;
    static constexpr int lds_size   = lds_height * lds_width;
    static constexpr int lds_stride = lds_width;
};

using config_w = wmma_config<kernel_type::wmma_shared_warp>;

/**
 * @brief Half-precision GEMM using WMMA with shared memory and warp-level tiling
 *
 * This kernel extends the shared memory WMMA implementation with warp-level tiling,
 * where each warp processes multiple WMMA tiles. This approach increases arithmetic
 * intensity and improves register reuse while maintaining efficient shared memory
 * access patterns.
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_shared_warp'
 * @param[out] C  Output matrix of size M × N
 * @param[in]  A  Input matrix A of size M × K
 * @param[in]  B  Input matrix B of size K × N (stored in column-major format)
 * @param[in]  M  Number of rows in matrices A and C
 * @param[in]  N  Number of columns in matrices B and C
 * @param[in]  K  Number of columns in matrix A/rows in matrix B
 *
 * @note Each warp processes a 4×2 grid of 16×16 WMMA tiles
 * @note Uses shared memory tiles of size (block_m × block_k) for A and (block_k × block_n) for B
 * @note Employs a 2×4 warp grid configuration within each thread block
 */
template<kernel_type K_TYPE>
__global__ auto __launch_bounds__(warpSize * config_w::total_warps)
    kernel_hgemm(half* C, const half* A, const half* B, int M, int N, int K) ->
    typename std::enable_if<(K_TYPE == kernel_type::wmma_shared_warp), void>::type
{
    // Single unified shared memory buffer
    __shared__ half lds_mem[config_wd::lds_size];

    // Create pointers for A and B regions
    half* a_tiles = &lds_mem[0];
    half* b_tiles = &lds_mem[config_wd::block_m * config_wd::lds_stride];

    const int tid         = threadIdx.y * blockDim.y + threadIdx.x;
    const int num_threads = blockDim.x * blockDim.y;

    const int block_row = blockIdx.x * config_w::block_m;
    const int block_col = blockIdx.y * config_w::block_n;

    const half* A_base = A + block_row; // Column-major A
    const half* B_base = B + block_col; // Row-major B

    const int warp_row = threadIdx.x / warpSize;
    const int warp_col = threadIdx.y;

    constexpr int half_warp    = warpSize / 2;
    const int     half_warp_id = (threadIdx.x % warpSize) / half_warp;
    const int     half_lane    = threadIdx.x % half_warp;

    const int warp_m_base = warp_row * config_w::warp_tile_m * wmma_tile;
    const int warp_n_base = warp_col * config_w::warp_tile_n * wmma_tile;

    half16 c_frags[config_w::warp_tile_m][config_w::warp_tile_n] = {};

    // Main K-dimension loop
    for(int k_tile = 0; k_tile < K; k_tile += config_w::block_k)
    {
        const half* A_next = A_base + k_tile * M;
        const half* B_next = B_base + k_tile * N;

        // Load A tile (column-major)
        for(int i = tid; i < (config_w::block_m * config_w::block_k); i += num_threads)
        {
            const int col = i / config_w::block_m;
            const int row = i % config_w::block_m;

            if(block_row + row < M && col < K)
            {
                *(a_tiles + col * config_w::lds_stride + row) = *(A_next + col * M + row);
            }
            else
            {
                *(a_tiles + col * config_w::lds_stride + row) = static_cast<half>(0.0f);
            }
        }

        // Load B tile (row-major)
        for(int i = tid; i < (config_w::block_k * config_w::block_n); i += num_threads)
        {
            const int row = i / config_w::block_n;
            const int col = i % config_w::block_n;

            if(row < K && block_col + col < N)
            {
                *(b_tiles + row * config_w::lds_stride + col) = *(B_next + row * N + col);
            }
            else
            {
                *(b_tiles + row * config_w::lds_stride + col) = static_cast<half>(0.0f);
            }
        }

        __syncthreads();

        // Process K-dimension in wmma_tile chunks
        for(int k = 0; k < config_w::block_k; k += wmma_tile)
        {
            // Loop over warp tiles
            for(int wm = 0; wm < config_w::warp_tile_m; ++wm)
            {
                for(int wn = 0; wn < config_w::warp_tile_n; ++wn)
                {
                    half16 a_frag = {};
                    half16 b_frag = {};

                    const int m_offset = warp_m_base + wm * wmma_tile;
                    const int n_offset = warp_n_base + wn * wmma_tile;

                    // For A (column-major loading)
                    for(int wm = 0; wm < config_w::warp_tile_m; ++wm)
                    {
                        const half* src = a_tiles + k * config_w::lds_stride
                                          + (warp_m_base + wm * wmma_tile + half_lane);
                        half* dest = reinterpret_cast<half*>(&a_frag);

#pragma unroll
                        for(int i = 0; i < wmma_tile; ++i)
                        {
                            *dest++ = *src;
                            src += config_w::lds_stride; // Move down column
                        }
                    }

                    // For B (row-major loading)
                    for(int wn = 0; wn < config_w::warp_tile_n; ++wn)
                    {
                        const half* src = b_tiles + k * config_w::lds_stride
                                          + (warp_n_base + wn * wmma_tile + half_lane);
                        half* dest = reinterpret_cast<half*>(&b_frag);

#pragma unroll
                        for(int i = 0; i < wmma_tile; ++i)
                        {
                            *dest++ = *src;
                            src += config_w::lds_stride;
                        }
                    }

                    // Perform WMMA operation
                    c_frags[wm][wn] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag,
                                                                                 b_frag,
                                                                                 c_frags[wm][wn],
                                                                                 false);

                }
            }
        }

        A_base += config_w::block_k * M; // Column-major stride for A
        B_base += config_w::block_k * N; // Row-major stride for B
        __syncthreads();
    }

    // Store results
    for(int wm = 0; wm < config_w::warp_tile_m; ++wm)
    {
        for(int wn = 0; wn < config_w::warp_tile_n; ++wn)
        {
            const int m_offset = warp_m_base + wm * wmma_tile;
            const int n_offset = warp_n_base + wn * wmma_tile;

#pragma unroll
            for(int i = 0; i < wmma_tile / 2; ++i)
            {
                const int r = i * 2 + half_warp_id;
                if(block_row + m_offset + r < M && block_col + n_offset + half_lane < N)
                {
                    C[(block_row + m_offset + r) * N + (block_col + n_offset + half_lane)]
                        = c_frags[wm][wn][i * 2];
                }
            }
        }
    }
}

/**
 * Function Definition for calling WMMA + Shared + Warp-Tiling GEMM kernel
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_shared_warp'
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 * @param stream  HIP stream to execute kernel
 */
template<>
__host__ void hgemm_gpu<kernel_type::wmma_shared_warp>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    constexpr int warp_size = 32;
    dim3          block_dim(warp_size * config_w::warps_m, config_w::warps_n);
    dim3          grid_dim(ceil_div(M, config_w::block_m), ceil_div(N, config_w::block_n));

    kernel_hgemm<kernel_type::wmma_shared_warp>
        <<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}

#endif // HIP_WMMA_SHARED_WARP_HPP
