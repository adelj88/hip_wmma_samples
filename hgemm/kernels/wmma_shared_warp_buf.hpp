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

#ifndef HIP_WMMA_SHARED_WARP_BUF_HPP
#define HIP_WMMA_SHARED_WARP_BUF_HPP

#include <common/matrix.hpp>
#include <hgemm/kernels/common.hpp>

template<>
struct wmma_config<kernel_type::wmma_shared_warp_buf>
{
    static constexpr int warps_m     = 2;
    static constexpr int warps_n     = 4;
    static constexpr int total_warps = warps_m * warps_n;

    static constexpr int warp_tile_m = 4;
    static constexpr int warp_tile_n = 2;

    static constexpr int block_m = warps_m * warp_tile_m * wmma_tile;
    static constexpr int block_n = warps_n * warp_tile_n * wmma_tile;
    static constexpr int block_k = 16;

    // Shared memory layout
    static constexpr int lds_width  = block_k;
    static constexpr int lds_height = block_m + block_n;
    static constexpr int lds_size   = lds_height * lds_width;
    static constexpr int lds_stride = lds_width;
};

using config_wd = wmma_config<kernel_type::wmma_shared_warp_buf>;

/**
 * @brief Half-precision GEMM using WMMA with shared memory, double buffering and warp tiling
 *
 * This kernel combines WMMA operations with shared memory, double buffering
 * and warp-level tiling. It uses double buffering to overlap computation with memory
 * operations, maximizing hardware utilization and hiding memory latency.
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_shared_warp_buf'
 * @param[out] C  Output matrix of size M × N
 * @param[in]  A  Input matrix A of size M × K (stored in column-major format)
 * @param[in]  B  Input matrix B of size K × N (stored in row-major format)
 * @param[in]  M  Number of rows in matrices A and C
 * @param[in]  N  Number of columns in matrices B and C
 * @param[in]  K  Number of columns in matrix A/rows in matrix B
 *
 * @note Implements double-buffering at global->shared
 * @note Each warp processes a 4×2 grid of 16×16 WMMA tiles
 * @note Uses shared memory tiles of size (block_m × block_k) for A and (block_k × block_n) for B
 * @note Employs a 2×4 warp grid configuration within each thread block
 */
template<kernel_type K_TYPE>
__global__ auto __launch_bounds__(warpSize * config_wd::total_warps)
    kernel_hgemm(half* C, const half* A, const half* B, int M, int N, int K) ->
    typename std::enable_if<(K_TYPE == kernel_type::wmma_shared_warp_buf), void>::type
{
    // Single unified shared memory buffer
    __shared__ half lds_mem[2][config_wd::lds_size];

    // Create pointers for A and B regions
    half* a_tiles = &lds_mem[0][0];
    half* b_tiles = &lds_mem[0][config_wd::block_m * config_wd::lds_stride];

    const int tid         = threadIdx.y + blockDim.y + threadIdx.x;
    const int num_threads = blockDim.x * blockDim.y;

    const int block_row = blockIdx.x * config_wd::block_m;
    const int block_col = blockIdx.y * config_wd::block_n;

    const half* A_base = A + block_row; // Column-major A
    const half* B_base = B + block_col; // Row-major B
    half*       C_base = C + block_row * N + block_col;

    const int warp_row = threadIdx.x / warpSize;
    const int warp_col = threadIdx.y;

    constexpr int half_warp    = warpSize / 2;
    const int     half_warp_id = (threadIdx.x % warpSize) / half_warp;
    const int     half_lane    = threadIdx.x % half_warp;

    const int warp_m_base = warp_row * config_wd::warp_tile_m * wmma_tile;
    const int warp_n_base = warp_col * config_wd::warp_tile_n * wmma_tile;

    half16 c_frags[config_wd::warp_tile_m][config_wd::warp_tile_n] = {};
    half16 a_frag[config_wd::warp_tile_m]                          = {};
    half16 b_frag[config_wd::warp_tile_n]                          = {};

    // Load first tiles into buffer 0
    {
        // Load A tile (column-major)
        for(int i = tid; i < (config_wd::block_m * config_wd::block_k); i += num_threads)
        {
            const int col = i / config_wd::block_m;
            const int row = i % config_wd::block_m;

            if(block_row + row < M && col < K)
            {
                *(a_tiles + col * config_wd::lds_stride + row) = *(A_base + col * M + row);
            }
            else
            {
                *(a_tiles + col * config_wd::lds_stride + row) = static_cast<half>(0.0f);
            }
        }

        // Load B tile (row-major)
        for(int i = tid; i < (config_wd::block_k * config_wd::block_n); i += num_threads)
        {
            const int row = i / config_wd::block_n;
            const int col = i % config_wd::block_n;

            if(row < K && block_col + col < N)
            {
                *(b_tiles + row * config_wd::lds_stride + col) = *(B_base + row * N + col);
            }
            else
            {
                *(b_tiles + row * config_wd::lds_stride + col) = static_cast<half>(0.0f);
            }
        }
    }

    __syncthreads();

    int         current_tile = 0;
    const half* A_tile_ptr   = A_base;
    const half* B_tile_ptr   = B_base;

    for(int k_tile = 0; k_tile < K; k_tile += config_wd::block_k)
    {
        if(k_tile + config_wd::block_k < K)
        {
            // Load next tiles into the opposite buffer
            half* next_a_tile = &lds_mem[1 - current_tile][0];
            half* next_b_tile
                = &lds_mem[1 - current_tile][config_wd::block_m * config_wd::lds_stride];
            const half* A_next = A_tile_ptr + (k_tile + config_wd::block_k) * M;
            const half* B_next = B_tile_ptr + (k_tile + config_wd::block_k) * N;

            // Load next A tile (column-major)
            for(int i = tid; i < (config_wd::block_m * config_wd::block_k); i += num_threads)
            {
                const int col = i / config_wd::block_m;
                const int row = i % config_wd::block_m;

                if(block_row + row < M && k_tile + config_wd::block_k + col < K)
                {
                    *(next_a_tile + col * config_wd::lds_stride + row) = *(A_next + row);
                }
                else
                {
                    *(next_a_tile + col * config_wd::lds_stride + row) = static_cast<half>(0.0f);
                }
            }

            // Load next B tile (row-major)
            for(int i = tid; i < (config_wd::block_k * config_wd::block_n); i += num_threads)
            {
                const int row = i / config_wd::block_n;
                const int col = i % config_wd::block_n;

                if(k_tile + config_wd::block_k + row < K && block_col + col < N)
                {
                    *(next_b_tile + row * config_wd::lds_stride + col) = *(B_next + row * N + col);
                }
                else
                {
                    *(next_b_tile + row * config_wd::lds_stride + col) = static_cast<half>(0.0f);
                }
            }
        }

        // Pre-calculate shared memory base pointers for current tile
        const half* const a_base_ptr = &lds_mem[current_tile][0];
        const half* const b_base_ptr
            = &lds_mem[current_tile][config_wd::block_m * config_wd::lds_stride];

        // Process K-dimension
        for(int k = 0; k < config_wd::block_k; k += wmma_tile)
        {
            // For A (column-major loading)
            for(int wm = 0; wm < config_wd::warp_tile_m; ++wm)
            {
                const half* src = a_base_ptr + k * config_wd::lds_stride
                                  + (warp_m_base + wm * wmma_tile + half_lane);
                half* dest = reinterpret_cast<half*>(&a_frag[wm]);

#pragma unroll
                for(int i = 0; i < wmma_tile; ++i)
                {
                    *dest++ = *src;
                    src += config_wd::lds_stride; // Move down column
                }
            }

            // For B (row-major loading)
            for(int wn = 0; wn < config_wd::warp_tile_n; ++wn)
            {
                const half* src = b_base_ptr + k * config_wd::lds_stride
                                  + (warp_n_base + wn * wmma_tile + half_lane);
                half* dest = reinterpret_cast<half*>(&b_frag[wn]);

#pragma unroll
                for(int i = 0; i < wmma_tile; ++i)
                {
                    *dest++ = *src;
                    src += config_wd::lds_stride;
                }
            }

            // Compute matrix multiplication
            for(int wm = 0; wm < config_wd::warp_tile_m; ++wm)
            {
                for(int wn = 0; wn < config_wd::warp_tile_n; ++wn)
                {
                    c_frags[wm][wn] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag[wm],
                                                                                 b_frag[wn],
                                                                                 c_frags[wm][wn],
                                                                                 false);
                }
            }
        }

        A_tile_ptr += config_wd::block_k * M; // Column-major stride for A
        B_tile_ptr += config_wd::block_k * N; // Row-major stride for B
        current_tile = 1 - current_tile;
        __syncthreads();
    }

    // Store results
    half* C_warp = C_base + warp_m_base * N + warp_n_base;
    for(int wm = 0; wm < config_wd::warp_tile_m; wm++)
    {
        half* C_row = C_warp + wm * wmma_tile * N;
        for(int wn = 0; wn < config_wd::warp_tile_n; wn++)
        {
            const int n_offset = wn * wmma_tile + half_lane;

            for(int i = 0; i < wmma_tile / 2; ++i)
            {
                const int row = i * 2 + half_warp_id;
                if(row < config_wd::block_m && n_offset < config_wd::block_n
                   && block_row + warp_m_base + row < M && block_col + n_offset < N)
                {
                    *(C_row + row * N + n_offset) = c_frags[wm][wn][i * 2];
                }
            }
        }
    }
}

/**
 * Function Definition for calling WMMA + Shared + Warp-Tiling + Double Buffering GEMM kernel
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_shared_warp_buf'
 * @param C       Output matrix
 * @param A       Input matrix A (stored in column-major format)
 * @param B       Input matrix B (stored in row-major format)
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 * @param stream  HIP stream to execute kernel
 */
template<>
__host__ void hgemm_gpu<kernel_type::wmma_shared_warp_buf>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    constexpr int warp_size = 32;
    dim3          block_dim(warp_size * config_wd::warps_m, config_wd::warps_n);
    dim3          grid_dim(ceil_div(M, config_wd::block_m), ceil_div(N, config_wd::block_n));

    kernel_hgemm<kernel_type::wmma_shared_warp_buf>
        <<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}

#endif // HIP_WMMA_SHARED_WARP_BUF_HPP
