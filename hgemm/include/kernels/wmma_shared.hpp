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
#include <kernels/common.hpp>

template<>
struct wmma_config<kernel_type::wmma_shared>
{
    static constexpr int block_m = 128; // or any new value
    static constexpr int block_n = 64;
    static constexpr int block_k = 64;

    // Use separate strides as needed (from previous fix)
    static constexpr int lds_stride_A
        = block_m; // A: column-major, each column has block_m elements
    static constexpr int lds_stride_B = block_n; // B: row-major, each row has block_n elements
    static constexpr int lds_size     = (block_m * block_k) + (block_k * block_n);

    // Compute warps dynamically:
    static constexpr int warps_m     = block_m / wmma_tile;
    static constexpr int warps_n     = block_n / wmma_tile;
    static constexpr int total_warps = warps_m * warps_n;
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
 * @note Uses 128x64×64 shared memory tiles with 16×16 WMMA operations
 * @note Employs a 8×4 warp grid configuration for better occupancy
 */
template<>
__global__ void kernel_hgemm<kernel_type::wmma_shared>(
    half* C, const half* A, const half* B, int M, int N, int K)
{
    __shared__ half lds_mem[config_s::lds_size];
    half*           a_tile = lds_mem;
    half*           b_tile = lds_mem + (config_s::block_m * config_s::block_k);

    const int warp_size = 32;
    // Compute a unique warp id from the 2D thread index:
    const int warp_id = (threadIdx.y * (blockDim.x / warp_size)) + (threadIdx.x / warp_size);
    // Dynamic warp tiling:
    const int warps_m       = config_s::block_m / wmma_tile;
    const int warps_n       = config_s::block_n / wmma_tile;
    const int warp_m_idx    = warp_id % warps_m;
    const int warp_n_idx    = warp_id / warps_m;
    const int warp_m_offset = warp_m_idx * wmma_tile;
    const int warp_n_offset = warp_n_idx * wmma_tile;

    // Compute lane within warp:
    const int     lane         = threadIdx.x % warp_size;
    constexpr int half_warp    = warp_size / 2;
    const int     half_warp_id = lane / half_warp;
    const int     half_lane    = lane % half_warp;

    // Compute overall thread id (if needed)
    const int tid         = threadIdx.y * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * blockDim.y;

    // Block base indices
    const int   block_row = blockIdx.x * config_s::block_m;
    const int   block_col = blockIdx.y * config_s::block_n;
    const half* A_base    = A + block_row; // A is column-major
    const half* B_base    = B + block_col; // B is row-major

    if(warp_n_offset < config_s::block_n)
    {
        half16 c_frag = {}; // Initialize the accumulator fragment

        for(int k_tile = 0; k_tile < K; k_tile += config_s::block_k)
        {
            const half* A_curr = A_base + k_tile * M;
            const half* B_curr = B_base + k_tile * N;

            // Load A tile into shared memory
            for(int i = tid; i < (config_s::block_m * config_s::block_k); i += num_threads)
            {
                const int col = i / config_s::block_m;
                const int row = i % config_s::block_m;
                if(block_row + row < M && k_tile + col < K)
                    a_tile[col * config_s::lds_stride_A + row] = A_curr[col * M + row];
                else
                    a_tile[col * config_s::lds_stride_A + row] = static_cast<half>(0.0f);
            }

            // Load B tile into shared memory
            for(int i = tid; i < (config_s::block_k * config_s::block_n); i += num_threads)
            {
                const int row = i / config_s::block_n;
                const int col = i % config_s::block_n;
                if(k_tile + row < K && block_col + col < N)
                    b_tile[row * config_s::lds_stride_B + col] = B_curr[row * N + col];
                else
                    b_tile[row * config_s::lds_stride_B + col] = static_cast<half>(0.0f);
            }

            __syncthreads();

            // Process the current tile in chunks of size wmma_tile
            for(int k = 0; k < config_s::block_k; k += wmma_tile)
            {
                half16 a_frag = {};
                half16 b_frag = {};

                // Load fragment from A tile (column-major)
                if(warp_m_offset + half_lane < config_s::block_m)
                {
                    const half* src
                        = a_tile + k * config_s::lds_stride_A + (warp_m_offset + half_lane);
                    half* dest = reinterpret_cast<half*>(&a_frag);
                    for(int i = 0; i < wmma_tile; ++i)
                    {
                        *dest++ = *src;
                        src += config_s::lds_stride_A;
                    }
                }

                // Load fragment from B tile (row-major)
                if(warp_n_offset + half_lane < config_s::block_n)
                {
                    const half* src
                        = b_tile + k * config_s::lds_stride_B + warp_n_offset + half_lane;
                    half* dest = reinterpret_cast<half*>(&b_frag);
                    for(int i = 0; i < wmma_tile; ++i)
                    {
                        *dest++ = *src;
                        src += config_s::lds_stride_B;
                    }
                }

                // Perform the WMMA operation
                c_frag = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag, b_frag, c_frag, false);
            }
            __syncthreads();
        }

        // Store the computed fragment to global memory
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
    dim3 block_dim(warp_size * config_s::warps_m, config_s::warps_n);
    dim3 grid_dim(ceil_div(M, config_s::block_m), ceil_div(N, config_s::block_n));

    kernel_hgemm<kernel_type::wmma_shared><<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}

#endif // HIP_WMMA_SHARED_HPP
