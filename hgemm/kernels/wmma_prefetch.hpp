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

#ifndef HIP_WMMA_PREFETCH_HPP
#define HIP_WMMA_PREFETCH_HPP

#include <common/matrix.hpp>
#include <hgemm/kernels/common.hpp>

template<>
struct wmma_config<kernel_type::wmma_prefetch>
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

    static constexpr int vector_width = 16;
    using vector_type                 = half16;
};

using config_p = wmma_config<kernel_type::wmma_prefetch>;

/**
 * @brief Half-precision GEMM using WMMA with all the principles from previous kernels
 * and adds global prefetching to registers before writing to shared memory.
 *
 * This kernel combines WMMA operations with shared memory, double buffering,
 * warp-level tiling and vectorized global loads. It uses double buffering to overlap computation
 * with memory operations, maximizing hardware utilization and hiding memory latency.
 * Additionally, global memory is prefetched to registers first before writing to shared memory.
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_prefetch'
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
 * @note Shared memory tiles for A and B use a layout of column-major and row-major
 * @note Employs a 2×4 warp grid configuration within each thread block
 */
template<kernel_type K_TYPE>
__global__ auto __launch_bounds__(warpSize * config_p::total_warps)
    kernel_hgemm(half* C, const half* A, const half* B, int M, int N, int K) ->
    typename std::enable_if<(K_TYPE == kernel_type::wmma_prefetch), void>::type
{
    using vector_type       = typename config_p::vector_type;
    constexpr int vec_width = config_p::vector_width;

    // Single unified shared memory buffer
    __shared__ half lds_mem[2][config_p::lds_size];

    // Create pointers for A and B regions
    half* a_tiles = &lds_mem[0][0];
    half* b_tiles = &lds_mem[0][config_p::block_m * config_p::lds_stride];

    const int tid         = threadIdx.y * blockDim.x + threadIdx.x;
    const int num_threads = blockDim.x * blockDim.y;

    const int block_row = blockIdx.x * config_p::block_m;
    const int block_col = blockIdx.y * config_p::block_n;

    const half* A_base = A + block_row; // Column-major A
    const half* B_base = B + block_col; // Row-major B
    half*       C_base = C + block_row * N + block_col;

    const int warp_row    = threadIdx.x / warpSize;
    const int warp_col    = threadIdx.y;
    const int warp_m_base = warp_row * config_p::warp_tile_m * wmma_tile;
    const int warp_n_base = warp_col * config_p::warp_tile_n * wmma_tile;

    constexpr int half_warp    = warpSize / 2;
    const int     half_warp_id = (threadIdx.x % warpSize) / half_warp;
    const int     half_lane    = threadIdx.x % half_warp;

    // Calculate vectors per thread
    constexpr int total_vectors_a = (config_p::block_m * config_p::block_k) / vec_width;
    constexpr int total_vectors_b = (config_p::block_n * config_p::block_k) / vec_width;

    constexpr int block_threads            = warpSize * config_p::total_warps;
    constexpr int max_vectors_per_thread_a = (total_vectors_a + block_threads - 1) / block_threads;
    constexpr int max_vectors_per_thread_b = (total_vectors_b + block_threads - 1) / block_threads;

    vector_type a_reg_buf[max_vectors_per_thread_a];
    vector_type b_reg_buf[max_vectors_per_thread_b];

    half16 c_frags[config_p::warp_tile_m][config_p::warp_tile_n]        = {};
    half16 a_frag[config_p::block_k / wmma_tile][config_p::warp_tile_m] = {};
    half16 b_frag[config_p::block_k / wmma_tile][config_p::warp_tile_n] = {};

    // Initial load to registers
    {
        // Load A tile portion to registers (column-major)
        for(int i = tid; i < total_vectors_a; i += num_threads)
        {
            const int col       = i / (config_p::block_m / vec_width);
            const int row       = (i % (config_p::block_m / vec_width)) * vec_width;
            const int local_idx = i / num_threads;

            if(block_row + row < M)
            {
                const half* src_ptr = A_base + col * M + row;
                if(K - col >= vec_width)
                {
                    a_reg_buf[local_idx] = *reinterpret_cast<const vector_type*>(src_ptr);
                }
                else
                {
                    half      tmp[vec_width];
                    const int remaining_k = K - col;
                    for(int j = 0; j < remaining_k; ++j)
                    {
                        tmp[j] = src_ptr[j * M]; // Column-major stride
                    }
                    for(int j = remaining_k; j < vec_width; ++j)
                    {
                        tmp[j] = static_cast<half>(0.0f);
                    }
                    a_reg_buf[local_idx] = *reinterpret_cast<const vector_type*>(tmp);
                }
            }
        }

        // Load B tile portion to registers (row-major)
        for(int i = tid; i < total_vectors_b; i += num_threads)
        {
            const int row       = i / (config_p::block_n / vec_width);
            const int col       = (i % (config_p::block_n / vec_width)) * vec_width;
            const int local_idx = i / num_threads;

            if(block_col + col < N)
            {
                const half* src_ptr = B_base + row * N + col;
                if(N - (block_col + col) >= vec_width)
                {
                    b_reg_buf[local_idx] = *reinterpret_cast<const vector_type*>(src_ptr);
                }
                else
                {
                    half      tmp[vec_width];
                    const int remaining_n = N - (block_col + col);
                    for(int j = 0; j < remaining_n; ++j)
                    {
                        tmp[j] = src_ptr[j];
                    }
                    for(int j = remaining_n; j < vec_width; ++j)
                    {
                        tmp[j] = static_cast<half>(0.0f);
                    }
                    b_reg_buf[local_idx] = *reinterpret_cast<const vector_type*>(tmp);
                }
            }
        }

        // Store thread's registers to shared memory
        {
            // Store A registers to shared memory (maintain column-major)
            for(int i = tid; i < total_vectors_a; i += num_threads)
            {
                const int col       = i / (config_p::block_m / vec_width);
                const int row       = (i % (config_p::block_m / vec_width)) * vec_width;
                const int local_idx = i / num_threads;

                vector_type* dest_ptr
                    = reinterpret_cast<vector_type*>(a_tiles + col * config_p::lds_stride + row);
                *dest_ptr = a_reg_buf[local_idx];
            }

            // Store B registers to shared memory (maintain row-major)
            for(int i = tid; i < total_vectors_b; i += num_threads)
            {
                const int row       = i / (config_p::block_n / vec_width);
                const int col       = (i % (config_p::block_n / vec_width)) * vec_width;
                const int local_idx = i / num_threads;

                vector_type* dest_ptr
                    = reinterpret_cast<vector_type*>(b_tiles + row * config_p::lds_stride + col);
                *dest_ptr = b_reg_buf[local_idx];
            }
        }
    }
    __syncthreads();

    int         current_tile = 0;
    const half* A_tile_ptr   = A_base;
    const half* B_tile_ptr   = B_base;

    for(int k_tile = 0; k_tile < K; k_tile += config_p::block_k)
    {
        const half* const a_base_ptr = &lds_mem[current_tile][0];
        const half* const b_base_ptr
            = &lds_mem[current_tile][config_p::block_m * config_p::lds_stride];

        // Load fragments
        for(int k = 0; k < config_p::block_k; k += wmma_tile)
        {
            int frag = k / wmma_tile;

            // For A (column-major loading)
            for(int wm = 0; wm < config_p::warp_tile_m; ++wm)
            {
                const half* src = a_base_ptr + k * config_p::lds_stride
                                  + (warp_m_base + wm * wmma_tile + half_lane);
                half* dest = reinterpret_cast<half*>(&a_frag[frag][wm]);

#pragma unroll
                for(int i = 0; i < wmma_tile; ++i)
                {
                    *dest++ = *src;
                    src += config_p::lds_stride; // Move down column
                }
            }

            // For B (row-major loading)
            for(int wn = 0; wn < config_p::warp_tile_n; ++wn)
            {
                const half* src = b_base_ptr + k * config_p::lds_stride
                                  + (warp_n_base + wn * wmma_tile + half_lane);
                half* dest = reinterpret_cast<half*>(&b_frag[frag][wn]);

#pragma unroll
                for(int i = 0; i < wmma_tile; ++i)
                {
                    *dest++ = *src;
                    src += config_p::lds_stride; // Move by N-sized stride
                }
            }
        }

        // Global prefetch for next tile
        if(k_tile + config_p::block_k < K)
        {
            const half* A_next
                = A_tile_ptr + (k_tile + config_p::block_k) * M; // Column-major stride
            const half* B_next = B_tile_ptr + (k_tile + config_p::block_k) * N; // Row-major stride

            // Load A tile portion to registers (column-major)
            for(int i = tid; i < total_vectors_a; i += num_threads)
            {
                const int col       = i / (config_p::block_m / vec_width);
                const int row       = (i % (config_p::block_m / vec_width)) * vec_width;
                const int local_idx = i / num_threads;

                if(block_row + row < M)
                {
                    const half* src_ptr = A_next + row;
                    if(K - (k_tile + config_p::block_k + col) >= vec_width)
                    {
                        a_reg_buf[local_idx] = *reinterpret_cast<const vector_type*>(src_ptr);
                    }
                    else
                    {
                        half      tmp[vec_width];
                        const int remaining_k = K - (k_tile + config_p::block_k + col);
                        for(int j = 0; j < remaining_k; ++j)
                        {
                            tmp[j] = src_ptr[j * M];
                        }
                        for(int j = remaining_k; j < vec_width; ++j)
                        {
                            tmp[j] = static_cast<half>(0.0f);
                        }
                        a_reg_buf[local_idx] = *reinterpret_cast<const vector_type*>(tmp);
                    }
                }
            }

            // Load B tile portion to registers (row-major)
            for(int i = tid; i < total_vectors_b; i += num_threads)
            {
                const int row       = i / (config_p::block_n / vec_width);
                const int col       = (i % (config_p::block_n / vec_width)) * vec_width;
                const int local_idx = i / num_threads;

                if(block_col + col < N)
                {
                    const half* src_ptr = B_next + row * N + col;
                    if(N - (block_col + col) >= vec_width)
                    {
                        b_reg_buf[local_idx] = *reinterpret_cast<const vector_type*>(src_ptr);
                    }
                    else
                    {
                        half      tmp[vec_width];
                        const int remaining_n = N - (block_col + col);
                        for(int j = 0; j < remaining_n; ++j)
                        {
                            tmp[j] = src_ptr[j];
                        }
                        for(int j = remaining_n; j < vec_width; ++j)
                        {
                            tmp[j] = static_cast<half>(0.0f);
                        }
                        b_reg_buf[local_idx] = *reinterpret_cast<const vector_type*>(tmp);
                    }
                }
            }
        }

        // Compute matrix multiplication
        for(int k = 0; k < config_p::block_k; k += wmma_tile)
        {
            int frag = k / wmma_tile;
            for(int wm = 0; wm < config_p::warp_tile_m; ++wm)
            {
                for(int wn = 0; wn < config_p::warp_tile_n; ++wn)
                {
                    c_frags[wm][wn] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag[frag][wm],
                                                                                 b_frag[frag][wn],
                                                                                 c_frags[wm][wn],
                                                                                 false);
                }
            }
        }

        // Store prefetched data to shared memory
        {
            // Store A registers to shared memory (maintain column-major)
            for(int i = tid; i < total_vectors_a; i += num_threads)
            {
                const int col       = i / (config_p::block_m / vec_width);
                const int row       = (i % (config_p::block_m / vec_width)) * vec_width;
                const int local_idx = i / num_threads;

                vector_type* dest_ptr = reinterpret_cast<vector_type*>(
                    &lds_mem[1 - current_tile][0] + col * config_p::lds_stride + row);
                *dest_ptr = a_reg_buf[local_idx];
            }

            // Store B registers to shared memory (maintain row-major)
            for(int i = tid; i < total_vectors_b; i += num_threads)
            {
                const int row       = i / (config_p::block_n / vec_width);
                const int col       = (i % (config_p::block_n / vec_width)) * vec_width;
                const int local_idx = i / num_threads;

                vector_type* dest_ptr = reinterpret_cast<vector_type*>(
                    &lds_mem[1 - current_tile][config_p::block_m * config_p::lds_stride]
                    + row * config_p::lds_stride + col);
                *dest_ptr = b_reg_buf[local_idx];
            }
        }

        A_tile_ptr += config_p::block_k * M; // Column-major stride for A
        B_tile_ptr += config_p::block_k * N; // Row-major stride for B
        current_tile = 1 - current_tile;
        __syncthreads();
    }

    // Store results
    half* C_warp = C_base + warp_m_base * N + warp_n_base;
    for(int wm = 0; wm < config_p::warp_tile_m; wm++)
    {
        half* C_row = C_warp + wm * wmma_tile * N;
        for(int wn = 0; wn < config_p::warp_tile_n; wn++)
        {
            const int n_offset = wn * wmma_tile + half_lane;
#pragma unroll
            for(int i = 0; i < wmma_tile / 2; ++i)
            {
                const int row = i * 2 + half_warp_id;
                if(row < config_p::block_m && n_offset < config_p::block_n
                   && block_row + warp_m_base + row < M && block_col + n_offset < N)
                {
                    *(C_row + row * N + n_offset) = c_frags[wm][wn][i * 2];
                }
            }
        }
    }
}

/**
 * Function Definition for calling WMMA Prefetch GEMM kernel
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_prefetch'
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 * @param stream  HIP stream to execute kernel
 */
template<>
__host__ void hgemm_gpu<kernel_type::wmma_prefetch>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    constexpr int warp_size = 32;
    dim3          block_dim(warp_size * config_p::warps_m, config_p::warps_n);
    dim3          grid_dim(ceil_div(M, config_p::block_m), ceil_div(N, config_p::block_n));

    kernel_hgemm<kernel_type::wmma_prefetch><<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}

#endif // HIP_WMMA_PREFETCH_HPP
