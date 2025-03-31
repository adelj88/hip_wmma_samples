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

#ifndef HIP_WMMA_OPT_2_HPP
#define HIP_WMMA_OPT_2_HPP

#include <common/matrix.hpp>
#include <kernels/common.hpp>

template<>
struct wmma_config<kernel_type::wmma_opt_2>
{
    static constexpr int warps_m     = 4;
    static constexpr int warps_n     = 4;
    static constexpr int total_warps = warps_m * warps_n;

    static constexpr int warp_tile_m = 4;
    static constexpr int warp_tile_n = 4;

    static constexpr int block_m = warps_m * warp_tile_m * wmma_tile; // 4*4*16 = 256
    static constexpr int block_n = warps_n * warp_tile_n * wmma_tile; // 4*4*16 = 256
    static constexpr int block_k = 32;

    // For A (stored column-major), each column has block_m elements.
    static constexpr int lds_stride_A = block_m;
    // For B (stored row-major), each row has block_n elements.
    static constexpr int lds_stride_B = block_n;
    // Total shared memory size: region for A plus region for B.
    static constexpr int lds_size = (block_m * block_k) + (block_k * block_n);

    // Vector loading configuration (256-bits = 2 128-bit loads)
    using vector_type                 = float8;
    static constexpr int vector_width = (sizeof(float8) / sizeof(half));

};

using config_o2 = wmma_config<kernel_type::wmma_opt_2>;

/**
   * @brief Half-precision GEMM using WMMA with shared memory, shared double buffering,
   * warp tiling, cooperative loading, Hilbert-curve mapping, and vectorized global loads using half16 vectors
   *
   * This kernel combines WMMA operations with shared memory, double buffering,
   * warp-level tiling and vectorized global loads. It uses double buffering at the shared
   * level to overlap computation with memory operations, maximizing hardware utilization and hiding
   * memory latency. Additionally, cooperative loading is used to load both A and B to shared memory
   * in parallel. The kernel also incorporates Hilbert-curve mapping for improved L2 cache locality.
   * This kernel also re-orders fragment loading to improve efficiency and uses
   * __launch_bounds__ to limit register pressure.
   *
   * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_opt_2'
   * @param[out] C  Output matrix of size M × N
   * @param[in]  A  Input matrix A of size M × K (stored in column-major format)
   * @param[in]  B  Input matrix B of size K × N (stored in row-major format)
   * @param[in]  M  Number of rows in matrices A and C
   * @param[in]  N  Number of columns in matrices B and C
   * @param[in]  K  Number of columns in matrix A/rows in matrix B
   *
   * @note Implements double-buffering at global->shared
   * @note Each warp processes a 4×4 grid of 16×16 WMMA tiles
   * @note Uses shared memory tiles of size (block_m × block_k) for A and (block_k × block_n) for B
   * @note Employs a 4×4 warp grid configuration within each thread block
   * @note Uses Hilbert-curve mapping for improved cache locality
   */
template<>
__global__ void
    __launch_bounds__(warp_size* config_o2::total_warps) kernel_hgemm<kernel_type::wmma_opt_2>(
        half* C, const half* A, const half* B, int M, int N, int K)
{
    // Calculate grid dimensions
    const int grid_m  = (M + config_o2::block_m - 1) / config_o2::block_m;
    const int grid_n  = (N + config_o2::block_n - 1) / config_o2::block_n;
    const int tile_id = blockIdx.x;

    // Get block coordinates using hilbert mapping
    int block_row, block_col;
    hilbert_tile_mapping<config_o2::block_m, config_o2::block_n>(tile_id,
                                                                 grid_m,
                                                                 grid_n,
                                                                 &block_row,
                                                                 &block_col);

    // Allocate a unified shared memory buffer.
    __shared__ half lds_mem[2 * config_o2::lds_size];

    // Partition the shared memory with manual offset calculations:
    // A tiles occupy the first region in each buffer
    half* a_tiles_0 = lds_mem;
    half* a_tiles_1 = lds_mem + config_o2::lds_size;
    // B tiles start after A's region in each buffer
    half* b_tiles_0 = lds_mem + (config_o2::block_m * config_o2::block_k);
    half* b_tiles_1 = lds_mem + config_o2::lds_size + (config_o2::block_m * config_o2::block_k);

    // Each block is launched with a one-dimensional thread block.
    const int tid         = threadIdx.x;
    const int num_threads = blockDim.x;
    const int half_block  = num_threads / 2;
    const int cid         = tid % half_block;

    const half* A_base = A + block_row; // A is in column-major order
    const half* B_base = B + block_col; // B is in row-major order
    half*       C_base = C + block_row * N + block_col;

    // Compute warp ID from the 1D thread index.
    const int warp_id  = tid / warp_size;
    const int warp_row = warp_id / config_o2::warps_n;
    const int warp_col = warp_id % config_o2::warps_n;

    constexpr int half_warp    = warp_size / 2;
    const int     lane_id      = (tid % warp_size);
    const int     half_warp_id = lane_id / half_warp;
    const int     half_lane    = tid % half_warp;

    // Determine the base offsets for this warp's set of WMMA tiles.
    const int warp_m_base = warp_row * config_o2::warp_tile_m * wmma_tile;
    const int warp_n_base = warp_col * config_o2::warp_tile_n * wmma_tile;

    // Declare fragment storage.
    half16 c_frags[config_o2::warp_tile_m][config_o2::warp_tile_n] = {};
    half16 a_frag[config_o2::warp_tile_m]                          = {};
    half16 b_frag[config_o2::warp_tile_n]                          = {};

    // Base pointers for the current A and B tiles.
    const half* A_tile_ptr = A_base;
    const half* B_tile_ptr = B_base;

    if(tid < half_block)
    {
        // Load A tile (of size block_m × block_k) into shared memory.
        for(int i = cid * config_o2::vector_width; i < (config_o2::block_m * config_o2::block_k);
            i += half_block * config_o2::vector_width)
        {
            const int col = i / config_o2::block_m;
            const int row = i % config_o2::block_m;

            int gload  = col * M + row;
            int swrite = col * config_o2::lds_stride_A + row;

            if((block_row + row + config_o2::vector_width - 1) < M && col < K)
            {
                // Load full vector
                *reinterpret_cast<config_o2::vector_type*>(a_tiles_0 + swrite)
                    = *reinterpret_cast<const config_o2::vector_type*>(A_tile_ptr + gload);
            }
            else
            {
                // Handle the boundary case element by element
                for(int v = 0; v < config_o2::vector_width; v++)
                {
                    if(block_row + row + v < M && col < K)
                    {
                        a_tiles_0[swrite + v] = A_tile_ptr[gload + v];
                    }
                    else
                    {
                        a_tiles_0[swrite + v] = static_cast<half>(0.0f);
                    }
                }
            }
        }
    }
    else
    {
        // Load B tile (row-major) using vectorized loads
        for(int i = cid * config_o2::vector_width; i < (config_o2::block_k * config_o2::block_n);
            i += half_block * config_o2::vector_width)
        {
            const int row = i / config_o2::block_n;
            const int col = i % config_o2::block_n;

            int gload  = row * N + col;
            int swrite = row * config_o2::lds_stride_B + col;

            if(row < K && (block_col + col + config_o2::vector_width - 1) < N)
            {
                // Load full vector
                *reinterpret_cast<config_o2::vector_type*>(b_tiles_0 + swrite)
                    = *reinterpret_cast<const config_o2::vector_type*>(B_tile_ptr + gload);
            }
            else
            {
                // Handle the boundary case element by element
                for(int v = 0; v < config_o2::vector_width; v++)
                {
                    if(row < K && block_col + col + v < N)
                    {
                        b_tiles_0[swrite + v] = B_tile_ptr[gload + v];
                    }
                    else
                    {
                        b_tiles_0[swrite + v] = static_cast<half>(0.0f);
                    }
                }
            }
        }
    }
    __syncthreads();

    half* current_a = a_tiles_0;
    half* current_b = b_tiles_0;
    half* next_a    = a_tiles_1;
    half* next_b    = b_tiles_1;

    // Main loop over k-dimension
    for(int k_tile = 0; k_tile < K; k_tile += config_o2::block_k)
    {
        if(tid >= half_block && k_tile + config_o2::block_k < K)
        {
            const half* next_A = A_tile_ptr + M * config_o2::block_k;
            // Load A tile (of size block_m × block_k) into shared memory.
            for(int i = cid * config_o2::vector_width;
                i < (config_o2::block_m * config_o2::block_k);
                i += half_block * config_o2::vector_width)
            {
                const int col = i / config_o2::block_m;
                const int row = i % config_o2::block_m;

                int gload  = col * M + row;
                int swrite = col * config_o2::lds_stride_A + row;

                if((block_row + row + config_o2::vector_width - 1) < M
                   && (k_tile + config_o2::block_k + col) < K)
                {
                    *reinterpret_cast<config_o2::vector_type*>(next_a + swrite)
                        = *reinterpret_cast<const config_o2::vector_type*>(next_A + gload);
                }
                else
                {
                    for(int v = 0; v < config_o2::vector_width; v++)
                    {
                        if(block_row + row + v < M && k_tile + config_o2::block_k + col < K)
                        {
                            next_a[swrite + v] = next_A[gload + v];
                        }
                        else
                        {
                            next_a[swrite + v] = static_cast<half>(0.0f);
                        }
                    }
                }
            }
        }

        // Process the loaded block_k in wmma_tile chunks
        for(int k_offset = 0; k_offset < config_o2::block_k; k_offset += wmma_tile)
        {
            const half* curr_a
                = current_a + k_offset * config_o2::lds_stride_A + (warp_m_base + half_lane);
            const half* curr_b
                = current_b + k_offset * config_o2::lds_stride_B + (warp_n_base + half_lane);

#pragma unroll
            for(int i = 0; i < wmma_tile; ++i)
            {
                const half* srca = curr_a + (i * config_o2::lds_stride_A);
                for(int wm = 0; wm < config_o2::warp_tile_m; ++wm)
                {
                    a_frag[wm][i] = *srca;
                    srca += wmma_tile;
                }

                const half* srcb = curr_b + (i * config_o2::lds_stride_B);
                for(int wn = 0; wn < config_o2::warp_tile_n; ++wn)
                {
                    b_frag[wn][i] = *srcb;
                    srcb += wmma_tile;
                }
            }

            // Compute: each warp performs WMMA on its fragments.
            for(int wm = 0; wm < config_o2::warp_tile_m; ++wm)
            {
                for(int wn = 0; wn < config_o2::warp_tile_n; ++wn)
                {
                    c_frags[wm][wn] = __builtin_amdgcn_wmma_f16_16x16x16_f16_w32(a_frag[wm],
                                                                                 b_frag[wn],
                                                                                 c_frags[wm][wn],
                                                                                 false);
                }
            }
        }

        if(tid < half_block && k_tile + config_o2::block_k < K)
        {
            const half* next_B = B_tile_ptr + N * config_o2::block_k;
            // Load B tile (row-major) using vectorized loads
            for(int i = cid * config_o2::vector_width;
                i < (config_o2::block_k * config_o2::block_n);
                i += half_block * config_o2::vector_width)
            {
                const int row = i / config_o2::block_n;
                const int col = i % config_o2::block_n;

                int gload  = row * N + col;
                int swrite = row * config_o2::lds_stride_B + col;

                if((k_tile + config_o2::block_k + row) < K
                   && (block_col + col + config_o2::vector_width - 1) < N)
                {
                    *reinterpret_cast<config_o2::vector_type*>(next_b + swrite)
                        = *reinterpret_cast<const config_o2::vector_type*>(next_B + gload);
                }
                else
                {

                    for(int v = 0; v < config_o2::vector_width; v++)
                    {
                        if(k_tile + config_o2::block_k + row < K && block_col + col + v < N)
                        {
                            next_b[swrite + v] = next_B[gload + v];
                        }
                        else
                        {
                            next_b[swrite + v] = static_cast<half>(0.0f);
                        }
                    }
                }
            }
        }

        // Advance the global pointers for A and B tiles.
        A_tile_ptr += M * config_o2::block_k;
        B_tile_ptr += N * config_o2::block_k;
        half* temp_a = current_a;
        half* temp_b = current_b;
        current_a    = next_a;
        current_b    = next_b;
        next_a       = temp_a;
        next_b       = temp_b;
        __syncthreads();
    }

    // Calculate the total size of the output tile
    constexpr int total_tile_elements = config_o2::block_m * config_o2::block_n;

    // Maximum shared memory available is the entire shared memory buffer
    constexpr int max_shared_elements = 2 * config_o2::lds_size;

    // Determine if we need to process in chunks or can handle the entire tile at once
    constexpr bool needs_chunking = total_tile_elements > max_shared_elements;

    // If chunking is needed, calculate how many rows we can process at once
    // Otherwise, process the entire tile
    constexpr int rows_per_chunk
        = needs_chunking ? max_shared_elements / config_o2::block_n : config_o2::block_m;

    // Reuse shared memory for storing C values
    half* c_tile = lds_mem;

    // Process the matrix in chunks
    for(int chunk_idx = 0; chunk_idx < config_o2::block_m; chunk_idx += rows_per_chunk)
    {
        // Calculate row range for this chunk
        const int row_start    = chunk_idx;
        const int row_end      = min(row_start + rows_per_chunk, config_o2::block_m);
        const int chunk_height = row_end - row_start;

        // Step 1: Store WMMA fragments to shared memory
        for(int wm = 0; wm < config_o2::warp_tile_m; ++wm)
        {
            const int warp_m_global = warp_m_base + wm * wmma_tile;

            // Skip warps not in the current chunk
            if(warp_m_global < row_start || warp_m_global >= row_end)
            {
                continue;
            }

            // Calculate local row offset within current chunk
            const int warp_m_local = warp_m_global - row_start;

            for(int wn = 0; wn < config_o2::warp_tile_n; ++wn)
            {
                const int warp_n_base_local = warp_n_base + wn * wmma_tile;

#pragma unroll
                for(int i = 0; i < wmma_tile / 2; ++i)
                {
                    const int row_local = warp_m_local + i * 2 + half_warp_id;
                    const int col_local = warp_n_base_local + half_lane;

                    // Store fragments directly to shared memory
                    c_tile[row_local * config_o2::block_n + col_local] = c_frags[wm][wn][i * 2];
                }
            }
        }
        __syncthreads();

        // Step 2: Perform vectorized writes from shared memory to global memory
        // Each thread processes multiple vectors
        for(int i = tid * config_o2::vector_width; i < (chunk_height * config_o2::block_n);
            i += num_threads * config_o2::vector_width)
        {

            const int row_local = i / config_o2::block_n;
            const int col_local = i % config_o2::block_n;

            // Calculate global position
            const int row_global = block_row + row_start + row_local;
            const int col_global = block_col + col_local;

            // Check if this vector is entirely within bounds
            if(row_global < M && col_global + config_o2::vector_width - 1 < N)
            {
                // Full vector write
                *reinterpret_cast<config_o2::vector_type*>(C_base + (row_start + row_local) * N
                                                           + col_local)
                    = *reinterpret_cast<const config_o2::vector_type*>(
                        c_tile + row_local * config_o2::block_n + col_local);
            }
            else if(row_global < M)
            {
                // Handle boundary case element by element
                for(int v = 0; v < config_o2::vector_width; v++)
                {
                    if(col_global + v < N)
                    {
                        C_base[(row_start + row_local) * N + col_local + v]
                            = c_tile[row_local * config_o2::block_n + col_local + v];
                    }
                }
            }
        }
        __syncthreads();
    }
}

/**
     * Function Definition for calling WMMA Optimized V2 GEMM kernel
     *
     * @tparam K_TYPE The type of kernel, should be 'kernel_type::wmma_opt_2'
     * @param C       Output matrix
     * @param A       Input matrix A (stored in column-major format)
     * @param B       Input matrix B (stored in row-major format)
     * @param M       Number of rows in matrices A and C
     * @param N       Number of columns in matrices B and C
     * @param K       Number of columns in matrix A/rows in matrix B
     * @param stream  HIP stream to execute kernel
     */
template<>
__host__ void hgemm_gpu<kernel_type::wmma_opt_2>(
    half* C, half* A, half* B, size_t M, size_t N, size_t K, hipStream_t& stream)
{
    // Calculate grid dimensions
    int grid_m       = (M + config_o2::block_m - 1) / config_o2::block_m;
    int grid_n       = (N + config_o2::block_n - 1) / config_o2::block_n;
    int total_blocks = grid_m * grid_n;

    dim3 grid_dim(total_blocks);
    dim3 block_dim(warp_size * config_o2::total_warps);

    kernel_hgemm<kernel_type::wmma_opt_2><<<grid_dim, block_dim, 0, stream>>>(C, A, B, M, N, K);
}

#endif // HIP_WMMA_OPT_2_HPP
