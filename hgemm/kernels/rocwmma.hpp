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

#ifndef HIP_ROCWMMA_HPP
#define HIP_ROCWMMA_HPP

#include <common/matrix.hpp>
#include <hgemm/kernels/common.hpp>
#include <rocwmma/rocwmma.hpp>
#include <rocwmma/rocwmma_coop.hpp>
#include <rocwmma/rocwmma_transforms.hpp>

template<>
struct wmma_config<kernel_type::rocwmma>
{
    static constexpr int warps_m     = 2;
    static constexpr int warps_n     = 4;
    static constexpr int total_warps = warps_m * warps_n;

    static constexpr int warp_tile_m = 4;
    static constexpr int warp_tile_n = 2;

    static constexpr int block_m  = warps_m * warp_tile_m * wmma_tile;
    static constexpr int block_n  = warps_n * warp_tile_n * wmma_tile;
    static constexpr int block_k  = 16;
    static constexpr int bank_pad = 4;
};

using config_rocwmma = wmma_config<kernel_type::rocwmma>;

/**
 * @brief Half-precision GEMM using rocWMMA.
 *
 * This kernel uses the rocWMMA library and is taken from the samples/perf_hgemm.cpp
 * file provided in their repository. It's simply meant to be a reference for
 * benchmarking purposes.
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::rocwmma'
 * @param[out] C  Output matrix of size M × N
 * @param[in]  A  Input matrix A of size M × K
 * @param[in]  B  Input matrix B of size K × N (stored in column-major format)
 * @param[in]  M  Number of rows in matrices A and C
 * @param[in]  N  Number of columns in matrices B and C
 * @param[in]  K  Number of columns in matrix A/rows in matrix B
 *
 * @note Implements double-buffering at global->shared
 * @note Each warp processes a 4×2 grid of 16×16 WMMA tiles
 * @note Uses shared memory tiles of size (block_m × block_k) for A and (block_k × block_n) for B
 * @note Employs a 2×4 warp grid configuration within each thread block
 */
template<kernel_type k_type>
__global__ auto __launch_bounds__(warpSize * config_rocwmma::total_warps)
    kernel_hgemm(half* c_out, const half* a_in, const half* b_in, int m, int n, int k) ->
    typename std::enable_if<(k_type == kernel_type::rocwmma), void>::type
{
    using namespace rocwmma;

    // Tile size constants
    constexpr auto warp_tile_size  = make_coord2d(config_rocwmma::warp_tile_m * wmma_tile,
                                                 config_rocwmma::warp_tile_n * wmma_tile);
    constexpr auto macro_tile_size = make_coord2d(config_rocwmma::block_m, config_rocwmma::block_n);

    // Local warp coordinate setup
    constexpr auto warp_dims = make_coord2d(config_rocwmma::warps_m, config_rocwmma::warps_n);
    auto           local_warp_coord  = make_coord2d(threadIdx.x / warpSize, threadIdx.y);
    auto           local_warp_offset = local_warp_coord * warp_tile_size;

    // Global matrix coordinates
    auto macro_tile_coord = make_coord2d(blockIdx.x, blockIdx.y) * macro_tile_size;
    auto warp_tile_coord  = macro_tile_coord + local_warp_offset;

    // Bounds check
    auto warp_tile_bound = warp_tile_coord + warp_tile_size;
    if(get<0>(warp_tile_bound) > m || get<1>(warp_tile_bound) > n)
    {
        return;
    }

    // Global read fragment types
    using gr_buff_a
        = fragment<matrix_a, config_rocwmma::block_m, wmma_tile, wmma_tile, half, col_major>;
    using gr_buff_b
        = fragment<matrix_b, wmma_tile, config_rocwmma::block_n, wmma_tile, half, row_major>;

    // MFMA fragment types
    using mfma_frag_a   = fragment<matrix_a, wmma_tile, wmma_tile, wmma_tile, half, col_major>;
    using mfma_frag_b   = fragment<matrix_b, wmma_tile, wmma_tile, wmma_tile, half, row_major>;
    using mfma_frag_acc = fragment<accumulator, wmma_tile, wmma_tile, wmma_tile, half, row_major>;

    // Local write fragment types
    using lw_buff_a = ApplyDataLayout_t<gr_buff_a, col_major>;
    using lw_buff_b = ApplyDataLayout_t<ApplyTranspose_t<gr_buff_b>, col_major>;

    // Local read fragment types
    using lr_frag_a = ApplyDataLayout_t<mfma_frag_a, col_major>;
    using lr_frag_b = ApplyDataLayout_t<ApplyTranspose_t<mfma_frag_b>, col_major>;

    // LDS setup for double buffering
    constexpr uint32_t lds_width  = config_rocwmma::block_k;
    constexpr uint32_t lds_height = config_rocwmma::block_m + config_rocwmma::block_n;
    constexpr uint32_t lds_size   = lds_height * lds_width;
    constexpr uint32_t lds_stride = lds_width; // For col_major layout

    __shared__ half lds_mem[2][lds_size]; // Double buffer

    // Warp scheduling for cooperative loads
    const uint32_t warp_idx
        = get<0>(local_warp_coord) * get<1>(warp_dims) + get<1>(local_warp_coord);

    // Initial global load
    gr_buff_a global_a;
    gr_buff_b global_b;

    // Calculate initial global offsets
    auto global_offset_a = get<0>(macro_tile_coord) * k;
    auto global_offset_b = get<1>(macro_tile_coord);

    // Load initial data cooperatively
    load_matrix_coop_sync<config_rocwmma::total_warps>(global_a,
                                                       a_in + global_offset_a,
                                                       k,
                                                       warp_idx);
    load_matrix_coop_sync<config_rocwmma::total_warps>(global_b,
                                                       b_in + global_offset_b,
                                                       n,
                                                       warp_idx);

    // Store to first buffer using local write fragments
    {
        lw_buff_a lw_a = applyDataLayout<col_major>(global_a);
        store_matrix_coop_sync<config_rocwmma::total_warps>(lds_mem[0], lw_a, lds_stride, warp_idx);

        lw_buff_b lw_b = applyDataLayout<col_major>(applyTranspose(global_b));
        store_matrix_coop_sync<config_rocwmma::total_warps>(
            lds_mem[0] + config_rocwmma::block_m * lds_stride,
            lw_b,
            lds_stride,
            warp_idx);
    }

    synchronize_workgroup();

    // Initialize accumulator fragments
    mfma_frag_acc accum[config_rocwmma::warp_tile_m][config_rocwmma::warp_tile_n];

    for(int i = 0; i < config_rocwmma::warp_tile_m; i++)
    {
        for(int j = 0; j < config_rocwmma::warp_tile_n; j++)
        {
            fill_fragment(accum[i][j], static_cast<half>(0.0f));
        }
    }

    // Main loop
    int current_buf = 0;
    for(int k_step = wmma_tile; k_step < k; k_step += wmma_tile)
    {
        // Load next global data while computing current
        load_matrix_coop_sync<config_rocwmma::total_warps>(global_a,
                                                           a_in + global_offset_a + k_step,
                                                           k,
                                                           warp_idx);
        load_matrix_coop_sync<config_rocwmma::total_warps>(global_b,
                                                           b_in + global_offset_b + k_step * n,
                                                           n,
                                                           warp_idx);

        // Load computation fragments from current buffer
        mfma_frag_a frags_a[config_rocwmma::warp_tile_m];
        mfma_frag_b frags_b[config_rocwmma::warp_tile_n];

        // Load from LDS with proper layout transformations
        {
            using frag_shape = GetIOShape_t<lr_frag_a>;
            using mapper_1d  = GetDataLayout_t<lr_frag_a>;
            auto block_step
                = mapper_1d::fromMatrixCoord(make_coord2d(frag_shape::BlockHeight, 0u), lds_stride);

            for(int i = 0; i < config_rocwmma::warp_tile_m; i++)
            {
                lr_frag_a tmp;
                auto      lds_addr_a = lds_mem[current_buf] + i * wmma_tile * lds_stride;
                load_matrix_sync(tmp, lds_addr_a, lds_stride);
                frags_a[i] = applyDataLayout<col_major>(tmp);
            }
        }

        {
            using frag_shape = GetIOShape_t<lr_frag_b>;
            using mapper_1d  = GetDataLayout_t<lr_frag_b>;
            auto block_step
                = mapper_1d::fromMatrixCoord(make_coord2d(frag_shape::BlockHeight, 0u), lds_stride);

            for(int i = 0; i < config_rocwmma::warp_tile_n; i++)
            {
                lr_frag_b tmp;
                auto      lds_addr_b = lds_mem[current_buf] + config_rocwmma::block_m * lds_stride
                                  + i * wmma_tile * lds_stride;
                load_matrix_sync(tmp, lds_addr_b, lds_stride);
                frags_b[i] = applyDataLayout<row_major>(applyTranspose(tmp));
            }
        }

        // Compute matrix multiply-accumulate with explicit loops
        for(int i = 0; i < config_rocwmma::warp_tile_m; i++)
        {
            for(int j = 0; j < config_rocwmma::warp_tile_n; j++)
            {
                mma_sync(accum[i][j], frags_a[i], frags_b[j], accum[i][j]);
            }
        }

        // Store next data to other buffer with layout transformations
        {
            lw_buff_a lw_a = applyDataLayout<col_major>(global_a);
            store_matrix_coop_sync<config_rocwmma::total_warps>(lds_mem[1 - current_buf],
                                                                lw_a,
                                                                lds_stride,
                                                                warp_idx);

            lw_buff_b lw_b = applyDataLayout<col_major>(applyTranspose(global_b));
            store_matrix_coop_sync<config_rocwmma::total_warps>(
                lds_mem[1 - current_buf] + config_rocwmma::block_m * lds_stride,
                lw_b,
                lds_stride,
                warp_idx);
        }

        synchronize_workgroup();
        current_buf = 1 - current_buf;

        // Update global offsets for next iteration
        global_offset_a += wmma_tile;
        global_offset_b += wmma_tile * n;
    }

    // Store final results with proper striding
    using frag_shape = GetIOShape_t<mfma_frag_acc>;
    using mapper_1d  = GetDataLayout_t<mfma_frag_acc>;

    auto block_step_x = mapper_1d::fromMatrixCoord(make_coord2d(frag_shape::BlockHeight, 0u), n);
    auto block_step_y = mapper_1d::fromMatrixCoord(make_coord2d(0u, frag_shape::BlockWidth), n);

    half* c_warp = c_out + get<0>(warp_tile_coord) * n + get<1>(warp_tile_coord);

    for(int i = 0; i < config_rocwmma::warp_tile_m; i++)
    {
        auto offset_y = 0u;
        for(int j = 0; j < config_rocwmma::warp_tile_n; j++)
        {
            store_matrix_sync(c_warp + offset_y, accum[i][j], n);
            offset_y += block_step_y;
        }
        c_warp += block_step_x;
    }
}

/**
 * Function Definition for calling rocWMMA GEMM kernel
 *
 * @tparam K_TYPE The type of kernel, should be 'kernel_type::rocwmma'
 * @param C       Output matrix
 * @param A       Input matrix A
 * @param B       Input matrix B
 * @param M       Number of rows in matrices A and C
 * @param N       Number of columns in matrices B and C
 * @param K       Number of columns in matrix A/rows in matrix B
 * @param stream  HIP stream to execute kernel
 */
template<>
__host__ void hgemm_gpu<kernel_type::rocwmma>(
    half* c_out, half* a_in, half* b_in, size_t m, size_t n, size_t k, hipStream_t& stream)
{
    dim3 block_dim(32 * config_rocwmma::warps_m, config_rocwmma::warps_n);
    dim3 grid_dim(ceil_div(m, config_rocwmma::block_m), ceil_div(n, config_rocwmma::block_n));

    kernel_hgemm<kernel_type::rocwmma>
        <<<grid_dim, block_dim, 0, stream>>>(c_out, a_in, b_in, m, n, k);
}

#endif // HIP_ROCWMMA_HPP
