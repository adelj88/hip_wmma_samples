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
    wmma_opt_3,
    wmma_opt_4,
    rocblas
};

// Tile size used for wmma kernel
constexpr int wmma_tile = 16;

constexpr int warp_size = 32;

typedef _Float16 half4 __attribute__((ext_vector_type(4)));
typedef _Float16 half8 __attribute__((ext_vector_type(8)));
typedef _Float16 half16 __attribute__((ext_vector_type(16)));

typedef float float8 __attribute__((ext_vector_type(8)));
typedef float float16 __attribute__((ext_vector_type(16)));

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
 * @brief Helper function for swizzled tile mapping
 *
 * Computes block indices using a swizzled mapping to improve L2 cache locality.
 * This is applied at the grid level to change the order in which tiles are processed.
 *
 * @param[in]  tile_id    Linear block ID
 * @param[in]  grid_m     Number of blocks in M dimension
 * @param[in]  grid_n     Number of blocks in N dimension
 * @param[out] block_row  Computed block row (M dimension)
 * @param[out] block_col  Computed block column (N dimension)
 */
template<int GROUP_SIZE, int BLOCK_M, int BLOCK_N>
__device__ __forceinline__ void
    swizzle_tile_mapping(int tile_id, int grid_m, int grid_n, int* block_row, int* block_col)
{
    // Group tiles along the M dimension for better cache locality
    int width      = GROUP_SIZE * grid_n;
    int group_id   = tile_id / width;
    int group_size = min(GROUP_SIZE, grid_m - group_id * GROUP_SIZE);

    // Compute swizzled indices
    int pid_m = group_id * GROUP_SIZE + (tile_id % group_size);
    int pid_n = (tile_id % width) / group_size;

    // Convert to actual block coordinates
    *block_row = pid_m * BLOCK_M;
    *block_col = pid_n * BLOCK_N;
}

/**
 * @brief Optimized Hilbert curve d2xy mapping using bit manipulation
 *
 * Converts a distance along the Hilbert curve to (x,y) coordinates using
 * GPU-optimized bit manipulation techniques.
 *
 * @param[in]  n      Size of the grid (must be a power of 2)
 * @param[in]  index  Distance along the Hilbert curve
 * @param[out] x      Resulting x coordinate
 * @param[out] y      Resulting y coordinate
 */
__device__ __forceinline__ void
    hilbert_d2xy_optimized(uint32_t n, uint32_t index, uint32_t* x, uint32_t* y)
{
    *x = 0;
    *y = 0;

#pragma unroll 16 // Unroll for better instruction-level parallelism
    for(uint32_t i = 0; i < 16; ++i)
    { // Assuming max 2^16 x 2^16 grid
        if((n >> i) == 0)
        {
            break; // Early termination
        }

        // Extract 2 bits from index
        uint32_t bits = (index >> (i * 2)) & 3;

        // Use lookup table approach for the rotation logic (better for GPU)
        switch(bits)
        {
            case 0:
                { // Lower left quadrant (reflect and swap)
                    uint32_t temp = *x;
                    *x            = *y;
                    *y            = temp;
                    break;
                }
            case 1:
                { // Lower right quadrant
                    *y |= (1U << i);
                    break;
                }
            case 2:
                { // Upper right quadrant
                    *x |= (1U << i);
                    *y |= (1U << i);
                    break;
                }
            case 3:
                { // Upper left quadrant (reflect and swap)
                    uint32_t temp = (1U << i) - 1 - *y;
                    *y            = (1U << i) - 1 - *x;
                    *x            = temp;
                    *x |= (1U << i);
                    break;
                }
        }
    }
}

/**
 * @brief Helper function for finding the largest power of 2 not exceeding n
 *
 * Uses efficient bit manipulation instead of looping.
 *
 * @param[in] n  Input value
 * @return    Largest power of 2 <= n
 */
__device__ __forceinline__ uint32_t largest_power_of_2(uint32_t n)
{
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    return (n + 1) >> 1;
}

/**
 * @brief Helper function for Hilbert curve tile mapping with core + remainder approach
 *
 * Computes block indices using a hybrid approach with two parts:
 * 1. A power-of-two core grid that uses optimal Hilbert curve mapping for spatial locality
 * 2. Remainder regions that use simpler mapping schemes
 *
 * This approach preserves exact block count while providing good locality for most tiles.
 * The Hilbert curve calculation uses optimized bit manipulation techniques for GPU efficiency.
 *
 * @param[in]  tile_id    Linear block ID
 * @param[in]  grid_m     Number of blocks in M dimension (scaled to BLOCK_M)
 * @param[in]  grid_n     Number of blocks in N dimension (scaled to BLOCK_N)
 * @param[out] block_row  Computed block row (M dimension)
 * @param[out] block_col  Computed block column (N dimension)
 */
template<int BLOCK_M, int BLOCK_N>
__device__ __forceinline__ void
    hilbert_tile_mapping(int tile_id, int grid_m, int grid_n, int* block_row, int* block_col)
{
    // Check bounds
    int total_tiles = grid_m * grid_n;
    if(tile_id >= total_tiles)
    {
        *block_row = 0;
        *block_col = 0;
        return;
    }

    // Special fast path for perfect power-of-two square grids
    if(grid_m == grid_n && (grid_m & (grid_m - 1)) == 0)
    {
        // Direct optimized Hilbert calculation - no remainder needed
        uint32_t x, y;
        hilbert_d2xy_optimized(grid_m, tile_id, &x, &y);

        // Convert to actual block coordinates with bit shift for multiplication
        *block_row = y << (__ffs(BLOCK_M) - 1);
        *block_col = x << (__ffs(BLOCK_N) - 1);
        return;
    }

    // Find the largest power-of-two dimensions that fit within our grid
    uint32_t core_m = largest_power_of_2(grid_m);
    uint32_t core_n = largest_power_of_2(grid_n);

    // Calculate size of the core grid (power-of-two in both dimensions)
    uint32_t core_size  = min(core_m, core_n);
    uint32_t core_tiles = core_size * core_size;

    // Check if we're in the core grid or in the remainder
    if(tile_id < core_tiles)
    {
        // We're in the power-of-two core grid - use optimized Hilbert curve mapping
        uint32_t x, y;
        hilbert_d2xy_optimized(core_size, tile_id, &x, &y);

        // Convert to actual block coordinates with bit shift for multiplication
        *block_row = y << (__ffs(BLOCK_M) - 1);
        *block_col = x << (__ffs(BLOCK_N) - 1);
    }
    else
    {
        // We're in the remainder regions - use simpler mapping
        int remainder_id = tile_id - core_tiles;

        // Define the three remainder regions
        int right_region_width   = grid_n - core_size;
        int bottom_region_height = grid_m - core_size;
        int right_region_tiles   = core_size * right_region_width;
        int bottom_region_tiles  = bottom_region_height * core_size;

        int row, col;

        // Map to the appropriate region
        if(remainder_id < right_region_tiles)
        {
            // Right region - core_size rows, right_region_width columns
            row = remainder_id / right_region_width;
            col = core_size + (remainder_id % right_region_width);
        }
        else if(remainder_id < right_region_tiles + bottom_region_tiles)
        {
            // Bottom region - bottom_region_height rows, core_size columns
            int local_id = remainder_id - right_region_tiles;
            row          = core_size + (local_id / core_size);
            col          = local_id % core_size;
        }
        else
        {
            // Corner region - bottom_region_height rows, right_region_width columns
            int local_id = remainder_id - right_region_tiles - bottom_region_tiles;
            row          = core_size + (local_id / right_region_width);
            col          = core_size + (local_id % right_region_width);
        }

        // Convert to actual block coordinates
        *block_row = row * BLOCK_M;
        *block_col = col * BLOCK_N;
    }
}

/**
 * @brief Calculate ceiling division of two integers
 * @param a Dividend
 * @param b Divisor
 * @return Ceiling of a/b
 */
int inline ceil_div(int a, int b)
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
