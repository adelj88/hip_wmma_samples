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

#include <common/hip_utils.hpp>
#include <common/matrix.hpp>
#include <hgemm/hgemm.hpp>

template<kernel_type K_TYPE>
struct layout_selector
{
    static constexpr matrix_layout a_layout = matrix_layout::col_major;
    static constexpr matrix_layout b_layout = matrix_layout::row_major;
    static constexpr matrix_layout c_layout = matrix_layout::row_major;
};

template<>
struct layout_selector<kernel_type::shared>
{
    static constexpr matrix_layout a_layout = matrix_layout::row_major;
    static constexpr matrix_layout b_layout = matrix_layout::col_major;
    static constexpr matrix_layout c_layout = matrix_layout::row_major;
};

template<>
struct layout_selector<kernel_type::wmma_naive>
{
    static constexpr matrix_layout a_layout = matrix_layout::row_major;
    static constexpr matrix_layout b_layout = matrix_layout::col_major;
    static constexpr matrix_layout c_layout = matrix_layout::row_major;
};

// Specialize for rocBLAS
template<>
struct layout_selector<kernel_type::rocblas>
{
    static constexpr matrix_layout a_layout = matrix_layout::col_major;
    static constexpr matrix_layout b_layout = matrix_layout::col_major;
    static constexpr matrix_layout c_layout = matrix_layout::col_major;
};

// Template function to run matrix multiplication with different kernel types and tile sizes
template<kernel_type K_TYPE, bool VERIFY = false>
int run_hgemm(size_t M, size_t N, size_t K)
{
    constexpr int runs = 50; // Number of runs for timing

    // Allocate memory on host using std::vector
    matrix<half, layout_selector<K_TYPE>::a_layout> h_A(M, K);
    matrix<half, layout_selector<K_TYPE>::b_layout> h_B(K, N);
    matrix<half, layout_selector<K_TYPE>::c_layout> h_C(M, N);
    matrix<half, layout_selector<K_TYPE>::c_layout> h_C_ref(M, N);

    // Initialize input matrices with random values
    init_matrix(h_A.data(), h_A.size());
    init_matrix(h_B.data(), h_B.size());

    hipStream_t stream;
    HIP_CHECK(hipStreamCreate(&stream));

    // Allocate memory on device
    half *d_A, *d_B, *d_C;
    HIP_CHECK(hipMalloc(&d_A, h_A.size() * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_B, h_B.size() * sizeof(half)));
    HIP_CHECK(hipMalloc(&d_C, h_C.size() * sizeof(half)));

    // Copy data from host to device
    HIP_CHECK(hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), hipMemcpyHostToDevice));
    HIP_CHECK(hipDeviceSynchronize());

    gpu_timer timer;

    // Warmup only
    for(int i = 0; i < 5; ++i)
    {
        hgemm_gpu<K_TYPE>(d_C, d_A, d_B, M, N, K, stream);
        HIP_CHECK(hipPeekAtLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());

    // Launch kernel multiple times to average execution time
    timer.start(stream);
    for(int i = 0; i < runs; ++i)
    {
        // Execute the matrix multiplication kernel
        hgemm_gpu<K_TYPE>(d_C, d_A, d_B, M, N, K, stream);
        HIP_CHECK(hipPeekAtLastError());
    }
    float elapsed_time = timer.stop(stream);

    std::cout << "Kernel execution time for sizes (" << M << ", " << N << ", " << K
              << "): " << elapsed_time / runs << " ms" << std::endl;

    // Copy the result back to host
    HIP_CHECK(hipMemcpy(h_C.data(), d_C, M * N * sizeof(half), hipMemcpyDeviceToHost));

    // Verify result if requested
    if(VERIFY)
    {
        // Calculate reference result on CPU
        hgemm_cpu(h_C_ref, h_A, h_B);
        verify_results(h_C, h_C_ref);
    }

    // Free device memory and destroy stream
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));

    return 0;
}

struct test_config
{
    // Sizes where verification will be performed
    std::vector<std::tuple<size_t, size_t, size_t>> verify_sizes = {
        {128, 128, 128}
    };

    // Sizes for pure benchmarking
    std::vector<std::tuple<size_t, size_t, size_t>> benchmark_sizes = {
        {1024, 1024, 1024},
        {4096, 4096, 4096}
    };
};

// Helper function to convert kernel type to string
inline const char* kernel_type_string(kernel_type type)
{
    switch(type)
    {
        case kernel_type::shared: return "Shared Memory";
        case kernel_type::wmma_naive: return "WMMA Naive";
        case kernel_type::wmma_shared: return "WMMA + Shared Memory";
        case kernel_type::wmma_shared_warp: return "WMMA + Shared Memory + Warp Tiling";
        case kernel_type::wmma_shared_warp_buf:
            return "WMMA + Shared Memory + Warp Tiling + Double Buffering";
        case kernel_type::wmma_shared_warp_buf_vec:
            return "WMMA + Shared Memory + Warp Tiling + Double Buffering + Global "
                   "Vectorized Loads";
        case kernel_type::wmma_prefetch: return "WMMA Prefetch";
#ifdef HAS_ROCWMMA
        case kernel_type::rocwmma: return "rocWMMA";
#endif
        case kernel_type::rocblas: return "rocBLAS";
        default: return "Unknown";
    }
}

// Variadic template to run multiple kernel types
template<kernel_type... Types>
void run_all_kernels(const test_config& config)
{
    (run_kernel_tests<Types>(config), ...);
}

// Helper function to run a single kernel type with the given configuration
template<kernel_type K_TYPE>
void run_kernel_tests(const test_config& config)
{
    std::cout << "GEMM Kernel Type: " << kernel_type_string(K_TYPE) << std::endl;
    std::cout << std::string(80, '-') << std::endl;

    // Run verification tests
    for(const auto& [M, N, K] : config.verify_sizes)
    {
        run_hgemm<K_TYPE, true>(M, N, K);
    }

    // Run benchmark tests
    for(const auto& [M, N, K] : config.benchmark_sizes)
    {
        run_hgemm<K_TYPE, false>(M, N, K);
    }

    std::cout << std::string(80, '-') << std::endl;
}

int main(int argc, char** argv)
{
    test_config config{
        .verify_sizes    = {{128, 128, 128}},
        .benchmark_sizes = {{256, 256, 256},
                            {512, 512, 512},
                            {1024, 1024, 1024},
                            {2048, 2048, 2048},
                            {4096, 4096, 4096}}
    };

    run_all_kernels<kernel_type::shared,
                    kernel_type::wmma_naive,
                    kernel_type::wmma_shared,
                    kernel_type::wmma_shared_warp,
                    kernel_type::wmma_shared_warp_buf,
                    kernel_type::wmma_shared_warp_buf_vec,
                    kernel_type::wmma_prefetch
#ifdef HAS_ROCWMMA
                    ,
                    kernel_type::rocwmma
#endif
                    >(config);

    init_rocblas();
    run_kernel_tests<kernel_type::rocblas>(config);
    cleanup_rocblas();

    return 0;
}
