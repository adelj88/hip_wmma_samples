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

#include <benchmark/benchmark.h>
#include <common/hip_utils.hpp>
#include <common/matrix.hpp>
#include <hgemm.hpp>
#include <iomanip>

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
    static constexpr matrix_layout b_layout = matrix_layout::row_major;
    static constexpr matrix_layout c_layout = matrix_layout::col_major;
};

template<kernel_type K_TYPE>
void run_benchmark(benchmark::State& state, size_t M, size_t N, size_t K)
{
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

    if(K_TYPE == kernel_type::rocblas)
    {
        init_rocblas();
    }

    // Warmup only
    for(int i = 0; i < 5; ++i)
    {
        hgemm_gpu<K_TYPE>(d_C, d_A, d_B, M, N, K, stream);
        HIP_CHECK(hipPeekAtLastError());
    }
    HIP_CHECK(hipDeviceSynchronize());

    double total_tflops = 0.0;
    double total_flops  = 2.0 * M * N * K; // 2 operations per element (multiply and add)

    for(auto _ : state)
    {
        timer.start(stream);
        hgemm_gpu<K_TYPE>(d_C, d_A, d_B, M, N, K, stream);
        HIP_CHECK(hipPeekAtLastError());
        float elapsed_time = timer.stop(stream);
        HIP_CHECK(hipDeviceSynchronize());

        double seconds = elapsed_time / 1000.0;
        state.SetIterationTime(seconds);
        double tflops = (total_flops / seconds) * 1e-12;
        total_tflops += tflops;
    }

    state.counters["TFLOPS"] = total_tflops / state.iterations();
    state.SetBytesProcessed(state.iterations() * ((M * K) + (K * N) + (M * N)) * sizeof(half));

    if(K_TYPE == kernel_type::rocblas)
    {
        cleanup_rocblas();
    }

    // Free device memory and destroy stream
    HIP_CHECK(hipStreamDestroy(stream));
    HIP_CHECK(hipFree(d_A));
    HIP_CHECK(hipFree(d_B));
    HIP_CHECK(hipFree(d_C));
}

#define CREATE_BENCHMARK(K_TYPE, M, N, K)                                          \
    benchmark::RegisterBenchmark("{hgemm:" #K_TYPE ",m:" #M ",n:" #N ",k:" #K "}", \
                                 run_benchmark<K_TYPE>,                            \
                                 M,                                                \
                                 N,                                                \
                                 K)

#define BENCHMARK_SIZE(k_type)                  \
    CREATE_BENCHMARK(k_type, 1024, 1024, 1024), \
    CREATE_BENCHMARK(k_type, 2048, 2048, 2048), \
    CREATE_BENCHMARK(k_type, 4096, 4096, 4096), \
    CREATE_BENCHMARK(k_type, 8192, 8192, 8192)

int main(int argc, char* argv[])
{
    // Parse argv
    benchmark::Initialize(&argc, argv);
    int trials = -1;

    std::vector<benchmark::internal::Benchmark*> benchmarks
        = {BENCHMARK_SIZE(kernel_type::shared),
           BENCHMARK_SIZE(kernel_type::wmma_naive),
           BENCHMARK_SIZE(kernel_type::wmma_shared),
           BENCHMARK_SIZE(kernel_type::wmma_shared_warp),
           BENCHMARK_SIZE(kernel_type::wmma_shared_warp_buf),
           BENCHMARK_SIZE(kernel_type::wmma_shared_warp_vec),
           BENCHMARK_SIZE(kernel_type::wmma_shared_warp_buf_vec),
           BENCHMARK_SIZE(kernel_type::wmma_prefetch),
           BENCHMARK_SIZE(kernel_type::wmma_opt_1),
           BENCHMARK_SIZE(kernel_type::wmma_opt_2),
           BENCHMARK_SIZE(kernel_type::wmma_opt_3),
           BENCHMARK_SIZE(kernel_type::rocblas)};

    // Use manual timing
    for(auto& b : benchmarks)
    {
        b->UseManualTime();
        b->Unit(benchmark::kMillisecond);
    }

    // Force number of iterations
    if(trials > 0)
    {
        for(auto& b : benchmarks)
        {
            b->Iterations(trials);
        }
    }

    // Run benchmarks
    benchmark::RunSpecifiedBenchmarks();
    return 0;
}
