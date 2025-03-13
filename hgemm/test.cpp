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
#include <gtest/gtest.h>
#include <hgemm.hpp>

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

template<>
struct layout_selector<kernel_type::rocblas>
{
    static constexpr matrix_layout a_layout = matrix_layout::col_major;
    static constexpr matrix_layout b_layout = matrix_layout::row_major;
    static constexpr matrix_layout c_layout = matrix_layout::col_major;
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
        case kernel_type::wmma_shared_warp_vec:
            return "WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads";
        case kernel_type::wmma_shared_warp_buf_vec:
            return "WMMA + Shared Memory + Warp Tiling + Double Buffering + Global "
                   "Vectorized Loads";
        case kernel_type::wmma_prefetch: return "WMMA Prefetch";
        case kernel_type::wmma_opt_1: return "WMMA Optimized V1";
        case kernel_type::wmma_opt_2: return "WMMA Optimized V2";
        case kernel_type::rocblas: return "rocBLAS";
        default: return "Unknown";
    }
}

// Base template for kernel type wrapper
template<kernel_type KT>
struct KernelTypeWrapper
{
    static constexpr kernel_type value = KT;
};

using SharedMemoryKernel         = KernelTypeWrapper<kernel_type::shared>;
using WmmaNaiveKernel            = KernelTypeWrapper<kernel_type::wmma_naive>;
using WmmaSharedKernel           = KernelTypeWrapper<kernel_type::wmma_shared>;
using WmmaSharedWarpKernel       = KernelTypeWrapper<kernel_type::wmma_shared_warp>;
using WmmaSharedWarpBufKernel    = KernelTypeWrapper<kernel_type::wmma_shared_warp_buf>;
using WmmaSharedWarpVecKernel    = KernelTypeWrapper<kernel_type::wmma_shared_warp_vec>;
using WmmaSharedWarpBufVecKernel = KernelTypeWrapper<kernel_type::wmma_shared_warp_buf_vec>;
using WmmaPrefetchKernel         = KernelTypeWrapper<kernel_type::wmma_prefetch>;
using WmmaOpt1Kernel             = KernelTypeWrapper<kernel_type::wmma_opt_1>;
using WmmaOpt2Kernel             = KernelTypeWrapper<kernel_type::wmma_opt_2>;
using RocblasKernel              = KernelTypeWrapper<kernel_type::rocblas>;

// Test fixture for HGEMM testing
template<typename KernelTypeT>
class HGEMMTest : public ::testing::Test
{
protected:
    static constexpr kernel_type K_TYPE     = KernelTypeT::value;
    static constexpr bool        is_rocblas = (K_TYPE == kernel_type::rocblas);

    void SetUp() override
    {
        if constexpr(is_rocblas)
        {
            init_rocblas();
        }

        HIP_CHECK(hipStreamCreate(&stream));
    }

    void TearDown() override
    {
        HIP_CHECK(hipStreamDestroy(stream));

        if constexpr(is_rocblas)
        {
            cleanup_rocblas();
        }
    }

    // Template function to run matrix multiplication and verify results
    void VerifyHGEMM(size_t M, size_t N, size_t K)
    {
        // Allocate memory on host using std::vector
        matrix<half, layout_selector<K_TYPE>::a_layout> h_A(M, K);
        matrix<half, layout_selector<K_TYPE>::b_layout> h_B(K, N);
        matrix<half, layout_selector<K_TYPE>::c_layout> h_C(M, N);
        matrix<half, layout_selector<K_TYPE>::c_layout> h_C_ref(M, N);

        // Initialize input matrices with random values
        init_matrix(h_A.data(), h_A.size());
        init_matrix(h_B.data(), h_B.size());

        // Allocate memory on device
        half *d_A, *d_B, *d_C;
        HIP_CHECK(hipMalloc(&d_A, h_A.size() * sizeof(half)));
        HIP_CHECK(hipMalloc(&d_B, h_B.size() * sizeof(half)));
        HIP_CHECK(hipMalloc(&d_C, h_C.size() * sizeof(half)));

        // Copy data from host to device
        HIP_CHECK(hipMemcpy(d_A, h_A.data(), h_A.size() * sizeof(half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(d_B, h_B.data(), h_B.size() * sizeof(half), hipMemcpyHostToDevice));
        HIP_CHECK(hipDeviceSynchronize());

        // Execute the matrix multiplication kernel
        hgemm_gpu<K_TYPE>(d_C, d_A, d_B, M, N, K, stream);
        HIP_CHECK(hipPeekAtLastError());
        HIP_CHECK(hipDeviceSynchronize());

        // Copy the result back to host
        HIP_CHECK(hipMemcpy(h_C.data(), d_C, M * N * sizeof(half), hipMemcpyDeviceToHost));
        HIP_CHECK(hipDeviceSynchronize());

        // Calculate reference result on CPU
        hgemm_cpu(h_C_ref, h_A, h_B);

        // Use the original verification function
        bool verification_result = verify_results(h_C, h_C_ref);

        // Use a gtest assertion
        ASSERT_TRUE(verification_result)
            << "Matrix verification failed for kernel: " << kernel_type_string(K_TYPE)
            << " with size " << M << "x" << N << "x" << K;

        // Free device memory
        HIP_CHECK(hipFree(d_A));
        HIP_CHECK(hipFree(d_B));
        HIP_CHECK(hipFree(d_C));
    }

    hipStream_t stream;
};

// Add or comment out type wrappers here
using KernelTypes = ::testing::Types<SharedMemoryKernel,
                                     WmmaNaiveKernel,
                                     WmmaSharedKernel,
                                     WmmaSharedWarpKernel,
                                     WmmaSharedWarpBufKernel,
                                     WmmaSharedWarpVecKernel,
                                     WmmaSharedWarpBufVecKernel,
                                     WmmaPrefetchKernel,
                                     WmmaOpt1Kernel,
                                     WmmaOpt2Kernel,
                                     RocblasKernel>;

TYPED_TEST_SUITE(HGEMMTest, KernelTypes);

// Test cases for the specified matrix sizes
TYPED_TEST(HGEMMTest, Size256)
{
    constexpr size_t M = 256;
    constexpr size_t N = 256;
    constexpr size_t K = 256;

    std::cout << "Testing " << kernel_type_string(TestFixture::K_TYPE) << " with size " << M << "x"
              << N << "x" << K << std::endl;

    this->VerifyHGEMM(M, N, K);
}

TYPED_TEST(HGEMMTest, Size512)
{
    constexpr size_t M = 512;
    constexpr size_t N = 512;
    constexpr size_t K = 512;

    std::cout << "Testing " << kernel_type_string(TestFixture::K_TYPE) << " with size " << M << "x"
              << N << "x" << K << std::endl;

    this->VerifyHGEMM(M, N, K);
}

int main(int argc, char** argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
