# Understanding HIP and WMMA Intrinsics

This project is a personal exploration of HIP programming and the RDNA3 Wave Matrix Multiply-Accumulate (WMMA) intrinsic. The primary goal was to deepen my understanding of the WMMA intrinsic and extend the fixed-size example provided in the [GPUOpen tutorial](https://gpuopen.com/learn/wmma_on_rdna3/) to support arbitrary matrix dimensions. While this project is primarily for personal learning, it may also serve as a helpful reference for others interested in exploring the WMMA intrinsic.

**Note:** The WMMA intrinsic is specific to RDNA3 GPUs for now, so running this project requires an RDNA3-compatible GPU. A future feature may include testing this implementation on RDNA4 hardware when it becomes available. For production-grade GPU matrix multiplication, it is highly recommended to use [rocWMMA](https://github.com/ROCm/rocWMMA), which provides a robust and optimized abstraction over the WMMA functionality.

## Objective

This project aims to:
1. Provide a simple example of HIP programming and WMMA usage for GPU-accelerated computation.
2. Extend beyond the fixed-size example in the [GPUOpen tutorial](https://gpuopen.com/learn/wmma_on_rdna3/) by supporting arbitrary matrix dimensions (¨M, N, K¨).
3. Enhance understanding of the WMMA intrinsic's mechanics, especially around data loading and storing.

## Features

- **Matrix Multiplication with WMMA Intrinsic:** Demonstrates how to use HIP-specific WMMA (various implementations).
- **Support for Arbitrary Sizes:** Goes beyond the fixed-size (16x16) example, allowing users to experiment with any matrix dimensions.
- **Shared Memory and WMMA Comparison:** Runs both a shared memory kernel and a WMMA kernel for performance comparison.
- **Verification Mode:** Compares GPU results with CPU reference computations to ensure correctness.

## Future Plans

If time permits, the following enhancements are planned:
1. **Further optimisations of WMMA HGEMM kernel:** Explore other optimisations to further improve WMMA HGEMM kernel.
2. **Explore WMMA in Other Kernel Types:** Investigate the use of WMMA intrinsics in other GPU workloads beyond matrix multiplication.
3. **Test on RDNA4:** Extend the implementation to test and validate the WMMA intrinsic on future RDNA4 hardware.

## Bugs

Currently, the tests for the WMMA HGEMM kernels that use shared memory seem to have an issue when `K > M, N`. I will fix this when time permits.

## How to Build and Run

### Prerequisites

- AMD ROCm installed with HIP support.
- CMake version 3.10 or higher.
- AMD RDNA3 GPU

### Steps

1. Clone the repository and navigate to its root directory:
   ```bash
   git clone https://github.com/adelj88/hip_wmma_samples.git
   cd hip_wmma_samples
   ```
2. Build the project:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
   ```
3. Run the executable:
   ```bash
   ./hgemm
   ```

## Usage

The program outputs performance metrics for all kernels, such as:
```
GEMM Kernel Type: XXX
----------------------------------------------------------------------------------
Kernel execution time for sizes (128, 128, 128): X.XX ms
...
Kernel execution time for sizes (1024, 1024, 1024): X.XX ms
...
Kernel execution time for sizes (4096, 4096, 4096): X.XX ms
----------------------------------------------------------------------------------

```

You can add specific sizes to the`.verify_sizes` list to validate the output, and add specific sizes to the `.benchmark_sizes` list to validate performance in `main.cpp`.

## Key Insights

- This project emphasizes understanding the mechanics of HIP programming and RDNA3 WMMA intrinsics, particularly how to handle data loading and storage effectively.
- It is intended as a learning tool and not as an optimized implementation for production use.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This project was inspired by:
- The [GPUOpen RDNA3 WMMA Tutorial](https://gpuopen.com/learn/wmma_on_rdna3/).
- The excellent work on [rocWMMA](https://github.com/ROCm/rocWMMA), which should be used for real-world applications involving WMMA.
