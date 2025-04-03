# Understanding HIP and WMMA Intrinsics

This project is a personal exploration of HIP programming and the RDNA3 Wave Matrix Multiply-Accumulate (WMMA) intrinsic. The primary goal was to deepen my understanding of the WMMA intrinsic and extend the fixed-size example provided in the [GPUOpen tutorial](https://gpuopen.com/learn/wmma_on_rdna3/) to support arbitrary matrix dimensions. While this project is primarily for personal learning, it may also serve as a helpful reference for others interested in exploring the WMMA intrinsic.

**Note:** The WMMA intrinsic is specific to RDNA3 GPUs for now, so running this project requires an RDNA3-compatible GPU. A future feature may include testing this implementation on RDNA4 hardware when it becomes available.

## Objectives
This project aims to:
1. Provide a simple example of HIP programming and WMMA usage for GPU-accelerated computation
2. Extend beyond the fixed-size example in the GPUOpen tutorial by supporting arbitrary matrix dimensions (M, N, K)
3. Enhance understanding of the WMMA intrinsic's mechanics, especially around data loading and storing

## Features

- **Flexible Matrix Dimensions:** Supports arbitrary matrix sizes (M, N, K) beyond the basic 16x16 example
- **Multiple Implementations:**
  - Basic WMMA implementation
  - Shared memory optimized WMMA
  - Other optimizations combined with WMMA
  - Traditional shared memory implementation (for comparison)
- **Performance Benchmarking:** Built-in benchmarking capabilities for comparing different implementations
- **Correctness Verification:** CPU reference implementation for result validation

## Performance Highlights

Performance measured on AMD Radeon RX 7900 GRE on Windows and WSL2 (HIP SDK 6.2.4). All implementations use half precision (FP16).

Note: No tuning has been done for different sizes.

### Square Matrix Performance Progression

The table below shows key performance points in my optimization progression:

| Implementation | 2048x2048 (TFLOPs/s) | 4096x4096 (TFLOPs/s) | 8192x8192 (TFLOPs/s) |
|----------------|---------------------|---------------------|---------------------|
| Shared Memory  | 3.64 | 3.70 | 3.37 |
| WMMA Naive     | 4.95 | 6.14 | 5.61 |
| WMMA + Shared Memory | 10.48 | 13.22 | 11.68 |
| ... | ... | ... | ... |
| WMMA Optimized V2 | 49.35 | 63.10 | 76.63 |
| WMMA Optimized V3 | 49.47 | 63.21 | 76.41 |
| rocBLAS | 55.15 | 70.93 | 76.45 |

[View detailed square matrix benchmarks](docs/general.md)

### LLM-Focused Performance

The most optimized WMMA implementation `wmma_opt_3` is compared against `rocBLAS` on matrix dimensions common in transformer/LLM architectures:

| Operation Type | Matrix Dimensions | `wmma_opt_3` (TFLOPs/s) | `rocBLAS` (TFLOPs/s) | `wmma_opt_3`/`rocBLAS` |
|----------------|-------------------|-----------------|-------------------|--------------|
| FFN Second Layer | m=4096, n=4096, k=16384 | 66.62 | 53.89 | 123.6% |
| Very Long Context | m=65536, n=2048, k=2048 | 80.54 | 62.01 | 129.9% |
| Attention Score | m=4096, n=2048, k=64 | 10.92 | 11.91 | 91.7% |
| Attention Score (Large Batch) | m=8192, n=4096, k=128 | 32.68 | 41.82 | 78.2% |

On average, `wmma_opt_3` achieves decent performance relative to `rocBLAS` across all tested LLM workloads without tuning.

[View detailed LLM benchmarks](docs/llm_focus.md)

## Verification Process

The project implements a comprehensive verification system to ensure kernel correctness and numerical stability across all implementations. The verification process includes:

### 1. Element-wise Validation
- **Comparison Method:** Each element of the GPU result matrix is compared with a CPU reference implementation
- **Adaptive Tolerance:** Different tolerances are applied based on matrix size (e.g., 0.04 for 256x256, 0.0425 for 512x512)
- **Detailed Metrics:**
  - Maximum relative error: Identifies the largest discrepancy and its location
  - Average relative error: Measures overall precision across all matrix elements
  - Number of valid comparisons: Ensures all elements are verified

### 2. Matrix Norm Validation
- **Relative Frobenius Norm Error:** Computes the difference between GPU and CPU results using matrix norms
- **Threshold-based Check:** Ensures the global error magnitude stays below acceptable limits
- **Mathematical Robustness:** Provides a single metric that captures overall numerical stability

### 3. Pattern Validation
- **Structural Similarity (SSIM):** Borrowed from image processing, this metric evaluates if the GPU result preserves the mathematical pattern of the reference
- **Threshold Check:** SSIM must be above 0.95 (95% similarity) to pass
- **Error Pattern Analysis:** Helps identify systematic issues like precision loss or algorithmic flaws

### 4. Comprehensive Reporting
The verification system provides detailed feedback for each test:
- Specific error locations and values
- Statistical summary of errors
- Pass/fail status for each validation method
- Combined overall validation status

## Known Issues

1. Some test cases are skipped for `shared` and `wmma_naive`, as there are no intentions to fix them.
2. `rocBLAS` fails some test cases (or throws an exception), so tests are skipped.

## Usage

Run the executable after building:
```bash
# Assumes you're currently in /build directory
# To run unit tests
./hgemm/test

# Additionally, tests are registered with ctest
# Assumes you're currently in /build directory
cd hgemm
ctest

# To run unit benchmarks
./hgemm/bench
```

## Future Improvements

1. **WMMA HGEMM Optimization:**
   - Explore additional optimization techniques
   - Tuning `wmma_opt_2` for different matrix sizes
   - Add more LLM-specific matrix dimension benchmark cases
   - Further tuning for LLM-specific matrix dimensions
   - Investigate performance on future RDNA4 hardware
