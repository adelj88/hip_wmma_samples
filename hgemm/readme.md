# Understanding HIP and WMMA Intrinsics

This project is a personal exploration of HIP programming and the RDNA3 Wave Matrix Multiply-Accumulate (WMMA) intrinsic. The primary goal was to deepen my understanding of the WMMA intrinsic and extend the fixed-size example provided in the [GPUOpen tutorial](https://gpuopen.com/learn/wmma_on_rdna3/) to support arbitrary matrix dimensions. While this project is primarily for personal learning, it may also serve as a helpful reference for others interested in exploring the WMMA intrinsic.

**Note:** The WMMA intrinsic is specific to RDNA3 GPUs for now, so running this project requires an RDNA3-compatible GPU. A future feature may include testing this implementation on RDNA4 hardware when it becomes available.

## Objectives
This project aims to:
1. Provide a simple example of HIP programming and WMMA usage for GPU-accelerated computation
2. Extend beyond the fixed-size example in the GPUOpen tutorial by supporting arbitrary power-of-two matrix dimensions (M, N, K)
3. Enhance understanding of the WMMA intrinsic's mechanics, especially around data loading and storing

## Features

- **Flexible Matrix Dimensions:** Supports arbitrary power-of-two matrix sizes (M, N, K) beyond the basic 16x16 example
- **Multiple Implementations:**
  - Basic WMMA implementation
  - Shared memory optimized WMMA
  - Other optimizations combined with WMMA
  - Traditional shared memory implementation (for comparison)
- **Performance Benchmarking:** Built-in benchmarking capabilities for comparing different implementations
- **Correctness Verification:** CPU reference implementation for result validation

## Performance Results

Performance measured on AMD Radeon RX 7900 GRE on Windows and WSL2 (HIP SDK 6.2.4). All implementations use half precision (FP16).

### Performance for 1024x1024 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 0.668 | 3.21 | 0.678 | 3.17 |
| WMMA Naive | 0.541 | 3.97 | 0.538 | 3.99 |
| WMMA + Shared Memory | 0.390 | 5.51 | 0.332 | 6.47 |
| WMMA + Shared Memory + Warp Tiling | 0.522 | 4.11 | 0.448 | 4.79 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.491 | 4.37 | 0.458 | 4.69 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.219 | 9.81 | 0.210 | 10.23 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.223 | 9.63 | 0.220 | 9.76 |
| WMMA Prefetch | 0.236 | 9.10 | 0.207 | 10.37 |
| WMMA Optimized V1 | 0.186 | 11.55 | 0.193 | 11.13 |
| WMMA Optimized V2 | 0.174 | 12.34 | 0.192 | 11.18 |
| rocBLAS | 0.115 | 18.67 | 0.141 | 15.23 |

### Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 4.840 | 3.55 | 4.690 | 3.66 |
| WMMA Naive | 3.920 | 4.38 | 3.420 | 5.02 |
| WMMA + Shared Memory | 1.700 | 10.11 | 1.750 | 9.82 |
| WMMA + Shared Memory + Warp Tiling | 0.950 | 18.08 | 0.899 | 19.11 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.972 | 17.67 | 0.850 | 20.21 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.545 | 31.52 | 0.454 | 37.84 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.500 | 34.36 | 0.439 | 39.13 |
| WMMA Prefetch | 0.532 | 32.29 | 0.447 | 38.43 |
| WMMA Optimized V1 | 0.438 | 39.22 | 0.422 | 40.71 |
| WMMA Optimized V2 | 0.400 | 42.95 | 0.425 | 40.42 |
| rocBLAS | 0.330 | 52.06 | 0.349 | 49.23 |

### Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 37.200 | 3.69 | 43.100 | 3.19 |
| WMMA Naive | 23.300 | 5.90 | 21.500 | 6.39 |
| WMMA + Shared Memory | 11.500 | 11.95 | 12.300 | 11.17 |
| WMMA + Shared Memory + Warp Tiling | 6.050 | 22.72 | 6.480 | 21.21 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 6.610 | 20.79 | 6.360 | 21.61 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.550 | 53.90 | 2.440 | 56.33 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 2.400 | 57.27 | 2.340 | 58.73 |
| WMMA Prefetch | 2.460 | 55.87 | 2.340 | 58.73 |
| WMMA Optimized V1 | 2.320 | 59.24 | 2.200 | 62.47 |
| WMMA Optimized V2 | 2.160 | 63.63 | 2.140 | 64.22 |
| rocBLAS | 1.930 | 71.21 | 1.860 | 73.89 |

### Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 324.000 | 3.39 | 327.000 | 3.36 |
| WMMA Naive | 196.000 | 5.61 | 200.000 | 5.50 |
| WMMA + Shared Memory | 94.000 | 11.70 | 93.800 | 11.72 |
| WMMA + Shared Memory + Warp Tiling | 42.900 | 25.63 | 42.600 | 25.81 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 41.600 | 26.43 | 41.000 | 26.82 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 17.800 | 61.77 | 17.500 | 62.83 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 17.300 | 63.56 | 17.100 | 64.30 |
| WMMA Prefetch | 17.400 | 63.19 | 17.300 | 63.56 |
| WMMA Optimized V1 | 15.800 | 69.59 | 16.000 | 68.72 |
| WMMA Optimized V2 | 15.600 | 70.48 | 15.800 | 69.59 |
| rocBLAS | 14.300 | 76.89 | 14.200 | 77.43 |

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

1. Current implementations are limited to the work tile configurations (e.g. if a tile of 256x256x16 is used, it will only work for dimensions that are multiples of 256).

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
