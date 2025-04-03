# Square Matrix Performance Benchmarks

Performance measured on AMD Radeon RX 7900 GRE on Windows and WSL2 (HIP SDK 6.2.4). All implementations use half precision (FP16).

Note: Kernel parameters haven't been tuned for different sizes in the following tables.

## Performance for 1024x1024 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 0.639 | 3.37 | 0.651 | 3.31 |
| WMMA Naive | 0.474 | 4.53 | 0.520 | 4.23 |
| WMMA + Shared Memory | 0.277 | 7.76 | 0.307 | 7.19 |
| WMMA + Shared Memory + Warp Tiling | 0.380 | 5.67 | 0.403 | 5.39 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.371 | 5.79 | 0.407 | 5.32 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.165 | 13.13 | 0.183 | 11.92 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.163 | 13.23 | 0.205 | 11.21 |
| WMMA Prefetch | 0.173 | 12.45 | 0.211 | 10.76 |
| WMMA Optimized V1 | 0.157 | 13.79 | 0.173 | 12.65 |
| WMMA Optimized V2 | 0.204 | 10.59 | 0.222 | 9.93 |
| WMMA Optimized V3 | 0.202 | 10.66 | 0.220 | 9.92 |
| rocBLAS | 0.100 | 21.59 | 0.125 | 18.35 |

## Performance for 2048x2048 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 4.960 | 3.47 | 4.820 | 3.58 |
| WMMA Naive | 3.820 | 4.54 | 3.470 | 4.96 |
| WMMA + Shared Memory | 1.390 | 12.34 | 1.690 | 10.18 |
| WMMA + Shared Memory + Warp Tiling | 0.833 | 20.65 | 0.823 | 20.94 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 0.809 | 21.25 | 0.810 | 21.27 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 0.406 | 42.41 | 0.424 | 40.91 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 0.396 | 43.46 | 0.436 | 40.91 |
| WMMA Prefetch | 0.418 | 41.14 | 0.425 | 40.69 |
| WMMA Optimized V1 | 0.377 | 45.65 | 0.401 | 43.67 |
| WMMA Optimized V2 | 0.349 | 49.35 | 0.351 | 49.17 |
| WMMA Optimized V3 | 0.348 | 49.47 | 0.377 | 47.77 |
| rocBLAS | 0.313 | 55.15 | 0.322 | 54.92 |

## Performance for 4096x4096 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 37.300 | 3.69 | 41.800 | 3.33 |
| WMMA Naive | 23.600 | 5.83 | 22.200 | 6.20 |
| WMMA + Shared Memory | 10.600 | 12.95 | 12.600 | 10.95 |
| WMMA + Shared Memory + Warp Tiling | 6.400 | 21.48 | 6.610 | 20.81 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 6.250 | 22.00 | 6.390 | 21.53 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 2.370 | 57.93 | 2.490 | 55.27 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 2.310 | 59.50 | 2.340 | 59.02 |
| WMMA Prefetch | 2.350 | 58.61 | 2.340 | 58.89 |
| WMMA Optimized V1 | 2.160 | 63.58 | 2.150 | 63.99 |
| WMMA Optimized V2 | 2.180 | 63.10 | 2.200 | 62.71 |
| WMMA Optimized V3 | 2.180 | 63.21 | 2.150 | 63.95 |
| rocBLAS | 1.940 | 70.93 | 1.850 | 74.73 |

## Performance for 8192x8192 Matrix Multiplication
| Implementation | Windows Time (ms) | Windows TFLOPs/s | WSL2 Time (ms) | WSL2 TFLOPs/s |
|----------------|-------------------|-------------------|----------------|---------------|
| Shared Memory | 326.000 | 3.37 | 328.000 | 3.35 |
| WMMA Naive | 198.000 | 5.56 | 200.000 | 5.51 |
| WMMA + Shared Memory | 94.100 | 11.68 | 94.200 | 11.67 |
| WMMA + Shared Memory + Warp Tiling | 42.900 | 25.65 | 43.000 | 25.59 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering | 41.400 | 26.54 | 41.200 | 26.70 |
| WMMA + Shared Memory + Warp Tiling + Global Vectorized Loads | 18.000 | 60.98 | 17.600 | 62.66 |
| WMMA + Shared Memory + Warp Tiling + Double Buffering + Global Vectorized Loads | 17.300 | 63.59 | 17.200 | 64.02 |
| WMMA Prefetch | 17.400 | 63.39 | 17.400 | 63.34 |
| WMMA Optimized V1 | 15.900 | 69.39 | 15.900 | 69.29 |
| WMMA Optimized V2 | 14.400 | 76.63 | 14.400 | 76.17 |
| WMMA Optimized V3 | 14.400 | 76.41 | 14.400 | 76.22 |
| rocBLAS | 14.400 | 76.45 | 14.300 | 76.76 |

## Analysis

### Optimization Progress
- From the baseline shared memory implementation to the best optimized version, achieved a **~22.5x speedup** for larger matrices
- WMMA Optimized V3 is now the best performing implementation for 8192x8192 matrices at 76.41 TFLOPs/s on Windows and 76.22 TFLOPs/s on WSL2
- The gap between the best implementation and rocBLAS has narrowed significantly, with near-parity achieved on 8192x8192 matrices

### Platform Differences
- Windows and WSL2 performance is mostly comparable
