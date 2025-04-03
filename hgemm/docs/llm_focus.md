# LLM-Focused Benchmarks

The latest HGEMM implementation (`wmma_opt_3`) has been benchmarked against `rocBLAS` using matrix dimensions commonly found in transformer/LLM architectures.

## Performance on Transformer/LLM Matrix Shapes

Below are benchmarks comparing my best implementation against `rocBLAS` on non-square matrix shapes typical in transformer models:

| Matrix Dimensions | Operation Type | `wmma_opt_3` (TFLOPs/s) | `rocBLAS` (TFLOPs/s) | `wmma_opt_3`/`rocBLAS` Ratio |
|------------------|----------------|-----------------|-------------------|-------------------|
| m=4096, n=4096, k=1024 | QKV Projection | 51.20 | 65.96 | 77.6% |
| m=8192, n=8192, k=1024 | QKV Projection (Large Batch) | 67.48 | 72.58 | 93.0% |
| m=4096, n=2048, k=64 | Attention Score | 10.92 | 11.91 | 91.7% |
| m=8192, n=4096, k=128 | Attention Score (Large Batch) | 32.68 | 41.82 | 78.2% |
| m=4096, n=16384, k=4096 | FFN First Layer | 76.30 | 75.29 | 101.3% |
| m=4096, n=4096, k=16384 | FFN Second Layer | 66.62 | 53.89 | 123.6% |
| m=2048, n=5120, k=5120 | Model with 5120 Hidden Dim | 76.47 | 72.05 | 106.1% |
| m=4096, n=5120, k=5120 | Model with 5120 Hidden Dim (Larger Batch) | 77.32 | 66.26 | 116.7% |
| m=32768, n=4096, k=4096 | Long Context Processing | 78.54 | 76.10 | 103.2% |
| m=65536, n=2048, k=2048 | Very Long Context Processing | 80.54 | 62.01 | 129.9% |

## Raw Benchmark Data

Below is the raw benchmark data for reference:

```
----------------------------------------------------------------------------------------------------------------------------
Benchmark                                                                  Time             CPU   Iterations UserCounters...
----------------------------------------------------------------------------------------------------------------------------
{hgemm:kernel_type::wmma_opt_3,m:4096,n:4096,k:1024}/manual_time       0.673 ms        0.649 ms          987 TFLOPS=51.2042 bytes_per_second=69.6253Gi/s
{hgemm:kernel_type::wmma_opt_3,m:8192,n:8192,k:1024}/manual_time        2.04 ms         2.05 ms          365 TFLOPS=67.4756 bytes_per_second=76.6093Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:2048,k:64}/manual_time         0.100 ms        0.100 ms         7162 TFLOPS=10.9192 bytes_per_second=164.063Gi/s
{hgemm:kernel_type::wmma_opt_3,m:8192,n:4096,k:128}/manual_time        0.264 ms        0.268 ms         2619 TFLOPS=32.6842 bytes_per_second=247.463Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:16384,k:4096}/manual_time       7.22 ms         7.28 ms          103 TFLOPS=76.2983 bytes_per_second=38.9532Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:4096,k:16384}/manual_time       8.26 ms         8.35 ms           88 TFLOPS=66.6227 bytes_per_second=34.0583Gi/s
{hgemm:kernel_type::wmma_opt_3,m:2048,n:5120,k:5120}/manual_time        1.41 ms         1.40 ms          514 TFLOPS=76.4704 bytes_per_second=62.5355Gi/s
{hgemm:kernel_type::wmma_opt_3,m:4096,n:5120,k:5120}/manual_time        2.78 ms         2.80 ms          262 TFLOPS=77.3214 bytes_per_second=45.6572Gi/s
{hgemm:kernel_type::wmma_opt_3,m:32768,n:4096,k:4096}/manual_time       14.0 ms         13.9 ms           53 TFLOPS=78.5394 bytes_per_second=37.904Gi/s
{hgemm:kernel_type::wmma_opt_3,m:65536,n:2048,k:2048}/manual_time       6.83 ms         6.83 ms          103 TFLOPS=80.5406 bytes_per_second=74.3633Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:1024}/manual_time          0.523 ms        0.525 ms         1399 TFLOPS=65.9577 bytes_per_second=89.5896Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:8192,k:1024}/manual_time           1.90 ms         1.90 ms          338 TFLOPS=72.5826 bytes_per_second=82.3954Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:2048,k:64}/manual_time            0.092 ms        0.095 ms         7770 TFLOPS=11.9069 bytes_per_second=177.316Gi/s
{hgemm:kernel_type::rocblas,m:8192,n:4096,k:128}/manual_time           0.207 ms        0.206 ms         3338 TFLOPS=41.8162 bytes_per_second=315.925Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:16384,k:4096}/manual_time          7.31 ms         7.27 ms          101 TFLOPS=75.2907 bytes_per_second=38.4683Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:4096,k:16384}/manual_time          10.2 ms         10.1 ms           68 TFLOPS=53.8943 bytes_per_second=27.5202Gi/s
{hgemm:kernel_type::rocblas,m:2048,n:5120,k:5120}/manual_time           1.49 ms         1.50 ms          479 TFLOPS=72.0482 bytes_per_second=58.8557Gi/s
{hgemm:kernel_type::rocblas,m:4096,n:5120,k:5120}/manual_time           3.27 ms         3.26 ms          225 TFLOPS=66.2591 bytes_per_second=38.8676Gi/s
{hgemm:kernel_type::rocblas,m:32768,n:4096,k:4096}/manual_time          14.5 ms         14.4 ms           51 TFLOPS=76.0952 bytes_per_second=36.7173Gi/s
{hgemm:kernel_type::rocblas,m:65536,n:2048,k:2048}/manual_time          8.87 ms         8.81 ms           78 TFLOPS=62.0142 bytes_per_second=57.2786Gi/s
```
