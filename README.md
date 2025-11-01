# GPU Performance Predictor - GEMM Benchmark

This project benchmarks matrix multiplication (GEMM) performance using two implementations:
- **cuBLAS**: NVIDIA's optimized library implementation
- **Custom Tiled**: A tiled shared-memory CUDA kernel implementation

## Environment Setup

```bash
module avail cuda
module load cuda-12.4
```

## Compilation

From the `GEMM/` directory:

```bash
nvcc benchmark.cu -o benchmark -lcublas
```

Or with optimizations:

```bash
nvcc -O3 -std=c++17 benchmark.cu -lcublas -o benchmark
```

## Usage

```bash
./benchmark [method] [num_samples]
```

### Arguments:
- **method** (optional): 
  - `0` = cuBLAS implementation (default: 1)
  - `1` = Custom tiled implementation (default: 1)
- **num_samples** (optional): Number of matrix samples to benchmark (default: 10000)

### Examples:

```bash
# Run custom tiled implementation with 10000 samples (default)
./benchmark

# Run cuBLAS implementation with 10000 samples
./benchmark 0

# Run custom implementation with 5000 samples
./benchmark 1 5000

# Run cuBLAS with 100 samples
./benchmark 0 100
```

## Implementation Details

### cuBLAS (method=0)
- Uses NVIDIA's optimized `cublasSgemm` function
- High performance, well-optimized library implementation

### Custom Tiled (method=1)
- Tiled shared-memory CUDA kernel (`matrixMulKernel`)
- Tile size: 16x16
- Uses shared memory with padding to reduce bank conflicts
- Implements standard tiled matrix multiplication algorithm

## Output

The benchmark generates a CSV file `benchmark_results.csv` containing:
- GPU hardware specifications
- Matrix dimensions (M, N, K)
- Algorithm used
- Performance metrics (time, GFLOPS, memory throughput, efficiency)
- Derived metrics (arithmetic intensity, compute/memory efficiency)

The CSV is automatically appended to (preserves existing data), and checkpoints are saved every 1000 samples during long runs.

## Matrix Size Distribution

The benchmark randomly samples matrix sizes:
- Range: 128 to 8192 for each dimension (M, N, K)
- Distribution: Quadratic sampling (smaller sizes are more likely)
- Formula: `size = 128 + random² * (8192 - 128)`
