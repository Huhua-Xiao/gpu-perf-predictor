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
./benchmark <method> <num_samples>
```

### Arguments:
- **method** (required): 
  - `0` = cuBLAS implementation
  - `1` = Custom tiled implementation
- **num_samples** (required): Number of matrix samples to benchmark (must be > 0)

### Examples:

```bash
# Run cuBLAS implementation with 10000 samples
./benchmark 0 10000

# Run custom tiled implementation with 5000 samples
./benchmark 1 5000

# Run cuBLAS with 100 samples
./benchmark 0 100

# Run custom implementation with 500 samples
./benchmark 1 500
```

## Implementation Details

### cuBLAS (method=0)
- Uses NVIDIA's optimized `cublasSgemm` function
- High performance, well-optimized library implementation
- Computes: C = A(M×K) × B(K×N)

### Custom Tiled (method=1)
- Tiled shared-memory CUDA kernel (`matrixMulKernel`)
- Tile size: 16×16 (`TILE_WIDTH = 16`)
- Uses shared memory with padding (`TILE_WIDTH + 1`) to reduce bank conflicts
- Implements standard tiled matrix multiplication algorithm
- Computes: C = A(M×K) × B(K×N)

## Output

The benchmark generates a CSV file `benchmark_results.csv` containing:
- GPU hardware specifications (name, compute capability, SM count, memory specs, etc.)
- Matrix dimensions (M, N, K)
- Algorithm used (cuBLAS or Custom-MM)
- Performance metrics:
  - Time statistics (mean, median, p95, stddev in milliseconds)
  - GFLOPS (mean, median, p95)
  - Memory throughput (GB/s)
  - Memory efficiency (%)
  - Compute efficiency (%)
- Derived metrics:
  - Total FLOPS
  - Theoretical I/O bytes
  - Arithmetic intensity (FLOPs per Byte)
  - GFLOPS over peak ratio

The CSV is automatically appended to (preserves existing data). During long runs, checkpoints are saved every 1000 samples to prevent data loss.

## Matrix Size Distribution

The benchmark randomly samples matrix sizes using a quadratic distribution:
- Range: 128 to 8192 for each dimension (M, N, K)
- Distribution: Quadratic sampling (smaller sizes are more likely to be sampled)
- Formula: `size = 128 + random² × (8192 - 128)`
- This creates a bias toward smaller matrices while still covering the full range

## Benchmarking Process

1. **Warm-up**: 10 iterations to stabilize performance
2. **Timing**: Multiple runs (default: 10 repeats) with inner iterations (default: 100) per run
3. **Statistics**: Calculates mean, median, 95th percentile, and standard deviation
4. **Metrics**: Computes GFLOPS, memory throughput, and efficiency metrics
5. **Checkpointing**: Saves progress every 1000 samples during long runs
