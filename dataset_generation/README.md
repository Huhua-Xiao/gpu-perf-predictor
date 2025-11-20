# GPU Performance Predictor - Dataset Generation

This directory contains CUDA benchmark implementations for generating performance datasets used to train GPU performance prediction models. The benchmarks support two representative workloads:

- **GEMM (General Matrix Multiply)**: Matrix multiplication benchmarking with cuBLAS and custom tiled implementations
- **NTT (Number Theoretic Transform)**: Fast Fourier Transform-like operations for polynomial multiplication

## Overview

These benchmarks collect comprehensive performance metrics including:
- GPU hardware specifications (compute capability, SM count, memory specs, etc.)
- Kernel configuration parameters (matrix dimensions, NTT size, etc.)
- Runtime performance metrics (execution time, GFLOPS, memory throughput, etc.)
- Real-time GPU telemetry (clock speeds, temperature, power consumption) via NVML

All results are saved to CSV files for downstream machine learning model training.

## Prerequisites

- **CUDA Toolkit** (version 12.4 recommended)
- **NVIDIA GPU** with CUDA support
- **nvcc** compiler
- **cuBLAS** library (for GEMM benchmarks)
- **NVML** library (`libnvidia-ml`) for runtime telemetry (optional, but recommended)

### Environment Setup

On systems with module support:

```bash
module avail cuda
module load cuda-12.4
```


## GEMM Benchmarks

### Compilation

From the `GEMM/` directory:

```bash
cd GEMM
make
```

Or manually:

```bash
nvcc -O3 -std=c++17 benchmark.cu -lcublas -lnvidia-ml -o benchmark
```

### Usage

```bash
./benchmark <method> <num_samples> [seed]
```

#### Arguments:
- **method** (required): 
  - `0` = cuBLAS implementation (NVIDIA's optimized library)
  - `1` = Custom tiled implementation (shared-memory CUDA kernel)
- **num_samples** (required): Number of matrix samples to benchmark (must be > 0)
- **seed** (optional): Random seed for reproducibility. If omitted, a random seed is used.

#### Examples:

```bash
# Run cuBLAS implementation with 10000 samples
./benchmark 0 10000

# Run custom tiled implementation with 5000 samples
./benchmark 1 5000

# Run cuBLAS with 100 samples using seed 42
./benchmark 0 100 42

# Run custom implementation with 500 samples using seed 12345
./benchmark 1 500 12345
```

### Benchmarking Process

1. **Warm-up**: 10 iterations to stabilize performance
2. **Timing**: Multiple runs (default: 10 repeats) with inner iterations (default: 100) per run
3. **Statistics**: Calculates mean, median, 95th percentile, and standard deviation
4. **Metrics**: Computes GFLOPS, memory throughput, and efficiency metrics
5. **Telemetry**: Collects runtime GPU metrics via NVML (if available)
6. **Checkpointing**: Saves progress every 100 samples during long runs

---

## NTT Benchmarks

### Compilation

From the `NTT/` directory:

```bash
cd NTT
make
```

Or manually:

```bash
nvcc -O3 -std=c++17 benchmark.cu -lnvidia-ml -o benchmark
```

### Usage

```bash
./benchmark <method> <num_samples> [N] [repeats] [inner_iters] [seed]
```

#### Arguments:
- **method** (required): 
  - `1` = Custom-NTT implementation
- **num_samples** (required): Number of NTT samples to benchmark (must be > 0)
- **N** (optional): Fixed NTT size (must be a power of 2). If omitted, random sizes are used
- **repeats** (optional): Number of repetitions per sample (default: 10)
- **inner_iters** (optional): Inner iterations per repeat (default: 50)
- **seed** (optional): Random seed for reproducibility. If omitted, a random seed is used.

#### Examples:

```bash
# Run 100 samples with random NTT sizes (2^k where k ∈ [15,24])
./benchmark 1 100

# Run 100 samples with fixed N=65536
./benchmark 1 100 65536

# Run 100 samples with fixed N=65536, custom repeats and inner_iters
./benchmark 1 100 65536 10 50

# Run 100 samples with fixed N=65536, custom parameters, and seed
./benchmark 1 100 65536 10 50 42

# Run 10 samples with N=65536, default repeats and inner_iters
./benchmark 1 10 65536
```


### Benchmarking Process

1. **Warm-up**: 10 iterations to stabilize performance
2. **Timing**: Multiple runs (default: 10 repeats) with inner iterations (default: 50) per run
3. **Statistics**: Calculates mean, median, 95th percentile, and standard deviation
4. **Metrics**: Computes modular operations per second, memory throughput, and efficiency
5. **Telemetry**: Collects runtime GPU metrics via NVML (if available)
6. **Checkpointing**: Saves progress every 100 samples during long runs