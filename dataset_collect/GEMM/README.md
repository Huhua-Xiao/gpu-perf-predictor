# GEMM Performance Benchmark

This directory contains CUDA benchmarking code for General Matrix Multiply (GEMM) operations. The benchmark compares two implementations: NVIDIA's cuBLAS library and a custom tiled shared-memory kernel.

## Table of Contents

- [Overview](#overview)
- [Environment Setup](#environment-setup)
- [Compilation](#compilation)
- [Running the Benchmark](#running-the-benchmark)
- [Implementation Details](#implementation-details)
- [Output Format](#output-format)
- [Understanding the Results](#understanding-the-results)
- [Troubleshooting](#troubleshooting)

---

## Overview

This benchmark measures the performance of matrix multiplication `C = A × B` where:
- `A` is an `M × K` matrix
- `B` is a `K × N` matrix
- `C` is the resulting `M × N` matrix

**Two implementations are tested:**

1. **cuBLAS** - NVIDIA's highly optimized library implementation
2. **Custom Tiled** - A hand-written tiled shared-memory CUDA kernel

The benchmark randomly samples matrix dimensions, measures execution time over multiple runs, and collects comprehensive performance metrics.

---

## Environment Setup

### Prerequisites

- NVIDIA GPU with compute capability 6.0 or higher
- CUDA Toolkit 11.0+ (tested with CUDA 12.4)
- GCC/G++ compiler
- cuBLAS library (included with CUDA Toolkit)

### Load CUDA Module (on HPC systems)

```bash
# Check available CUDA versions
module avail cuda

# Load CUDA (example for CUDA 12.4)
module load cuda-12.4
```

### Verify CUDA Installation

```bash
nvcc --version
which nvcc
```

---

## Compilation

### Using Make (Recommended)

```bash
cd dataset_collect/GEMM
make
```

This compiles the benchmark with optimization flags (`-O3`) and produces the `benchmark` executable.

### Manual Compilation

```bash
nvcc -O3 -std=c++17 benchmark.cu -lcublas -o benchmark
```

**Compilation flags:**
- `-O3`: Maximum optimization
- `-std=c++17`: C++17 standard
- `-lcublas`: Link cuBLAS library
- `-o benchmark`: Output executable name

### Clean Build Artifacts

```bash
make clean
```

---

## Running the Benchmark

### Basic Usage

```bash
./benchmark <method> <num_samples> [output_name] [seed]
```

**Required Arguments:**
- `method`: Implementation to benchmark
  - `0` = cuBLAS (NVIDIA's optimized library)
  - `1` = Custom Tiled (16×16 tiled kernel)
- `num_samples`: Number of random matrix samples to test (must be > 0)

**Optional Arguments:**
- `output_name`: Custom name for output file (default: none)
- `seed`: Random seed for reproducibility (default: time-based)

### Output File Naming

The benchmark generates CSV files with the following naming convention:

```
benchmark_results_[output_name_]N{num_samples}[_S{seed}].csv
```

**Examples:**
- No optional args: `benchmark_results_N1000.csv`
- With output_name: `benchmark_results_cublas_test_N1000.csv`
- With seed: `benchmark_results_cublas_test_N1000_S42.csv`

### Usage Examples

```bash
# Basic: cuBLAS with 1000 samples (default output name)
./benchmark 0 1000
# Output: benchmark_results_N1000.csv

# Custom name: cuBLAS with 10,000 samples
./benchmark 0 10000 cublas_run1
# Output: benchmark_results_cublas_run1_N10000.csv

# With seed: Custom tiled with 5,000 samples
./benchmark 1 5000 custom_test 42
# Output: benchmark_results_custom_test_N5000_S42.csv

# Multiple runs with different seeds
./benchmark 0 10000 cublas_run1 42
./benchmark 0 10000 cublas_run2 123
./benchmark 0 10000 cublas_run3 456

# Different GPUs
./benchmark 0 10000 rtx4090_cublas
./benchmark 0 10000 a100_cublas
./benchmark 0 10000 h100_cublas
```

### What Happens During Execution

1. **GPU Detection**: Prints GPU name and compute capability
2. **Random Matrix Generation**: Creates random M, N, K dimensions
3. **Warm-up Phase**: Runs 10 iterations to stabilize GPU clocks
4. **Benchmarking Loop**: For each sample:
   - Generates random matrices A and B
   - Runs the kernel 10 times with 100 inner iterations each
   - Measures execution time
   - Computes performance metrics
5. **Checkpointing**: Saves results every 1,000 samples
6. **Final Output**: Appends all results to the output CSV file

### Expected Runtime

- **100 samples**: ~1-2 minutes
- **1,000 samples**: ~10-15 minutes
- **10,000 samples**: ~2-3 hours
- **20,000 samples**: ~5-7 hours
- **50,000 samples**: ~10-15 hours

**Note:** Execution time varies based on GPU performance and matrix sizes.

---

## Implementation Details

### cuBLAS Implementation (method=0)

Uses NVIDIA's `cublasSgemm` function:
- Highly optimized for all NVIDIA GPUs
- Automatically selects best algorithm
- Leverages Tensor Cores on newer GPUs (Volta+)
- Typically achieves 80-95% of theoretical peak performance

**Code location:** `benchmark.cu` (lines with `cublasSgemm`)

### Custom Tiled Implementation (method=1)

Hand-written CUDA kernel with tiling:
- **Tile size**: 16×16 (`TILE_WIDTH = 16`)
- **Shared memory**: Padded to reduce bank conflicts (`TILE_WIDTH + 1`)
- **Algorithm**: Standard tiled matrix multiplication
  - Load tiles from A and B into shared memory
  - Compute partial products
  - Accumulate results
- **Performance**: Typically 30-60% of peak (educational implementation)

**Code location:** `gemm.cu` (`matrixMulKernel`)

**Key optimizations:**
- Shared memory usage to reduce global memory access
- Bank conflict avoidance with padding
- Thread coarsening within tiles

---

## Output Format

### CSV Columns (52 features)

#### GPU Hardware Specifications
- `gpu_name`: GPU model name (e.g., "NVIDIA RTX 4090")
- `device_id`: CUDA device ID
- `cc_major`, `cc_minor`: Compute capability (e.g., 8.9)
- `sm_count`: Number of streaming multiprocessors
- `l2_size_bytes`: L2 cache size
- `shared_mem_per_sm`: Shared memory per SM (bytes)
- `total_global_mem`: Total GPU memory (bytes)
- `mem_clock_khz`: Memory clock frequency (kHz)
- `mem_bus_width`: Memory bus width (bits)
- `peak_mem_bandwidth_GBps`: Theoretical peak bandwidth
- `base_clock_mhz`: Base GPU clock (MHz)
- `boost_clock_mhz`: Boost GPU clock (MHz)
- `peak_flops_fp32_GFLOPs`: Theoretical peak FLOPS (FP32)

#### Workload Parameters
- `M`, `N`, `K`: Matrix dimensions
- `algorithm`: "cuBLAS" or "Custom-MM"
- `precision`: "FP32"
- `driver_version`: NVIDIA driver version
- `cuda_runtime_version`: CUDA runtime version

#### Performance Measurements
- `time_ms_mean`: Mean execution time (milliseconds)
- `time_ms_median`: Median execution time
- `time_ms_p95`: 95th percentile execution time
- `time_ms_stddev`: Standard deviation of execution time
- `gflops_mean`: Mean performance (GFLOPS)
- `gflops_median`: Median performance (GFLOPS)
- `gflops_p95`: 95th percentile performance (GFLOPS)

#### Efficiency Metrics
- `memory_throughput_GBps`: Measured memory throughput
- `memory_efficiency_pct`: Memory efficiency (%)
- `compute_efficiency_pct`: Theoretical compute efficiency (%)
- `gops_over_peak_mean`: Ratio of achieved to peak GFLOPS

#### Derived Features
- `total_ops`: Total floating-point operations (2×M×N×K)
- `total_io_bytes`: Total memory I/O (bytes)
- `arithmetic_intensity`: FLOPs per byte
- `repeats`: Number of timing repeats (10)
- `inner_iters`: Iterations per repeat (100)
- `seed`: Random seed for reproducibility

#### Kernel Configuration (Custom Tiled only)
- `tile_width`: Tile dimension (16)
- `block_x`, `block_y`: Thread block dimensions
- `grid_x`, `grid_y`: Grid dimensions
- `shared_mem_per_block`: Shared memory per block (bytes)
- `registers_per_thread`: Registers used per thread

#### Runtime Metrics
- `actual_clock_mhz`: Measured GPU clock during execution
- `actual_mem_clock_mhz`: Measured memory clock
- `temperature_c`: GPU temperature (Celsius)
- `power_watts`: GPU power consumption (Watts)
- `peak_flops_fp32_GFLOPs_actual`: Peak FLOPS at actual clocks
- `compute_efficiency_actual_pct`: Efficiency vs actual clocks

---

## Understanding the Results

### Matrix Size Distribution

Matrix dimensions (M, N, K) are sampled using a **quadratic distribution**:

```python
size = 128 + (random() ** 2) * (8192 - 128)
```

**Characteristics:**
- **Range**: 128 to 8192
- **Distribution**: Heavily biased toward smaller sizes
- **Rationale**: Smaller matrices are more common in practice (DNNs, etc.)

**Example distribution:**
- ~70% of samples: 128-2048
- ~20% of samples: 2048-4096
- ~10% of samples: 4096-8192

### Key Metrics to Analyze

1. **GFLOPS (gflops_mean)**
   - Measures computational throughput
   - Higher is better
   - cuBLAS typically achieves 70-95% of peak
   - Custom kernel typically achieves 30-60% of peak

2. **Memory Throughput (memory_throughput_GBps)**
   - Measures memory bandwidth utilization
   - Compare to `peak_mem_bandwidth_GBps`
   - Important for memory-bound operations (small matrices)

3. **Arithmetic Intensity (arithmetic_intensity)**
   - FLOPs per byte of data transferred
   - Higher values → compute-bound (good for optimization)
   - Lower values → memory-bound (limited by bandwidth)
   - Formula: `(2×M×N×K) / (4×(M×K + K×N + M×N))`

4. **Compute Efficiency (compute_efficiency_pct)**
   - Percentage of peak GFLOPS achieved
   - Indicates how well the kernel utilizes the GPU

### Performance Analysis Tips

**Compare implementations:**
```bash
# Run both implementations
./benchmark 0 1000 cublas_compare
./benchmark 1 1000 custom_compare

# Analyze results (create your own analysis script)
python analyze_results.py
```

**Look for:**
- cuBLAS should outperform custom kernel by 2-3×
- Performance should increase with matrix size (better cache utilization)
- Large matrices should achieve higher compute efficiency
- Small matrices are often memory-bound

---

## Troubleshooting

### Common Issues

**1. CUDA not found**
```
nvcc: command not found
```
**Solution:** Load CUDA module or add to PATH:
```bash
module load cuda-12.4
# OR
export PATH=/usr/local/cuda/bin:$PATH
```

**2. cuBLAS linking error**
```
undefined reference to `cublasSgemm`
```
**Solution:** Ensure `-lcublas` flag is included:
```bash
nvcc benchmark.cu -lcublas -o benchmark
```

**3. Out of memory**
```
CUDA error: out of memory
```
**Solution:** System is generating very large matrices. This is rare with the quadratic distribution, but if it happens, reduce `num_samples` or modify the size range in the code.

**4. Invalid device**
```
CUDA error: invalid device
```
**Solution:** No NVIDIA GPU detected. Run on a machine with an NVIDIA GPU.

**5. File already exists**
The benchmark **appends** to existing CSV files. To start fresh:
```bash
rm benchmark_results_*.csv
```

---

## Data Collection Best Practices

### For Machine Learning Training

1. **Collect diverse samples:**
   ```bash
   # Collect 10k samples from each implementation
   ./benchmark 0 10000 cublas_$(hostname)
   ./benchmark 1 10000 custom_$(hostname)
   ```

2. **Use different GPUs:**
   - Run on multiple GPU architectures (Pascal, Volta, Ampere, Ada, Hopper)
   - This improves model generalization
   ```bash
   # On RTX 3090
   ./benchmark 0 10000 rtx3090_cublas

   # On A100
   ./benchmark 0 10000 a100_cublas

   # On H100
   ./benchmark 0 10000 h100_cublas
   ```

3. **Set random seeds for reproducibility:**
   ```bash
   ./benchmark 0 10000 run1 42
   ./benchmark 0 10000 run2 123
   ./benchmark 0 10000 run3 456
   ```

4. **Check for consistency:**
   - Run multiple times and verify similar distributions
   - Check that no samples have anomalous timing values

5. **Monitor progress:**
   ```bash
   # Watch file size grow
   watch -n 5 ls -lh benchmark_results_*.csv

   # Monitor latest entries
   tail -f benchmark_results_*.csv
   ```

### Recommended Sample Sizes

- **Quick test**: 100 samples (~2 minutes)
- **Development**: 1,000 samples (~15 minutes)
- **Training dataset**: 10,000-20,000 samples (~3-7 hours)
- **Production dataset**: 30,000+ samples (~10+ hours)

---

## Next Steps

After collecting data:

1. **Copy to model training directory:**
   ```bash
   cp benchmark_results_*.csv ../../model_train/data/
   ```

2. **Train ML models:**
   ```bash
   cd ../../model_train
   python train_gemm_model_v1.py --dataset data/benchmark_results_cublas_N10000.csv
   ```

3. **Analyze results:**
   - Use the generated visualizations
   - Check feature importance
   - Evaluate model performance

---

## References

- **CUDA C Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **cuBLAS Documentation**: https://docs.nvidia.com/cuda/cublas/
- **Matrix Multiplication Tutorial**: https://siboehm.com/articles/22/CUDA-MMM
- **Dataset Collection Guide**: [../README.md](../README.md)
- **Model Training Guide**: [../../model_train/README.md](../../model_train/README.md)
