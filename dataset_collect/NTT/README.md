# NTT Benchmarking Tool

This directory contains a CUDA-based benchmarking tool for **Number Theoretic Transform (NTT)** operations. The tool collects comprehensive performance metrics from different GPU architectures to train machine learning models that predict NTT performance.

## Table of Contents

- [Quick Start](#quick-start)
- [What is NTT?](#what-is-ntt)
- [Build Instructions](#build-instructions)
- [Usage](#usage)
- [Output File Naming Convention](#output-file-naming-convention)
- [Output Features](#output-features)
- [Implementation Details](#implementation-details)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# Navigate to NTT directory
cd dataset_collect/NTT

# Build the benchmark tool
make

# Run with default settings (1000 random-sized samples)
./benchmark 1 1000

# Run with custom output name
./benchmark 1 1000 ntt_v100_test

# Run with fixed size and reproducible seed
./benchmark 1 500 65536 10 50 42
```

## What is NTT?

**Number Theoretic Transform (NTT)** is the modular arithmetic version of the FFT (Fast Fourier Transform), operating in finite fields instead of complex numbers. It's widely used in:

- **Cryptography**: Lattice-based cryptography (e.g., CRYSTALS-Kyber, Dilithium)
- **Homomorphic Encryption**: Polynomial multiplication in encrypted domains
- **Large Integer Multiplication**: Efficient multiplication of multi-precision integers
- **Signal Processing**: When working with modular arithmetic

### Algorithm Complexity

- **Butterflies**: `(N/2) × log₂(N)` butterfly operations
- **Modular Operations**: `3 × butterflies` (1 mul_mod + 1 add_mod + 1 sub_mod per butterfly)
- **Memory Access**: `~8 × N × log₂(N)` bytes (read + write per element per stage)

The NTT benchmark uses the **Cooley-Tukey radix-2 decimation-in-time** algorithm with:
- **Modulus**: 998244353 (a 30-bit prime with primitive root 3)
- **Primitive Root**: 3
- **Transform Sizes**: Powers of 2 from 2^15 to 2^24 (32,768 to 16,777,216 elements)

## Build Instructions

### Prerequisites

- NVIDIA GPU with CUDA support (compute capability ≥ 3.0)
- CUDA Toolkit (version 11.0 or later recommended)
- GCC/G++ compiler with C++17 support
- NVIDIA Management Library (NVML) for runtime metrics

### Compilation

```bash
# Build the benchmark executable
make

# Clean build artifacts
make clean
```

The build process compiles `benchmark.cu` with:
- Optimization level: `-O3`
- C++ standard: C++17
- Libraries: `-lcublas -lnvidia-ml`

### Verify Build

```bash
# Check if executable was created
ls -lh benchmark

# Test with minimal run
./benchmark 1 10
```

## Usage

### Command Syntax

```bash
./benchmark <method> <num_samples> [output_name] [N] [repeats] [inner_iters] [seed]
```

### Parameters

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `method` | Yes | - | Algorithm selection (1 = Custom-NTT) |
| `num_samples` | Yes | - | Number of benchmark samples to collect |
| `output_name` | No | "" | Optional custom name for output file |
| `N` | No | Random | Fixed NTT size (power of 2). If omitted, uses random sizes |
| `repeats` | No | 10 | Number of repetitions per sample |
| `inner_iters` | No | 50 | Inner iterations for timing precision |
| `seed` | No | Random | Random seed for reproducibility (0 = random) |

### Usage Examples

#### Example 1: Basic Random Sampling (Default Output)
```bash
./benchmark 1 1000
# Output: benchmark_results_N1000.csv
# 1000 samples with random N ∈ {2^15, 2^16, ..., 2^24}
```

#### Example 2: Custom Output Name
```bash
./benchmark 1 1000 v100_baseline
# Output: benchmark_results_v100_baseline_N1000.csv
# Useful for organizing results by GPU or test scenario
```

#### Example 3: Reproducible Run with Seed
```bash
./benchmark 1 500 experiment1 32768 10 50 42
# Output: benchmark_results_experiment1_N500_S42.csv
# N=32768 (2^15), seed=42, repeats=10, inner_iters=50
```

#### Example 4: Fixed Size Benchmarking
```bash
./benchmark 1 200 65536
# Output: benchmark_results_N200.csv
# 200 samples, all with N=65536 (2^16)
```

#### Example 5: High-Precision Benchmarking
```bash
./benchmark 1 100 precision_test 131072 20 100 12345
# Output: benchmark_results_precision_test_N100_S12345.csv
# N=131072 (2^17), 20 repeats, 100 inner iterations
```

### Understanding Random vs Fixed N Mode

**Random N Mode** (N not specified):
- Uniformly samples N from {2^15, 2^16, ..., 2^24}
- Better for training ML models (diverse problem sizes)
- Each sample may have different N

**Fixed N Mode** (N specified):
- All samples use the same N value
- Better for performance analysis of specific size
- Useful for comparing GPUs on same workload

## Output File Naming Convention

The benchmark automatically generates descriptive filenames based on your input parameters:

### Naming Pattern

```
benchmark_results_[output_name_]N{num_samples}[_S{seed}].csv
```

### Components

- **`benchmark_results_`**: Fixed prefix
- **`[output_name_]`**: Optional custom identifier (only if provided)
- **`N{num_samples}`**: Number of samples collected
- **`[_S{seed}]`**: Optional seed value (only if seed ≠ 0)
- **`.csv`**: File extension

### Naming Examples

| Command | Output Filename | Notes |
|---------|----------------|-------|
| `./benchmark 1 1000` | `benchmark_results_N1000.csv` | Default naming |
| `./benchmark 1 1000 rtx4090` | `benchmark_results_rtx4090_N1000.csv` | With output name |
| `./benchmark 1 500 test 65536 10 50 42` | `benchmark_results_test_N500_S42.csv` | With name and seed |
| `./benchmark 1 2000 v100_exp 32768 10 50 0` | `benchmark_results_v100_exp_N2000.csv` | Seed=0 omitted |

### Why Custom Names?

Use `output_name` to organize benchmarks by:
- **GPU Model**: `rtx4090`, `v100`, `a100`
- **Test Scenario**: `baseline`, `optimized`, `comparison`
- **Configuration**: `high_freq`, `low_power`, `overclocked`
- **Date/Version**: `2025jan`, `v2`, `batch1`

## Output Features

The benchmark collects **36 features** per sample, saved as CSV with the following columns:

### Static GPU Information (15 features)

| Feature | Description | Example |
|---------|-------------|---------|
| `gpu_name` | GPU model name | "NVIDIA RTX 4090" |
| `device_id` | CUDA device ID | 0 |
| `cc_major` | Compute capability major version | 8 |
| `cc_minor` | Compute capability minor version | 9 |
| `sm_count` | Number of streaming multiprocessors | 128 |
| `l2_size_bytes` | L2 cache size in bytes | 73400320 |
| `shared_mem_per_sm` | Shared memory per SM in bytes | 102400 |
| `total_global_mem` | Total global memory in bytes | 25757220864 |
| `mem_bandwidth_GBps_est` | Estimated memory bandwidth (GB/s) | 1008.0 |
| `peak_flops_fp32_GFLOPs_est` | Estimated peak FP32 GFLOPS | 82580.0 |
| `driver_version` | CUDA driver version | 12040 |
| `cuda_runtime_version` | CUDA runtime version | 12040 |
| `max_threads_per_sm` | Max threads per SM | 1536 |
| `max_threads_per_block` | Max threads per block | 1024 |
| `warp_size` | Warp size | 32 |

### Kernel Configuration (6 features)

| Feature | Description | Example |
|---------|-------------|---------|
| `N` | NTT size (number of elements) | 65536 |
| `modulus` | Modular arithmetic prime | 998244353 |
| `primitive_root` | Primitive root for NTT | 3 |
| `repeats` | Number of outer repetitions | 10 |
| `inner_iters` | Inner iterations for timing | 50 |
| `algorithm` | Implementation type | "Custom-NTT" |
| `seed` | Random seed used | 42 |

### Runtime Performance Metrics (8 features)

| Feature | Description | Unit |
|---------|-------------|------|
| `time_ms_mean` | Mean execution time | milliseconds |
| `time_ms_median` | Median execution time | milliseconds |
| `time_ms_p95` | 95th percentile execution time | milliseconds |
| `time_ms_stddev` | Standard deviation of execution time | milliseconds |
| `actual_clock_mhz` | Measured GPU core clock | MHz |
| `actual_mem_clock_mhz` | Measured memory clock | MHz |
| `temperature_c` | GPU temperature during run | Celsius |
| `power_watts` | GPU power consumption | Watts |

### Derived Performance Metrics (7 features)

| Feature | Description | Formula |
|---------|-------------|---------|
| `butterflies_total` | Total butterfly operations | `(N/2) × log₂(N)` |
| `modops_total` | Total modular operations | `butterflies_total × 3` |
| `modops_per_sec` | Modular ops throughput | `modops_total / (time_ms_mean / 1000)` |
| `bytes_total_theoretical` | Theoretical memory traffic | `8 × N × log₂(N)` bytes |
| `memory_throughput_GBps` | Achieved memory bandwidth | `(bytes_total / 1e9) / (time_ms_mean / 1000)` |
| `memory_efficiency_pct` | Memory bandwidth utilization | `(memory_throughput / mem_bandwidth_est) × 100` |

## Implementation Details

### NTT Algorithm

The benchmark implements the **Cooley-Tukey radix-2 decimation-in-time** NTT algorithm with:

1. **Bit-Reversal Permutation**: Reorder input array
2. **Butterfly Operations**: `log₂(N)` stages of butterfly computations
3. **Modular Arithmetic**: All operations mod 998244353
4. **Twiddle Factors**: Precomputed powers of primitive root

### CUDA Kernel Optimizations

- **Shared Memory**: Fast on-chip storage for intermediate results
- **Warp-Level Parallelism**: Efficient use of 32-thread warps
- **Memory Coalescing**: Aligned memory access patterns
- **Register Usage**: Minimize register pressure for high occupancy

### Timing Methodology

1. **Warmup**: 10 iterations to stabilize GPU clocks
2. **Timed Runs**: `repeats` × `inner_iters` total executions
3. **CUDA Events**: High-precision GPU timing
4. **Statistics**: Mean, median, 95th percentile, standard deviation
5. **Runtime Metrics**: NVML queries after timing (non-interfering)

### Checkpointing

The benchmark automatically saves progress every **100 samples**:
- Prevents data loss from crashes
- Allows resuming long runs
- Appends to existing CSV files

## Best Practices

### For Machine Learning Dataset Collection

1. **Use Random N Mode** for diverse training data:
   ```bash
   ./benchmark 1 10000 training_set
   ```

2. **Set Reproducible Seed** for validation sets:
   ```bash
   ./benchmark 1 2000 validation_set 0 10 50 42
   ```

3. **Collect Multiple GPUs** with descriptive names:
   ```bash
   # On RTX 4090
   ./benchmark 1 5000 rtx4090_train

   # On V100
   ./benchmark 1 5000 v100_train
   ```

### For Performance Analysis

1. **Use Fixed N Mode** for fair GPU comparison:
   ```bash
   ./benchmark 1 500 131072  # Same size across all GPUs
   ```

2. **Increase Precision** for low-variance measurements:
   ```bash
   ./benchmark 1 100 65536 20 100  # 20 repeats × 100 inner iterations
   ```

3. **Monitor Temperature** between runs:
   - Let GPU cool between intensive benchmarks
   - Thermal throttling affects performance consistency

### For Reproducibility

1. **Always Specify Seed** for reproducible research:
   ```bash
   ./benchmark 1 1000 experiment1 0 10 50 42
   ```

2. **Document Parameters** in filenames:
   ```bash
   ./benchmark 1 1000 n65k_r20_i100 65536 20 100 12345
   ```

3. **Record System State**:
   - GPU driver version
   - CUDA version
   - System temperature and load
   - Power/thermal settings

## Troubleshooting

### Build Issues

**Error: `nvcc: command not found`**
```bash
# Check CUDA installation
which nvcc
echo $PATH

# Add CUDA to PATH (adjust version as needed)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Error: `cannot find -lnvidia-ml`**
```bash
# NVML library missing - check CUDA installation
ls /usr/local/cuda/lib64/libnvidia-ml.so

# If missing, install full CUDA toolkit
```

### Runtime Issues

**Error: `NVML not available - runtime metrics will be set to 0`**

This is a **warning**, not an error. The benchmark continues with:
- `actual_clock_mhz = 0`
- `actual_mem_clock_mhz = 0`
- `temperature_c = 0`
- `power_watts = 0`

**Solution**: Install proper NVIDIA drivers with NVML support.

**Error: `out of memory`**

Large N values (2^24 = 16M elements) require significant GPU memory:
- Each element: 4 bytes (uint32_t)
- N=2^24: ~64 MB per array (+ twiddle factors)

**Solution**: Reduce max N in random mode or use smaller fixed N.

**Performance Inconsistency**

Possible causes:
1. **Thermal Throttling**: GPU overheating reduces clocks
2. **Background Processes**: Other GPU workloads interfere
3. **Power Limits**: GPU hitting power cap

**Solution**:
```bash
# Check GPU status
nvidia-smi

# Monitor during run
watch -n 1 nvidia-smi
```

### Data Quality Issues

**High Standard Deviation** (`time_ms_stddev`)

Indicates timing variance. Improve by:
```bash
# Increase repeats and inner_iters
./benchmark 1 100 65536 20 100
```

**Memory Efficiency > 100%**

Theoretical model underestimates actual memory traffic. This can happen with:
- Cache effects
- Bandwidth boost technologies
- Measurement precision issues

**Zero Runtime Metrics**

NVML not available (see above). Non-critical for ML training.

## File Structure

```
NTT/
├── benchmark.cu          # Main benchmark implementation
├── ntt.cu               # NTT kernel implementation
├── Makefile             # Build configuration
├── README.md            # This file
└── benchmark_results_*.csv  # Output data files
```

## Next Steps

After collecting NTT benchmark data:

1. **Move Data to Training Directory**:
   ```bash
   cp benchmark_results_*.csv ../../model_train/data/
   ```

2. **Train Prediction Models**:
   ```bash
   cd ../../model_train
   python train_ntt_model_v1.py --dataset ./data/benchmark_results_*.csv
   ```

3. **Evaluate on Unseen GPUs**:
   ```bash
   python evaluate_ntt_unseen_gpu.py --model output/output_NTT_*/xgboost_model.pkl
   ```

See the main [README](../../README.md) for complete workflow documentation.

## References

- **NTT Algorithm**: Cooley-Tukey FFT adapted for modular arithmetic
- **Modulus Choice**: 998244353 = 119 × 2^23 + 1 (30-bit NTT-friendly prime)
- **Applications**: [CRYSTALS-Kyber](https://pq-crystals.org/kyber/), [SEAL](https://github.com/microsoft/SEAL)

---

For questions or issues, please refer to the main project [README](../../README.md) or check the [troubleshooting section](#troubleshooting).
