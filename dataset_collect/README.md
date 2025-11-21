# Dataset Collection

This directory contains CUDA benchmarking code for collecting performance data from GPU kernels. The collected data is used to train machine learning models for performance prediction.

## Table of Contents

- [Overview](#overview)
- [Directory Structure](#directory-structure)
- [Quick Start](#quick-start)
- [Detailed Instructions](#detailed-instructions)
- [Output Data Format](#output-data-format)
- [Best Practices](#best-practices)

---

## Overview

This directory provides benchmarking tools for two computational kernels:

1. **GEMM** (General Matrix Multiply) - Compute-bound matrix multiplication
2. **NTT** (Number Theoretic Transform) - Memory-bound number-theoretic transform

Each benchmark:
- Runs on NVIDIA GPUs with CUDA support
- Collects comprehensive performance metrics
- Outputs CSV files for ML training
- Supports checkpointing for long runs

---

## Directory Structure

```
dataset_collect/
├── gemm_benchmark.cu          # GEMM benchmarking main code
├── gemm.cu                    # GEMM kernel implementations
├── ntt_benchmark.cu           # NTT benchmarking main code
├── ntt.cu                     # NTT kernel implementation
├── Makefile                   # Unified build configuration
└── README.md                  # This file
```

---

## Quick Start

### 1. Environment Setup

```bash
# Load CUDA module (on NYU CIMS CUDA clusters)
module load cuda-12.4

# Verify CUDA version
# This benchmark expected to run under CUDA 12.x, modify the code if you want to run under CUDA 13.0
nvcc --version
```

### 2. Compile Benchmarks

```bash
# Compile both benchmarks
# nvcc -O3 -std=c++17 -lcublas -lnvidia-ml
make all

# Or compile individually
make gemm
make ntt
```

### 3. Run Benchmarks

```bash
# Run GEMM benchmark (cuBLAS, 1000 samples)
./gemm_benchmark 0 1000 cublas_test

# Run NTT benchmark (5000 samples)
./ntt_benchmark 5000 ntt_test
```

### 4. Collect Output

```bash
# GEMM output: benchmark_results_cublas_test_N1000_S{seed}.csv
# NTT output: benchmark_results_ntt_test_N5000_S{seed}.csv

# Copy to model training directory
cp benchmark_results_*.csv ../model_train/data/
```

---

## Detailed Instructions

### Compilation

```bash
# Compile both benchmarks
# nvcc -O3 -std=c++17 -lcublas -lnvidia-ml
make all

# Compile only GEMM
make gemm

# Compile only NTT
make ntt

# Clean all compiled files
make clean
```

**Build outputs:**
- `gemm_benchmark` - GEMM executable
- `ntt_benchmark` - NTT executable

### GEMM Benchmarking

The GEMM benchmark tests matrix multiplication performance with two implementations.

**Usage:**
```bash
./gemm_benchmark <method> <num_samples> [output_name] [seed]
```

**Parameters:**
- `method`: `0` = cuBLAS (optimized), `1` = Custom 16×16 tiled kernel
- `num_samples`: Number of configurations to benchmark
- `output_name`: Optional output file prefix
- `seed`: Optional random seed for reproducibility

**Examples:**
```bash
# cuBLAS implementation with 10k samples
./gemm_benchmark 0 10000 gemm_run1

# Custom tiled implementation with 10k samples
./gemm_benchmark 1 10000 gemm_run1

# With fixed random seed
./gemm_benchmark 0 10000 gemm_run1 42

# Default naming (no output_name)
./gemm_benchmark 0 1000
```

**Output features:** 52 columns including GPU specs, matrix dimensions (M, N, K), timing metrics, GFLOPS, efficiency

### NTT Benchmarking

The NTT benchmark tests number-theoretic transform performance using Cooley-Tukey radix-2 algorithm.

**Usage:**
```bash
./ntt_benchmark <num_samples> [output_name] [N] [repeats] [inner_iters] [seed]
```

**Parameters:**
- `num_samples`: Number of configurations to benchmark
- `output_name`: Optional output file prefix
- `N`: Optional fixed NTT size (power of 2). If not specified, random sizes will be used
- `repeats`: Optional number of repetitions (default: 10)
- `inner_iters`: Optional inner iterations (default: 50)
- `seed`: Optional random seed for reproducibility (default: random)

**Examples:**
```bash
# Run with 5000 samples (random N sizes)
./ntt_benchmark 5000 ntt_run1

# With fixed random seed
./ntt_benchmark 5000 ntt_run1 42

# Default naming
./ntt_benchmark 1000

# Fixed N=65536 with custom settings
./ntt_benchmark 500 test 65536 10 50 42

# Seed only (treats numeric arg as seed if not power of 2)
./ntt_benchmark 1000 42
```

**Output features:** 36 columns including GPU specs, NTT size (N), timing metrics, butterfly operations, modular operations

---

## Output Data Format

### File Naming Convention

Both benchmarks use the following naming pattern:

```
benchmark_results_[output_name_]N{num_samples}[_S{seed}].csv
```

**Examples:**
```bash
# GEMM with default name
./gemm_benchmark 0 1000
→ benchmark_results_N1000_S{random}.csv

# GEMM with custom name
./gemm_benchmark 0 1000 cublas_test
→ benchmark_results_cublas_test_N1000_S{random}.csv

# GEMM with custom name and seed
./gemm_benchmark 0 1000 cublas_test 42
→ benchmark_results_cublas_test_N1000_S42.csv

# NTT with custom name
./ntt_benchmark 5000 ntt_experiment
→ benchmark_results_ntt_experiment_N5000_S{random}.csv

# NTT with custom name and seed
./ntt_benchmark 5000 ntt_experiment 123
→ benchmark_results_ntt_experiment_N5000_S123.csv
```

### CSV Format

**GEMM output (52 features):**
- **GPU hardware**: name, compute capability, SM count, memory specs, cache sizes, theoretical peak FLOPS/bandwidth
- **Kernel config**: M, N, K dimensions, precision (FP32), algorithm type, grid/block dimensions
- **Execution timing**: mean, median, 95th percentile, stddev (via CUDA event timing)
- **Derived metrics**: total FLOPs (2MNK), theoretical I/O bytes, arithmetic intensity (FLOPs/byte)
- **Runtime metrics**: actual clock frequencies, temperature, power consumption (via NVML)

**NTT output (36 features):**
- **GPU hardware**: Same architectural descriptors as GEMM
- **Kernel config**: Transform size N, modulus p, primitive root ω, butterfly stages (log₂ N)
- **Execution timing**: Same timing methodology as GEMM
- **Derived metrics**: butterfly operations (N/2 × log₂ N), modular operations (3 × butterflies), theoretical memory traffic (8N × log₂ N bytes)

---

## Best Practices

### For Machine Learning Training

1. **Collect diverse samples across multiple GPUs:**
   ```bash
   # On GPU 1 (e.g., RTX 2080 Ti)
   ./gemm_benchmark 0 10000 rtx2080ti_cublas 42

   # On GPU 2 (e.g., TITAN V)
   ./gemm_benchmark 0 10000 titanv_cublas 42

   # On GPU 3 (e.g., RTX 4070)
   ./gemm_benchmark 0 10000 rtx4070_cublas 42
   ```

2. **Use descriptive output names:**
   ```bash
   # Good naming (includes GPU, algorithm, workload)
   ./gemm_benchmark 0 20000 rtx4090_cublas_fp32
   ./ntt_benchmark 10000 a100_ntt_large

   # Avoid generic names
   ./gemm_benchmark 0 20000 test1
   ./ntt_benchmark 5000 run
   ```

3. **Set random seeds for reproducibility:**
   ```bash
   # Different seeds for different runs
   ./gemm_benchmark 0 10000 cublas_run1 42
   ./gemm_benchmark 0 10000 cublas_run2 123
   ./gemm_benchmark 0 10000 cublas_run3 456
   ```

4. **Monitor progress:**
   - Benchmarks checkpoint every 100 samples
   - Check CSV files periodically during long runs
   - Use `tail -f` to monitor:
     ```bash
     tail -f benchmark_results_*.csv
     ```

### Recommended Sample Sizes

**For development/testing:**
- Quick test: 100 samples (~2 minutes)
- Development: 1,000 samples (~15 minutes)

**For ML training:**
- Small dataset: 5,000 samples (~1 hour)
- Medium dataset: 10,000 samples (~2-3 hours)
- Large dataset: 20,000 samples (~5-7 hours)
- Production dataset: 30,000+ samples (~10+ hours)

### Complete Data Collection Workflow

```bash
# 1. Set up environment
cd dataset_collect
module load cuda-12.4

# 2. Compile both benchmarks
make all

# 3. Run GEMM benchmarks (both implementations)
./gemm_benchmark 0 20000 cublas_$(hostname) 42
./gemm_benchmark 1 20000 custom_$(hostname) 42

# 4. Run NTT benchmark
./ntt_benchmark 10000 ntt_$(hostname) 42

# 5. Verify outputs
ls -lh benchmark_results_*.csv

# 6. Copy to model training directory
mkdir -p ../model_train/data
cp benchmark_results_*.csv ../model_train/data/

# 7. Merge multiple runs (if needed)
cd ../model_train/data
cat gemm_run1.csv > gemm_combined.csv
tail -n +2 gemm_run2.csv >> gemm_combined.csv  # Skip header
tail -n +2 gemm_run3.csv >> gemm_combined.csv
```

---

## Troubleshooting

### Common Issues

**CUDA not found:**
```bash
# Load CUDA module
module load cuda-12.4

# Or add to PATH manually
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Compilation errors:**
```bash
# Clean and rebuild
make clean
make all

# Check CUDA version
nvcc --version
```

**Out of memory:**
```bash
# Reduce number of samples
./gemm_benchmark 0 1000  # Instead of 10000
./ntt_benchmark 1000     # Instead of 5000
```

**Benchmarks run too slowly:**
```bash
# Check GPU utilization
nvidia-smi

# Reduce samples for quick testing
./gemm_benchmark 0 100
./ntt_benchmark 100
```

**Permission denied for executables:**
```bash
# Make executables runnable
chmod +x gemm_benchmark ntt_benchmark
```

---

## Algorithm Details

### GEMM Sampling Strategy

Matrix dimensions sampled stochastically:
```
M, N, K ~ Uniform²[128, 8192]
```
Quadratic transformation biases sampling toward larger matrices that expose compute-bound behavior while including smaller configurations.

### NTT Sampling Strategy

Transform sizes sampled logarithmically:
```
k ~ DiscreteUniform[15, 24]
N = 2^k
```
Range: 32,768 to 16,777,216 elements, capturing cache-resident to DRAM-bound execution.

---

## Next Steps

After collecting data:

1. **Move to model training:**
   ```bash
   cd ../model_train
   ```

2. **Download pre-collected datasets (optional):**
   ```bash
   python download_dataset.py
   ```

3. **Train models:**
   ```bash
   python train_gemm_model_v1.py --dataset data/gemm_dataset_train_20k.csv
   python train_ntt_model_v1.py --dataset data/ntt_dataset_train_20k.csv
   ```

See [../model_train/README.md](../model_train/README.md) for training instructions.

---

## References

- **Model Training**: [../model_train/README.md](../model_train/README.md)
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
- **cuBLAS Documentation**: https://docs.nvidia.com/cuda/cublas/
