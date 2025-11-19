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

1. **GEMM** (General Matrix Multiply) - Matrix multiplication benchmarking
2. **NTT** (Number Theoretic Transform) - Number-theoretic transform benchmarking

Each benchmark:
- Runs on NVIDIA GPUs with CUDA support
- Collects comprehensive performance metrics
- Outputs CSV files for ML training
- Supports checkpointing for long runs

---

## Directory Structure

```
dataset_collect/
├── GEMM/                      # GEMM benchmarking
│   ├── benchmark.cu           # Main benchmarking code
│   ├── gemm.cu                # GEMM kernel implementations
│   ├── Makefile               # Build configuration
│   └── README.md              # Detailed GEMM instructions
├── NTT/                       # NTT benchmarking
│   ├── benchmark.cu           # Main benchmarking code
│   ├── ntt.cu                 # NTT kernel implementation
│   ├── Makefile               # Build configuration
│   └── README.md              # Detailed NTT instructions
└── README.md                  # This file
```

---

## Quick Start

### 1. Environment Setup

```bash
# Load CUDA module (on HPC systems)
module load cuda-12.4

# Verify CUDA installation
nvcc --version
```

### 2. Compile Benchmarks

```bash
# Compile GEMM benchmark
cd GEMM
make
cd ..

# Compile NTT benchmark
cd NTT
make
cd ..
```

### 3. Run Benchmarks

```bash
# Run GEMM benchmark (cuBLAS, 1000 samples)
cd GEMM
./benchmark 0 1000 cublas_test

# Run NTT benchmark (5000 samples)
cd NTT
./benchmark 5000 ntt_test
```

### 4. Collect Output

```bash
# GEMM output: benchmark_results_cublas_test_N1000.csv
# NTT output: benchmark_results_ntt_test_N5000.csv

# Copy to model training directory
cp GEMM/benchmark_results_*.csv ../model_train/data/
cp NTT/benchmark_results_*.csv ../model_train/data/
```

---

## Detailed Instructions

### GEMM Benchmarking

The GEMM benchmark tests matrix multiplication performance.

**See [GEMM/README.md](GEMM/README.md) for complete details.**

**Quick reference:**
```bash
cd GEMM

# Compile
make

# Run cuBLAS implementation with 10k samples
./benchmark 0 10000 cublas_run1

# Run custom tiled implementation with 10k samples
./benchmark 1 10000 custom_run1

# Output: benchmark_results_cublas_run1_N10000.csv
#         benchmark_results_custom_run1_N10000.csv
```

**Methods:**
- `0` = cuBLAS (NVIDIA's optimized library)
- `1` = Custom Tiled (16×16 shared-memory kernel)

**Output features:** 52 columns including GPU specs, matrix dimensions, timing metrics, GFLOPS, efficiency

### NTT Benchmarking

The NTT benchmark tests number-theoretic transform performance.

**See [NTT/README.md](NTT/README.md) for complete details.**

**Quick reference:**
```bash
cd NTT

# Compile
make

# Run with 5000 samples
./benchmark 5000 ntt_run1

# Output: benchmark_results_ntt_run1_N5000.csv
```

**Output features:** 36 columns including GPU specs, NTT size, timing metrics, throughput

---

## Output Data Format

### File Naming Convention

Both benchmarks now support custom output file naming:

**GEMM:**
```bash
./benchmark <method> <num_samples> [output_name] [seed]

# Default: benchmark_results_N{num_samples}.csv
# With name: benchmark_results_{output_name}_N{num_samples}.csv
# With seed: benchmark_results_{output_name}_N{num_samples}_S{seed}.csv
```

**NTT:**
```bash
./benchmark <num_samples> [output_name] [seed]

# Default: benchmark_results_N{num_samples}.csv
# With name: benchmark_results_{output_name}_N{num_samples}.csv
# With seed: benchmark_results_{output_name}_N{num_samples}_S{seed}.csv
```

**Examples:**
```bash
# GEMM with default name
./benchmark 0 1000
# Output: benchmark_results_N1000.csv

# GEMM with custom name
./benchmark 0 1000 cublas_test
# Output: benchmark_results_cublas_test_N1000.csv

# GEMM with custom name and seed
./benchmark 0 1000 cublas_test 42
# Output: benchmark_results_cublas_test_N1000_S42.csv

# NTT with custom name
./benchmark 5000 ntt_experiment
# Output: benchmark_results_ntt_experiment_N5000.csv
```

### CSV Format

**GEMM output (52 features):**
- GPU hardware: name, compute capability, SM count, memory specs, clocks
- Workload: M, N, K dimensions, algorithm (cuBLAS or Custom-MM)
- Performance: time statistics, GFLOPS, memory throughput, efficiency
- Derived: total ops, I/O bytes, arithmetic intensity

**NTT output (36 features):**
- GPU hardware: name, compute capability, SM count, memory specs
- Workload: N (size), modulus, primitive root
- Performance: time statistics, modops/sec, memory throughput
- Derived: butterflies, modular operations, theoretical bytes

---

## Best Practices

### For Machine Learning Training

1. **Collect diverse samples:**
   ```bash
   # Run on multiple GPUs (different architectures)
   # Pascal, Volta, Ampere, Ada Lovelace, Hopper

   # On GPU 1 (e.g., RTX 3090)
   ./benchmark 0 10000 rtx3090_cublas

   # On GPU 2 (e.g., A100)
   ./benchmark 0 10000 a100_cublas

   # On GPU 3 (e.g., RTX 4090)
   ./benchmark 0 10000 rtx4090_cublas
   ```

2. **Use descriptive names:**
   ```bash
   # Good naming
   ./benchmark 0 20000 rtx4090_cublas_fp32
   ./benchmark 5000 a100_ntt_large

   # Avoid generic names
   ./benchmark 0 20000 test1
   ./benchmark 5000 run
   ```

3. **Set random seeds for reproducibility:**
   ```bash
   # Different seeds for different runs
   ./benchmark 0 10000 cublas_run1 42
   ./benchmark 0 10000 cublas_run2 123
   ./benchmark 0 10000 cublas_run3 456
   ```

4. **Monitor progress:**
   - Benchmarks checkpoint every 1,000 samples
   - Check CSV files periodically during long runs
   - Use `tail -f` to monitor progress:
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

### Data Collection Workflow

**Complete workflow for collecting training data:**

```bash
# 1. Set up environment
cd dataset_collect
module load cuda-12.4

# 2. Compile both benchmarks
cd GEMM && make && cd ..
cd NTT && make && cd ..

# 3. Run GEMM benchmarks (both implementations)
cd GEMM
./benchmark 0 20000 cublas_$(hostname) 42
./benchmark 1 20000 custom_$(hostname) 42
cd ..

# 4. Run NTT benchmark
cd NTT
./benchmark 10000 ntt_$(hostname) 42
cd ..

# 5. Verify outputs
ls -lh GEMM/benchmark_results_*.csv
ls -lh NTT/benchmark_results_*.csv

# 6. Copy to model training directory
mkdir -p ../model_train/data
cp GEMM/benchmark_results_*.csv ../model_train/data/
cp NTT/benchmark_results_*.csv ../model_train/data/

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

# Or add to PATH
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

**Compilation errors:**
```bash
# Clean and rebuild
make clean
make

# Check CUDA version compatibility
nvcc --version
```

**Out of memory:**
```bash
# Reduce number of samples
./benchmark 0 1000  # Instead of 10000

# Or modify code to use smaller matrices
```

**Benchmarks run too slowly:**
```bash
# Check GPU is not busy
nvidia-smi

# Reduce number of samples for testing
./benchmark 0 100  # Quick test
```

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

- **GEMM Benchmarking**: [GEMM/README.md](GEMM/README.md)
- **NTT Benchmarking**: [NTT/README.md](NTT/README.md)
- **Model Training**: [../model_train/README.md](../model_train/README.md)
- **CUDA Programming Guide**: https://docs.nvidia.com/cuda/cuda-c-programming-guide/
