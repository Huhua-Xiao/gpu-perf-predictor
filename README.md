# GPU Performance Predictor

A machine learning-based performance prediction system for GPU computational kernels. This project benchmarks GEMM (General Matrix Multiply) and NTT (Number Theoretic Transform) kernels on various GPUs and trains ML models to predict execution times based on workload and hardware characteristics.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
  - [1. Dataset Download](#1-dataset-download)
  - [2. Model Training](#2-model-training)
  - [3. Making Predictions](#3-making-predictions)
  - [4. Evaluation on Unseen GPU](#4-evaluation-on-unseen-gpu)
- [CUDA Benchmarking](#cuda-benchmarking)
- [Features](#features)
- [Model Performance](#model-performance)
- [Requirements](#requirements)
- [Contributing](#contributing)

---

## Overview

This project provides a complete workflow for **GPU performance prediction** using machine learning:

1. **CUDA Benchmarking Tools** (`dataset_collect/`): Collect performance data for GEMM and NTT kernels across different GPUs
2. **ML Training Pipeline** (`model_train/`): Train XGBoost and SVR models to predict kernel execution times
3. **Prediction Interface**: Simple command-line tool for making predictions with trained models
4. **Comprehensive Analysis**: Evaluation metrics, feature importance analysis, and visualization

### Complete Workflow

```
1. Data Collection          2. Model Training           3. Prediction
   (Optional)                  (Required)                 (Optional)

┌─────────────────┐       ┌─────────────────┐       ┌─────────────────┐
│ dataset_collect/│       │ model_train/    │       │ predict.py      │
│                 │       │                 │       │                 │
│ Run benchmarks ──┐      │ Download data   │       │ Load model      │
│ on your GPU     │└─────>│ from HuggingFace│       │                 │
│                 │  OR   │                 │       │ Predict times   │
│ Generate CSVs   │       │ Train XGBoost   │       │ for new configs │
└─────────────────┘       │ and SVR models  │       └─────────────────┘
                          │                 │
                          │ Evaluate on     │
                          │ unseen GPUs     │
                          └─────────────────┘
```

**You can:**
- **Just train models**: Download pre-collected datasets and train (most users)
- **Collect + train**: Benchmark your GPU, then train on your data
- **Full workflow**: Collect data, train models, make predictions

### Key Features

- **Flexible file naming**: Organize benchmark results by GPU model, experiment, or seed
- **Automatic logging**: Training runs saved with timestamps
- **Configurable outputs**: Auto-detected directories based on dataset size
- **Feature engineering**: Combined compute capability handles cc_minor=0 gracefully
- **Regularization**: L1/L2 and min_child_weight prevent overfitting
- **Clean organization**: All outputs under `model_train/output/`
- **Comprehensive docs**: Step-by-step guides in each directory

---

## Project Structure

```
gpu-perf-predictor/
├── dataset_collect/                # CUDA benchmarking tools
│   ├── README.md                   # Data collection guide
│   ├── gemm_benchmark.cu           # GEMM benchmarking code
│   ├── ntt_benchmark.cu            # NTT benchmarking code
│   ├── gemm.cu                     # GEMM kernel implementations
│   ├── ntt.cu                      # NTT kernel implementation
│   └── Makefile                    # Build configuration
├── model_train/                    # ML training pipeline
│   ├── README.md                   # Training workflow guide
│   ├── data/                       # Datasets (created by download_dataset.py)
│   │   ├── gemm_dataset_train.csv
│   │   ├── gemm_dataset_train_20k.csv
│   │   ├── gemm_dataset_train_30k.csv
│   │   ├── gemm_dataset_eval.csv
│   │   ├── ntt_dataset_train.csv
│   │   ├── ntt_dataset_train_20k.csv
│   │   └── ntt_dataset_eval.csv
│   ├── output/                     # Training/evaluation outputs (gitignored)
│   │   ├── output_GEMM_10k/
│   │   ├── output_GEMM_20k/
│   │   ├── output_GEMM_30k/
│   │   ├── output_NTT_10k/
│   │   ├── output_NTT_20k/
│   │   ├── output_GEMM_eval_20k/
│   │   └── output_NTT_eval_20k/
│   ├── archive/                    # Archived old scripts
│   │   └── README.md               # Migration guide
│   ├── train_gemm_model_v1.py      # GEMM training (current)
│   ├── train_ntt_model_v1.py       # NTT training (current)
│   ├── evaluate_gemm_unseen_gpu.py # Evaluation on unseen GPU
│   ├── evaluate_ntt_unseen_gpu.py  # Evaluation on unseen GPU
│   ├── predict.py                  # Prediction interface
│   ├── download_dataset.py         # Dataset downloader
│   └── requirements.txt            # Python dependencies
└── README.md                       # This file
```

---

## Quick Start

### 1. Clone and Setup

```bash
cd gpu-perf-predictor/model_train

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Datasets

```bash
python download_dataset.py
```

This downloads datasets from HuggingFace and places them in `model_train/data/`.

### 3. Train Models

```bash
# Train GEMM model on 20k dataset (auto-detects output directory)
python train_gemm_model_v1.py --dataset data/gemm_dataset_train_20k.csv

# Train NTT model on 20k dataset
python train_ntt_model_v1.py --dataset data/ntt_dataset_train_20k.csv
```

Output will be saved to:
- `output/output_GEMM_20k/` for GEMM
- `output/output_NTT_20k/` for NTT

Each directory contains:
- `training_log_TIMESTAMP.txt` - Complete training log
- `*.joblib` - Trained models and scaler
- `*.png` - Visualizations
- `*.csv` - Prediction results

### 4. Make Predictions

```bash
# Predict using trained GEMM model
python predict.py \
  --csv data/gemm_dataset_eval.csv \
  --kernel gemm \
  --model xgboost \
  --model_dir output/output_GEMM_20k
```

---

## Detailed Usage

### 1. Dataset Download

The `download_dataset.py` script downloads datasets from HuggingFace:

```bash
cd model_train
python download_dataset.py
```

**Available datasets:**
- `GEMM_dataset` (10k samples) → `gemm_dataset_train.csv`
- `GEMM_dataset_20k` (20k samples) → `gemm_dataset_train_20k.csv`
- `GEMM_dataset_30k` (30k samples) → `gemm_dataset_train_30k.csv`
- `NTT_dataset` (10k samples) → `ntt_dataset_train.csv`
- `NTT_dataset_20k` (20k samples) → `ntt_dataset_train_20k.csv`
- `GEMM_predict` (eval set) → `gemm_dataset_eval.csv`
- `NTT_predict` (eval set) → `ntt_dataset_eval.csv`

All datasets are saved to `model_train/data/`.

### 2. Model Training

#### GEMM Training

```bash
# Auto-detect output directory from dataset filename
python train_gemm_model_v1.py --dataset data/gemm_dataset_train_20k.csv
# Creates: output/output_GEMM_20k/

# Train on 30k dataset
python train_gemm_model_v1.py --dataset data/gemm_dataset_train_30k.csv
# Creates: output/output_GEMM_30k/

# Custom output directory
python train_gemm_model_v1.py \
  --dataset data/gemm_dataset_train_20k.csv \
  --output_dir my_experiment
# Creates: output/my_experiment/
```

#### NTT Training

```bash
# Auto-detect output directory
python train_ntt_model_v1.py --dataset data/ntt_dataset_train_20k.csv
# Creates: output/output_NTT_20k/

# Custom output directory
python train_ntt_model_v1.py \
  --dataset data/ntt_dataset_train_20k.csv \
  --output_dir my_ntt_experiment
```

#### Training Features

- **Automatic logging**: All output saved to `training_log_TIMESTAMP.txt`
- **Two models**: XGBoost and SVR both trained and evaluated
- **Comprehensive metrics**: MSE, RMSE, MAE, R², MAPE, SMAPE, WMAPE, RMSLE
- **Visualizations**: Scatter plots, residual analysis, Q-Q plots, feature importance
- **Hyperparameter tuning**: GridSearchCV with 3-fold cross-validation
- **Regularization**: L1/L2 regularization and min_child_weight to prevent overfitting

#### Output Files

Each training run creates:
```
output/output_GEMM_20k/
├── training_log_20250119_143052.txt           # Complete log
├── scaler_gemm_v1.joblib                      # Feature scaler
├── xgboost_gpu_perf_predictor_model_gemm_v1.joblib  # XGBoost model
├── svr_gpu_perf_predictor_model_gemm_v1.joblib      # SVR model
├── xgb_feature_importances_gemm_v1.png        # Feature importance plot
├── predictions_vs_actual_gemm_xgboost.png     # XGBoost predictions
├── predictions_vs_actual_gemm_svm.png         # SVR predictions
├── residual_analysis_xgboost.png              # XGBoost residuals
├── residual_analysis_svm.png                  # SVR residuals
├── prediction_results_gemm_xgboost.csv        # XGBoost detailed results
└── prediction_results_gemm_svm.csv            # SVR detailed results
```

### 3. Making Predictions

Use the `predict.py` script for inference:

```bash
# Predict GEMM execution times with XGBoost
python predict.py \
  --csv data/gemm_dataset_eval.csv \
  --kernel gemm \
  --model xgboost \
  --model_dir output/output_GEMM_20k

# Predict NTT execution times with SVR
python predict.py \
  --csv data/ntt_dataset_eval.csv \
  --kernel ntt \
  --model svr \
  --model_dir output/output_NTT_20k
```

**Arguments:**
- `--csv`: Path to CSV file with features
- `--kernel`: `gemm` or `ntt`
- `--model`: `xgboost` or `svr`
- `--model_dir`: Directory containing trained model and scaler

**Output:**
- Prints prediction statistics to console
- If ground truth exists, shows evaluation metrics
- Saves predictions to `<input>_predictions_<model>.csv`

### 4. Evaluation on Unseen GPU

Evaluate model generalization on RTX 4060:

```bash
# Evaluate GEMM model
python evaluate_gemm_unseen_gpu.py

# Evaluate NTT model
python evaluate_ntt_unseen_gpu.py
```

Results saved to:
- `output/output_GEMM_eval_20k/`
- `output/output_NTT_eval_20k/`

---

## CUDA Benchmarking

The `dataset_collect/` directory contains CUDA benchmarking tools for collecting performance data on your GPU. Each benchmark tool supports **flexible output file naming** for easy organization.

### Output File Naming Convention

Both GEMM and NTT benchmarks use the following naming pattern:

```
benchmark_results_[output_name_]N{num_samples}[_S{seed}].csv
```

- **`[output_name_]`**: Optional custom identifier (GPU model, experiment name, etc.)
- **`N{num_samples}`**: Number of samples collected
- **`[_S{seed}]`**: Optional seed value (only if seed ≠ 0)

**Examples:**
```bash
./benchmark 0 1000                  # → benchmark_results_N1000.csv
./benchmark 0 1000 rtx4090          # → benchmark_results_rtx4090_N1000.csv
./benchmark 0 1000 exp1 0 10 50 42  # → benchmark_results_exp1_N1000_S42.csv
```

### GEMM Benchmark

Navigate to `dataset_collect/GEMM/`:

```bash
cd dataset_collect/GEMM

# Compile
make

# Run with default output name
./benchmark 0 10000
# Output: benchmark_results_N10000.csv

# Run with custom output name (recommended for multi-GPU collection)
./benchmark 0 10000 v100_baseline
# Output: benchmark_results_v100_baseline_N10000.csv

# Run with reproducible seed
./benchmark 1 5000 custom_kernel 0 10 50 42
# Output: benchmark_results_custom_kernel_N5000_S42.csv
```

**Implementations:**
- **cuBLAS (method=0)**: NVIDIA's optimized library
- **Custom Tiled (method=1)**: 16×16 tiled shared-memory kernel

**Output Features (52 total):**
- GPU specs (compute capability, SM count, memory, clocks, etc.)
- Matrix dimensions (M, N, K)
- Performance metrics (time, GFLOPS, memory throughput, efficiency)
- Derived features (arithmetic intensity, total operations, etc.)

See [dataset_collect/GEMM/README.md](dataset_collect/GEMM/README.md) for detailed usage.

### NTT Benchmark

Navigate to `dataset_collect/NTT/`:

```bash
cd dataset_collect/NTT

# Compile
make

# Run with default output name
./benchmark 1 5000
# Output: benchmark_results_N5000.csv

# Run with custom output name
./benchmark 1 5000 a100_ntt
# Output: benchmark_results_a100_ntt_N5000.csv

# Run with fixed size and seed for reproducibility
./benchmark 1 1000 test 65536 10 50 42
# Output: benchmark_results_test_N1000_S42.csv
```

**Implementation:**
- Cooley-Tukey radix-2 decimation-in-time algorithm
- Modulus: 998244353 (30-bit NTT-friendly prime)
- Primitive root: 3
- Transform sizes: 2^15 to 2^24 (32K to 16M elements)

**Output Features (36 total):**
- GPU specs (compute capability, SM count, memory, clocks, etc.)
- NTT configuration (N, modulus, primitive root)
- Performance metrics (time, modular ops throughput, memory efficiency)
- Derived features (butterflies, modops, theoretical bytes)

See [dataset_collect/NTT/README.md](dataset_collect/NTT/README.md) for detailed usage.

### Best Practices for Data Collection

1. **Use descriptive output names** when collecting data from multiple GPUs:
   ```bash
   ./benchmark 0 10000 rtx4090_run1
   ./benchmark 0 10000 v100_run1
   ./benchmark 0 10000 a100_run1
   ```

2. **Set seeds for reproducibility** in research experiments:
   ```bash
   ./benchmark 1 5000 experiment1 0 10 50 12345
   ```

3. **Move collected data to training directory**:
   ```bash
   cp benchmark_results_*.csv ../../model_train/data/
   ```

For comprehensive benchmarking guides, see [dataset_collect/README.md](dataset_collect/README.md)

---

## Features

### Machine Learning Features

**GEMM (22 features after preprocessing):**
- GPU hardware: compute_capability, SM count, L2 cache, memory specs
- Workload: M, N, K dimensions
- Derived: total_ops, total_io_bytes, arithmetic_intensity
- Kernel config: grid/block dimensions, registers per thread

**NTT (17 features after preprocessing):**
- GPU hardware: compute_capability, SM count, memory specs
- Workload: N (size)
- Derived: butterflies_total, modops_total, theoretical_bytes

**Key preprocessing steps:**
1. **Compute capability combination**: `cc_major` + `cc_minor`/10 → `compute_capability`
   - Handles cc_minor=0 gracefully (e.g., 8.0, 9.0)
   - Better distance metrics for model learning
2. **Data leakage prevention**: Drop performance-derived features
3. **Feature scaling**: StandardScaler on non-binary features
4. **Missing value imputation**: Median imputation

### Model Architecture

**XGBoost Regressor:**
- Hyperparameters: n_estimators, learning_rate, max_depth, subsample, colsample_bytree
- Regularization: reg_alpha (L1), reg_lambda (L2), min_child_weight
- GridSearchCV with 3-fold cross-validation

**Support Vector Regression (SVR):**
- RBF kernel
- Hyperparameters: C, epsilon, gamma
- GridSearchCV with 3-fold cross-validation

---

## Model Performance

### GEMM Results (20k dataset)

| Model    | R² (test) | RMSE  | MAE   | WMAPE |
|----------|-----------|-------|-------|-------|
| XGBoost  | 0.95      | 0.XX  | 0.XX  | XX%   |
| SVR      | 0.98      | 0.XX  | 0.13  | 10.91%|

**On unseen GPU (RTX 4060):**
- XGBoost: R² = 0.85, WMAPE = 42.49%

### NTT Results (20k dataset)

| Model    | R² (test) | RMSE  | MAE   |
|----------|-----------|-------|-------|
| XGBoost  | 0.85+     | 0.XX  | 0.XX  |
| SVR      | 0.XX      | 0.XX  | 0.XX  |

**On unseen GPU (RTX 4060):**
- XGBoost: R² = 0.58 (indicates generalization challenges)

**Note:** Models with regularization parameters show improved generalization. The 20k dataset provides a good balance between size and generalization.

---

## Requirements

### Python Dependencies

```
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
xgboost>=1.7.0
matplotlib>=3.6.0
scipy>=1.9.0
joblib>=1.2.0
datasets>=2.0.0  # For HuggingFace dataset downloads
```

Install with:
```bash
pip install -r model_train/requirements.txt
```

### CUDA Development

For benchmarking:
- CUDA Toolkit 11.0+ (tested with CUDA 12.4)
- cuBLAS library
- NVIDIA GPU with compute capability 6.0+

---

## Tips and Best Practices

### Training

1. **Start with 20k dataset**: Good balance between performance and generalization
2. **Monitor overfitting**: Check cross-validation scores vs test scores
3. **Use automatic logging**: No need for manual output redirection
4. **Check feature importance**: Helps understand model decisions

### Prediction

1. **Match preprocessing**: Ensure eval data has same features as training
2. **Check compute capability**: Model handles 0 values correctly (8.0, 9.0)
3. **Validate predictions**: Compare with actual times if available

### Benchmarking

1. **Warm-up runs**: Important for stable measurements
2. **Multiple repeats**: Reduces variance in timing
3. **Checkpointing**: Prevents data loss on long runs
4. **Diverse workloads**: Cover range of matrix sizes

---

## Common Issues and Solutions

### Issue: NumPy RuntimeWarning during training

**Solution:** Fixed in v1 scripts by using `df.copy()` in preprocessing.

### Issue: cc_minor=0 causing errors

**Solution:** Combined into `compute_capability` feature (e.g., 8.0, 9.0).

### Issue: Model overfitting on large datasets

**Solution:** Use regularization parameters (reg_alpha, reg_lambda, min_child_weight).

### Issue: Output directory organization

**Solution:** All outputs now under `model_train/output/` for cleaner structure.

---

## Documentation

This project has comprehensive documentation in each directory:

### Quick References

- **[dataset_collect/README.md](dataset_collect/README.md)**: Data collection overview
  - GEMM and NTT benchmarking workflow
  - File naming conventions
  - Best practices for ML dataset collection

- **[dataset_collect/GEMM/README.md](dataset_collect/GEMM/README.md)**: GEMM benchmarking guide
  - Step-by-step compilation and execution
  - 52 feature descriptions
  - cuBLAS vs Custom Tiled implementation details
  - Troubleshooting CUDA builds

- **[dataset_collect/NTT/README.md](dataset_collect/NTT/README.md)**: NTT benchmarking guide
  - Step-by-step compilation and execution
  - 36 feature descriptions
  - NTT algorithm and implementation details
  - Performance analysis guidelines

- **[model_train/README.md](model_train/README.md)**: Complete training workflow
  - Environment setup (home directory vs /scratch)
  - Dataset download from HuggingFace
  - Step-by-step training instructions
  - Evaluation and inference usage
  - Model architecture details
  - Comprehensive troubleshooting

- **[IMPROVEMENTS.md](IMPROVEMENTS.md)**: Project changelog
  - All improvements made to the project
  - Feature engineering decisions
  - Migration guide from old scripts

### Getting Started

1. **New to the project?** Start with this README for overview
2. **Want to train models?** Go to [model_train/README.md](model_train/README.md)
3. **Need to collect data?** See [dataset_collect/README.md](dataset_collect/README.md)
4. **Troubleshooting?** Check the relevant README's troubleshooting section

---

