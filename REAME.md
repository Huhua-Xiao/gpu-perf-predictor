# GPU Performance Predictor

Accurately predicting GPU kernel runtime performance is essential for optimizing compute workloads and understanding hardware utilization. This project develops regression-based models to predict kernel performance as a function of both GPU specifications and kernel-level configuration parameters. We focus on two representative workloads: Matrix Multiplication (MM), which is compute-bound, and the Number Theoretic Transform (NTT), which is memory-bound. By collecting runtime measurements across multiple accelerators and configurations, we train and evaluate models such as XGBoost and Support Vector Regression (SVR) to estimate kernel execution time, while analyzing feature importance to understand how architectural and configuration-level parameters impact GPU performance. This work contributes a reproducible framework for kernel benchmarking, dataset generation, and performance prediction across heterogeneous GPU architectures.

## Overview

This project provides:
- **Dataset Generation**: CUDA benchmarks for GEMM and NTT operations
- **Machine Learning Models**: Trained models (XGBoost and SVM) to predict GPU performance based on hardware specifications and operation parameters
- **Evaluation Tools**: Scripts to evaluate models on unseen GPUs and new datasets

### Supported Operations

- **GEMM (General Matrix Multiply)**: Matrix multiplication benchmarking with cuBLAS implementation (NVIDIA's optimized library)
  
- **NTT (Number Theoretic Transform)**: Fast Fourier Transform-like operations for polynomial multiplication

## Project Structure

```
gpu-perf-predictor/
├── dataset_generation/          # CUDA benchmarking code
│   ├── GEMM/                    # GEMM benchmark implementation
│   └── NTT/                     # NTT benchmark implementation
├── model_train/                 # Machine learning training and evaluation
│   ├── train_gemm_model_v1.py  # Train GEMM performance predictor
│   ├── train_ntt_model_v1.py   # Train NTT performance predictor
│   ├── evaluate_gemm_unseen_gpu.py  # Evaluate on new GPUs
│   ├── evaluate_ntt_unseen_gpu.py   # Evaluate on new GPUs
│   ├── download_dataset.py     # Download training datasets
│   └── output_*/               # Model outputs and checkpoints
├── run.sh                       # Main script for model to train and eval
└── REAME.md                     # This file
```

## Prerequisites

- **CUDA Toolkit** (version 12.4 recommended)
- **Python 3** (3.8+)
- **NVIDIA GPU** with CUDA support
- **nvcc** compiler
- **cuBLAS** library

## Quick Start

### 1. Clone Repository

```bash
git clone <repository-url>
cd gpu-perf-predictor
```

### 2. Run Model Training Script

The main script sets up the environment, downloads datasets, and trains both GEMM and NTT models:

```bash
chmod +x run.sh
./run.sh
```

## Setup and Usage

For detailed setup instructions and usage, please refer to the README files in each subdirectory:

- **[Dataset Generation](dataset_generation/README.md)**: Instructions for setting up CUDA environment and running benchmarks
- **[Model Training](model_train/README.md)**: Instructions for Python environment setup, dataset download, and model training/evaluation

## Directory Structure Details

### `model_train/`
- Training scripts for GEMM and NTT models
- Evaluation scripts for unseen GPU testing
- Output directories containing trained models and results
- Jupyter notebooks for exploratory analysis

### `dataset_generation/`
- CUDA benchmark implementations
- Makefiles for compilation
- Test suites for validation

## Model Outputs

After training, models and evaluation results are saved in `model_train/output_*/` directories:

- **Trained Models**: `*_gpu_perf_predictor_model_*.joblib`
- **Scalers**: `scaler_*.joblib` (for preprocessing new data)
- **Evaluation Metrics**: CSV files with prediction results
- **Visualizations**: 
  - Prediction vs actual plots
  - Residual analysis plots
  - Feature importance plots (XGBoost)
