# Model Training - GPU Performance Prediction

This directory contains training scripts, evaluation tools, and inference utilities for **GEMM** and **NTT** GPU performance prediction models. The models use XGBoost and Support Vector Regression (SVR) to predict execution times based on GPU hardware characteristics and kernel configurations.

## Table of Contents

- [Quick Start](#quick-start)
- [Directory Structure](#directory-structure)
- [Step-by-Step Workflow](#step-by-step-workflow)
- [Training Scripts](#training-scripts)
- [Evaluation and Inference](#evaluation-and-inference)
- [Output Organization](#output-organization)
- [Model Architecture](#model-architecture)
- [Troubleshooting](#troubleshooting)

## Quick Start

```bash
# 1. Set up Python environment
cd model_train
python3 -m venv .venv
source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download datasets
python download_dataset.py

# 4. Train GEMM model (20k dataset recommended)
python train_gemm_model_v1.py --dataset data/gemm_dataset_train_20k.csv

# 5. Train NTT model (20k dataset recommended)
python train_ntt_model_v1.py --dataset data/ntt_dataset_train_20k.csv

# 6. Use trained model for prediction
python predict.py --model output/output_GEMM_20k/xgboost_model.pkl \
                  --task gemm \
                  --input test_data.csv
```

## Directory Structure

```
model_train/
├── train_gemm_model_v1.py        # GEMM model training script
├── train_ntt_model_v1.py         # NTT model training script
├── predict.py                    # Inference script for both GEMM and NTT
├── evaluate_gemm_unseen_gpu.py   # Evaluate GEMM model on new GPU
├── evaluate_ntt_unseen_gpu.py    # Evaluate NTT model on new GPU
├── download_dataset.py           # Download training datasets from HuggingFace
├── requirements.txt              # Python dependencies (local setup)
├── requirements_scratch.txt      # Python dependencies (scratch disk setup)
│
├── data/                         # Downloaded datasets (created by download_dataset.py)
│   ├── gemm_dataset_train.csv          # 10k GEMM samples
│   ├── gemm_dataset_train_20k.csv      # 20k GEMM samples (recommended)
│   ├── gemm_dataset_train_30k.csv      # 30k GEMM samples
│   ├── ntt_dataset_train.csv           # 10k NTT samples
│   ├── ntt_dataset_train_20k.csv       # 20k NTT samples (recommended)
│   ├── gemm_dataset_eval.csv           # GEMM evaluation data (unseen GPU)
│   └── ntt_dataset_eval.csv            # NTT evaluation data (unseen GPU)
│
├── output/                       # Training outputs (auto-created)
│   ├── output_GEMM_10k/               # Results from 10k GEMM training
│   │   ├── xgboost_model.pkl
│   │   ├── svr_model.pkl
│   │   ├── scaler_xgboost.pkl
│   │   ├── scaler_svr.pkl
│   │   ├── predictions_train.csv
│   │   ├── predictions_test.csv
│   │   ├── feature_importance.png
│   │   └── training_log_YYYYMMDD_HHMMSS.txt
│   │
│   ├── output_GEMM_20k/               # Results from 20k GEMM training
│   │   └── [same structure as above]
│   │
│   ├── output_NTT_10k/                # Results from 10k NTT training
│   │   └── [same structure as above]
│   │
│   ├── output_NTT_20k/                # Results from 20k NTT training
│   │   └── [same structure as above]
│   │
│   └── output_GEMM_eval_20k/          # Evaluation on unseen GPU
│       ├── predictions_unseen_xgboost.csv
│       ├── predictions_unseen_svr.csv
│       └── evaluation_log_YYYYMMDD_HHMMSS.txt
│
└── archive/                      # Deprecated scripts (for reference)
    ├── train_gemm_model.py
    ├── train_ntt_model.py
    └── README.md
```

## Step-by-Step Workflow

### Step 1: Environment Setup

You can set up the environment either in your **home directory** or on **/scratch** (recommended for CIMS GPU machines with limited home disk space).

#### Option A: Home Directory Setup

```bash
# Navigate to model_train directory
cd model_train

# Create Python virtual environment
python3 -m venv .venv

# Activate environment
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Option B: Scratch Disk Setup (Recommended for CIMS)

```bash
# 1. Create scratch directory (first time only)
mkdir -p /scratch/$USER

# 2. Create dedicated model_train directory
mkdir -p /scratch/$USER/model_train

# 3. Navigate to scratch directory
cd /scratch/$USER/model_train

# 4. Create Python virtual environment
python3 -m venv .venv

# 5. Activate environment
source .venv/bin/activate

# 6. Install dependencies (adjust path to your project location)
pip install -r /path/to/gpu-perf-predictor/model_train/requirements_scratch.txt

# 7. For future sessions, activate from anywhere:
source /scratch/$USER/model_train/.venv/bin/activate
cd /path/to/gpu-perf-predictor/model_train
```

**Verify Installation:**
```bash
python -c "import xgboost, sklearn, pandas; print('All packages installed successfully!')"
```

### Step 2: Download Training Datasets

The `download_dataset.py` script downloads pre-collected benchmark data from HuggingFace:

```bash
python download_dataset.py
```

**What This Does:**
- Creates `data/` directory inside `model_train/`
- Downloads datasets from [https://huggingface.co/NYUGPUClass](https://huggingface.co/NYUGPUClass)
- Saves 7 CSV files (~500 MB total)

**Downloaded Files:**

| File | Size | Samples | Description |
|------|------|---------|-------------|
| `gemm_dataset_train.csv` | ~50 MB | 10,000 | Initial GEMM training set |
| `gemm_dataset_train_20k.csv` | ~100 MB | 20,000 | **Recommended** GEMM training set |
| `gemm_dataset_train_30k.csv` | ~150 MB | 30,000 | Extended GEMM set (may overfit) |
| `ntt_dataset_train.csv` | ~40 MB | 10,000 | Initial NTT training set |
| `ntt_dataset_train_20k.csv` | ~80 MB | 20,000 | **Recommended** NTT training set |
| `gemm_dataset_eval.csv` | ~20 MB | ~3,000 | GEMM evaluation (unseen GPU) |
| `ntt_dataset_eval.csv` | ~15 MB | ~2,500 | NTT evaluation (unseen GPU) |

**Recommendation**: Use **20k datasets** for training - they provide the best balance between model performance and generalization.

### Step 3: Train GEMM Model

Train the GEMM performance prediction model using XGBoost and SVR:

```bash
python train_gemm_model_v1.py --dataset data/gemm_dataset_train_20k.csv
```

**Options:**
```bash
# Specify custom output directory
python train_gemm_model_v1.py \
    --dataset data/gemm_dataset_train_20k.csv \
    --output_dir output/custom_gemm_run

# Use different dataset
python train_gemm_model_v1.py --dataset data/gemm_dataset_train_30k.csv
```

**What Happens During Training:**

1. **Data Loading**: Reads CSV with 52 features per sample
2. **Preprocessing**:
   - Combines `cc_major` and `cc_minor` into `compute_capability`
   - Drops correlated features (`actual_time_us`, `gflops`, `memory_throughput_GBps`)
   - Applies StandardScaler normalization
3. **Train/Test Split**: 80% train, 20% test (stratified by GPU)
4. **Hyperparameter Tuning**:
   - GridSearchCV with 3-fold cross-validation
   - Tests 1,000+ hyperparameter combinations
   - Optimized for negative MAE
5. **Model Training**:
   - XGBoost regressor with regularization (L1/L2)
   - SVR with RBF kernel
6. **Evaluation**:
   - Predictions on train and test sets
   - MAE, RMSE, R² metrics
   - Feature importance visualization
7. **Saving Artifacts**:
   - Trained models (`.pkl`)
   - Scalers (`.pkl`)
   - Predictions (`.csv`)
   - Feature importance plot (`.png`)
   - Training log with timestamp

**Training Time**: 30-60 minutes on typical workstation (depends on GridSearchCV)

**Expected Output:**
```
=== Final GEMM XGBoost Results ===
Train MAE: 12.34 μs
Test MAE:  15.67 μs
Test RMSE: 23.45 μs
Test R²:   0.9876

=== Final GEMM SVR Results ===
Train MAE: 18.90 μs
Test MAE:  21.23 μs
Test RMSE: 31.45 μs
Test R²:   0.9654
```

### Step 4: Train NTT Model

Train the NTT performance prediction model:

```bash
python train_ntt_model_v1.py --dataset data/ntt_dataset_train_20k.csv
```

**Options:**
```bash
# Specify custom output directory
python train_ntt_model_v1.py \
    --dataset data/ntt_dataset_train_20k.csv \
    --output_dir output/custom_ntt_run
```

The training process is identical to GEMM but uses:
- 36 features (NTT-specific metrics)
- NTT-specific feature engineering
- Different hyperparameter search space

**Training Time**: 25-50 minutes

**Expected Output:**
```
=== Final NTT XGBoost Results ===
Train MAE: 8.92 μs
Test MAE:  11.45 μs
Test RMSE: 17.23 μs
Test R²:   0.9823

=== Final NTT SVR Results ===
Train MAE: 14.56 μs
Test MAE:  17.89 μs
Test RMSE: 25.67 μs
Test R²:   0.9512
```

### Step 5: Evaluate on Unseen GPU

Test model generalization on a GPU **not seen during training**:

#### GEMM Evaluation
```bash
python evaluate_gemm_unseen_gpu.py \
    --model output/output_GEMM_20k/xgboost_model.pkl \
    --scaler output/output_GEMM_20k/scaler_xgboost.pkl \
    --unseen_data data/gemm_dataset_eval.csv
```

#### NTT Evaluation
```bash
python evaluate_ntt_unseen_gpu.py \
    --model output/output_NTT_20k/xgboost_model.pkl \
    --scaler output/output_NTT_20k/scaler_xgboost.pkl \
    --unseen_data data/ntt_dataset_eval.csv
```

**What This Tests:**
- Model's ability to generalize to new GPU architectures
- Prediction accuracy on different compute capabilities
- Robustness across varying problem sizes

**Expected Output:**
```
=== Unseen GPU Evaluation ===
GPU: NVIDIA RTX 4090 (not in training set)
Samples: 3,247

Unseen GPU MAE:  23.45 μs
Unseen GPU RMSE: 35.67 μs
Unseen GPU R²:   0.9234

Saved predictions to: output/output_GEMM_eval_20k/predictions_unseen_xgboost.csv
```

### Step 6: Inference on New Data

Use trained models to predict performance on new configurations:

```bash
python predict.py \
    --model output/output_GEMM_20k/xgboost_model.pkl \
    --scaler output/output_GEMM_20k/scaler_xgboost.pkl \
    --task gemm \
    --input new_gemm_configs.csv
```

**Input CSV Format:**

Your `new_gemm_configs.csv` should have the same features as training data. **Required columns for GEMM**:
```csv
gpu_name,device_id,cc_major,cc_minor,sm_count,l2_size_bytes,...,M,N,K,algorithm
NVIDIA A100,0,8,0,108,41943040,...,2048,2048,2048,Custom-Tiled
```

**Required columns for NTT**:
```csv
gpu_name,device_id,cc_major,cc_minor,sm_count,l2_size_bytes,...,N,modulus,primitive_root,algorithm
NVIDIA A100,0,8,0,108,41943040,...,65536,998244353,3,Custom-NTT
```

**Output:**

The script adds a `predicted_time_ms` column:
```csv
gpu_name,M,N,K,algorithm,predicted_time_ms
NVIDIA A100,2048,2048,2048,Custom-Tiled,0.234
```

## Training Scripts

### train_gemm_model_v1.py

**Features:**
- Automatic output directory detection from dataset filename
- Combined `compute_capability` feature engineering
- L1/L2 regularization to prevent overfitting
- Automatic logging to timestamped file
- Feature importance visualization

**Key Arguments:**
```bash
--dataset PATH          # Path to training CSV (required)
--output_dir PATH       # Custom output directory (optional, auto-detected)
```

**Auto-Detection Logic:**
- `gemm_dataset_train.csv` → `output/output_GEMM_10k/`
- `gemm_dataset_train_20k.csv` → `output/output_GEMM_20k/`
- `gemm_dataset_train_30k.csv` → `output/output_GEMM_30k/`

### train_ntt_model_v1.py

Identical structure to GEMM training but for NTT workloads.

**Auto-Detection Logic:**
- `ntt_dataset_train.csv` → `output/output_NTT_10k/`
- `ntt_dataset_train_20k.csv` → `output/output_NTT_20k/`

## Evaluation and Inference

### evaluate_gemm_unseen_gpu.py / evaluate_ntt_unseen_gpu.py

**Purpose**: Test model on GPU architectures not seen during training.

**Arguments:**
```bash
--model PATH            # Trained model (.pkl)
--scaler PATH           # Trained scaler (.pkl)
--unseen_data PATH      # Evaluation dataset CSV
--output_dir PATH       # Where to save results (optional)
```

**Output Files:**
- `predictions_unseen_xgboost.csv`: Predictions with actual vs predicted
- `evaluation_log_YYYYMMDD_HHMMSS.txt`: Timestamped evaluation metrics

### predict.py

**Purpose**: Inference on new data without ground truth labels.

**Arguments:**
```bash
--model PATH            # Trained model (.pkl)
--scaler PATH           # Trained scaler (.pkl)
--task {gemm,ntt}       # Which task to predict
--input PATH            # Input CSV with features
--output PATH           # Output CSV (optional, defaults to <input>_predictions.csv)
```

**Example:**
```bash
python predict.py \
    --model output/output_GEMM_20k/xgboost_model.pkl \
    --scaler output/output_GEMM_20k/scaler_xgboost.pkl \
    --task gemm \
    --input my_gpu_configs.csv \
    --output results.csv
```

## Output Organization

All training outputs are organized under `output/` with descriptive subdirectories:

```
output/
├── output_GEMM_20k/
│   ├── xgboost_model.pkl              # Trained XGBoost model
│   ├── svr_model.pkl                  # Trained SVR model
│   ├── scaler_xgboost.pkl             # StandardScaler for XGBoost
│   ├── scaler_svr.pkl                 # StandardScaler for SVR
│   ├── predictions_train.csv          # Train set predictions
│   ├── predictions_test.csv           # Test set predictions
│   ├── feature_importance.png         # XGBoost feature importance plot
│   └── training_log_20250119_143022.txt  # Timestamped training log
│
└── output_GEMM_eval_20k/
    ├── predictions_unseen_xgboost.csv
    ├── predictions_unseen_svr.csv
    └── evaluation_log_20250119_150500.txt
```

**File Descriptions:**

| File | Description |
|------|-------------|
| `xgboost_model.pkl` | Serialized XGBoost regressor (best hyperparameters) |
| `svr_model.pkl` | Serialized SVR model with RBF kernel |
| `scaler_*.pkl` | StandardScaler fitted on training data |
| `predictions_*.csv` | CSV with columns: `actual_time_ms`, `predicted_time_ms`, `gpu_name`, `M`, `N`, `K` |
| `feature_importance.png` | Bar plot of top 20 most important features |
| `training_log_*.txt` | Complete stdout from training (timestamped) |

## Model Architecture

### XGBoost Regressor

**Hyperparameter Search Space:**
```python
{
    "n_estimators": [100, 200],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 7],
    "subsample": [0.7, 0.8],
    "colsample_bytree": [0.7, 0.8],
    "reg_alpha": [0, 0.1, 1.0],        # L1 regularization
    "reg_lambda": [1.0, 2.0, 5.0],     # L2 regularization
    "min_child_weight": [1, 3, 5],
}
```

**Why XGBoost?**
- Handles non-linear relationships between GPU specs and performance
- Built-in feature importance for interpretability
- Robust to outliers and missing values
- Regularization prevents overfitting

### Support Vector Regression (SVR)

**Hyperparameter Search Space:**
```python
{
    "C": [0.1, 1, 10, 100],
    "epsilon": [0.01, 0.1, 0.5],
    "gamma": ["scale", "auto"],
}
```

**Why SVR?**
- Strong baseline for comparison
- Effective with high-dimensional feature spaces
- Good generalization with proper regularization

### Feature Engineering

**Compute Capability Combination:**
```python
# Before: cc_major=8, cc_minor=0 treated as separate features
# After:  compute_capability = 8.0 (combined semantic meaning)

compute_capability = cc_major + cc_minor / 10.0
```

**Why?**
- Captures ordinal relationship (8.0 > 7.5 > 7.0)
- Handles cc_minor=0 gracefully
- Improves distance metrics for tree-based models

**Dropped Features (prevent data leakage):**
- `actual_time_us`: Direct target variable
- `gflops`: Derived from execution time
- `memory_throughput_GBps`: Derived from execution time

## Troubleshooting

### Installation Issues

**Error: `ModuleNotFoundError: No module named 'xgboost'`**

```bash
# Ensure virtual environment is activated
source .venv/bin/activate  # or: source /scratch/$USER/model_train/.venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

**Error: `pip: command not found`**

```bash
# Use python -m pip instead
python -m pip install -r requirements.txt
```

### Training Issues

**Error: `FileNotFoundError: data/gemm_dataset_train_20k.csv`**

```bash
# Download datasets first
python download_dataset.py

# Verify download
ls -lh data/
```

**Error: `RuntimeWarning: invalid value encountered in cast`**

This warning is **harmless** and has been mitigated in v1 scripts with `df.copy()`. It occurs during StandardScaler transformations.

**Training Takes Too Long**

Reduce GridSearchCV search space in the script:
```python
# Edit train_gemm_model_v1.py
param_grid = {
    "n_estimators": [100],       # Reduced from [100, 200]
    "learning_rate": [0.1],      # Reduced from [0.05, 0.1, 0.2]
    "max_depth": [5],            # Reduced from [3, 5, 7]
    # ...
}
```

### Prediction Issues

**Error: `ValueError: X has different number of features`**

Input CSV is missing required features. Check:
```bash
# Compare columns
python -c "
import pandas as pd
train = pd.read_csv('data/gemm_dataset_train_20k.csv')
test = pd.read_csv('your_input.csv')
print('Missing:', set(train.columns) - set(test.columns))
print('Extra:', set(test.columns) - set(train.columns))
"
```

**Poor Prediction Accuracy**

Possible causes:
1. **GPU architecture very different from training data**
   - Solution: Collect benchmark data on new GPU and retrain
2. **Problem sizes outside training distribution**
   - Solution: Ensure M, N, K values are within training range
3. **Incorrect feature values**
   - Solution: Verify GPU specs match `nvidia-smi` output

### Memory Issues

**Error: `MemoryError` or system freeze during training**

```bash
# Reduce dataset size
head -n 10000 data/gemm_dataset_train_20k.csv > data/gemm_dataset_train_small.csv

# Train on smaller dataset
python train_gemm_model_v1.py --dataset data/gemm_dataset_train_small.csv
```

## Best Practices

1. **Always Use 20k Datasets** for training (best performance/generalization trade-off)
2. **Activate Virtual Environment** before running scripts
3. **Check Logs** in `output/*/training_log_*.txt` for detailed diagnostics
4. **Use Evaluation Scripts** to validate model on unseen GPUs before deployment
5. **Save Predictions** for later analysis and debugging
6. **Version Control Output Directories** by renaming before re-training

## Next Steps

After training models:

1. **Evaluate Generalization**:
   ```bash
   python evaluate_gemm_unseen_gpu.py --model output/output_GEMM_20k/xgboost_model.pkl
   ```

2. **Deploy for Inference**:
   - Copy `.pkl` files to production environment
   - Use `predict.py` for batch predictions
   - Integrate into GPU selection pipeline

3. **Collect More Data**:
   - Run benchmarks on new GPUs (see `dataset_collect/README.md`)
   - Append to existing datasets
   - Retrain for improved accuracy

4. **Experiment with Hyperparameters**:
   - Modify `param_grid` in training scripts
   - Try different feature engineering approaches
   - Compare XGBoost vs SVR performance

## References

- **XGBoost Documentation**: [https://xgboost.readthedocs.io/](https://xgboost.readthedocs.io/)
- **Scikit-learn User Guide**: [https://scikit-learn.org/stable/user_guide.html](https://scikit-learn.org/stable/user_guide.html)
- **Dataset Source**: [https://huggingface.co/NYUGPUClass](https://huggingface.co/NYUGPUClass)

---

For questions or issues, please refer to the main project [README](../README.md) or check the [troubleshooting section](#troubleshooting).
