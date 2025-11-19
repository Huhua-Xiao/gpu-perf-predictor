# GPU Performance Predictor - Recent Improvements

## Summary

This document describes the improvements made to the GPU performance prediction project to address code quality, overfitting issues, and usability.

## 1. Configurable Output Directories

### Problem
- Output directories were hardcoded as `output_GEMM_20k` and `output_NTT_20k`
- Required manual code editing to train on different dataset sizes (10k, 30k)
- Inflexible and error-prone

### Solution
- Added `--output_dir` CLI argument to both `train_gemm_model_v1.py` and `train_ntt_model_v1.py`
- Implemented automatic output directory detection based on dataset filename:
  - `gemm_dataset_train.csv` or `*10k*` → `output_GEMM_10k`
  - `*20k*` → `output_GEMM_20k`
  - `*30k*` → `output_GEMM_30k`
- Users can still override with custom directory: `--output_dir my_custom_output`

### Usage
```bash
# Auto-detect output directory from dataset
python train_gemm_model_v1.py --dataset ../data/gemm_dataset_train_20k.csv

# Manual override
python train_gemm_model_v1.py --dataset ../data/gemm_dataset_train_30k.csv --output_dir output_GEMM_30k_experiment
```

## 2. Regularization to Prevent Overfitting

### Problem
- Models showed signs of overfitting when scaling from 20k to 30k samples
- NTT model R² dropped from 0.85 (test) to 0.58 (unseen GPU)
- GEMM model dropped from 0.97 (test) to 0.85 (unseen GPU)

### Solution
Added three new hyperparameters to XGBoost GridSearchCV:

1. **L1 Regularization (`reg_alpha`)**: `[0, 0.1, 1.0]`
   - Helps with feature selection and sparsity
   - Reduces model complexity

2. **L2 Regularization (`reg_lambda`)**: `[1.0, 2.0, 5.0]`
   - Penalizes large weights
   - Smooths the model

3. **Minimum Child Weight (`min_child_weight`)**: `[1, 3, 5]`
   - Prevents overfitting to small sample groups
   - Controls tree depth indirectly

### Impact
- GridSearchCV will now explore 3 × 3 × 3 = 27 additional hyperparameter combinations
- Expected to improve generalization to unseen GPUs
- May slightly reduce training set performance but should improve validation/test performance

### Recommendation
**For the overfitting concern:**
- **Suggested approach**: Use the 20k model with the new regularization parameters
- The 20k dataset appears to provide good balance between size and generalization
- The 30k dataset may benefit from the regularization but should be tested
- Consider training both and comparing cross-validation scores

To test the regularization effectiveness:
```bash
# Train on 20k with new regularization
python train_gemm_model_v1.py --dataset ../data/gemm_dataset_train_20k.csv

# Train on 30k with new regularization
python train_gemm_model_v1.py --dataset ../data/gemm_dataset_train_30k.csv

# Compare the best_params_ and cross-validation scores
```

## 3. Simple Prediction Interface

### Problem
- No simple way to make predictions with trained models
- `evaluate_unseen_gpu.py` scripts were hardcoded for specific use case
- Users needed to understand the full preprocessing pipeline

### Solution
Created `model_train/predict.py` - a user-friendly CLI tool for making predictions:

**Features:**
- Simple command-line interface
- Supports both GEMM and NTT kernels
- Supports both XGBoost and SVR models
- Automatic preprocessing matching the training pipeline
- Handles CSV input files
- Outputs predictions to new CSV with `_predictions_{model}.csv` suffix
- Shows statistics and evaluation metrics (if ground truth available)

### Usage
```bash
# Predict GEMM execution times using XGBoost model trained on 20k dataset
python predict.py \
  --csv ../data/gemm_dataset_eval.csv \
  --kernel gemm \
  --model xgboost \
  --model_dir output_GEMM_20k

# Predict NTT execution times using SVR model
python predict.py \
  --csv ../data/ntt_dataset_eval.csv \
  --kernel ntt \
  --model svr \
  --model_dir output_NTT_20k
```

**Output:**
- Prediction statistics (mean, median, std, min, max)
- If ground truth exists: MSE, RMSE, MAE, R² scores
- CSV file with predictions added as new column

## 4. Code Cleanup and Organization

### Changes Made

1. **Removed commented code**
   - Deleted large blocks of commented one-hot encoding code (lines 110-119 in both files)
   - Code was confusing and not being used

2. **Archived outdated files**
   - Created `model_train/archive/` directory
   - Moved `train_gemm_model.py` and `train_ntt_model.py` (non-v1 versions)
   - Added `archive/README.md` explaining what's archived and why

3. **Added .gitignore**
   - Prevents tracking of large model files (*.joblib)
   - Excludes output directories
   - Excludes Python cache files
   - Excludes compiled CUDA binaries
   - Keeps repository clean

### Project Structure After Cleanup
```
model_train/
├── archive/                      # Outdated code (kept for reference)
│   ├── README.md
│   ├── train_gemm_model.py
│   └── train_ntt_model.py
├── train_gemm_model_v1.py        # Current GEMM training (use this)
├── train_ntt_model_v1.py         # Current NTT training (use this)
├── evaluate_gemm_unseen_gpu.py   # Evaluation on RTX 4060
├── evaluate_ntt_unseen_gpu.py    # Evaluation on RTX 4060
├── predict.py                    # NEW: Simple prediction interface
├── download_dataset.py
├── requirements.txt
├── output_GEMM_20k/              # Training results (gitignored)
└── output_NTT_20k/               # Training results (gitignored)
```

## Migration Guide

### For Users of Old Scripts

**Before:**
```bash
# Had to edit code to change output directory
python train_gemm_model.py --dataset ../data/gemm_dataset_train_20k.csv
```

**After:**
```bash
# Auto-detects output directory
python train_gemm_model_v1.py --dataset ../data/gemm_dataset_train_20k.csv

# Or specify custom directory
python train_gemm_model_v1.py --dataset ../data/gemm_dataset_train_30k.csv --output_dir my_experiment
```

### For Making Predictions

**Before:**
```bash
# Had to modify evaluate_unseen_gpu.py or write custom code
# Edit hardcoded paths, run evaluation script
```

**After:**
```bash
# Simple one-liner
python predict.py --csv my_data.csv --kernel gemm --model xgboost --model_dir output_GEMM_20k
```

## Next Steps

### Immediate Actions
1. **Test regularization effectiveness:**
   - Retrain models on 20k and 30k datasets
   - Compare cross-validation scores
   - Evaluate on unseen GPU (RTX 4060)

2. **Verify predict.py:**
   - Test on evaluation datasets
   - Confirm outputs match evaluate_unseen scripts

### Future Improvements
1. **Address generalization gap further:**
   - Collect data from more diverse GPUs (different architectures)
   - Consider GPU architecture family as a categorical feature
   - Try ensemble methods or domain adaptation techniques

2. **Add learning curve analysis:**
   - Plot training vs validation error over dataset sizes
   - Determine if more data helps or if regularization is sufficient

3. **Create comprehensive test suite:**
   - Unit tests for preprocessing functions
   - Integration tests for training pipeline
   - Correctness tests for CUDA kernels

4. **Documentation:**
   - Add examples to main README
   - Create tutorial notebook
   - Document feature engineering decisions

## Conclusion

These improvements make the project more:
- **Flexible**: Configurable output directories
- **Robust**: Regularization to prevent overfitting
- **Usable**: Simple prediction interface
- **Maintainable**: Clean code organization, archived old versions
- **Professional**: Proper .gitignore, documentation

The models should now generalize better to unseen GPUs while maintaining the clean architecture and good practices established in the original project.
