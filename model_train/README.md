# GPU Performance Predictor - Model Training

This directory contains the training scripts and utilities for GEMM and NTT models.

## Output Directories

- **output_GEMM/**: Stores training results for GEMM models trained on 10k dataset
- **output_GEMM_20k/**: Stores training results for GEMM models trained on 20k dataset
- **output_GEMM_eval/**: Stores evaluation results using models from output_GEMM/
- **output_GEMM_eval_20k/**: Stores evaluation results using models from output_GEMM_20k/
- **output_NTT/**: Stores training results for NTT models trained on 10k dataset
- **output_NTT_20k/**: Stores training results for NTT models trained on 20k dataset
- **output_NTT_eval/**: Stores evaluation results using models from output_NTT/
- **output_NTT_eval_20k/**: Stores evaluation results using models from output_NTT_20k/ 

## Quick Start

### 1. Environment Setup

Navigate to the project directory and set up a Python virtual environment. Sometimes disk space on home directory might not be sufficient.
So, this project can be run either directly in your **home directory** or using the **/scratch** directory for larger storage (recommended on CIMS GPU machines).

## **Option A: Setup in Home Directory**
```bash
cd model_train
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## **Option B: Setup on /scratch**

```bash
# 1. Create your scratch directory (first-time only)
mkdir -p /scratch/$USER

# 2. Verify it exists
ls /scratch/$USER

# 3. Create a dedicated model_train directory
mkdir -p /scratch/$USER/model_train

# 4. Move into scratch directory 
cd /scratch/$USER/model_train

# 5. Create a Python virtual environment on scratch
python3 -m venv .venv

# 6. Activate the environment
source .venv/bin/activate # On Windows: .venv\Scripts\activate

# 7. Install dependencies from the project directory 
# Note: Update the path to match your project location!
pip install -r <path-to-project>/gpu-perf-predictor/model_train/requirements_scratch.txt

# 8. Close the current terminal then open a new terminal to reactivate scratch environment (works from anywhere)
source /scratch/$USER/model_train/.venv/bin/activate

# Navigate to the model_train directory
# Note: Update the path to match your project location!
cd <path-to-project>/gpu-perf-predictor/model_train
```

### 2. Download Dataset

Download the required training datasets (automatically creates `../data/` directory):
```bash
python download_dataset.py
```

**Downloaded files:**
- `gemm_dataset_train.csv` (10k samples)
- `ntt_dataset_train.csv` (10k samples)
- `gemm_dataset_train_20k.csv` (20k samples)
- `ntt_dataset_train_20k.csv` (20k samples)
- `gemm_dataset_train_30k.csv` (30k samples)
- `gemm_dataset_eval.csv` (evaluation dataset)
- `ntt_dataset_eval.csv` (evaluation dataset)

> **Dataset Source**: All datasets are available at [https://huggingface.co/NYUGPUClass](https://huggingface.co/NYUGPUClass)


### 3. Train Models

#### GEMM Model

Train on 10k dataset:
```bash
python train_gemm_model_v1.py --dataset ../data/gemm_dataset_train.csv > run_output_gemm.txt
```

Train on 20k dataset:
```bash
python train_gemm_model_v1.py --dataset ../data/gemm_dataset_train_20k.csv > run_output_gemm.txt
```

**Note**: The output directory is hardcoded in the script. For 20k dataset, modify `output_dir` in `train_gemm_model_v1.py` to `"output_GEMM_20k"` (or use the default if already set).

#### NTT Model

Train on 10k dataset:
```bash
python train_ntt_model_v1.py --dataset ../data/ntt_dataset_train.csv > run_output_ntt.txt
```

Train on 20k dataset:
```bash
python train_ntt_model_v1.py --dataset ../data/ntt_dataset_train_20k.csv > run_output_ntt.txt
```

**Note**: The output directory is hardcoded in the script. For 20k dataset, modify `output_dir` in `train_ntt_model_v1.py` to `"output_NTT_20k"` (or use the default if already set).

### 4. Evaluate Models on Unseen GPUs

After training, you can evaluate models on evaluation datasets:

#### GEMM Evaluation
```bash
python evaluate_gemm_unseen_gpu.py
```

#### NTT Evaluation
```bash
python evaluate_ntt_unseen_gpu.py
```

**Note**: Make sure the evaluation scripts point to the correct model and scaler paths from your training output directories.

## Output

Each output directory contains:
- **Trained Models**: 
  - `xgboost_gpu_perf_predictor_model_*_v1.joblib` (XGBoost model)
  - `svr_gpu_perf_predictor_model_*_v1.joblib` (SVR model)
- **Preprocessing**: 
  - `scaler_*_v1.joblib` (StandardScaler for feature normalization)
- **Evaluation Results**:
  - `prediction_results_*_xgboost.csv` (XGBoost predictions)
  - `prediction_results_*_svm.csv` (SVR predictions)
  - `predictions_vs_actual_*.png` (Prediction vs actual plots)
  - `residual_analysis_*.png` (Residual analysis plots)
- **Feature Importance** (XGBoost only):
  - `xgb_feature_importances_*_v1.png`
- **Training Logs**:
  - `run_output_*.txt` (Training stdout)
  - `eval_output_*.txt` (Evaluation stdout)

## Tips

- Make sure your virtual environment is activated before running any commands
- Training may take significant time depending on your hardware
- Check the output logs for training progress and metrics
- The training scripts use hardcoded output directories - modify them in the script if you want to use different directories
