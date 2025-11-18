# Model Training

This directory contains the training scripts and utilities for GEMM and NTT models.

output_GEMM stores the 10k GEMM results, and output_GEMM_20k stores the 20k GEMM results. 

output_NTT stores the 10k NTT results, and output_NTT_25k stores the 25k NTT results. 

##  Quick Start

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
# Note: Use your own path!!
pip install -r /home/$USER/Gpus/gpu-perf-predictor/model_train/requirements_scratch.txt

# 8. Close the current terminal then open a new terminal to reactivate scratch environment (works from anywhere)
source /scratch/$USER/model_train/.venv/bin/activate

# Navigate to the model_train directory
# Note: Use your own path!!
cd /home/$USER/Gpus/gpu-perf-predictor/model_train
```

### 2. Download Dataset

Download the required training datasets (automatically creates `../data/` directory):
```bash
python download_dataset.py
```

**Downloaded files:**
- `gemm_dataset_train.csv`
- `ntt_dataset_train.csv`
- `gemm_dataset_train_20k.csv`
- `ntt_dataset_train_20k.csv`
- `gemm_dataset_train_30k.csv`
- `gemm_dataset_eval.csv`
- `ntt_dataset_eval.csv`
>  **Dataset Source**: All datasets are available at [https://huggingface.co/NYUGPUClass](https://huggingface.co/NYUGPUClass)


### 3. Train Models

#### GEMM Model
```bash
python train_gemm_model_v1.py --dataset ../data/gemm_dataset_train.csv > run_output_gemm.txt
```

#### NTT Model
```bash
python train_ntt_model_v1.py --dataset ../data/ntt_dataset_train.csv > run_output_ntt.txt
```

## 📁 Directory Structure
```
model_train/
├── output_GEMM/          # GEMM training logs and checkpoints
├── output_GEMM/          # GEMM training logs and checkpoints. //todo
├── output_NTT/           # NTT training logs and checkpoints
├── run_output_gemm.txt   # GEMM training stdout
└── run_output_ntt.txt    # NTT training stdout
```

##  Output

Training outputs and logs are automatically saved to:
- **GEMM outputs**: `output_GEMM/`
- **NTT outputs**: `output_NTT/`
- **Training logs**: `run_output_gemm.txt` and `run_output_ntt.txt`

##  Tips

- Make sure your virtual environment is activated before running any commands
- Training may take significant time depending on your hardware
- Check the output logs for training progress and metrics
