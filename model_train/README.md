# Model Training

This directory contains the training scripts and utilities for GEMM and NTT models.

output_GEMM stores the 10k GEMM results, and output_GEMM_20k stores the 20k GEMM results. 

output_NTT stores the 10k NTT results, and output_NTT_25k stores the 25k NTT results. 

##  Quick Start

### 1. Environment Setup

Navigate to the project directory and set up a Python virtual environment:
```bash
cd model_train
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Dataset

Download the required training datasets (automatically creates `../data/` directory):
```bash
python download_dataset.py
```

**Downloaded files:**
- `gemm_dataset_train.csv`
- `ntt_dataset_train.csv`
>  **Dataset Source**: All datasets are available at [https://huggingface.co/NYUGPUClass](https://huggingface.co/NYUGPUClass)


### 3. Train Models

#### GEMM Model
```bash
python train_gemm_model_v1.py > run_output_gemm.txt
```

#### NTT Model
```bash
python train_tnn_model_v1.py > run_output_ntt.txt
```

## 📁 Directory Structure
```
model_train/
├── output_GEMM/          # GEMM training logs and checkpoints
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
