#!/bin/bash

set -xe

# Setting up the environment
cd model_train
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements_scratch.txt

# Download dataset
python download_dataset.py

# Train model for GEMM
python train_gemm_model_v1.py --dataset ../data/gemm_dataset_train.csv > run_output_gemm.txt

# Train model for NTT
python train_tnn_model_v1.py --dataset ../data/ntt_dataset_train.csv > run_output_ntt.txt