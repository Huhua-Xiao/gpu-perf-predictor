## Environment Setup

# 1. Move into the model_train directory
cd model_train

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate the environment
source .venv/bin/activate

# 4. Install all dependencies
pip install -r requirements.txt


## Download Dataset

# Run the dataset download script before training
python download_dataset.py


## Run Training

# Train the GEMM model
# All stdout logs will be saved into run_output_gemm.txt
python train_gemm_model_v1.py > run_output_gemm.txt

# Train the NTT model
# All stdout logs will be saved into run_output_ntt.txt
python train_tnn_model_v1.py > run_output_ntt.txt


## Output Logs

# Training logs and model outputs will be stored in:
#   output_GEMM/
#   output_NTT/
