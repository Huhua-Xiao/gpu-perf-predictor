## Environment Setup

We recommend using a virtual environment to ensure reproducibility.

```bash
# 1. Cd model_train
cd model_train

# 2. Create a virtual environment
python3 -m venv .venv

# 3. Activate it
source .venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run training
python train_ntt_model.py
python train_ntt_model_v1.py
python train_gemm_model.py

