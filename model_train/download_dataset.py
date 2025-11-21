from datasets import load_dataset
import os

# Create data directory inside model_train instead of root
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

dataset_gemm = load_dataset("NYUGPUClass/GEMM_dataset")
gemm_csv_path = os.path.join(output_dir, "gemm_dataset_train.csv")
dataset_gemm["train"].to_csv(gemm_csv_path, index=False)

dataset_ntt = load_dataset("NYUGPUClass/NTT_dataset")
ntt_csv_path = os.path.join(output_dir, "ntt_dataset_train.csv")
dataset_ntt["train"].to_csv(ntt_csv_path, index=False)

dataset_gemm = load_dataset("NYUGPUClass/GEMM_dataset_20k")
gemm_csv_path = os.path.join(output_dir, "gemm_dataset_train_20k.csv")
dataset_gemm["train"].to_csv(gemm_csv_path, index=False)

dataset_ntt = load_dataset("NYUGPUClass/NTT_dataset_20k")
ntt_csv_path = os.path.join(output_dir, "ntt_dataset_train_20k.csv")
dataset_ntt["train"].to_csv(ntt_csv_path, index=False)

dataset_gemm = load_dataset("NYUGPUClass/GEMM_dataset_30k")
gemm_csv_path = os.path.join(output_dir, "gemm_dataset_train_30k.csv")
dataset_gemm["train"].to_csv(gemm_csv_path, index=False)

dataset_gemm = load_dataset("NYUGPUClass/GEMM_predict")
gemm_csv_path = os.path.join(output_dir, "gemm_dataset_eval.csv")
dataset_gemm["train"].to_csv(gemm_csv_path, index=False)

dataset_ntt = load_dataset("NYUGPUClass/NTT_predict")
ntt_csv_path = os.path.join(output_dir, "ntt_dataset_eval.csv")
dataset_ntt["train"].to_csv(ntt_csv_path, index=False)

print(f"Saving datasets to: {os.path.abspath(output_dir)}")