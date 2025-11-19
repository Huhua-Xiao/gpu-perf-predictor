# Archive

This folder contains outdated code that has been replaced by improved versions.

## Archived Files

### Training Scripts (Replaced by v1 versions)

- `train_gemm_model.py` - Original GEMM training script
  - **Replaced by:** `train_gemm_model_v1.py`
  - **Reason:** v1 adds configurable output directory, removed commented code, added regularization parameters

- `train_ntt_model.py` - Original NTT training script
  - **Replaced by:** `train_ntt_model_v1.py`
  - **Reason:** v1 adds configurable output directory, removed commented code, added regularization parameters

## Why Archive?

These files are kept for reference purposes but should not be used for new training runs. The v1 versions include:
- Automatic output directory detection based on dataset size
- Manual override via `--output_dir` argument
- Additional regularization parameters (reg_alpha, reg_lambda, min_child_weight) to prevent overfitting
- Cleaner code with commented sections removed

## Migration Notes

If you previously used these scripts, update your commands to use the v1 versions:

**Old:**
```bash
python train_gemm_model.py --dataset ../data/gemm_dataset_train_20k.csv
```

**New:**
```bash
python train_gemm_model_v1.py --dataset ../data/gemm_dataset_train_20k.csv
# Output directory will auto-detect as output_GEMM_20k

# Or specify manually:
python train_gemm_model_v1.py --dataset ../data/gemm_dataset_train_30k.csv --output_dir output_GEMM_30k
```
