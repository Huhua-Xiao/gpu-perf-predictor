#!/usr/bin/env python3
"""
Simple prediction script for GPU performance prediction models.
Provides a command-line interface to predict execution time for GEMM or NTT workloads.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler


def preprocess_gemm(df: pd.DataFrame, scaler):
    """Preprocess GEMM features to match training pipeline."""
    columns_to_drop = [
        'device_id', 'driver_version', 'precision', 'algorithm',
        'time_ms_mean','time_ms_median', 'time_ms_p95', 'time_ms_stddev',
        "gflops_mean", "gflops_median", "gflops_p95",
        "gops_over_peak_mean", "memory_throughput_GBps",
        "memory_efficiency_pct", "compute_efficiency_pct",
        "peak_flops_fp32_GFLOPs_actual", "compute_efficiency_actual_pct",
        "actual_clock_mhz", "actual_mem_clock_mhz",
        "temperature_c", "power_watts", "gpu_name", "cuda_runtime_version",
        "repeats", "inner_iters", "seed",
        "tile_width", "block_x", "block_y", "shared_mem_per_block",
    ]

    X = df.drop(columns=columns_to_drop, errors="ignore")

    # Handle missing values
    missing_numerical_cols = X.select_dtypes(include=['number']).columns[
        X.select_dtypes(include=['number']).isnull().any()
    ].tolist()

    for col in missing_numerical_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)

    # Scale features
    numeric_cols_after = X.select_dtypes(include=["number"]).columns.tolist()
    non_boolean_numeric_cols = []
    for col in numeric_cols_after:
        unique_vals = X[col].unique()
        if (X[col].dtype != "bool") and (X[col].nunique() > 2 or not set(unique_vals).issubset({0, 1})):
            non_boolean_numeric_cols.append(col)

    X[non_boolean_numeric_cols] = scaler.transform(X[non_boolean_numeric_cols])

    return X


def preprocess_ntt(df: pd.DataFrame, scaler):
    """Preprocess NTT features to match training pipeline."""
    columns_to_drop = [
        "time_ms_mean", "time_ms_median", "time_ms_p95", "time_ms_stddev",
        "modops_per_sec", "memory_throughput_GBps", "memory_efficiency_pct",
        "actual_clock_mhz", "actual_mem_clock_mhz",
        "temperature_c", "power_watts",
        "device_id", "modulus", "primitive_root",
        "repeats", "inner_iters", "algorithm", "seed", "gpu_name",
    ]

    X = df.drop(columns=columns_to_drop, errors="ignore")

    # Handle missing values
    missing_numerical_cols = X.select_dtypes(include=['number']).columns[
        X.select_dtypes(include=['number']).isnull().any()
    ].tolist()

    for col in missing_numerical_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)

    # Scale features
    numeric_cols_after = X.select_dtypes(include=["number"]).columns.tolist()
    non_boolean_numeric_cols = []
    for col in numeric_cols_after:
        unique_vals = X[col].unique()
        if (X[col].dtype != "bool") and (X[col].nunique() > 2 or not set(unique_vals).issubset({0, 1})):
            non_boolean_numeric_cols.append(col)

    X[non_boolean_numeric_cols] = scaler.transform(X[non_boolean_numeric_cols])

    return X


def predict_from_csv(csv_path: str, model_type: str, kernel_type: str, model_dir: str):
    """
    Load a dataset CSV and predict execution times.

    Args:
        csv_path: Path to CSV file with features
        model_type: "xgboost" or "svr"
        kernel_type: "gemm" or "ntt"
        model_dir: Directory containing trained models and scaler
    """
    # Load the appropriate model and scaler
    if model_type.lower() == "xgboost":
        model_filename = f"xgboost_gpu_perf_predictor_model_{kernel_type}_v1.joblib"
    elif model_type.lower() == "svr":
        model_filename = f"svr_gpu_perf_predictor_model_{kernel_type}_v1.joblib"
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'xgboost' or 'svr'.")

    scaler_filename = f"scaler_{kernel_type}_v1.joblib"

    model_path = os.path.join(model_dir, model_filename)
    scaler_path = os.path.join(model_dir, scaler_filename)

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler not found: {scaler_path}")

    print(f"Loading model from: {model_path}")
    model = joblib.load(model_path)

    print(f"Loading scaler from: {scaler_path}")
    scaler = joblib.load(scaler_path)

    # Load and preprocess data
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded. Shape: {df.shape}")

    # Check if ground truth is available
    has_ground_truth = "time_ms_mean" in df.columns

    if kernel_type.lower() == "gemm":
        X = preprocess_gemm(df, scaler)
    elif kernel_type.lower() == "ntt":
        X = preprocess_ntt(df, scaler)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}. Use 'gemm' or 'ntt'.")

    # Make predictions
    print(f"\nMaking predictions with {model_type.upper()} model...")
    predictions = model.predict(X)

    # Display results
    print(f"\n{'='*60}")
    print(f"PREDICTION RESULTS ({kernel_type.upper()}, {model_type.upper()})")
    print(f"{'='*60}")
    print(f"Number of predictions: {len(predictions)}")
    print(f"Predicted execution time statistics (ms):")
    print(f"  Mean:   {predictions.mean():.4f}")
    print(f"  Median: {np.median(predictions):.4f}")
    print(f"  Std:    {predictions.std():.4f}")
    print(f"  Min:    {predictions.min():.4f}")
    print(f"  Max:    {predictions.max():.4f}")

    # If ground truth is available, show comparison
    if has_ground_truth:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        y_true = df["time_ms_mean"].values
        mse = mean_squared_error(y_true, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, predictions)
        r2 = r2_score(y_true, predictions)

        print(f"\n{'='*60}")
        print("EVALUATION METRICS (vs ground truth)")
        print(f"{'='*60}")
        print(f"MSE:  {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE:  {mae:.4f}")
        print(f"R²:   {r2:.4f}")

    # Save predictions to CSV
    output_df = df.copy()
    output_df[f'predicted_time_ms_{model_type}'] = predictions

    output_path = csv_path.replace('.csv', f'_predictions_{model_type}.csv')
    output_df.to_csv(output_path, index=False)
    print(f"\nPredictions saved to: {output_path}")

    return predictions


def predict_single(features: dict, model_type: str, kernel_type: str, model_dir: str):
    """
    Predict execution time for a single set of features.

    Args:
        features: Dictionary of feature values
        model_type: "xgboost" or "svr"
        kernel_type: "gemm" or "ntt"
        model_dir: Directory containing trained models and scaler
    """
    # Create a single-row DataFrame
    df = pd.DataFrame([features])

    # Add dummy columns that will be dropped anyway
    if "time_ms_mean" not in df.columns:
        df["time_ms_mean"] = 0  # Dummy value

    # Load model and scaler
    if model_type.lower() == "xgboost":
        model_filename = f"xgboost_gpu_perf_predictor_model_{kernel_type}_v1.joblib"
    elif model_type.lower() == "svr":
        model_filename = f"svr_gpu_perf_predictor_model_{kernel_type}_v1.joblib"
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    scaler_filename = f"scaler_{kernel_type}_v1.joblib"

    model_path = os.path.join(model_dir, model_filename)
    scaler_path = os.path.join(model_dir, scaler_filename)

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Preprocess
    if kernel_type.lower() == "gemm":
        X = preprocess_gemm(df, scaler)
    elif kernel_type.lower() == "ntt":
        X = preprocess_ntt(df, scaler)
    else:
        raise ValueError(f"Unknown kernel type: {kernel_type}")

    # Predict
    prediction = model.predict(X)[0]

    print(f"\nPredicted execution time: {prediction:.4f} ms")

    return prediction


def main():
    parser = argparse.ArgumentParser(
        description="Predict GPU kernel execution times using trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict from CSV file
  python predict.py --csv data/gemm_dataset_eval.csv --kernel gemm --model xgboost --model_dir output_GEMM_20k

  # Predict from CSV with NTT model
  python predict.py --csv data/ntt_dataset_eval.csv --kernel ntt --model svr --model_dir output_NTT_20k
        """
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to CSV file containing features for prediction"
    )
    parser.add_argument(
        "--kernel",
        type=str,
        required=True,
        choices=["gemm", "ntt"],
        help="Kernel type: 'gemm' or 'ntt'"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost",
        choices=["xgboost", "svr"],
        help="Model type: 'xgboost' (default) or 'svr'"
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Directory containing trained model and scaler (e.g., output_GEMM_20k)"
    )

    args = parser.parse_args()

    try:
        predictions = predict_from_csv(
            csv_path=args.csv,
            model_type=args.model,
            kernel_type=args.kernel,
            model_dir=args.model_dir
        )
        print("\nPrediction complete!")

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
