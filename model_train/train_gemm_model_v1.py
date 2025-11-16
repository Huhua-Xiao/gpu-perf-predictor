import os
import sys
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import joblib
from sklearn.svm import SVR


# =========================
# 1. Data loading and EDA
# =========================

def load_dataset(csv_path: str):
    # train_csv_path = "/home/nyu_id/Gpus/gpu-perf-predictor/data/gemm_dataset_train.csv"
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print("Dataset loaded. Shape:", df.shape)
    return df


def basic_eda(df: pd.DataFrame):
    print("\n=== DataFrame Overview ===")
    print(df.info())

    print("\n=== Descriptive statistics for numerical columns ===")
    print(df.describe())

    print("\n=== Missing values in each column ===")
    print(df.isnull().sum())

    print("\n=== First 5 rows of the DataFrame ===")
    print(df.head())

    print("\n=== Unique Values ===")
    categorical_cols = ['gpu_name', 'algorithm']
    for col in categorical_cols:
        print(f"Unique values for '{col}':")
        print(df[col].unique())
        print("\n")
    

# =========================
# 2. Preprocessing
# =========================

def preprocess(df: pd.DataFrame):
    # 1. Check if time_ms_mean in dataset
    if "time_ms_mean" not in df.columns:
        raise ValueError("Column 'time_ms_mean' not found;")
    y = df["time_ms_mean"]

    # 2. Separate features and target variable
    columns_to_drop = [
        'device_id', # Constant column
        'driver_version', # Constant column
        'precision', # Constant categorical column
        'algorithm', # Constant categorical column
        'time_ms_mean','time_ms_median', 'time_ms_p95', 'time_ms_stddev', # Performance metrics leading to data leakage
        "gflops_mean",
        "gflops_median",
        "gflops_p95",
        "gops_over_peak_mean",
        "memory_throughput_GBps",
        "memory_efficiency_pct",
        "compute_efficiency_pct",
        "peak_flops_fp32_GFLOPs_actual",
        "compute_efficiency_actual_pct",
        "actual_clock_mhz",
        "actual_mem_clock_mhz",
        "temperature_c",
        "power_watts",
        "gpu_name", 
        "cuda_runtime_version",
        "repeats",
        "inner_iters",
        "seed",
        "tile_width",
        "block_x",
        "block_y",
        "shared_mem_per_block",
    ]

    X = df.drop(columns=columns_to_drop, errors="ignore")
    print("Features (X) and target (y) separated successfully.")
    print(f"Shape of X: {X.shape}")
    print(f"Shape of y: {y.shape}")

    # 3. Handle missing values in numerical columns
    missing_numerical_cols = X.select_dtypes(include=['number']).columns[X.select_dtypes(include=['number']).isnull().any()].tolist()

    print("\n Numerical columns in X with missing values:")
    print(missing_numerical_cols)

    print("Imputing missing numerical values with median...")
    for col in missing_numerical_cols:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        print(f"Column '{col}': filled missing values with median={median_val}. Remaining missing: {X[col].isnull().sum()}")

    print("Missing values in numerical columns after imputation:")
    print(X[missing_numerical_cols].isnull().sum())

    # # 4. One-hot encode categorical columns
    # categorical_cols_to_encode = X.select_dtypes(include=["object"]).columns.tolist()
    # print("\nCategorical columns in X to be encoded:")
    # print(categorical_cols_to_encode)

    # if categorical_cols_to_encode:
    #     X = pd.get_dummies(X, columns=categorical_cols_to_encode, drop_first=True)
    #     print("Applied one-hot encoding. New X shape:", X.shape)
    # else:
    #     print("No categorical columns to encode.")

    # 5. Standard-scale numeric features (excluding 0/1 dummy columns)
    numeric_cols_after = X.select_dtypes(include=["number"]).columns.tolist()
    non_boolean_numeric_cols = []
    for col in numeric_cols_after:
        unique_vals = X[col].unique()
        if (X[col].dtype != "bool") and (X[col].nunique() > 2 or not set(unique_vals).issubset({0, 1})):
            non_boolean_numeric_cols.append(col)

    print("\nNumeric columns to scale (standardization):")
    print(non_boolean_numeric_cols)

    scaler = StandardScaler()
    X[non_boolean_numeric_cols] = scaler.fit_transform(X[non_boolean_numeric_cols])
    print("Standardization complete.")

    return X, y

# =========================
# 3. Model training (XGBoost)
# =========================

def train_xgb(X_train, y_train):
    xgb_model = XGBRegressor(random_state=42)

    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.7, 0.8],
        "colsample_bytree": [0.7, 0.8],
    }


    print("\n=== XGBoost Regressor model initialized and parameter grid defined ===")

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="neg_mean_squared_error",
        cv=3,
        verbose=1,
        n_jobs=-1,
        return_train_score=True
    )

    print("Starting GridSearchCV fit...")
    grid_search.fit(X_train, y_train)
    print("GridSearchCV fit complete.")
    print("Best parameters found: ", grid_search.best_params_)
    print("Best cross-validation score (negative mean squared error): ", grid_search.best_score_)

    best_model = grid_search.best_estimator_

    return best_model

# =========================
# 4. Evaluation
# =========================
def plot_feature_importance(model, feature_names, output_dir):

    importances = model.feature_importances_
    
    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    # Sort features by importance
    importance_df = importance_df.sort_values(by='Importance', ascending=False)

    top_k = min(25, len(importance_df))
    subset = importance_df.head(top_k)

    plt.figure(figsize=(10, 8))
    plt.barh(subset["Feature"][::-1], subset["Importance"][::-1])
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("XGBoost Feature Importances (Top 25)")
    plt.tight_layout()
    out_path = os.path.join(output_dir, "xgb_feature_importances_gemm_v1.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Feature importance plot saved to: {out_path}\n")

def evaluate_model(model, X_test, y_test, output_dir, name="XGBoost"):

    print(f"\n=== Evaluation for {name} on test set ===")
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print("Model Evaluation on Test Set:")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R-squared (R2): {r2:.4f}")

    errors = y_test - y_pred
    abs_errors = np.abs(errors)
    rel_errors = np.abs(errors) / (y_test + 1e-10) * 100
    smape = np.mean(2 * abs_errors / (np.abs(y_test) + np.abs(y_pred) + 1e-10)) * 100
    rmsle = np.sqrt(np.mean((np.log1p(y_pred) - np.log1p(y_test))**2))
    wmape = np.sum(abs_errors) / np.sum(y_test) * 100
    
    print(f"\n=== Error Analysis ===")
    print(f"Mean Absolute Percentage Error (MAPE): {np.mean(rel_errors):.2f}%")
    print(f"Symmetric MAPE (SMAPE): {smape:.2f}%")
    print(f"Weighted MAPE (WMAPE): {wmape:.2f}%")
    print(f"Root Mean Squared Log Error (RMSLE): {rmsle:.6f}")
    print(f"Median Absolute Error: {np.median(abs_errors):.4f} ms")
    print(f"90th Percentile Absolute Error: {np.percentile(abs_errors, 90):.4f} ms")
    print(f"95th Percentile Absolute Error: {np.percentile(abs_errors, 95):.4f} ms")
    print(f"Max Absolute Error: {np.max(abs_errors):.4f} ms")

    print(f"\n=== Prediction Statistics ===")
    print(f"Actual - Mean: {y_test.mean():.4f}, Std: {y_test.std():.4f}, Range: [{y_test.min():.4f}, {y_test.max():.4f}]")
    print(f"Predicted - Mean: {y_pred.mean():.4f}, Std: {y_pred.std():.4f}, Range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    
    plt.figure(figsize=(10, 8))
    plt.scatter(y_test, y_pred, alpha=0.5, s=10)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('Actual Time (ms)')
    plt.ylabel('Predicted Time (ms)')
    plt.title(f'{name}: Predicted vs Actual (GEMM)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    output_path_f1 = os.path.join(output_dir, 'predictions_vs_actual_gemm.png')
    plt.savefig(output_path_f1, dpi=300)
    plt.close()
    print(f"Saved to: {output_path_f1}\n")


    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    axes[0, 0].hist(errors, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(x=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_xlabel('Residuals (Actual - Predicted)')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Residuals')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].scatter(y_pred, errors, alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('Predicted Time (ms)')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].set_title('Residuals vs Predicted Values')
    axes[0, 1].grid(True, alpha=0.3)
    
    axes[1, 0].hist(rel_errors, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=np.median(rel_errors), color='r', linestyle='--', linewidth=2, label=f'Median: {np.median(rel_errors):.2f}%')
    axes[1, 0].set_xlabel('Absolute Percentage Error (%)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Relative Errors')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    stats.probplot(errors, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot (Residuals vs Normal Distribution)')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path_f2 = os.path.join(output_dir, 'residual_analysis.png')
    plt.savefig(output_path_f2, dpi=300)
    plt.close()
    print(f"Saved: {output_path_f2}\n")
    
    print(f"\n=== Error Analysis by Prediction Range ===")
    percentiles = [0, 25, 50, 75, 100]
    bins = np.percentile(y_test, percentiles)
    
    for i in range(len(bins) - 1):
        mask = (y_test >= bins[i]) & (y_test < bins[i+1])
        if i == len(bins) - 2: 
            mask = (y_test >= bins[i]) & (y_test <= bins[i+1])
        
        if mask.sum() > 0:
            range_mae = np.mean(abs_errors[mask])
            range_mape = np.mean(rel_errors[mask])
            range_r2 = r2_score(y_test[mask], y_pred[mask])
            print(f"Range [{bins[i]:.4f}, {bins[i+1]:.4f}] (n={mask.sum()}): MAE={range_mae:.4f}, MAPE={range_mape:.2f}%, R²={range_r2:.4f}")
    
    results_df = pd.DataFrame({
        'actual': y_test,
        'predicted': y_pred,
        'error': errors,
        'abs_error': abs_errors,
        'rel_error_pct': rel_errors
    })
    results_df = results_df.sort_values('abs_error', ascending=False)
    csv_output_path = os.path.join(output_dir, 'prediction_results_gemm.csv')
    results_df.to_csv(csv_output_path, index=False)
    print(f"\nSaved CSV to: {csv_output_path}\n")
    
    print(f"\n=== Top 10 Worst Predictions ===")
    print(results_df.head(10).to_string(index=False))
# =========================
# 5. Main entry point
# =========================

def main():
    output_dir = "output_GEMM"
    os.makedirs(output_dir, exist_ok=True)
    print(f"\nAll files and output will be saved to {output_dir}\n")

    # Path to your CSV on CIMS
    # Remind: replace hx2487 to your netid
    csv_path = "/home/hx2487/Gpus/gpu-perf-predictor/data/gemm_dataset_train.csv"

    # 1. Load data and run simple EDA (printed only)
    df = load_dataset(csv_path)
    basic_eda(df)

    # 2. Preprocess (cleaning, encoding, scaling)
    X, y = preprocess(df)

    # 3. Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets successfully.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"y_test shape: {y_test.shape}")

    # 4. Train XGBoost with GridSearchCV
    best_xgb_model = train_xgb(X_train, y_train)

    # 5. Plot feature importances (optional but useful for report)
    plot_feature_importance(best_xgb_model, X_train.columns, output_dir)

    # 6. Evaluate on test set
    evaluate_model(best_xgb_model, X_test, y_test, output_dir, name="XGBoost")

    # 7. Save model to disk
    model_filename =os.path.join(output_dir, "xgboost_gpu_perf_predictor_model_gemm_v1.joblib")
    joblib.dump(best_xgb_model, model_filename)
    print(f"\nSaved trained XGBoost model to: {model_filename}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
