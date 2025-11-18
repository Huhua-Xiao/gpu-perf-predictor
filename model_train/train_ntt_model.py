import os
import sys
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import joblib
import argparse


# =========================
# 1. Data loading and EDA
# =========================

def load_dataset(csv_path: str):
    # train_csv_path = "/home/nyu_id/Gpus/gpu-perf-predictor/data/ntt_dataset_train.csv"
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
        "device_id",      # constant column
        "driver_version", # constant column
        "algorithm",      # constant categorical column
        # Performance metrics leading to data leakage
        "time_ms_mean",
        "time_ms_median",
        "time_ms_p95",
        "time_ms_stddev",
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

    # 4. One-hot encode categorical columns
    categorical_cols_to_encode = X.select_dtypes(include=["object"]).columns.tolist()
    print("\nCategorical columns in X to be encoded:")
    print(categorical_cols_to_encode)

    if categorical_cols_to_encode:
        X = pd.get_dummies(X, columns=categorical_cols_to_encode, drop_first=True)
        print("Applied one-hot encoding. New X shape:", X.shape)
    else:
        print("No categorical columns to encode.")

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
def plot_feature_importance(model, feature_names, out_path="xgb_feature_importances_ntt.png"):

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
    plt.savefig(out_path)
    plt.close()
    print(f"Feature importance plot saved to: {out_path}")

def evaluate_model(model, X_test, y_test, name="XGBoost"):
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


# =========================
# 5. Main entry point
# =========================

def main():
    parser = argparse.ArgumentParser(description="GPU GEMM Performance Predictor")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to the GEMM dataset CSV file"
    )
    args = parser.parse_args()
    csv_path = args.dataset

    # Path to your CSV on CIMS
    # Remind: replace hx2487 to your netid
    # csv_path = "/home/hx2487/Gpus/gpu-perf-predictor/data/ntt_dataset_train.csv"

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
    plot_feature_importance(best_xgb_model, X_train.columns, out_path="xgb_feature_importances_ntt.png")

    # 6. Evaluate on test set
    evaluate_model(best_xgb_model, X_test, y_test, name="XGBoost")

    # 7. Save model to disk
    model_filename = "xgboost_gpu_perf_predictor_model_ntt.joblib"
    joblib.dump(best_xgb_model, model_filename)
    print(f"\nSaved trained XGBoost model to: {model_filename}")

    print("\n=== Done ===")


if __name__ == "__main__":
    main()
    