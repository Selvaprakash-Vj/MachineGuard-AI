import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


DATA_DIR = "data/processed_v2"
MODELS_DIR = "models_v2"
RESULTS_DIR = "results_v2"

RANDOM_SEED = 42

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def load_data(dataset):
    """Load the leakage-safe processed dataset."""

    dataset_path = os.path.join(
        DATA_DIR,
        f"{dataset}.pkl",
    )

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found: {dataset_path}\n"
            "Run preprocessing_v2.py first."
        )

    data = joblib.load(dataset_path)

    return (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
        data["test_units"],
    )


def flatten_windows(X):
    """
    Convert each 30 × 24 sensor window into one feature vector.

    Shape:
        Before: samples × 30 cycles × 24 features
        After:  samples × 720 features
    """

    return X.reshape(X.shape[0], -1)


def calculate_nasa_score(y_true, y_pred):
    """
    Calculate the asymmetric NASA C-MAPSS score.

    Overpredicting remaining life receives a stronger penalty.
    """

    error = y_pred - y_true

    penalties = np.where(
        error < 0,
        np.exp(-error / 13.0) - 1.0,
        np.exp(error / 10.0) - 1.0,
    )

    return float(np.sum(penalties))


def calculate_metrics(y_true, y_pred):
    """Calculate regression and engineering evaluation metrics."""

    y_pred = np.maximum(y_pred, 0.0)

    return {
        "rmse": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "mae": float(
            mean_absolute_error(y_true, y_pred)
        ),
        "r2": float(
            r2_score(y_true, y_pred)
        ),
        "nasa_score": calculate_nasa_score(
            y_true,
            y_pred,
        ),
        "mean_error": float(
            np.mean(y_pred - y_true)
        ),
    }


def build_model(model_name, ridge_alpha):
    """Create the requested classical baseline model."""

    if model_name == "mean":
        return DummyRegressor(
            strategy="mean",
        )

    if model_name == "ridge":
        return Ridge(
            alpha=ridge_alpha,
            random_state=RANDOM_SEED,
        )

    raise ValueError(
        f"Unsupported baseline model: {model_name}"
    )


def save_metrics(result):
    """Save one current row per model and dataset."""

    metrics_path = os.path.join(
        RESULTS_DIR,
        "baseline_metrics.csv",
    )

    new_row = pd.DataFrame([result])

    if os.path.exists(metrics_path):
        existing_results = pd.read_csv(metrics_path)

        duplicate_mask = (
            (existing_results["model"] == result["model"])
            & (existing_results["dataset"] == result["dataset"])
        )

        existing_results = existing_results.loc[
            ~duplicate_mask
        ]

        updated_results = pd.concat(
            [existing_results, new_row],
            ignore_index=True,
        )

    else:
        updated_results = new_row

    updated_results = updated_results.sort_values(
        by=["dataset", "test_rmse"],
    )

    updated_results.to_csv(
        metrics_path,
        index=False,
    )

    return metrics_path


def run_baseline(
    model_name,
    dataset,
    ridge_alpha=1.0,
):
    """Train and evaluate a classical ML baseline."""

    (
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
        test_units,
    ) = load_data(dataset)

    X_train_flat = flatten_windows(X_train)
    X_val_flat = flatten_windows(X_val)
    X_test_flat = flatten_windows(X_test)

    print(f"Training baseline: {model_name}")
    print(f"Dataset:           {dataset}")
    print(f"Training shape:    {X_train_flat.shape}")
    print(f"Validation shape:  {X_val_flat.shape}")
    print(f"Testing shape:     {X_test_flat.shape}")

    model = build_model(
        model_name=model_name,
        ridge_alpha=ridge_alpha,
    )

    model.fit(
        X_train_flat,
        y_train,
    )

    validation_predictions = model.predict(
        X_val_flat
    )

    test_predictions = model.predict(
        X_test_flat
    )

    validation_metrics = calculate_metrics(
        y_val,
        validation_predictions,
    )

    test_metrics = calculate_metrics(
        y_test,
        test_predictions,
    )

    model_path = os.path.join(
        MODELS_DIR,
        f"baseline_{model_name}_{dataset}.joblib",
    )

    joblib.dump(
        model,
        model_path,
    )

    predictions_path = os.path.join(
        RESULTS_DIR,
        f"predictions_baseline_{model_name}_{dataset}.csv",
    )

    predictions = pd.DataFrame(
        {
            "unit": test_units,
            "actual_rul": y_test,
            "predicted_rul": np.maximum(
                test_predictions,
                0.0,
            ),
            "error": test_predictions - y_test,
            "absolute_error": np.abs(
                test_predictions - y_test
            ),
        }
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    result = {
        "model": model_name,
        "dataset": dataset,
        "validation_rmse": validation_metrics["rmse"],
        "validation_mae": validation_metrics["mae"],
        "validation_r2": validation_metrics["r2"],
        "test_rmse": test_metrics["rmse"],
        "test_mae": test_metrics["mae"],
        "test_r2": test_metrics["r2"],
        "test_nasa_score": test_metrics["nasa_score"],
        "test_mean_error": test_metrics["mean_error"],
    }

    metrics_path = save_metrics(result)

    print("\nValidation results")
    print(
        f"RMSE: {validation_metrics['rmse']:.4f}"
    )
    print(
        f"MAE:  {validation_metrics['mae']:.4f}"
    )
    print(
        f"R²:   {validation_metrics['r2']:.4f}"
    )

    print("\nIndependent test results")
    print(
        f"RMSE:       {test_metrics['rmse']:.4f}"
    )
    print(
        f"MAE:        {test_metrics['mae']:.4f}"
    )
    print(
        f"R²:         {test_metrics['r2']:.4f}"
    )
    print(
        f"NASA score: {test_metrics['nasa_score']:.4f}"
    )
    print(
        f"Mean error: {test_metrics['mean_error']:.4f}"
    )

    print(f"\nModel saved to {model_path}")
    print(f"Predictions saved to {predictions_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["mean", "ridge"],
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["FD001", "FD002", "FD003", "FD004"],
    )

    parser.add_argument(
        "--ridge_alpha",
        type=float,
        default=1.0,
    )

    arguments = parser.parse_args()

    run_baseline(
        model_name=arguments.model,
        dataset=arguments.dataset,
        ridge_alpha=arguments.ridge_alpha,
    )