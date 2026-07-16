import argparse
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from tensorflow import keras


DATA_DIR = "data/processed_v2"
MODELS_DIR = "models_v2"
RESULTS_DIR = "results_v2"

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_test_data(dataset):
    """Load the independent test-engine sequences."""

    dataset_path = os.path.join(
        DATA_DIR,
        f"{dataset}.pkl",
    )

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Processed dataset not found: {dataset_path}\n"
            "Run preprocessing_v2.py first."
        )

    data = joblib.load(dataset_path)

    return (
        data["X_test"],
        data["y_test"],
        data["test_units"],
    )


def calculate_nasa_score(y_true, y_pred):
    """
    Calculate the asymmetric NASA C-MAPSS scoring function.

    Late failure predictions receive a larger penalty than early
    predictions because missed maintenance can be more dangerous.
    """

    prediction_error = y_pred - y_true

    score = np.where(
        prediction_error < 0,
        np.exp(-prediction_error / 13.0) - 1.0,
        np.exp(prediction_error / 10.0) - 1.0,
    )

    return float(np.sum(score))


def save_metrics(metrics):
    """Save one current result per model and dataset."""

    metrics_path = os.path.join(
        RESULTS_DIR,
        "metrics.csv",
    )

    new_row = pd.DataFrame([metrics])

    if os.path.exists(metrics_path):
        existing_results = pd.read_csv(metrics_path)

        duplicate_mask = (
            (existing_results["model"] == metrics["model"])
            & (existing_results["dataset"] == metrics["dataset"])
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
        by=["dataset", "rmse"],
    )

    updated_results.to_csv(
        metrics_path,
        index=False,
    )

    return metrics_path


def evaluate_model(model_name, dataset):
    """Evaluate one final RUL prediction per test engine."""

    print(f"Evaluating {model_name} on {dataset}")

    X_test, y_test, test_units = load_test_data(dataset)

    model_path = os.path.join(
        MODELS_DIR,
        f"{model_name}_{dataset}.keras",
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found: {model_path}"
        )

    model = keras.models.load_model(model_path)

    y_pred = model.predict(
        X_test,
        batch_size=64,
        verbose=1,
    ).reshape(-1)

    y_pred = np.maximum(y_pred, 0.0)

    rmse = float(
        np.sqrt(mean_squared_error(y_test, y_pred))
    )

    mae = float(
        mean_absolute_error(y_test, y_pred)
    )

    r2 = float(
        r2_score(y_test, y_pred)
    )

    nasa_score = calculate_nasa_score(
        y_test,
        y_pred,
    )

    mean_error = float(
        np.mean(y_pred - y_test)
    )

    metrics = {
        "model": model_name,
        "dataset": dataset,
        "test_engines": len(y_test),
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "nasa_score": nasa_score,
        "mean_error": mean_error,
    }

    predictions = pd.DataFrame(
        {
            "unit": test_units,
            "actual_rul": y_test,
            "predicted_rul": y_pred,
            "error": y_pred - y_test,
            "absolute_error": np.abs(y_pred - y_test),
        }
    )

    predictions_path = os.path.join(
        RESULTS_DIR,
        f"predictions_{model_name}_{dataset}.csv",
    )

    predictions.to_csv(
        predictions_path,
        index=False,
    )

    metrics_path = save_metrics(metrics)

    print("\nEvaluation results")
    print(f"Test engines: {len(y_test)}")
    print(f"RMSE:         {rmse:.4f}")
    print(f"MAE:          {mae:.4f}")
    print(f"R²:           {r2:.4f}")
    print(f"NASA score:   {nasa_score:.4f}")
    print(f"Mean error:   {mean_error:.4f}")

    print(f"\nPredictions saved to {predictions_path}")
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        required=True,
        choices=["lstm", "gru", "transformer"],
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["FD001", "FD002", "FD003", "FD004"],
    )

    arguments = parser.parse_args()

    evaluate_model(
        model_name=arguments.model,
        dataset=arguments.dataset,
    )