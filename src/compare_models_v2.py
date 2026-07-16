import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


RESULTS_DIR = "results_v2"
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")

os.makedirs(FIGURES_DIR, exist_ok=True)


def calculate_nasa_score(y_true, y_pred):
    """Calculate the asymmetric NASA C-MAPSS score."""

    error = y_pred - y_true

    penalties = np.where(
        error < 0,
        np.exp(-error / 13.0) - 1.0,
        np.exp(error / 10.0) - 1.0,
    )

    return float(np.sum(penalties))


def calculate_metrics(y_true, y_pred):
    """Calculate model-comparison metrics."""

    errors = y_pred - y_true

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
            np.mean(errors)
        ),
        "overprediction_rate": float(
            np.mean(errors > 0) * 100
        ),
        "large_overprediction_rate": float(
            np.mean(errors > 10) * 100
        ),
        "maximum_absolute_error": float(
            np.max(np.abs(errors))
        ),
    }


def load_predictions(dataset):
    """Load LSTM and Ridge predictions and merge them by engine."""

    lstm_path = os.path.join(
        RESULTS_DIR,
        f"predictions_lstm_{dataset}.csv",
    )

    ridge_path = os.path.join(
        RESULTS_DIR,
        f"predictions_baseline_ridge_{dataset}.csv",
    )

    if not os.path.exists(lstm_path):
        raise FileNotFoundError(
            f"LSTM predictions not found: {lstm_path}"
        )

    if not os.path.exists(ridge_path):
        raise FileNotFoundError(
            f"Ridge predictions not found: {ridge_path}"
        )

    lstm_df = pd.read_csv(lstm_path)
    ridge_df = pd.read_csv(ridge_path)

    lstm_df = lstm_df[
        ["unit", "actual_rul", "predicted_rul"]
    ].rename(
        columns={
            "predicted_rul": "lstm_prediction",
        }
    )

    ridge_df = ridge_df[
        ["unit", "actual_rul", "predicted_rul"]
    ].rename(
        columns={
            "actual_rul": "ridge_actual_rul",
            "predicted_rul": "ridge_prediction",
        }
    )

    comparison_df = lstm_df.merge(
        ridge_df,
        on="unit",
        how="inner",
        validate="one_to_one",
    )

    actual_values_match = np.allclose(
        comparison_df["actual_rul"],
        comparison_df["ridge_actual_rul"],
    )

    if not actual_values_match:
        raise ValueError(
            "LSTM and Ridge prediction files contain "
            "different actual RUL values."
        )

    comparison_df = comparison_df.drop(
        columns="ridge_actual_rul"
    )

    comparison_df["lstm_error"] = (
        comparison_df["lstm_prediction"]
        - comparison_df["actual_rul"]
    )

    comparison_df["ridge_error"] = (
        comparison_df["ridge_prediction"]
        - comparison_df["actual_rul"]
    )

    comparison_df["lstm_absolute_error"] = np.abs(
        comparison_df["lstm_error"]
    )

    comparison_df["ridge_absolute_error"] = np.abs(
        comparison_df["ridge_error"]
    )

    comparison_df["better_model"] = np.where(
        comparison_df["lstm_absolute_error"]
        < comparison_df["ridge_absolute_error"],
        "LSTM",
        np.where(
            comparison_df["ridge_absolute_error"]
            < comparison_df["lstm_absolute_error"],
            "Ridge",
            "Tie",
        ),
    )

    return comparison_df


def save_comparison_metrics(comparison_df, dataset):
    """Calculate and save summary metrics for both models."""

    y_true = comparison_df["actual_rul"].to_numpy()

    model_predictions = {
        "LSTM": comparison_df[
            "lstm_prediction"
        ].to_numpy(),
        "Ridge": comparison_df[
            "ridge_prediction"
        ].to_numpy(),
    }

    rows = []

    for model_name, predictions in model_predictions.items():
        metrics = calculate_metrics(
            y_true,
            predictions,
        )

        rows.append(
            {
                "model": model_name,
                "dataset": dataset,
                **metrics,
            }
        )

    metrics_df = pd.DataFrame(rows)

    metrics_df = metrics_df.sort_values(
        by="rmse",
    )

    metrics_path = os.path.join(
        RESULTS_DIR,
        f"comparison_metrics_{dataset}.csv",
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    return metrics_df, metrics_path


def plot_actual_vs_predicted(comparison_df, dataset):
    """Plot predicted RUL against actual RUL."""

    actual = comparison_df["actual_rul"]

    minimum_value = min(
        actual.min(),
        comparison_df["lstm_prediction"].min(),
        comparison_df["ridge_prediction"].min(),
    )

    maximum_value = max(
        actual.max(),
        comparison_df["lstm_prediction"].max(),
        comparison_df["ridge_prediction"].max(),
    )

    plt.figure(figsize=(9, 7))

    plt.scatter(
        actual,
        comparison_df["lstm_prediction"],
        alpha=0.70,
        label="LSTM",
    )

    plt.scatter(
        actual,
        comparison_df["ridge_prediction"],
        alpha=0.70,
        label="Ridge",
        marker="x",
    )

    plt.plot(
        [minimum_value, maximum_value],
        [minimum_value, maximum_value],
        linestyle="--",
        label="Perfect prediction",
    )

    plt.xlabel("Actual Remaining Useful Life")
    plt.ylabel("Predicted Remaining Useful Life")
    plt.title(
        f"Actual vs Predicted RUL — {dataset}"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        f"actual_vs_predicted_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_error_distribution(comparison_df, dataset):
    """Plot the error distributions for both models."""

    plt.figure(figsize=(9, 6))

    plt.hist(
        comparison_df["lstm_error"],
        bins=20,
        alpha=0.60,
        label="LSTM error",
    )

    plt.hist(
        comparison_df["ridge_error"],
        bins=20,
        alpha=0.60,
        label="Ridge error",
    )

    plt.axvline(
        0,
        linestyle="--",
        label="Zero error",
    )

    plt.xlabel(
        "Prediction Error "
        "(Predicted RUL − Actual RUL)"
    )

    plt.ylabel("Number of test engines")
    plt.title(
        f"Prediction Error Distribution — {dataset}"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        f"error_distribution_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_sorted_predictions(comparison_df, dataset):
    """Plot predictions after sorting engines by actual RUL."""

    sorted_df = comparison_df.sort_values(
        by="actual_rul"
    ).reset_index(drop=True)

    engine_position = np.arange(
        len(sorted_df)
    )

    plt.figure(figsize=(12, 6))

    plt.plot(
        engine_position,
        sorted_df["actual_rul"],
        linewidth=2.5,
        label="Actual RUL",
    )

    plt.plot(
        engine_position,
        sorted_df["lstm_prediction"],
        linewidth=1.8,
        label="LSTM prediction",
    )

    plt.plot(
        engine_position,
        sorted_df["ridge_prediction"],
        linewidth=1.8,
        label="Ridge prediction",
    )

    plt.xlabel(
        "Test engines sorted by actual RUL"
    )

    plt.ylabel("Remaining Useful Life")
    plt.title(
        f"RUL Prediction Comparison — {dataset}"
    )
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = os.path.join(
        FIGURES_DIR,
        f"sorted_predictions_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def save_worst_predictions(comparison_df, dataset):
    """Save engines with the largest model errors."""

    worst_lstm = comparison_df.nlargest(
        10,
        "lstm_absolute_error",
    ).copy()

    worst_lstm["model"] = "LSTM"

    worst_ridge = comparison_df.nlargest(
        10,
        "ridge_absolute_error",
    ).copy()

    worst_ridge["model"] = "Ridge"

    worst_predictions = pd.concat(
        [worst_lstm, worst_ridge],
        ignore_index=True,
    )

    output_path = os.path.join(
        RESULTS_DIR,
        f"worst_predictions_{dataset}.csv",
    )

    worst_predictions.to_csv(
        output_path,
        index=False,
    )

    return output_path


def compare_models(dataset):
    """Run the complete Ridge-versus-LSTM comparison."""

    comparison_df = load_predictions(dataset)

    comparison_path = os.path.join(
        RESULTS_DIR,
        f"model_comparison_{dataset}.csv",
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    metrics_df, metrics_path = save_comparison_metrics(
        comparison_df,
        dataset,
    )

    scatter_path = plot_actual_vs_predicted(
        comparison_df,
        dataset,
    )

    error_path = plot_error_distribution(
        comparison_df,
        dataset,
    )

    sorted_path = plot_sorted_predictions(
        comparison_df,
        dataset,
    )

    worst_path = save_worst_predictions(
        comparison_df,
        dataset,
    )

    lstm_wins = int(
        (comparison_df["better_model"] == "LSTM").sum()
    )

    ridge_wins = int(
        (comparison_df["better_model"] == "Ridge").sum()
    )

    ties = int(
        (comparison_df["better_model"] == "Tie").sum()
    )

    print("\nModel comparison")
    print(metrics_df.to_string(index=False))

    print("\nEngine-level comparison")
    print(f"LSTM more accurate:  {lstm_wins} engines")
    print(f"Ridge more accurate: {ridge_wins} engines")
    print(f"Ties:                {ties} engines")

    print("\nSaved outputs")
    print(f"Comparison data: {comparison_path}")
    print(f"Metrics:         {metrics_path}")
    print(f"Worst cases:     {worst_path}")
    print(f"Scatter plot:    {scatter_path}")
    print(f"Error plot:      {error_path}")
    print(f"Sorted plot:     {sorted_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["FD001", "FD002", "FD003", "FD004"],
    )

    arguments = parser.parse_args()

    compare_models(
        dataset=arguments.dataset,
    )