import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras


DATA_DIR = "data/processed_v2"
MODELS_DIR = "models_v2"
RESULTS_DIR = "results_v2"
UNCERTAINTY_DIR = os.path.join(
    RESULTS_DIR,
    "uncertainty",
)

RANDOM_SEED = 42
RUL_CAP = 125.0

os.makedirs(
    UNCERTAINTY_DIR,
    exist_ok=True,
)


def load_data(dataset):
    """Load validation and independent test data."""

    dataset_path = os.path.join(
        DATA_DIR,
        f"{dataset}.pkl",
    )

    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Processed dataset not found: {dataset_path}"
        )

    data = joblib.load(dataset_path)

    return {
        "X_val": data["X_val"],
        "y_val": data["y_val"],
        "X_test": data["X_test"],
        "y_test": data["y_test"],
        "test_units": data["test_units"],
        "rul_cap": float(
            data.get("rul_cap", RUL_CAP)
        ),
    }


def load_model(model_name, dataset):
    """Load the trained neural-network model."""

    model_path = os.path.join(
        MODELS_DIR,
        f"{model_name}_{dataset}.keras",
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = keras.models.load_model(
        model_path
    )

    return model, model_path


def select_validation_subset(
    X_val,
    y_val,
    max_samples,
):
    """Select a reproducible validation subset."""

    if len(X_val) <= max_samples:
        return X_val, y_val

    random_generator = np.random.default_rng(
        RANDOM_SEED
    )

    selected_indices = random_generator.choice(
        len(X_val),
        size=max_samples,
        replace=False,
    )

    return (
        X_val[selected_indices],
        y_val[selected_indices],
    )


def monte_carlo_predictions(
    model,
    X,
    passes,
    batch_size=64,
):
    """
    Run repeated stochastic forward passes.

    training=True keeps dropout active during inference.
    """

    prediction_runs = []

    for pass_number in range(passes):
        batch_predictions = []

        for start_index in range(
            0,
            len(X),
            batch_size,
        ):
            end_index = start_index + batch_size

            batch = tf.convert_to_tensor(
                X[start_index:end_index],
                dtype=tf.float32,
            )

            predictions = model(
                batch,
                training=True,
            )

            batch_predictions.append(
                predictions.numpy().reshape(-1)
            )

        complete_predictions = np.concatenate(
            batch_predictions
        )

        complete_predictions = np.clip(
            complete_predictions,
            0.0,
            RUL_CAP,
        )

        prediction_runs.append(
            complete_predictions
        )

        if (
            pass_number == 0
            or (pass_number + 1) % 20 == 0
            or pass_number + 1 == passes
        ):
            print(
                f"Completed uncertainty pass "
                f"{pass_number + 1}/{passes}"
            )

    return np.asarray(
        prediction_runs,
        dtype=np.float32,
    )


def calculate_conformal_multiplier(
    y_true,
    prediction_mean,
    prediction_std,
    confidence_level,
):
    """
    Calibrate adaptive uncertainty intervals on validation data.

    Larger Monte Carlo standard deviation produces a wider interval.
    Validation residuals calibrate the final interval width.
    """

    alpha = 1.0 - confidence_level

    uncertainty_scale = prediction_std + 1.0

    conformity_scores = (
        np.abs(
            y_true - prediction_mean
        )
        / uncertainty_scale
    )

    quantile_level = min(
        1.0,
        np.ceil(
            (len(conformity_scores) + 1)
            * (1.0 - alpha)
        )
        / len(conformity_scores),
    )

    multiplier = float(
        np.quantile(
            conformity_scores,
            quantile_level,
            method="higher",
        )
    )

    return multiplier


def assign_confidence_level(
    interval_width,
    mc_std,
):
    """Convert uncertainty values into a simple user-facing label."""

    if interval_width <= 25 and mc_std <= 5:
        return "High"

    if interval_width <= 50 and mc_std <= 10:
        return "Medium"

    return "Low"


def assign_health_status(
    predicted_rul,
):
    """Convert predicted RUL into a maintenance status."""

    if predicted_rul <= 20:
        return "Critical"

    if predicted_rul <= 50:
        return "Warning"

    if predicted_rul <= 80:
        return "Monitor"

    return "Healthy"


def assign_recommendation(
    predicted_rul,
    lower_bound,
    confidence,
):
    """Generate a maintenance recommendation."""

    if lower_bound <= 20:
        return "Immediate inspection recommended"

    if predicted_rul <= 50:
        return "Plan maintenance soon"

    if confidence == "Low":
        return "Manual review required"

    if predicted_rul <= 80:
        return "Increase monitoring frequency"

    return "Continue normal monitoring"


def build_uncertainty_results(
    units,
    y_true,
    prediction_runs,
    multiplier,
    rul_cap,
):
    """Create engine-level uncertainty and maintenance results."""

    prediction_mean = np.mean(
        prediction_runs,
        axis=0,
    )

    prediction_median = np.median(
        prediction_runs,
        axis=0,
    )

    prediction_std = np.std(
        prediction_runs,
        axis=0,
    )

    percentile_05 = np.percentile(
        prediction_runs,
        5,
        axis=0,
    )

    percentile_95 = np.percentile(
        prediction_runs,
        95,
        axis=0,
    )

    calibrated_half_width = (
        multiplier
        * (prediction_std + 1.0)
    )

    lower_bound = np.clip(
        prediction_mean - calibrated_half_width,
        0.0,
        rul_cap,
    )

    upper_bound = np.clip(
        prediction_mean + calibrated_half_width,
        0.0,
        rul_cap,
    )

    interval_width = (
        upper_bound - lower_bound
    )

    rows = []

    for index, unit_id in enumerate(units):
        confidence = assign_confidence_level(
            interval_width=float(
                interval_width[index]
            ),
            mc_std=float(
                prediction_std[index]
            ),
        )

        health_status = assign_health_status(
            predicted_rul=float(
                prediction_mean[index]
            )
        )

        recommendation = assign_recommendation(
            predicted_rul=float(
                prediction_mean[index]
            ),
            lower_bound=float(
                lower_bound[index]
            ),
            confidence=confidence,
        )

        rows.append(
            {
                "unit": int(unit_id),
                "actual_rul": float(
                    y_true[index]
                ),
                "predicted_rul_mean": float(
                    prediction_mean[index]
                ),
                "predicted_rul_median": float(
                    prediction_median[index]
                ),
                "mc_standard_deviation": float(
                    prediction_std[index]
                ),
                "raw_percentile_05": float(
                    percentile_05[index]
                ),
                "raw_percentile_95": float(
                    percentile_95[index]
                ),
                "calibrated_lower_bound": float(
                    lower_bound[index]
                ),
                "calibrated_upper_bound": float(
                    upper_bound[index]
                ),
                "interval_width": float(
                    interval_width[index]
                ),
                "prediction_error": float(
                    prediction_mean[index]
                    - y_true[index]
                ),
                "absolute_error": float(
                    abs(
                        prediction_mean[index]
                        - y_true[index]
                    )
                ),
                "confidence": confidence,
                "health_status": health_status,
                "recommendation": recommendation,
                "actual_inside_interval": bool(
                    lower_bound[index]
                    <= y_true[index]
                    <= upper_bound[index]
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_selected_units(
    results_df,
    selected_units,
    dataset,
    model_name,
):
    """Plot RUL prediction intervals for selected engines."""

    if selected_units:
        plot_df = results_df[
            results_df["unit"].isin(
                selected_units
            )
        ].copy()
    else:
        plot_df = (
            results_df
            .nlargest(
                10,
                "absolute_error",
            )
            .copy()
        )

    if plot_df.empty:
        raise ValueError(
            "No matching units found for plotting."
        )

    plot_df = plot_df.sort_values(
        by="unit"
    ).reset_index(drop=True)

    positions = np.arange(
        len(plot_df)
    )

    predictions = plot_df[
        "predicted_rul_mean"
    ].to_numpy()

    lower_errors = (
        predictions
        - plot_df[
            "calibrated_lower_bound"
        ].to_numpy()
    )

    upper_errors = (
        plot_df[
            "calibrated_upper_bound"
        ].to_numpy()
        - predictions
    )

    plt.figure(
        figsize=(11, 7)
    )

    plt.errorbar(
        positions,
        predictions,
        yerr=[
            lower_errors,
            upper_errors,
        ],
        fmt="o",
        capsize=5,
        label="Predicted RUL with calibrated interval",
    )

    plt.scatter(
        positions,
        plot_df["actual_rul"],
        marker="x",
        s=80,
        label="Actual RUL",
    )

    plt.xticks(
        positions,
        [
            f"Unit {unit}"
            for unit in plot_df["unit"]
        ],
    )

    plt.ylabel(
        "Remaining Useful Life"
    )

    plt.xlabel(
        "Test engine"
    )

    plt.title(
        f"Prediction Uncertainty — "
        f"{model_name.upper()} {dataset}"
    )

    plt.legend()
    plt.grid(
        axis="y",
        alpha=0.25,
    )
    plt.tight_layout()

    output_path = os.path.join(
        UNCERTAINTY_DIR,
        f"uncertainty_{model_name}_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def print_selected_units(
    results_df,
    selected_units,
):
    """Print selected engine uncertainty results."""

    if selected_units:
        display_df = results_df[
            results_df["unit"].isin(
                selected_units
            )
        ].copy()
    else:
        display_df = (
            results_df
            .nlargest(
                10,
                "absolute_error",
            )
            .copy()
        )

    columns = [
        "unit",
        "actual_rul",
        "predicted_rul_mean",
        "calibrated_lower_bound",
        "calibrated_upper_bound",
        "mc_standard_deviation",
        "confidence",
        "health_status",
        "recommendation",
        "actual_inside_interval",
    ]

    print(
        display_df[columns]
        .sort_values("unit")
        .to_string(index=False)
    )


def run_uncertainty_analysis(
    model_name,
    dataset,
    passes,
    confidence_level,
    max_validation_samples,
    selected_units,
):
    """Run uncertainty estimation and conformal calibration."""

    np.random.seed(
        RANDOM_SEED
    )

    tf.keras.utils.set_random_seed(
        RANDOM_SEED
    )

    data = load_data(
        dataset
    )

    model, model_path = load_model(
        model_name,
        dataset,
    )

    (
        X_val_subset,
        y_val_subset,
    ) = select_validation_subset(
        data["X_val"],
        data["y_val"],
        max_validation_samples,
    )

    print(
        f"Model: {model_path}"
    )

    print(
        f"Monte Carlo passes: {passes}"
    )

    print(
        f"Validation samples used: "
        f"{len(X_val_subset)}"
    )

    print(
        "\nRunning validation uncertainty..."
    )

    validation_runs = monte_carlo_predictions(
        model,
        X_val_subset,
        passes,
    )

    validation_mean = np.mean(
        validation_runs,
        axis=0,
    )

    validation_std = np.std(
        validation_runs,
        axis=0,
    )

    multiplier = calculate_conformal_multiplier(
        y_true=y_val_subset,
        prediction_mean=validation_mean,
        prediction_std=validation_std,
        confidence_level=confidence_level,
    )

    print(
        f"\nCalibrated interval multiplier: "
        f"{multiplier:.4f}"
    )

    print(
        "\nRunning test uncertainty..."
    )

    test_runs = monte_carlo_predictions(
        model,
        data["X_test"],
        passes,
    )

    results_df = build_uncertainty_results(
        units=data["test_units"],
        y_true=data["y_test"],
        prediction_runs=test_runs,
        multiplier=multiplier,
        rul_cap=data["rul_cap"],
    )

    coverage = float(
        results_df[
            "actual_inside_interval"
        ].mean()
    )

    average_width = float(
        results_df[
            "interval_width"
        ].mean()
    )

    results_path = os.path.join(
        UNCERTAINTY_DIR,
        f"uncertainty_results_"
        f"{model_name}_{dataset}.csv",
    )

    results_df.to_csv(
        results_path,
        index=False,
    )

    plot_path = plot_selected_units(
        results_df=results_df,
        selected_units=selected_units,
        dataset=dataset,
        model_name=model_name,
    )

    print(
        "\nUncertainty summary"
    )

    print(
        f"Requested confidence level: "
        f"{confidence_level:.1%}"
    )

    print(
        f"Observed test coverage:      "
        f"{coverage:.1%}"
    )

    print(
        f"Average interval width:      "
        f"{average_width:.2f} cycles"
    )

    print(
        "\nSelected engine results"
    )

    print_selected_units(
        results_df,
        selected_units,
    )

    print(
        "\nSaved outputs"
    )

    print(
        f"Results: {results_path}"
    )

    print(
        f"Plot:    {plot_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="lstm",
        choices=[
            "lstm",
            "gru",
            "transformer",
        ],
    )

    parser.add_argument(
        "--dataset",
        required=True,
        choices=[
            "FD001",
            "FD002",
            "FD003",
            "FD004",
        ],
    )

    parser.add_argument(
        "--passes",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--confidence",
        type=float,
        default=0.90,
    )

    parser.add_argument(
        "--max_validation_samples",
        type=int,
        default=1000,
    )

    parser.add_argument(
        "--units",
        type=int,
        nargs="*",
        default=None,
    )

    arguments = parser.parse_args()

    if not 0.0 < arguments.confidence < 1.0:
        raise ValueError(
            "Confidence must be between 0 and 1."
        )

    if arguments.passes < 10:
        raise ValueError(
            "Use at least 10 Monte Carlo passes."
        )

    run_uncertainty_analysis(
        model_name=arguments.model,
        dataset=arguments.dataset,
        passes=arguments.passes,
        confidence_level=arguments.confidence,
        max_validation_samples=(
            arguments.max_validation_samples
        ),
        selected_units=arguments.units,
    )