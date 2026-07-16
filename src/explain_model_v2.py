import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)
from tensorflow import keras


DATA_DIR = "data/processed_v2"
MODELS_DIR = "models_v2"
RESULTS_DIR = "results_v2"
EXPLANATIONS_DIR = os.path.join(
    RESULTS_DIR,
    "explanations",
)

RANDOM_SEED = 42

os.makedirs(
    EXPLANATIONS_DIR,
    exist_ok=True,
)


def load_data(dataset):
    """Load processed validation and test data."""

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
        "feature_names": data["feature_names"],
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


def calculate_rmse(
    y_true,
    y_pred,
):
    """Calculate root mean squared error."""

    return float(
        np.sqrt(
            mean_squared_error(
                y_true,
                y_pred,
            )
        )
    )


def predict(
    model,
    X,
):
    """Generate flattened non-negative RUL predictions."""

    predictions = model.predict(
        X,
        batch_size=64,
        verbose=0,
    ).reshape(-1)

    return np.maximum(
        predictions,
        0.0,
    )


def calculate_global_importance(
    model,
    X_val,
    y_val,
    feature_names,
):
    """
    Measure global feature importance using occlusion.

    Each feature is replaced by zero across all 30 cycles.
    Because the inputs are standardized, zero represents
    the average training value.
    """

    baseline_predictions = predict(
        model,
        X_val,
    )

    baseline_rmse = calculate_rmse(
        y_val,
        baseline_predictions,
    )

    baseline_mae = float(
        mean_absolute_error(
            y_val,
            baseline_predictions,
        )
    )

    rows = []

    for feature_index, feature_name in enumerate(
        feature_names
    ):
        occluded_data = X_val.copy()

        occluded_data[
            :,
            :,
            feature_index,
        ] = 0.0

        occluded_predictions = predict(
            model,
            occluded_data,
        )

        occluded_rmse = calculate_rmse(
            y_val,
            occluded_predictions,
        )

        occluded_mae = float(
            mean_absolute_error(
                y_val,
                occluded_predictions,
            )
        )

        prediction_change = (
            occluded_predictions
            - baseline_predictions
        )

        rows.append(
            {
                "feature": feature_name,
                "baseline_rmse": baseline_rmse,
                "occluded_rmse": occluded_rmse,
                "rmse_increase": (
                    occluded_rmse
                    - baseline_rmse
                ),
                "baseline_mae": baseline_mae,
                "occluded_mae": occluded_mae,
                "mae_increase": (
                    occluded_mae
                    - baseline_mae
                ),
                "mean_absolute_prediction_change": float(
                    np.mean(
                        np.abs(
                            prediction_change
                        )
                    )
                ),
                "mean_signed_prediction_change": float(
                    np.mean(
                        prediction_change
                    )
                ),
            }
        )

        print(
            f"Processed global feature: "
            f"{feature_name}"
        )

    importance_df = pd.DataFrame(
        rows
    )

    importance_df = importance_df.sort_values(
        by="rmse_increase",
        ascending=False,
    ).reset_index(drop=True)

    return (
        importance_df,
        baseline_rmse,
        baseline_mae,
    )


def calculate_local_explanation(
    model,
    X_test,
    y_test,
    test_units,
    feature_names,
    unit_id,
):
    """Explain one test-engine prediction."""

    matching_indices = np.where(
        test_units == unit_id
    )[0]

    if len(matching_indices) == 0:
        raise ValueError(
            f"Unit {unit_id} does not exist."
        )

    unit_index = int(
        matching_indices[0]
    )

    unit_window = X_test[
        unit_index:unit_index + 1
    ]

    actual_rul = float(
        y_test[unit_index]
    )

    original_prediction = float(
        predict(
            model,
            unit_window,
        )[0]
    )

    rows = []

    for feature_index, feature_name in enumerate(
        feature_names
    ):
        occluded_window = unit_window.copy()

        occluded_window[
            :,
            :,
            feature_index,
        ] = 0.0

        occluded_prediction = float(
            predict(
                model,
                occluded_window,
            )[0]
        )

        signed_effect = (
            original_prediction
            - occluded_prediction
        )

        rows.append(
            {
                "unit": unit_id,
                "feature": feature_name,
                "actual_rul": actual_rul,
                "original_prediction": (
                    original_prediction
                ),
                "occluded_prediction": (
                    occluded_prediction
                ),
                "signed_effect_on_prediction": (
                    signed_effect
                ),
                "absolute_effect_on_prediction": abs(
                    signed_effect
                ),
            }
        )

        print(
            f"Processed local feature: "
            f"{feature_name}"
        )

    explanation_df = pd.DataFrame(
        rows
    )

    explanation_df = explanation_df.sort_values(
        by="absolute_effect_on_prediction",
        ascending=False,
    ).reset_index(drop=True)

    return (
        explanation_df,
        actual_rul,
        original_prediction,
    )


def plot_global_importance(
    importance_df,
    dataset,
    model_name,
    top_features,
):
    """Plot the most globally important features."""

    plot_df = (
        importance_df
        .head(top_features)
        .sort_values(
            by="rmse_increase",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        plot_df["feature"],
        plot_df["rmse_increase"],
    )

    plt.xlabel(
        "Increase in validation RMSE after occlusion"
    )

    plt.ylabel(
        "Input feature"
    )

    plt.title(
        f"Global Occlusion Importance — "
        f"{model_name.upper()} {dataset}"
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        EXPLANATIONS_DIR,
        f"global_importance_{model_name}_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_local_explanation(
    explanation_df,
    dataset,
    model_name,
    unit_id,
    top_features,
):
    """Plot features affecting one engine prediction."""

    plot_df = (
        explanation_df
        .head(top_features)
        .sort_values(
            by="signed_effect_on_prediction",
            ascending=True,
        )
    )

    plt.figure(
        figsize=(10, 7)
    )

    plt.barh(
        plot_df["feature"],
        plot_df[
            "signed_effect_on_prediction"
        ],
    )

    plt.axvline(
        0,
        linestyle="--",
        linewidth=1,
    )

    plt.xlabel(
        "Effect on predicted RUL in cycles"
    )

    plt.ylabel(
        "Input feature"
    )

    plt.title(
        f"Local Occlusion Explanation — "
        f"{dataset} Unit {unit_id}"
    )

    plt.grid(
        axis="x",
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        EXPLANATIONS_DIR,
        f"local_explanation_{model_name}_"
        f"{dataset}_unit_{unit_id}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def explain_model(
    model_name,
    dataset,
    unit_id,
    max_samples,
    top_features,
):
    """Run global and local model explanations."""

    data = load_data(
        dataset
    )

    model, model_path = load_model(
        model_name,
        dataset,
    )

    X_val_subset, y_val_subset = (
        select_validation_subset(
            data["X_val"],
            data["y_val"],
            max_samples,
        )
    )

    print(
        f"Model: {model_path}"
    )

    print(
        f"Validation samples used: "
        f"{len(X_val_subset)}"
    )

    print(
        "\nCalculating global importance..."
    )

    (
        global_df,
        baseline_rmse,
        baseline_mae,
    ) = calculate_global_importance(
        model,
        X_val_subset,
        y_val_subset,
        data["feature_names"],
    )

    global_csv_path = os.path.join(
        EXPLANATIONS_DIR,
        f"global_importance_"
        f"{model_name}_{dataset}.csv",
    )

    global_df.to_csv(
        global_csv_path,
        index=False,
    )

    global_plot_path = plot_global_importance(
        global_df,
        dataset,
        model_name,
        top_features,
    )

    print(
        f"\nCalculating local explanation "
        f"for Unit {unit_id}..."
    )

    (
        local_df,
        actual_rul,
        original_prediction,
    ) = calculate_local_explanation(
        model,
        data["X_test"],
        data["y_test"],
        data["test_units"],
        data["feature_names"],
        unit_id,
    )

    local_csv_path = os.path.join(
        EXPLANATIONS_DIR,
        f"local_explanation_"
        f"{model_name}_{dataset}_"
        f"unit_{unit_id}.csv",
    )

    local_df.to_csv(
        local_csv_path,
        index=False,
    )

    local_plot_path = plot_local_explanation(
        local_df,
        dataset,
        model_name,
        unit_id,
        top_features,
    )

    print("\nGlobal explanation")
    print(
        f"Baseline validation RMSE: "
        f"{baseline_rmse:.4f}"
    )
    print(
        f"Baseline validation MAE:  "
        f"{baseline_mae:.4f}"
    )

    print(
        "\nMost important global features"
    )

    print(
        global_df[
            [
                "feature",
                "rmse_increase",
                "mean_absolute_prediction_change",
            ]
        ]
        .head(top_features)
        .to_string(index=False)
    )

    print(
        "\nLocal explanation"
    )

    print(
        f"Unit:                {unit_id}"
    )
    print(
        f"Actual RUL:          {actual_rul:.2f}"
    )
    print(
        f"Original prediction: "
        f"{original_prediction:.2f}"
    )

    print(
        "\nStrongest local feature effects"
    )

    print(
        local_df[
            [
                "feature",
                "occluded_prediction",
                "signed_effect_on_prediction",
            ]
        ]
        .head(top_features)
        .to_string(index=False)
    )

    print("\nSaved outputs")
    print(
        f"Global CSV:  {global_csv_path}"
    )
    print(
        f"Global plot: {global_plot_path}"
    )
    print(
        f"Local CSV:   {local_csv_path}"
    )
    print(
        f"Local plot:  {local_plot_path}"
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
        "--unit",
        type=int,
        default=79,
    )

    parser.add_argument(
        "--max_samples",
        type=int,
        default=512,
    )

    parser.add_argument(
        "--top_features",
        type=int,
        default=10,
    )

    arguments = parser.parse_args()

    explain_model(
        model_name=arguments.model,
        dataset=arguments.dataset,
        unit_id=arguments.unit,
        max_samples=arguments.max_samples,
        top_features=arguments.top_features,
    )