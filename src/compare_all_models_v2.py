import argparse
import itertools
import os

import joblib
import matplotlib.pyplot as plt
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
COMPARISON_DIR = os.path.join(
    RESULTS_DIR,
    "all_model_comparison",
)

MODEL_NAMES = [
    "ridge",
    "lstm",
    "gru",
    "transformer",
]

RUL_CAP = 125.0

os.makedirs(
    COMPARISON_DIR,
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
    }


def flatten_windows(X):
    """Flatten sequence windows for Ridge regression."""

    return X.reshape(
        X.shape[0],
        -1,
    )


def load_ridge_model(dataset):
    """Load the trained Ridge baseline."""

    model_path = os.path.join(
        MODELS_DIR,
        f"baseline_ridge_{dataset}.joblib",
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Ridge model not found: {model_path}"
        )

    return (
        joblib.load(model_path),
        model_path,
    )


def load_neural_model(
    model_name,
    dataset,
):
    """Load one trained neural-network model."""

    model_path = os.path.join(
        MODELS_DIR,
        f"{model_name}_{dataset}.keras",
    )

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    return (
        keras.models.load_model(model_path),
        model_path,
    )


def predict_neural_model(
    model,
    X,
):
    """Generate deterministic neural-network predictions."""

    predictions = model.predict(
        X,
        batch_size=64,
        verbose=0,
    ).reshape(-1)

    return np.clip(
        predictions,
        0.0,
        RUL_CAP,
    )


def calculate_nasa_score(
    y_true,
    y_pred,
):
    """Calculate the asymmetric NASA C-MAPSS score."""

    error = y_pred - y_true

    penalties = np.where(
        error < 0,
        np.exp(-error / 13.0) - 1.0,
        np.exp(error / 10.0) - 1.0,
    )

    return float(
        np.sum(penalties)
    )


def calculate_metrics(
    y_true,
    y_pred,
):
    """Calculate accuracy and safety-related metrics."""

    y_pred = np.clip(
        y_pred,
        0.0,
        RUL_CAP,
    )

    error = y_pred - y_true

    return {
        "rmse": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred,
                )
            )
        ),
        "mae": float(
            mean_absolute_error(
                y_true,
                y_pred,
            )
        ),
        "r2": float(
            r2_score(
                y_true,
                y_pred,
            )
        ),
        "nasa_score": calculate_nasa_score(
            y_true,
            y_pred,
        ),
        "mean_error": float(
            np.mean(error)
        ),
        "overprediction_rate": float(
            np.mean(error > 0)
        ),
        "dangerous_overprediction_rate": float(
            np.mean(error > 10)
        ),
        "maximum_absolute_error": float(
            np.max(
                np.abs(error)
            )
        ),
    }


def generate_predictions(dataset):
    """Generate validation and test predictions for all models."""

    data = load_data(
        dataset
    )

    validation_predictions = {}
    test_predictions = {}
    model_paths = {}

    ridge_model, ridge_path = load_ridge_model(
        dataset
    )

    validation_predictions["ridge"] = np.clip(
        ridge_model.predict(
            flatten_windows(
                data["X_val"]
            )
        ),
        0.0,
        RUL_CAP,
    )

    test_predictions["ridge"] = np.clip(
        ridge_model.predict(
            flatten_windows(
                data["X_test"]
            )
        ),
        0.0,
        RUL_CAP,
    )

    model_paths["ridge"] = ridge_path

    for model_name in [
        "lstm",
        "gru",
        "transformer",
    ]:
        model, model_path = load_neural_model(
            model_name,
            dataset,
        )

        validation_predictions[
            model_name
        ] = predict_neural_model(
            model,
            data["X_val"],
        )

        test_predictions[
            model_name
        ] = predict_neural_model(
            model,
            data["X_test"],
        )

        model_paths[
            model_name
        ] = model_path

    return (
        data,
        validation_predictions,
        test_predictions,
        model_paths,
    )


def generate_weight_combinations(
    number_of_models,
    weight_step,
):
    """
    Generate non-negative weights that sum to one.

    For a step of 0.05, weights are tested in increments of 5%.
    """

    divisions = int(
        round(
            1.0 / weight_step
        )
    )

    for combination in itertools.product(
        range(divisions + 1),
        repeat=number_of_models,
    ):
        if sum(combination) != divisions:
            continue

        yield np.asarray(
            combination,
            dtype=float,
        ) / divisions


def calculate_weighted_prediction(
    prediction_dictionary,
    weights,
):
    """Combine model predictions using supplied weights."""

    prediction_matrix = np.column_stack(
        [
            prediction_dictionary[
                model_name
            ]
            for model_name in MODEL_NAMES
        ]
    )

    return prediction_matrix @ weights


def find_best_ensemble_weights(
    validation_predictions,
    y_val,
    weight_step,
):
    """
    Select ensemble weights using validation data only.

    The independent test targets are not used during weight selection.
    """

    best_weights = None
    best_metrics = None

    combinations_tested = 0

    for weights in generate_weight_combinations(
        number_of_models=len(
            MODEL_NAMES
        ),
        weight_step=weight_step,
    ):
        ensemble_prediction = (
            calculate_weighted_prediction(
                validation_predictions,
                weights,
            )
        )

        metrics = calculate_metrics(
            y_val,
            ensemble_prediction,
        )

        combinations_tested += 1

        if best_metrics is None:
            best_weights = weights
            best_metrics = metrics
            continue

        current_key = (
            metrics["rmse"],
            metrics["nasa_score"],
            metrics["mae"],
        )

        best_key = (
            best_metrics["rmse"],
            best_metrics["nasa_score"],
            best_metrics["mae"],
        )

        if current_key < best_key:
            best_weights = weights
            best_metrics = metrics

    return (
        best_weights,
        best_metrics,
        combinations_tested,
    )


def build_metrics_table(
    y_val,
    y_test,
    validation_predictions,
    test_predictions,
    ensemble_validation_prediction,
    ensemble_test_prediction,
):
    """Build validation and test metrics for every model."""

    rows = []

    all_validation_predictions = {
        **validation_predictions,
        "ensemble": ensemble_validation_prediction,
    }

    all_test_predictions = {
        **test_predictions,
        "ensemble": ensemble_test_prediction,
    }

    for model_name in [
        *MODEL_NAMES,
        "ensemble",
    ]:
        validation_metrics = calculate_metrics(
            y_val,
            all_validation_predictions[
                model_name
            ],
        )

        test_metrics = calculate_metrics(
            y_test,
            all_test_predictions[
                model_name
            ],
        )

        rows.append(
            {
                "model": model_name,
                "validation_rmse": validation_metrics[
                    "rmse"
                ],
                "validation_mae": validation_metrics[
                    "mae"
                ],
                "validation_r2": validation_metrics[
                    "r2"
                ],
                "validation_nasa_score": validation_metrics[
                    "nasa_score"
                ],
                "test_rmse": test_metrics[
                    "rmse"
                ],
                "test_mae": test_metrics[
                    "mae"
                ],
                "test_r2": test_metrics[
                    "r2"
                ],
                "test_nasa_score": test_metrics[
                    "nasa_score"
                ],
                "test_mean_error": test_metrics[
                    "mean_error"
                ],
                "test_overprediction_rate": test_metrics[
                    "overprediction_rate"
                ],
                "test_dangerous_overprediction_rate": test_metrics[
                    "dangerous_overprediction_rate"
                ],
                "test_maximum_absolute_error": test_metrics[
                    "maximum_absolute_error"
                ],
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values(
            by="test_rmse"
        )
        .reset_index(drop=True)
    )


def build_engine_comparison(
    data,
    test_predictions,
    ensemble_prediction,
):
    """Create engine-level predictions and identify the best model."""

    comparison_df = pd.DataFrame(
        {
            "unit": data["test_units"],
            "actual_rul": data["y_test"],
            "ridge_prediction": test_predictions[
                "ridge"
            ],
            "lstm_prediction": test_predictions[
                "lstm"
            ],
            "gru_prediction": test_predictions[
                "gru"
            ],
            "transformer_prediction": test_predictions[
                "transformer"
            ],
            "ensemble_prediction": ensemble_prediction,
        }
    )

    model_columns = {
        "Ridge": "ridge_prediction",
        "LSTM": "lstm_prediction",
        "GRU": "gru_prediction",
        "Transformer": "transformer_prediction",
        "Ensemble": "ensemble_prediction",
    }

    for model_name, prediction_column in (
        model_columns.items()
    ):
        comparison_df[
            f"{model_name.lower()}_error"
        ] = (
            comparison_df[
                prediction_column
            ]
            - comparison_df[
                "actual_rul"
            ]
        )

        comparison_df[
            f"{model_name.lower()}_absolute_error"
        ] = np.abs(
            comparison_df[
                f"{model_name.lower()}_error"
            ]
        )

    absolute_error_columns = {
        model_name: (
            f"{model_name.lower()}_absolute_error"
        )
        for model_name in model_columns
    }

    comparison_df[
        "best_model"
    ] = comparison_df.apply(
        lambda row: min(
            absolute_error_columns,
            key=lambda model_name: row[
                absolute_error_columns[
                    model_name
                ]
            ],
        ),
        axis=1,
    )

    prediction_matrix = np.column_stack(
        [
            comparison_df[
                prediction_column
            ].to_numpy()
            for prediction_column in (
                model_columns.values()
            )
        ]
    )

    comparison_df[
        "model_prediction_spread"
    ] = (
        np.max(
            prediction_matrix,
            axis=1,
        )
        - np.min(
            prediction_matrix,
            axis=1,
        )
    )

    return comparison_df


def build_win_summary(
    comparison_df,
):
    """Count how often each model is closest to the true RUL."""

    win_counts = (
        comparison_df[
            "best_model"
        ]
        .value_counts()
        .rename_axis(
            "model"
        )
        .reset_index(
            name="engine_wins"
        )
    )

    win_counts[
        "percentage"
    ] = (
        win_counts[
            "engine_wins"
        ]
        / len(comparison_df)
        * 100
    )

    return win_counts


def plot_test_metrics(
    metrics_df,
    dataset,
):
    """Plot test RMSE and MAE for all models."""

    plot_df = metrics_df.sort_values(
        by="test_rmse"
    )

    positions = np.arange(
        len(plot_df)
    )

    width = 0.35

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        positions - width / 2,
        plot_df[
            "test_rmse"
        ],
        width=width,
        label="RMSE",
    )

    plt.bar(
        positions + width / 2,
        plot_df[
            "test_mae"
        ],
        width=width,
        label="MAE",
    )

    plt.xticks(
        positions,
        plot_df[
            "model"
        ].str.upper(),
    )

    plt.ylabel(
        "Prediction error in cycles"
    )

    plt.xlabel(
        "Model"
    )

    plt.title(
        f"Complete Model Benchmark — {dataset}"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        COMPARISON_DIR,
        f"complete_model_benchmark_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_sorted_predictions(
    comparison_df,
    dataset,
):
    """Plot model predictions after sorting engines by actual RUL."""

    sorted_df = comparison_df.sort_values(
        by="actual_rul"
    ).reset_index(drop=True)

    positions = np.arange(
        len(sorted_df)
    )

    plt.figure(
        figsize=(13, 7)
    )

    plt.plot(
        positions,
        sorted_df[
            "actual_rul"
        ],
        linewidth=2.8,
        label="Actual RUL",
    )

    plt.plot(
        positions,
        sorted_df[
            "gru_prediction"
        ],
        linewidth=1.8,
        label="GRU",
    )

    plt.plot(
        positions,
        sorted_df[
            "lstm_prediction"
        ],
        linewidth=1.5,
        label="LSTM",
    )

    plt.plot(
        positions,
        sorted_df[
            "ridge_prediction"
        ],
        linewidth=1.5,
        label="Ridge",
    )

    plt.plot(
        positions,
        sorted_df[
            "ensemble_prediction"
        ],
        linewidth=2.0,
        linestyle="--",
        label="Validation-selected ensemble",
    )

    plt.xlabel(
        "Test engines sorted by actual RUL"
    )

    plt.ylabel(
        "Remaining Useful Life"
    )

    plt.title(
        f"Model Predictions Across Test Engines — {dataset}"
    )

    plt.legend()

    plt.grid(
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        COMPARISON_DIR,
        f"sorted_all_model_predictions_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def print_selected_engines(
    comparison_df,
    selected_units,
):
    """Display model predictions for selected difficult engines."""

    selected_df = comparison_df[
        comparison_df["unit"].isin(
            selected_units
        )
    ].copy()

    columns = [
        "unit",
        "actual_rul",
        "ridge_prediction",
        "lstm_prediction",
        "gru_prediction",
        "transformer_prediction",
        "ensemble_prediction",
        "best_model",
        "model_prediction_spread",
    ]

    print(
        selected_df[
            columns
        ]
        .sort_values(
            by="unit"
        )
        .to_string(
            index=False
        )
    )


def compare_all_models(
    dataset,
    weight_step,
    selected_units,
):
    """Run the complete model and ensemble comparison."""

    (
        data,
        validation_predictions,
        test_predictions,
        model_paths,
    ) = generate_predictions(
        dataset
    )

    (
        best_weights,
        best_validation_metrics,
        combinations_tested,
    ) = find_best_ensemble_weights(
        validation_predictions,
        data["y_val"],
        weight_step,
    )

    ensemble_validation_prediction = (
        calculate_weighted_prediction(
            validation_predictions,
            best_weights,
        )
    )

    ensemble_test_prediction = (
        calculate_weighted_prediction(
            test_predictions,
            best_weights,
        )
    )

    metrics_df = build_metrics_table(
        y_val=data["y_val"],
        y_test=data["y_test"],
        validation_predictions=validation_predictions,
        test_predictions=test_predictions,
        ensemble_validation_prediction=(
            ensemble_validation_prediction
        ),
        ensemble_test_prediction=(
            ensemble_test_prediction
        ),
    )

    comparison_df = build_engine_comparison(
        data=data,
        test_predictions=test_predictions,
        ensemble_prediction=(
            ensemble_test_prediction
        ),
    )

    win_summary_df = build_win_summary(
        comparison_df
    )

    weights_df = pd.DataFrame(
        [
            {
                "model": model_name,
                "weight": float(
                    best_weights[index]
                ),
            }
            for index, model_name in enumerate(
                MODEL_NAMES
            )
        ]
    )

    metrics_path = os.path.join(
        COMPARISON_DIR,
        f"complete_metrics_{dataset}.csv",
    )

    comparison_path = os.path.join(
        COMPARISON_DIR,
        f"engine_comparison_{dataset}.csv",
    )

    weights_path = os.path.join(
        COMPARISON_DIR,
        f"ensemble_weights_{dataset}.csv",
    )

    wins_path = os.path.join(
        COMPARISON_DIR,
        f"engine_wins_{dataset}.csv",
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    weights_df.to_csv(
        weights_path,
        index=False,
    )

    win_summary_df.to_csv(
        wins_path,
        index=False,
    )

    metrics_plot_path = plot_test_metrics(
        metrics_df,
        dataset,
    )

    predictions_plot_path = (
        plot_sorted_predictions(
            comparison_df,
            dataset,
        )
    )

    print("Loaded models")

    for model_name, model_path in (
        model_paths.items()
    ):
        print(
            f"{model_name:12s}: {model_path}"
        )

    print(
        f"\nEnsemble combinations tested: "
        f"{combinations_tested}"
    )

    print(
        "\nValidation-selected ensemble weights"
    )

    print(
        weights_df.to_string(
            index=False
        )
    )

    print(
        "\nBest ensemble validation metrics"
    )

    print(
        f"RMSE:       "
        f"{best_validation_metrics['rmse']:.4f}"
    )

    print(
        f"MAE:        "
        f"{best_validation_metrics['mae']:.4f}"
    )

    print(
        f"NASA score: "
        f"{best_validation_metrics['nasa_score']:.4f}"
    )

    print(
        "\nComplete model benchmark"
    )

    print(
        metrics_df[
            [
                "model",
                "test_rmse",
                "test_mae",
                "test_r2",
                "test_nasa_score",
                "test_mean_error",
                "test_dangerous_overprediction_rate",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nEngine-level model wins"
    )

    print(
        win_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nSelected difficult engines"
    )

    print_selected_engines(
        comparison_df,
        selected_units,
    )

    print(
        "\nSaved outputs"
    )

    print(
        f"Metrics:       {metrics_path}"
    )

    print(
        f"Engine data:   {comparison_path}"
    )

    print(
        f"Weights:       {weights_path}"
    )

    print(
        f"Model wins:    {wins_path}"
    )

    print(
        f"Metrics plot:  {metrics_plot_path}"
    )

    print(
        f"Prediction plot: "
        f"{predictions_plot_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

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
        "--weight_step",
        type=float,
        default=0.05,
    )

    parser.add_argument(
        "--units",
        type=int,
        nargs="*",
        default=[
            25,
            45,
            67,
            79,
        ],
    )

    arguments = parser.parse_args()

    if not 0.0 < arguments.weight_step <= 0.25:
        raise ValueError(
            "Weight step must be greater than 0 "
            "and no larger than 0.25."
        )

    compare_all_models(
        dataset=arguments.dataset,
        weight_step=arguments.weight_step,
        selected_units=arguments.units,
    )