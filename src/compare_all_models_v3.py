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
    "all_model_comparison_v3",
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

    data = joblib.load(
        dataset_path
    )

    required_keys = [
        "X_val",
        "y_val",
        "validation_units",
        "validation_window_units",
        "X_test",
        "y_test",
        "test_units",
    ]

    missing_keys = [
        key
        for key in required_keys
        if key not in data
    ]

    if missing_keys:
        raise KeyError(
            "The processed dataset is missing required keys: "
            f"{missing_keys}"
        )

    return {
        "X_val": np.asarray(
            data["X_val"]
        ),
        "y_val": np.asarray(
            data["y_val"]
        ).reshape(-1),
        "validation_units": np.asarray(
            data["validation_units"]
        ).reshape(-1),
        "validation_window_units": np.asarray(
            data["validation_window_units"]
        ).reshape(-1),
        "X_test": np.asarray(
            data["X_test"]
        ),
        "y_test": np.asarray(
            data["y_test"]
        ).reshape(-1),
        "test_units": np.asarray(
            data["test_units"]
        ).reshape(-1),
    }


def select_validation_snapshots(
    data,
    random_seed=42,
):
    """
    Select one deployment-like snapshot from each validation engine.

    Validation engines contain complete run-to-failure histories.
    Selecting the final window from every engine would produce RUL
    targets close to zero and would not resemble the independent
    C-MAPSS test configuration.

    Instead, deterministic target RUL values are distributed between
    10 and 120 cycles. For every validation engine, the available
    window whose target is closest to the assigned RUL is selected.
    """

    validation_units = np.asarray(
        data["validation_units"]
    ).reshape(-1)

    validation_window_units = np.asarray(
        data["validation_window_units"]
    ).reshape(-1)

    y_val = np.asarray(
        data["y_val"]
    ).reshape(-1)

    if len(validation_window_units) != len(y_val):
        raise ValueError(
            "validation_window_units and y_val must have "
            "the same number of entries."
        )

    target_rul_values = np.linspace(
        10.0,
        120.0,
        len(validation_units),
    )

    random_generator = np.random.default_rng(
        random_seed
    )

    assigned_target_rul = (
        random_generator.permutation(
            target_rul_values
        )
    )

    selected_indices = []
    selected_actual_rul = []

    for unit, target_rul in zip(
        validation_units,
        assigned_target_rul,
    ):
        engine_indices = np.flatnonzero(
            validation_window_units == unit
        )

        if len(engine_indices) == 0:
            raise ValueError(
                "No validation windows were found for "
                f"validation engine {unit}."
            )

        engine_targets = y_val[
            engine_indices
        ]

        closest_local_index = int(
            np.argmin(
                np.abs(
                    engine_targets
                    - target_rul
                )
            )
        )

        selected_index = int(
            engine_indices[
                closest_local_index
            ]
        )

        selected_indices.append(
            selected_index
        )

        selected_actual_rul.append(
            float(
                y_val[
                    selected_index
                ]
            )
        )

    selected_indices = np.asarray(
        selected_indices,
        dtype=int,
    )

    selected_actual_rul = np.asarray(
        selected_actual_rul,
        dtype=float,
    )

    return {
        "X_val": data["X_val"][
            selected_indices
        ],
        "y_val": selected_actual_rul,
        "validation_units": validation_units,
        "assigned_target_rul": np.asarray(
            assigned_target_rul,
            dtype=float,
        ),
        "selected_indices": selected_indices,
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
        joblib.load(
            model_path
        ),
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

    model = keras.models.load_model(
        model_path
    )

    return model, model_path


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

    y_true = np.asarray(
        y_true
    ).reshape(-1)

    y_pred = np.asarray(
        y_pred
    ).reshape(-1)

    error = y_pred - y_true

    penalties = np.where(
        error < 0,
        np.exp(
            -error / 13.0
        )
        - 1.0,
        np.exp(
            error / 10.0
        )
        - 1.0,
    )

    return float(
        np.sum(
            penalties
        )
    )


def calculate_metrics(
    y_true,
    y_pred,
):
    """Calculate accuracy and maintenance-safety metrics."""

    y_true = np.asarray(
        y_true
    ).reshape(-1)

    y_pred = np.clip(
        np.asarray(
            y_pred
        ).reshape(-1),
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
            np.mean(
                error
            )
        ),
        "overprediction_rate": float(
            np.mean(
                error > 0
            )
        ),
        "dangerous_overprediction_rate": float(
            np.mean(
                error > 10
            )
        ),
        "maximum_absolute_error": float(
            np.max(
                np.abs(
                    error
                )
            )
        ),
    }


def generate_predictions(
    dataset,
    random_seed,
):
    """
    Generate validation-snapshot and independent test predictions.

    The validation predictions contain one selected snapshot per
    validation engine. The test predictions contain one final
    available snapshot per test engine.
    """

    full_data = load_data(
        dataset
    )

    snapshot_data = (
        select_validation_snapshots(
            data=full_data,
            random_seed=random_seed,
        )
    )

    validation_predictions = {}
    test_predictions = {}
    model_paths = {}

    ridge_model, ridge_path = load_ridge_model(
        dataset
    )

    validation_predictions[
        "ridge"
    ] = np.clip(
        ridge_model.predict(
            flatten_windows(
                snapshot_data[
                    "X_val"
                ]
            )
        ),
        0.0,
        RUL_CAP,
    )

    test_predictions[
        "ridge"
    ] = np.clip(
        ridge_model.predict(
            flatten_windows(
                full_data[
                    "X_test"
                ]
            )
        ),
        0.0,
        RUL_CAP,
    )

    model_paths[
        "ridge"
    ] = ridge_path

    for model_name in [
        "lstm",
        "gru",
        "transformer",
    ]:
        model, model_path = load_neural_model(
            model_name=model_name,
            dataset=dataset,
        )

        validation_predictions[
            model_name
        ] = predict_neural_model(
            model=model,
            X=snapshot_data[
                "X_val"
            ],
        )

        test_predictions[
            model_name
        ] = predict_neural_model(
            model=model,
            X=full_data[
                "X_test"
            ],
        )

        model_paths[
            model_name
        ] = model_path

    return (
        full_data,
        snapshot_data,
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

    With a step of 0.05, weights are tested in increments of 5%.
    """

    divisions = int(
        round(
            1.0 / weight_step
        )
    )

    for combination in itertools.product(
        range(
            divisions + 1
        ),
        repeat=number_of_models,
    ):
        if sum(
            combination
        ) != divisions:
            continue

        yield (
            np.asarray(
                combination,
                dtype=float,
            )
            / divisions
        )


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

    predictions = (
        prediction_matrix
        @ weights
    )

    return np.clip(
        predictions,
        0.0,
        RUL_CAP,
    )


def find_best_ensemble_weights(
    validation_predictions,
    y_val,
    weight_step,
):
    """
    Select ensemble weights using validation snapshots only.

    Test targets are never used during weight selection.
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
                prediction_dictionary=(
                    validation_predictions
                ),
                weights=weights,
            )
        )

        metrics = calculate_metrics(
            y_true=y_val,
            y_pred=ensemble_prediction,
        )

        combinations_tested += 1

        if best_metrics is None:
            best_weights = weights
            best_metrics = metrics
            continue

        current_key = (
            metrics[
                "rmse"
            ],
            metrics[
                "nasa_score"
            ],
            metrics[
                "mae"
            ],
        )

        best_key = (
            best_metrics[
                "rmse"
            ],
            best_metrics[
                "nasa_score"
            ],
            best_metrics[
                "mae"
            ],
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
    """Build validation-snapshot and test metrics."""

    rows = []

    all_validation_predictions = {
        **validation_predictions,
        "ensemble": (
            ensemble_validation_prediction
        ),
    }

    all_test_predictions = {
        **test_predictions,
        "ensemble": (
            ensemble_test_prediction
        ),
    }

    for model_name in [
        *MODEL_NAMES,
        "ensemble",
    ]:
        validation_metrics = calculate_metrics(
            y_true=y_val,
            y_pred=all_validation_predictions[
                model_name
            ],
        )

        test_metrics = calculate_metrics(
            y_true=y_test,
            y_pred=all_test_predictions[
                model_name
            ],
        )

        rows.append(
            {
                "model": model_name,
                "validation_snapshot_rmse": (
                    validation_metrics[
                        "rmse"
                    ]
                ),
                "validation_snapshot_mae": (
                    validation_metrics[
                        "mae"
                    ]
                ),
                "validation_snapshot_r2": (
                    validation_metrics[
                        "r2"
                    ]
                ),
                "validation_snapshot_nasa_score": (
                    validation_metrics[
                        "nasa_score"
                    ]
                ),
                "validation_snapshot_mean_error": (
                    validation_metrics[
                        "mean_error"
                    ]
                ),
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
                "test_overprediction_rate": (
                    test_metrics[
                        "overprediction_rate"
                    ]
                ),
                "test_dangerous_overprediction_rate": (
                    test_metrics[
                        "dangerous_overprediction_rate"
                    ]
                ),
                "test_maximum_absolute_error": (
                    test_metrics[
                        "maximum_absolute_error"
                    ]
                ),
            }
        )

    metrics_df = pd.DataFrame(
        rows
    )

    return (
        metrics_df.sort_values(
            by="test_rmse"
        )
        .reset_index(
            drop=True
        )
    )


def build_validation_snapshot_table(
    snapshot_data,
    validation_predictions,
    ensemble_prediction,
):
    """Build a table describing validation snapshots."""

    snapshot_df = pd.DataFrame(
        {
            "unit": snapshot_data[
                "validation_units"
            ],
            "selected_window_index": snapshot_data[
                "selected_indices"
            ],
            "assigned_target_rul": snapshot_data[
                "assigned_target_rul"
            ],
            "actual_rul": snapshot_data[
                "y_val"
            ],
            "ridge_prediction": (
                validation_predictions[
                    "ridge"
                ]
            ),
            "lstm_prediction": (
                validation_predictions[
                    "lstm"
                ]
            ),
            "gru_prediction": (
                validation_predictions[
                    "gru"
                ]
            ),
            "transformer_prediction": (
                validation_predictions[
                    "transformer"
                ]
            ),
            "ensemble_prediction": (
                ensemble_prediction
            ),
        }
    )

    snapshot_df[
        "target_selection_difference"
    ] = np.abs(
        snapshot_df[
            "actual_rul"
        ]
        - snapshot_df[
            "assigned_target_rul"
        ]
    )

    return snapshot_df.sort_values(
        by="actual_rul"
    ).reset_index(
        drop=True
    )


def build_engine_comparison(
    data,
    test_predictions,
    ensemble_prediction,
):
    """Create engine-level test predictions and errors."""

    comparison_df = pd.DataFrame(
        {
            "unit": data[
                "test_units"
            ],
            "actual_rul": data[
                "y_test"
            ],
            "ridge_prediction": (
                test_predictions[
                    "ridge"
                ]
            ),
            "lstm_prediction": (
                test_predictions[
                    "lstm"
                ]
            ),
            "gru_prediction": (
                test_predictions[
                    "gru"
                ]
            ),
            "transformer_prediction": (
                test_predictions[
                    "transformer"
                ]
            ),
            "ensemble_prediction": (
                ensemble_prediction
            ),
        }
    )

    model_columns = {
        "Ridge": "ridge_prediction",
        "LSTM": "lstm_prediction",
        "GRU": "gru_prediction",
        "Transformer": (
            "transformer_prediction"
        ),
        "Ensemble": (
            "ensemble_prediction"
        ),
    }

    for (
        model_name,
        prediction_column,
    ) in model_columns.items():
        error_column = (
            f"{model_name.lower()}_error"
        )

        absolute_error_column = (
            f"{model_name.lower()}"
            "_absolute_error"
        )

        comparison_df[
            error_column
        ] = (
            comparison_df[
                prediction_column
            ]
            - comparison_df[
                "actual_rul"
            ]
        )

        comparison_df[
            absolute_error_column
        ] = np.abs(
            comparison_df[
                error_column
            ]
        )

    absolute_error_columns = {
        model_name: (
            f"{model_name.lower()}"
            "_absolute_error"
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

    base_prediction_matrix = np.column_stack(
        [
            comparison_df[
                model_columns[
                    model_name
                ]
            ].to_numpy()
            for model_name in [
                "Ridge",
                "LSTM",
                "GRU",
                "Transformer",
            ]
        ]
    )

    comparison_df[
        "base_model_prediction_spread"
    ] = (
        np.max(
            base_prediction_matrix,
            axis=1,
        )
        - np.min(
            base_prediction_matrix,
            axis=1,
        )
    )

    return comparison_df


def build_win_summary(
    comparison_df,
):
    """Count how often each model is closest to actual RUL."""

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
        / len(
            comparison_df
        )
        * 100
    )

    return win_counts


def plot_test_metrics(
    metrics_df,
    dataset,
):
    """Plot independent test RMSE and MAE."""

    plot_df = metrics_df.sort_values(
        by="test_rmse"
    )

    positions = np.arange(
        len(
            plot_df
        )
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
        "Deployment-Aligned Model "
        f"Benchmark — {dataset}"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        COMPARISON_DIR,
        f"deployment_aligned_benchmark_"
        f"{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_validation_snapshots(
    snapshot_df,
    dataset,
):
    """Plot selected validation snapshots and predictions."""

    plot_df = snapshot_df.sort_values(
        by="actual_rul"
    ).reset_index(
        drop=True
    )

    positions = np.arange(
        len(
            plot_df
        )
    )

    plt.figure(
        figsize=(12, 7)
    )

    plt.plot(
        positions,
        plot_df[
            "actual_rul"
        ],
        linewidth=3,
        marker="o",
        label="Actual snapshot RUL",
    )

    plt.plot(
        positions,
        plot_df[
            "gru_prediction"
        ],
        linewidth=1.8,
        label="GRU",
    )

    plt.plot(
        positions,
        plot_df[
            "lstm_prediction"
        ],
        linewidth=1.6,
        label="LSTM",
    )

    plt.plot(
        positions,
        plot_df[
            "ridge_prediction"
        ],
        linewidth=1.6,
        label="Ridge",
    )

    plt.plot(
        positions,
        plot_df[
            "transformer_prediction"
        ],
        linewidth=1.4,
        label="Transformer",
    )

    plt.plot(
        positions,
        plot_df[
            "ensemble_prediction"
        ],
        linewidth=2.2,
        linestyle="--",
        label="Selected ensemble",
    )

    plt.xlabel(
        "Validation engines sorted by snapshot RUL"
    )

    plt.ylabel(
        "Remaining Useful Life"
    )

    plt.title(
        "One Deployment-Like Snapshot per "
        f"Validation Engine — {dataset}"
    )

    plt.legend()

    plt.grid(
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        COMPARISON_DIR,
        f"validation_snapshots_"
        f"{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_sorted_test_predictions(
    comparison_df,
    dataset,
):
    """Plot model predictions sorted by independent test RUL."""

    sorted_df = comparison_df.sort_values(
        by="actual_rul"
    ).reset_index(
        drop=True
    )

    positions = np.arange(
        len(
            sorted_df
        )
    )

    plt.figure(
        figsize=(13, 7)
    )

    plt.plot(
        positions,
        sorted_df[
            "actual_rul"
        ],
        linewidth=3,
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
            "transformer_prediction"
        ],
        linewidth=1.3,
        label="Transformer",
    )

    plt.plot(
        positions,
        sorted_df[
            "ensemble_prediction"
        ],
        linewidth=2.2,
        linestyle="--",
        label="Validation-snapshot ensemble",
    )

    plt.xlabel(
        "Test engines sorted by actual RUL"
    )

    plt.ylabel(
        "Remaining Useful Life"
    )

    plt.title(
        "Deployment-Aligned Predictions "
        f"Across Test Engines — {dataset}"
    )

    plt.legend()

    plt.grid(
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        COMPARISON_DIR,
        f"sorted_test_predictions_"
        f"{dataset}.png",
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
    """Display predictions for selected difficult engines."""

    selected_df = comparison_df[
        comparison_df[
            "unit"
        ].isin(
            selected_units
        )
    ].copy()

    if selected_df.empty:
        print(
            "No matching selected engines were found."
        )
        return

    columns = [
        "unit",
        "actual_rul",
        "ridge_prediction",
        "lstm_prediction",
        "gru_prediction",
        "transformer_prediction",
        "ensemble_prediction",
        "best_model",
        "base_model_prediction_spread",
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
    random_seed,
):
    """Run deployment-aligned model and ensemble comparison."""

    (
        data,
        snapshot_data,
        validation_predictions,
        test_predictions,
        model_paths,
    ) = generate_predictions(
        dataset=dataset,
        random_seed=random_seed,
    )

    (
        best_weights,
        best_validation_metrics,
        combinations_tested,
    ) = find_best_ensemble_weights(
        validation_predictions=(
            validation_predictions
        ),
        y_val=snapshot_data[
            "y_val"
        ],
        weight_step=weight_step,
    )

    ensemble_validation_prediction = (
        calculate_weighted_prediction(
            prediction_dictionary=(
                validation_predictions
            ),
            weights=best_weights,
        )
    )

    ensemble_test_prediction = (
        calculate_weighted_prediction(
            prediction_dictionary=(
                test_predictions
            ),
            weights=best_weights,
        )
    )

    metrics_df = build_metrics_table(
        y_val=snapshot_data[
            "y_val"
        ],
        y_test=data[
            "y_test"
        ],
        validation_predictions=(
            validation_predictions
        ),
        test_predictions=test_predictions,
        ensemble_validation_prediction=(
            ensemble_validation_prediction
        ),
        ensemble_test_prediction=(
            ensemble_test_prediction
        ),
    )

    snapshot_df = (
        build_validation_snapshot_table(
            snapshot_data=snapshot_data,
            validation_predictions=(
                validation_predictions
            ),
            ensemble_prediction=(
                ensemble_validation_prediction
            ),
        )
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
                    best_weights[
                        index
                    ]
                ),
            }
            for index, model_name in enumerate(
                MODEL_NAMES
            )
        ]
    )

    metrics_path = os.path.join(
        COMPARISON_DIR,
        f"complete_metrics_v3_"
        f"{dataset}.csv",
    )

    snapshot_path = os.path.join(
        COMPARISON_DIR,
        f"validation_snapshots_v3_"
        f"{dataset}.csv",
    )

    comparison_path = os.path.join(
        COMPARISON_DIR,
        f"engine_comparison_v3_"
        f"{dataset}.csv",
    )

    weights_path = os.path.join(
        COMPARISON_DIR,
        f"ensemble_weights_v3_"
        f"{dataset}.csv",
    )

    wins_path = os.path.join(
        COMPARISON_DIR,
        f"engine_wins_v3_"
        f"{dataset}.csv",
    )

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    snapshot_df.to_csv(
        snapshot_path,
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
        metrics_df=metrics_df,
        dataset=dataset,
    )

    snapshot_plot_path = (
        plot_validation_snapshots(
            snapshot_df=snapshot_df,
            dataset=dataset,
        )
    )

    prediction_plot_path = (
        plot_sorted_test_predictions(
            comparison_df=comparison_df,
            dataset=dataset,
        )
    )

    print(
        "Loaded models"
    )

    for (
        model_name,
        model_path,
    ) in model_paths.items():
        print(
            f"{model_name:12s}: "
            f"{model_path}"
        )

    print(
        "\nValidation snapshot configuration"
    )

    print(
        f"Validation engines: "
        f"{len(snapshot_data['validation_units'])}"
    )

    print(
        f"RUL range: "
        f"{snapshot_data['y_val'].min():.1f} "
        f"to "
        f"{snapshot_data['y_val'].max():.1f}"
    )

    print(
        "Average assigned-target difference: "
        f"{snapshot_df['target_selection_difference'].mean():.4f}"
    )

    print(
        "\nValidation snapshot preview"
    )

    print(
        snapshot_df[
            [
                "unit",
                "assigned_target_rul",
                "actual_rul",
                "target_selection_difference",
            ]
        ]
        .head(
            20
        )
        .to_string(
            index=False
        )
    )

    print(
        f"\nEnsemble combinations tested: "
        f"{combinations_tested}"
    )

    print(
        "\nValidation-snapshot-selected "
        "ensemble weights"
    )

    print(
        weights_df.to_string(
            index=False
        )
    )

    print(
        "\nBest ensemble validation-snapshot metrics"
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
        f"R²:         "
        f"{best_validation_metrics['r2']:.4f}"
    )

    print(
        f"NASA score: "
        f"{best_validation_metrics['nasa_score']:.4f}"
    )

    print(
        "\nComplete independent-test benchmark"
    )

    benchmark_columns = [
        "model",
        "validation_snapshot_rmse",
        "validation_snapshot_mae",
        "test_rmse",
        "test_mae",
        "test_r2",
        "test_nasa_score",
        "test_mean_error",
        "test_dangerous_overprediction_rate",
    ]

    print(
        metrics_df[
            benchmark_columns
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
        comparison_df=comparison_df,
        selected_units=selected_units,
    )

    print(
        "\nSaved outputs"
    )

    print(
        f"Metrics:              {metrics_path}"
    )

    print(
        f"Validation snapshots: {snapshot_path}"
    )

    print(
        f"Engine data:          {comparison_path}"
    )

    print(
        f"Weights:              {weights_path}"
    )

    print(
        f"Model wins:           {wins_path}"
    )

    print(
        f"Metrics plot:         {metrics_plot_path}"
    )

    print(
        f"Snapshot plot:        {snapshot_plot_path}"
    )

    print(
        f"Prediction plot:      {prediction_plot_path}"
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
        "--random_seed",
        type=int,
        default=42,
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

    if not (
        0.0
        < arguments.weight_step
        <= 0.25
    ):
        raise ValueError(
            "Weight step must be greater than zero "
            "and no larger than 0.25."
        )

    compare_all_models(
        dataset=arguments.dataset,
        weight_step=arguments.weight_step,
        selected_units=arguments.units,
        random_seed=arguments.random_seed,
    )