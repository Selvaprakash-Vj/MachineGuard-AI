import argparse
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RAW_DIR = "data/raw"
PROCESSED_DIR = "data/processed_v2"
RESULTS_DIR = "results_v2"

WINDOW_SIZE = 30


def load_raw_test_data(dataset):
    """Load the raw C-MAPSS test data."""

    test_path = os.path.join(
        RAW_DIR,
        f"test_{dataset}.txt",
    )

    test_df = pd.read_csv(
        test_path,
        sep=r"\s+",
        header=None,
    )

    column_names = (
        ["unit", "time"]
        + [f"op_setting_{index}" for index in range(1, 4)]
        + [f"sensor_{index}" for index in range(1, 22)]
    )

    test_df.columns = column_names

    return test_df


def load_processed_metadata(dataset):
    """Load feature names and the fitted training scaler."""

    processed_path = os.path.join(
        PROCESSED_DIR,
        f"{dataset}.pkl",
    )

    if not os.path.exists(processed_path):
        raise FileNotFoundError(
            f"Processed dataset not found: {processed_path}"
        )

    data = joblib.load(processed_path)

    return (
        data["feature_names"],
        data["scaler"],
        data["window_size"],
    )


def load_model_comparison(dataset):
    """Load engine-level LSTM and Ridge predictions."""

    comparison_path = os.path.join(
        RESULTS_DIR,
        f"model_comparison_{dataset}.csv",
    )

    if not os.path.exists(comparison_path):
        raise FileNotFoundError(
            f"Comparison file not found: {comparison_path}\n"
            "Run compare_models_v2.py first."
        )

    return pd.read_csv(comparison_path)


def scale_test_features(
    test_df,
    feature_names,
    scaler,
):
    """Apply the training-data scaler to the test engines."""

    scaled_df = test_df.copy()

    # StandardScaler returns floating-point values.
    # Convert feature columns before assigning scaled data.
    feature_dtypes = {
        feature: np.float64
        for feature in feature_names
    }

    scaled_df = scaled_df.astype(
        feature_dtypes
    )

    scaled_df.loc[:, feature_names] = scaler.transform(
        scaled_df[feature_names]
    )

    return scaled_df


def calculate_feature_diagnostics(
    unit_df,
    feature_names,
):
    """
    Rank features according to how strongly they change.

    The diagnostic score combines:
    - total change across the observed history,
    - linear trend magnitude,
    - abnormality in the final 30-cycle model window.
    """

    diagnostic_rows = []

    cycle_values = unit_df["time"].to_numpy(
        dtype=float
    )

    cycle_span = max(
        cycle_values[-1] - cycle_values[0],
        1.0,
    )

    for feature in feature_names:
        values = unit_df[feature].to_numpy(
            dtype=float
        )

        first_count = min(5, len(values))
        last_count = min(5, len(values))

        initial_mean = float(
            np.mean(values[:first_count])
        )

        final_mean = float(
            np.mean(values[-last_count:])
        )

        total_change = final_mean - initial_mean

        if len(values) > 1:
            slope = float(
                np.polyfit(
                    cycle_values,
                    values,
                    deg=1,
                )[0]
            )
        else:
            slope = 0.0

        final_window = values[-WINDOW_SIZE:]

        final_window_mean = float(
            np.mean(final_window)
        )

        final_window_max_abs = float(
            np.max(np.abs(final_window))
        )

        diagnostic_score = (
            abs(total_change)
            + abs(slope * cycle_span)
            + 0.25 * final_window_max_abs
        )

        diagnostic_rows.append(
            {
                "feature": feature,
                "initial_mean_scaled": initial_mean,
                "final_mean_scaled": final_mean,
                "total_change_scaled": total_change,
                "linear_slope_scaled": slope,
                "final_window_mean_scaled": final_window_mean,
                "final_window_max_abs_scaled": final_window_max_abs,
                "diagnostic_score": diagnostic_score,
            }
        )

    diagnostics_df = pd.DataFrame(
        diagnostic_rows
    )

    return diagnostics_df.sort_values(
        by="diagnostic_score",
        ascending=False,
    )


def plot_top_feature_histories(
    unit_df,
    top_features,
    prediction_row,
    dataset,
    output_directory,
):
    """Plot the most informative standardized sensor histories."""

    unit_id = int(prediction_row["unit"])

    plt.figure(figsize=(12, 7))

    for feature in top_features:
        plt.plot(
            unit_df["time"],
            unit_df[feature],
            linewidth=1.8,
            label=feature,
        )

    final_cycle = unit_df["time"].max()

    model_window_start = max(
        unit_df["time"].min(),
        final_cycle - WINDOW_SIZE + 1,
    )

    plt.axvspan(
        model_window_start,
        final_cycle,
        alpha=0.12,
        label="Final 30-cycle model input",
    )

    plt.axhline(
        0,
        linestyle="--",
        linewidth=1,
    )

    actual_rul = prediction_row["actual_rul"]
    lstm_prediction = prediction_row[
        "lstm_prediction"
    ]
    ridge_prediction = prediction_row[
        "ridge_prediction"
    ]

    plt.xlabel("Observed operating cycle")
    plt.ylabel("Standardized feature value")

    plt.title(
        f"{dataset} Unit {unit_id} — Sensor Diagnostics\n"
        f"Actual RUL: {actual_rul:.1f} | "
        f"LSTM: {lstm_prediction:.1f} | "
        f"Ridge: {ridge_prediction:.1f}"
    )

    plt.legend(
        loc="best",
        ncol=2,
    )

    plt.grid(alpha=0.25)
    plt.tight_layout()

    output_path = os.path.join(
        output_directory,
        f"unit_{unit_id}_top_feature_histories.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_final_window_heatmap(
    unit_df,
    feature_names,
    prediction_row,
    dataset,
    output_directory,
):
    """Visualize the exact standardized 30-cycle model input."""

    unit_id = int(prediction_row["unit"])

    final_window = (
        unit_df
        .sort_values("time")
        .tail(WINDOW_SIZE)
    )

    matrix = final_window[
        feature_names
    ].to_numpy(dtype=float).T

    plt.figure(figsize=(13, 8))

    image = plt.imshow(
        matrix,
        aspect="auto",
        interpolation="nearest",
    )

    plt.colorbar(
        image,
        label="Standardized feature value",
    )

    plt.yticks(
        ticks=np.arange(len(feature_names)),
        labels=feature_names,
    )

    plt.xlabel("Position inside final 30-cycle window")
    plt.ylabel("Input feature")

    plt.title(
        f"{dataset} Unit {unit_id} — Final Model Input"
    )

    plt.tight_layout()

    output_path = os.path.join(
        output_directory,
        f"unit_{unit_id}_final_window_heatmap.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def analyze_unit(
    unit_id,
    scaled_test_df,
    comparison_df,
    feature_names,
    dataset,
    top_feature_count,
    output_directory,
):
    """Run diagnostics for one selected test engine."""

    unit_rows = scaled_test_df[
        scaled_test_df["unit"] == unit_id
    ].copy()

    if unit_rows.empty:
        raise ValueError(
            f"Unit {unit_id} does not exist in {dataset}."
        )

    unit_rows = unit_rows.sort_values(
        "time"
    )

    prediction_rows = comparison_df[
        comparison_df["unit"] == unit_id
    ]

    if prediction_rows.empty:
        raise ValueError(
            f"No prediction data found for unit {unit_id}."
        )

    prediction_row = prediction_rows.iloc[0]

    diagnostics_df = calculate_feature_diagnostics(
        unit_rows,
        feature_names,
    )

    diagnostics_df.insert(
        0,
        "unit",
        unit_id,
    )

    diagnostics_path = os.path.join(
        output_directory,
        f"unit_{unit_id}_feature_diagnostics.csv",
    )

    diagnostics_df.to_csv(
        diagnostics_path,
        index=False,
    )

    top_features = diagnostics_df.head(
        top_feature_count
    )["feature"].tolist()

    history_plot_path = plot_top_feature_histories(
        unit_rows,
        top_features,
        prediction_row,
        dataset,
        output_directory,
    )

    heatmap_path = plot_final_window_heatmap(
        unit_rows,
        feature_names,
        prediction_row,
        dataset,
        output_directory,
    )

    summary = {
        "unit": unit_id,
        "observed_cycles": int(
            unit_rows["time"].max()
        ),
        "actual_rul": float(
            prediction_row["actual_rul"]
        ),
        "lstm_prediction": float(
            prediction_row["lstm_prediction"]
        ),
        "ridge_prediction": float(
            prediction_row["ridge_prediction"]
        ),
        "lstm_error": float(
            prediction_row["lstm_error"]
        ),
        "ridge_error": float(
            prediction_row["ridge_error"]
        ),
        "better_model": prediction_row[
            "better_model"
        ],
        "highest_ranked_feature": top_features[0],
        "top_features": ", ".join(top_features),
    }

    print(f"\nUnit {unit_id}")
    print(
        f"Actual RUL:       {summary['actual_rul']:.1f}"
    )
    print(
        f"LSTM prediction:  {summary['lstm_prediction']:.1f}"
    )
    print(
        f"Ridge prediction: {summary['ridge_prediction']:.1f}"
    )
    print(
        f"Top features:     {summary['top_features']}"
    )
    print(
        f"Diagnostics:      {diagnostics_path}"
    )
    print(
        f"History plot:     {history_plot_path}"
    )
    print(
        f"Window heatmap:   {heatmap_path}"
    )

    return summary


def analyze_failures(
    dataset,
    unit_ids,
    top_feature_count,
):
    """Analyze selected or automatically chosen failure cases."""

    feature_names, scaler, window_size = (
        load_processed_metadata(dataset)
    )

    if window_size != WINDOW_SIZE:
        raise ValueError(
            f"Expected window size {WINDOW_SIZE}, "
            f"but processed data uses {window_size}."
        )

    raw_test_df = load_raw_test_data(dataset)

    scaled_test_df = scale_test_features(
        raw_test_df,
        feature_names,
        scaler,
    )

    comparison_df = load_model_comparison(
        dataset
    )

    if not unit_ids:
        automatically_selected = (
            comparison_df
            .nlargest(
                4,
                "lstm_absolute_error",
            )["unit"]
            .astype(int)
            .tolist()
        )

        unit_ids = automatically_selected

    output_directory = os.path.join(
        RESULTS_DIR,
        "failure_analysis",
        dataset,
    )

    os.makedirs(
        output_directory,
        exist_ok=True,
    )

    summaries = []

    for unit_id in unit_ids:
        summary = analyze_unit(
            unit_id=int(unit_id),
            scaled_test_df=scaled_test_df,
            comparison_df=comparison_df,
            feature_names=feature_names,
            dataset=dataset,
            top_feature_count=top_feature_count,
            output_directory=output_directory,
        )

        summaries.append(summary)

    summary_df = pd.DataFrame(summaries)

    summary_path = os.path.join(
        output_directory,
        "failure_analysis_summary.csv",
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    print(
        f"\nFailure-analysis summary saved to "
        f"{summary_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        required=True,
        choices=["FD001", "FD002", "FD003", "FD004"],
    )

    parser.add_argument(
        "--units",
        type=int,
        nargs="*",
        default=None,
        help=(
            "Test-engine IDs to analyze. "
            "If omitted, the four largest LSTM errors are used."
        ),
    )

    parser.add_argument(
        "--top_features",
        type=int,
        default=6,
    )

    arguments = parser.parse_args()

    analyze_failures(
        dataset=arguments.dataset,
        unit_ids=arguments.units,
        top_feature_count=arguments.top_features,
    )