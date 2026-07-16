import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
)


RESULTS_DIR = "results_v2"
UNCERTAINTY_DIR = os.path.join(
    RESULTS_DIR,
    "uncertainty",
)

os.makedirs(
    UNCERTAINTY_DIR,
    exist_ok=True,
)


def load_results(model_name, dataset):
    """Load previously generated uncertainty results."""

    results_path = os.path.join(
        UNCERTAINTY_DIR,
        f"uncertainty_results_{model_name}_{dataset}.csv",
    )

    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"Uncertainty results not found: {results_path}\n"
            "Run uncertainty_v2.py first."
        )

    results_df = pd.read_csv(
        results_path
    )

    required_columns = [
        "unit",
        "actual_rul",
        "predicted_rul_mean",
        "mc_standard_deviation",
        "calibrated_lower_bound",
        "calibrated_upper_bound",
        "interval_width",
        "prediction_error",
        "absolute_error",
        "confidence",
        "actual_inside_interval",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in results_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing_columns)
        )

    if results_df[
        "actual_inside_interval"
    ].dtype == object:
        results_df[
            "actual_inside_interval"
        ] = (
            results_df[
                "actual_inside_interval"
            ]
            .astype(str)
            .str.lower()
            .eq("true")
        )

    return results_df, results_path


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


def safe_pearson_correlation(
    values_x,
    values_y,
):
    """Calculate Pearson correlation with constant-value protection."""

    values_x = np.asarray(
        values_x,
        dtype=float,
    )

    values_y = np.asarray(
        values_y,
        dtype=float,
    )

    if (
        len(values_x) < 2
        or np.std(values_x) == 0
        or np.std(values_y) == 0
    ):
        return float("nan")

    return float(
        np.corrcoef(
            values_x,
            values_y,
        )[0, 1]
    )


def calculate_spearman_correlation(
    values_x,
    values_y,
):
    """Calculate Spearman rank correlation without SciPy."""

    rank_x = pd.Series(
        values_x
    ).rank(
        method="average"
    )

    rank_y = pd.Series(
        values_y
    ).rank(
        method="average"
    )

    return safe_pearson_correlation(
        rank_x,
        rank_y,
    )


def build_overall_summary(
    results_df,
):
    """Calculate overall uncertainty-quality statistics."""

    actual_rul = results_df[
        "actual_rul"
    ].to_numpy()

    predicted_rul = results_df[
        "predicted_rul_mean"
    ].to_numpy()

    absolute_error = results_df[
        "absolute_error"
    ].to_numpy()

    mc_std = results_df[
        "mc_standard_deviation"
    ].to_numpy()

    interval_width = results_df[
        "interval_width"
    ].to_numpy()

    prediction_error = results_df[
        "prediction_error"
    ].to_numpy()

    summary = {
        "test_engines": len(results_df),
        "rmse": calculate_rmse(
            actual_rul,
            predicted_rul,
        ),
        "mae": float(
            mean_absolute_error(
                actual_rul,
                predicted_rul,
            )
        ),
        "interval_coverage": float(
            results_df[
                "actual_inside_interval"
            ].mean()
        ),
        "average_interval_width": float(
            np.mean(interval_width)
        ),
        "median_interval_width": float(
            np.median(interval_width)
        ),
        "average_mc_standard_deviation": float(
            np.mean(mc_std)
        ),
        "pearson_mc_std_vs_error": (
            safe_pearson_correlation(
                mc_std,
                absolute_error,
            )
        ),
        "spearman_mc_std_vs_error": (
            calculate_spearman_correlation(
                mc_std,
                absolute_error,
            )
        ),
        "pearson_interval_width_vs_error": (
            safe_pearson_correlation(
                interval_width,
                absolute_error,
            )
        ),
        "spearman_interval_width_vs_error": (
            calculate_spearman_correlation(
                interval_width,
                absolute_error,
            )
        ),
        "dangerous_overprediction_rate": float(
            np.mean(
                prediction_error > 10
            )
        ),
        "large_error_rate": float(
            np.mean(
                absolute_error > 20
            )
        ),
    }

    return summary


def build_rul_region_analysis(
    results_df,
):
    """Evaluate uncertainty coverage across RUL regions."""

    analysis_df = results_df.copy()

    analysis_df["rul_region"] = pd.cut(
        analysis_df["actual_rul"],
        bins=[
            -np.inf,
            20,
            50,
            80,
            np.inf,
        ],
        labels=[
            "Critical: 0–20",
            "Warning: 21–50",
            "Monitor: 51–80",
            "Healthy: 81+",
        ],
        include_lowest=True,
    )

    rows = []

    for region_name, region_df in analysis_df.groupby(
        "rul_region",
        observed=False,
    ):
        if region_df.empty:
            continue

        rows.append(
            {
                "rul_region": str(
                    region_name
                ),
                "engine_count": len(
                    region_df
                ),
                "rmse": calculate_rmse(
                    region_df["actual_rul"],
                    region_df[
                        "predicted_rul_mean"
                    ],
                ),
                "mae": float(
                    mean_absolute_error(
                        region_df["actual_rul"],
                        region_df[
                            "predicted_rul_mean"
                        ],
                    )
                ),
                "coverage": float(
                    region_df[
                        "actual_inside_interval"
                    ].mean()
                ),
                "average_interval_width": float(
                    region_df[
                        "interval_width"
                    ].mean()
                ),
                "average_absolute_error": float(
                    region_df[
                        "absolute_error"
                    ].mean()
                ),
                "dangerous_overprediction_rate": float(
                    (
                        region_df[
                            "prediction_error"
                        ]
                        > 10
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_confidence_analysis(
    results_df,
):
    """Compare accuracy between confidence categories."""

    confidence_order = [
        "High",
        "Medium",
        "Low",
    ]

    rows = []

    for confidence in confidence_order:
        confidence_df = results_df[
            results_df["confidence"]
            == confidence
        ]

        if confidence_df.empty:
            continue

        rows.append(
            {
                "confidence": confidence,
                "engine_count": len(
                    confidence_df
                ),
                "rmse": calculate_rmse(
                    confidence_df["actual_rul"],
                    confidence_df[
                        "predicted_rul_mean"
                    ],
                ),
                "mae": float(
                    mean_absolute_error(
                        confidence_df[
                            "actual_rul"
                        ],
                        confidence_df[
                            "predicted_rul_mean"
                        ],
                    )
                ),
                "coverage": float(
                    confidence_df[
                        "actual_inside_interval"
                    ].mean()
                ),
                "average_interval_width": float(
                    confidence_df[
                        "interval_width"
                    ].mean()
                ),
                "average_absolute_error": float(
                    confidence_df[
                        "absolute_error"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_rejection_curve(
    results_df,
):
    """
    Measure performance after rejecting uncertain predictions.

    Engines with the widest uncertainty intervals are removed first.
    If uncertainty is meaningful, RMSE should decrease as uncertain
    predictions are rejected.
    """

    sorted_df = results_df.sort_values(
        by="interval_width",
        ascending=True,
    ).reset_index(drop=True)

    retention_levels = [
        1.00,
        0.90,
        0.80,
        0.70,
        0.60,
        0.50,
        0.40,
        0.30,
        0.20,
    ]

    rows = []

    for retention_fraction in retention_levels:
        retained_count = max(
            1,
            int(
                np.ceil(
                    len(sorted_df)
                    * retention_fraction
                )
            ),
        )

        retained_df = sorted_df.iloc[
            :retained_count
        ]

        rows.append(
            {
                "retained_fraction": (
                    retention_fraction
                ),
                "retained_percentage": (
                    retention_fraction
                    * 100
                ),
                "retained_engines": len(
                    retained_df
                ),
                "rejected_engines": (
                    len(sorted_df)
                    - len(retained_df)
                ),
                "rmse": calculate_rmse(
                    retained_df["actual_rul"],
                    retained_df[
                        "predicted_rul_mean"
                    ],
                ),
                "mae": float(
                    mean_absolute_error(
                        retained_df[
                            "actual_rul"
                        ],
                        retained_df[
                            "predicted_rul_mean"
                        ],
                    )
                ),
                "average_absolute_error": float(
                    retained_df[
                        "absolute_error"
                    ].mean()
                ),
                "coverage": float(
                    retained_df[
                        "actual_inside_interval"
                    ].mean()
                ),
                "dangerous_overprediction_rate": float(
                    (
                        retained_df[
                            "prediction_error"
                        ]
                        > 10
                    ).mean()
                ),
                "maximum_interval_width": float(
                    retained_df[
                        "interval_width"
                    ].max()
                ),
            }
        )

    return pd.DataFrame(rows)


def identify_confidently_wrong_engines(
    results_df,
):
    """
    Identify engines with low predicted uncertainty but large errors.

    These are especially concerning because the model appears
    confident while being inaccurate.
    """

    low_uncertainty_threshold = float(
        results_df[
            "interval_width"
        ].quantile(0.25)
    )

    high_error_threshold = max(
        20.0,
        float(
            results_df[
                "absolute_error"
            ].quantile(0.75)
        ),
    )

    confidently_wrong_df = results_df[
        (
            results_df["interval_width"]
            <= low_uncertainty_threshold
        )
        & (
            results_df["absolute_error"]
            >= high_error_threshold
        )
    ].copy()

    confidently_wrong_df = (
        confidently_wrong_df
        .sort_values(
            by="absolute_error",
            ascending=False,
        )
    )

    return (
        confidently_wrong_df,
        low_uncertainty_threshold,
        high_error_threshold,
    )


def plot_uncertainty_vs_error(
    results_df,
    model_name,
    dataset,
):
    """Plot uncertainty interval width against absolute error."""

    plt.figure(
        figsize=(9, 7)
    )

    plt.scatter(
        results_df["interval_width"],
        results_df["absolute_error"],
        alpha=0.75,
    )

    worst_engines = results_df.nlargest(
        5,
        "absolute_error",
    )

    for _, row in worst_engines.iterrows():
        plt.annotate(
            f"Unit {int(row['unit'])}",
            (
                row["interval_width"],
                row["absolute_error"],
            ),
            xytext=(5, 5),
            textcoords="offset points",
        )

    plt.xlabel(
        "Calibrated uncertainty interval width"
    )

    plt.ylabel(
        "Absolute prediction error"
    )

    plt.title(
        f"Uncertainty vs Prediction Error — "
        f"{model_name.upper()} {dataset}"
    )

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_path = os.path.join(
        UNCERTAINTY_DIR,
        f"uncertainty_vs_error_"
        f"{model_name}_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_rejection_curve(
    rejection_df,
    model_name,
    dataset,
):
    """Plot model accuracy after rejecting uncertain predictions."""

    plt.figure(
        figsize=(9, 6)
    )

    plt.plot(
        rejection_df[
            "retained_percentage"
        ],
        rejection_df["rmse"],
        marker="o",
        label="RMSE",
    )

    plt.plot(
        rejection_df[
            "retained_percentage"
        ],
        rejection_df["mae"],
        marker="s",
        label="MAE",
    )

    plt.gca().invert_xaxis()

    plt.xlabel(
        "Predictions retained (%)"
    )

    plt.ylabel(
        "Prediction error in cycles"
    )

    plt.title(
        f"Uncertainty Rejection Curve — "
        f"{model_name.upper()} {dataset}"
    )

    plt.legend()

    plt.grid(
        alpha=0.25
    )

    plt.tight_layout()

    output_path = os.path.join(
        UNCERTAINTY_DIR,
        f"rejection_curve_"
        f"{model_name}_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_coverage_by_region(
    region_df,
    model_name,
    dataset,
):
    """Plot interval coverage for different actual-RUL regions."""

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        region_df["rul_region"],
        region_df["coverage"] * 100,
    )

    plt.axhline(
        90,
        linestyle="--",
        label="Requested 90% coverage",
    )

    plt.ylabel(
        "Observed interval coverage (%)"
    )

    plt.xlabel(
        "Actual RUL region"
    )

    plt.title(
        f"Uncertainty Coverage by RUL Region — "
        f"{model_name.upper()} {dataset}"
    )

    plt.ylim(
        0,
        105,
    )

    plt.xticks(
        rotation=15,
        ha="right",
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        UNCERTAINTY_DIR,
        f"coverage_by_region_"
        f"{model_name}_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def validate_uncertainty(
    model_name,
    dataset,
):
    """Run the complete uncertainty-validation analysis."""

    results_df, source_path = load_results(
        model_name,
        dataset,
    )

    summary = build_overall_summary(
        results_df
    )

    region_df = build_rul_region_analysis(
        results_df
    )

    confidence_df = build_confidence_analysis(
        results_df
    )

    rejection_df = build_rejection_curve(
        results_df
    )

    (
        confidently_wrong_df,
        low_uncertainty_threshold,
        high_error_threshold,
    ) = identify_confidently_wrong_engines(
        results_df
    )

    summary_path = os.path.join(
        UNCERTAINTY_DIR,
        f"validation_summary_"
        f"{model_name}_{dataset}.csv",
    )

    region_path = os.path.join(
        UNCERTAINTY_DIR,
        f"coverage_by_region_"
        f"{model_name}_{dataset}.csv",
    )

    confidence_path = os.path.join(
        UNCERTAINTY_DIR,
        f"confidence_analysis_"
        f"{model_name}_{dataset}.csv",
    )

    rejection_path = os.path.join(
        UNCERTAINTY_DIR,
        f"rejection_curve_"
        f"{model_name}_{dataset}.csv",
    )

    confidently_wrong_path = os.path.join(
        UNCERTAINTY_DIR,
        f"confidently_wrong_"
        f"{model_name}_{dataset}.csv",
    )

    pd.DataFrame(
        [summary]
    ).to_csv(
        summary_path,
        index=False,
    )

    region_df.to_csv(
        region_path,
        index=False,
    )

    confidence_df.to_csv(
        confidence_path,
        index=False,
    )

    rejection_df.to_csv(
        rejection_path,
        index=False,
    )

    confidently_wrong_df.to_csv(
        confidently_wrong_path,
        index=False,
    )

    scatter_path = plot_uncertainty_vs_error(
        results_df,
        model_name,
        dataset,
    )

    rejection_plot_path = plot_rejection_curve(
        rejection_df,
        model_name,
        dataset,
    )

    region_plot_path = plot_coverage_by_region(
        region_df,
        model_name,
        dataset,
    )

    print(
        f"Source results: {source_path}"
    )

    print("\nOverall uncertainty validation")

    print(
        f"Test engines:                   "
        f"{summary['test_engines']}"
    )

    print(
        f"RMSE:                           "
        f"{summary['rmse']:.4f}"
    )

    print(
        f"MAE:                            "
        f"{summary['mae']:.4f}"
    )

    print(
        f"Interval coverage:              "
        f"{summary['interval_coverage']:.1%}"
    )

    print(
        f"Average interval width:         "
        f"{summary['average_interval_width']:.2f}"
    )

    print(
        f"Pearson MC std vs error:        "
        f"{summary['pearson_mc_std_vs_error']:.4f}"
    )

    print(
        f"Spearman MC std vs error:       "
        f"{summary['spearman_mc_std_vs_error']:.4f}"
    )

    print(
        f"Pearson interval width vs error:"
        f" {summary['pearson_interval_width_vs_error']:.4f}"
    )

    print(
        f"Spearman interval width vs error:"
        f" {summary['spearman_interval_width_vs_error']:.4f}"
    )

    print(
        "\nCoverage by actual-RUL region"
    )

    print(
        region_df.to_string(
            index=False
        )
    )

    print(
        "\nUncertainty rejection curve"
    )

    print(
        rejection_df[
            [
                "retained_percentage",
                "retained_engines",
                "rmse",
                "mae",
                "coverage",
            ]
        ].to_string(
            index=False
        )
    )

    print(
        "\nConfidently wrong engines"
    )

    print(
        f"Low-uncertainty threshold: "
        f"{low_uncertainty_threshold:.2f}"
    )

    print(
        f"High-error threshold:      "
        f"{high_error_threshold:.2f}"
    )

    if confidently_wrong_df.empty:
        print(
            "No confidently wrong engines "
            "were identified."
        )
    else:
        print(
            confidently_wrong_df[
                [
                    "unit",
                    "actual_rul",
                    "predicted_rul_mean",
                    "absolute_error",
                    "interval_width",
                    "mc_standard_deviation",
                ]
            ].to_string(
                index=False
            )
        )

    print(
        "\nSaved outputs"
    )

    print(
        f"Summary:              {summary_path}"
    )

    print(
        f"RUL-region analysis:  {region_path}"
    )

    print(
        f"Confidence analysis:  {confidence_path}"
    )

    print(
        f"Rejection data:       {rejection_path}"
    )

    print(
        f"Confidently wrong:    {confidently_wrong_path}"
    )

    print(
        f"Uncertainty scatter:  {scatter_path}"
    )

    print(
        f"Rejection plot:       {rejection_plot_path}"
    )

    print(
        f"Coverage plot:        {region_plot_path}"
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

    arguments = parser.parse_args()

    validate_uncertainty(
        model_name=arguments.model,
        dataset=arguments.dataset,
    )