import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = "results_v2"

V3_DIR = os.path.join(
    RESULTS_DIR,
    "decisions_v3",
)

V4_DIR = os.path.join(
    RESULTS_DIR,
    "decisions_v4",
)

VALIDATION_DIR = os.path.join(
    RESULTS_DIR,
    "decision_validation",
)

os.makedirs(
    VALIDATION_DIR,
    exist_ok=True,
)


REVIEW_LEVELS = [
    "Required",
    "Recommended",
    "Not required",
]


def safe_divide(
    numerator,
    denominator,
):
    """Safely divide two numbers."""

    if denominator == 0:
        return 0.0

    return float(
        numerator
        / denominator
    )


def load_decision_results(
    version,
    model_name,
    dataset,
):
    """Load one decision-layer result file."""

    if version == "v3":
        source_directory = V3_DIR

    elif version == "v4":
        source_directory = V4_DIR

    else:
        raise ValueError(
            f"Unsupported version: {version}"
        )

    result_path = os.path.join(
        source_directory,
        f"decision_results_{version}_"
        f"{model_name}_{dataset}.csv",
    )

    if not os.path.exists(
        result_path
    ):
        raise FileNotFoundError(
            f"Decision results not found: {result_path}"
        )

    decision_df = pd.read_csv(
        result_path
    )

    required_columns = [
        "unit",
        "actual_rul",
        "predicted_rul_mean",
        "absolute_error",
        "prediction_error",
        "engine_condition",
        "prediction_trust",
        "review_flag",
        "risk_score",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column
        not in decision_df.columns
    ]

    if missing_columns:
        raise KeyError(
            f"{version.upper()} results are missing columns: "
            f"{missing_columns}"
        )

    decision_df[
        "unit"
    ] = decision_df[
        "unit"
    ].astype(int)

    if decision_df[
        "unit"
    ].duplicated().any():
        raise ValueError(
            f"{version.upper()} contains duplicate engine IDs."
        )

    if not decision_df[
        "review_flag"
    ].isin(
        REVIEW_LEVELS
    ).all():
        unexpected_values = (
            decision_df.loc[
                ~decision_df[
                    "review_flag"
                ].isin(
                    REVIEW_LEVELS
                ),
                "review_flag",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            "Unexpected review categories in "
            f"{version.upper()}: {unexpected_values}"
        )

    return decision_df, result_path


def add_evaluation_labels(
    decision_df,
    high_error_threshold,
    dangerous_threshold,
    severe_dangerous_threshold,
):
    """
    Add retrospective evaluation labels.

    These labels use actual RUL only for evaluation.
    They are never used to assign decisions.
    """

    evaluation_df = decision_df.copy()

    evaluation_df[
        "high_error"
    ] = (
        evaluation_df[
            "absolute_error"
        ]
        >= high_error_threshold
    )

    evaluation_df[
        "moderate_error"
    ] = (
        evaluation_df[
            "absolute_error"
        ]
        >= 10.0
    )

    evaluation_df[
        "dangerous_overprediction"
    ] = (
        evaluation_df[
            "prediction_error"
        ]
        > dangerous_threshold
    )

    evaluation_df[
        "severe_dangerous_overprediction"
    ] = (
        evaluation_df[
            "prediction_error"
        ]
        > severe_dangerous_threshold
    )

    evaluation_df[
        "required_review"
    ] = (
        evaluation_df[
            "review_flag"
        ]
        == "Required"
    )

    evaluation_df[
        "any_review"
    ] = (
        evaluation_df[
            "review_flag"
        ]
        != "Not required"
    )

    evaluation_df[
        "low_error_required_review"
    ] = (
        evaluation_df[
            "required_review"
        ]
        & (
            evaluation_df[
                "absolute_error"
            ]
            < 10.0
        )
    )

    evaluation_df[
        "missed_dangerous_prediction"
    ] = (
        evaluation_df[
            "dangerous_overprediction"
        ]
        & (
            evaluation_df[
                "review_flag"
            ]
            == "Not required"
        )
    )

    evaluation_df[
        "missed_high_error"
    ] = (
        evaluation_df[
            "high_error"
        ]
        & (
            evaluation_df[
                "review_flag"
            ]
            == "Not required"
        )
    )

    return evaluation_df


def calculate_validation_metrics(
    evaluation_df,
    version,
):
    """Calculate workload, recall and precision metrics."""

    total_engines = len(
        evaluation_df
    )

    required_count = int(
        evaluation_df[
            "required_review"
        ].sum()
    )

    any_review_count = int(
        evaluation_df[
            "any_review"
        ].sum()
    )

    high_error_count = int(
        evaluation_df[
            "high_error"
        ].sum()
    )

    dangerous_count = int(
        evaluation_df[
            "dangerous_overprediction"
        ].sum()
    )

    severe_dangerous_count = int(
        evaluation_df[
            "severe_dangerous_overprediction"
        ].sum()
    )

    required_high_error_count = int(
        (
            evaluation_df[
                "required_review"
            ]
            & evaluation_df[
                "high_error"
            ]
        ).sum()
    )

    any_review_high_error_count = int(
        (
            evaluation_df[
                "any_review"
            ]
            & evaluation_df[
                "high_error"
            ]
        ).sum()
    )

    required_dangerous_count = int(
        (
            evaluation_df[
                "required_review"
            ]
            & evaluation_df[
                "dangerous_overprediction"
            ]
        ).sum()
    )

    any_review_dangerous_count = int(
        (
            evaluation_df[
                "any_review"
            ]
            & evaluation_df[
                "dangerous_overprediction"
            ]
        ).sum()
    )

    required_severe_dangerous_count = int(
        (
            evaluation_df[
                "required_review"
            ]
            & evaluation_df[
                "severe_dangerous_overprediction"
            ]
        ).sum()
    )

    any_review_severe_dangerous_count = int(
        (
            evaluation_df[
                "any_review"
            ]
            & evaluation_df[
                "severe_dangerous_overprediction"
            ]
        ).sum()
    )

    low_error_required_count = int(
        evaluation_df[
            "low_error_required_review"
        ].sum()
    )

    missed_dangerous_count = int(
        evaluation_df[
            "missed_dangerous_prediction"
        ].sum()
    )

    missed_high_error_count = int(
        evaluation_df[
            "missed_high_error"
        ].sum()
    )

    not_required_df = evaluation_df[
        evaluation_df[
            "review_flag"
        ]
        == "Not required"
    ]

    required_df = evaluation_df[
        evaluation_df[
            "review_flag"
        ]
        == "Required"
    ]

    recommended_df = evaluation_df[
        evaluation_df[
            "review_flag"
        ]
        == "Recommended"
    ]

    return {
        "version": version.upper(),
        "total_engines": total_engines,
        "required_review_count": required_count,
        "required_review_rate": safe_divide(
            required_count,
            total_engines,
        ),
        "any_review_count": any_review_count,
        "any_review_rate": safe_divide(
            any_review_count,
            total_engines,
        ),
        "high_error_count": high_error_count,
        "high_error_required_recall": safe_divide(
            required_high_error_count,
            high_error_count,
        ),
        "high_error_any_review_recall": safe_divide(
            any_review_high_error_count,
            high_error_count,
        ),
        "high_error_required_precision": safe_divide(
            required_high_error_count,
            required_count,
        ),
        "missed_high_error_count": missed_high_error_count,
        "dangerous_overprediction_count": dangerous_count,
        "dangerous_required_recall": safe_divide(
            required_dangerous_count,
            dangerous_count,
        ),
        "dangerous_any_review_recall": safe_divide(
            any_review_dangerous_count,
            dangerous_count,
        ),
        "dangerous_required_precision": safe_divide(
            required_dangerous_count,
            required_count,
        ),
        "missed_dangerous_count": missed_dangerous_count,
        "severe_dangerous_count": severe_dangerous_count,
        "severe_dangerous_required_recall": safe_divide(
            required_severe_dangerous_count,
            severe_dangerous_count,
        ),
        "severe_dangerous_any_review_recall": safe_divide(
            any_review_severe_dangerous_count,
            severe_dangerous_count,
        ),
        "low_error_required_count": low_error_required_count,
        "low_error_required_rate_of_fleet": safe_divide(
            low_error_required_count,
            total_engines,
        ),
        "low_error_share_of_required_reviews": safe_divide(
            low_error_required_count,
            required_count,
        ),
        "required_average_absolute_error": (
            float(
                required_df[
                    "absolute_error"
                ].mean()
            )
            if not required_df.empty
            else np.nan
        ),
        "recommended_average_absolute_error": (
            float(
                recommended_df[
                    "absolute_error"
                ].mean()
            )
            if not recommended_df.empty
            else np.nan
        ),
        "not_required_average_absolute_error": (
            float(
                not_required_df[
                    "absolute_error"
                ].mean()
            )
            if not not_required_df.empty
            else np.nan
        ),
        "not_required_dangerous_rate": (
            float(
                not_required_df[
                    "dangerous_overprediction"
                ].mean()
            )
            if not not_required_df.empty
            else 0.0
        ),
        "not_required_high_error_rate": (
            float(
                not_required_df[
                    "high_error"
                ].mean()
            )
            if not not_required_df.empty
            else 0.0
        ),
    }


def build_review_performance_table(
    evaluation_df,
    version,
):
    """Evaluate each review category separately."""

    rows = []

    for review_flag in REVIEW_LEVELS:
        group_df = evaluation_df[
            evaluation_df[
                "review_flag"
            ]
            == review_flag
        ]

        if group_df.empty:
            continue

        rows.append(
            {
                "version": version.upper(),
                "review_flag": review_flag,
                "engine_count": len(
                    group_df
                ),
                "percentage": (
                    len(
                        group_df
                    )
                    / len(
                        evaluation_df
                    )
                    * 100.0
                ),
                "average_absolute_error": float(
                    group_df[
                        "absolute_error"
                    ].mean()
                ),
                "median_absolute_error": float(
                    group_df[
                        "absolute_error"
                    ].median()
                ),
                "high_error_rate": float(
                    group_df[
                        "high_error"
                    ].mean()
                ),
                "dangerous_overprediction_rate": float(
                    group_df[
                        "dangerous_overprediction"
                    ].mean()
                ),
                "severe_dangerous_rate": float(
                    group_df[
                        "severe_dangerous_overprediction"
                    ].mean()
                ),
                "average_risk_score": float(
                    group_df[
                        "risk_score"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_rul_region_table(
    evaluation_df,
    version,
):
    """Evaluate review behaviour across actual-RUL regions."""

    region_df = evaluation_df.copy()

    region_df[
        "actual_rul_region"
    ] = pd.cut(
        region_df[
            "actual_rul"
        ],
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
    )

    rows = []

    for (
        region,
        group_df,
    ) in region_df.groupby(
        "actual_rul_region",
        observed=True,
    ):
        rows.append(
            {
                "version": version.upper(),
                "actual_rul_region": str(
                    region
                ),
                "engine_count": len(
                    group_df
                ),
                "required_review_rate": float(
                    group_df[
                        "required_review"
                    ].mean()
                ),
                "any_review_rate": float(
                    group_df[
                        "any_review"
                    ].mean()
                ),
                "average_absolute_error": float(
                    group_df[
                        "absolute_error"
                    ].mean()
                ),
                "high_error_rate": float(
                    group_df[
                        "high_error"
                    ].mean()
                ),
                "dangerous_overprediction_rate": float(
                    group_df[
                        "dangerous_overprediction"
                    ].mean()
                ),
                "missed_dangerous_count": int(
                    group_df[
                        "missed_dangerous_prediction"
                    ].sum()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_transition_table(
    v3_df,
    v4_df,
):
    """Build engine-level V3-to-V4 review transitions."""

    v3_columns = [
        "unit",
        "review_flag",
        "engine_condition",
        "risk_score",
    ]

    v4_columns = [
        "unit",
        "review_flag",
        "engine_condition",
        "risk_score",
        "trajectory_flag",
        "trajectory_trust",
        "recent_slope",
        "recent_predicted_drop",
    ]

    merged_df = v3_df[
        v3_columns
    ].rename(
        columns={
            "review_flag": "v3_review_flag",
            "engine_condition": "v3_engine_condition",
            "risk_score": "v3_risk_score",
        }
    ).merge(
        v4_df[
            v4_columns
        ].rename(
            columns={
                "review_flag": "v4_review_flag",
                "engine_condition": "v4_engine_condition",
                "risk_score": "v4_risk_score",
            }
        ),
        on="unit",
        how="inner",
        validate="one_to_one",
    )

    evaluation_columns = [
        "unit",
        "actual_rul",
        "predicted_rul_mean",
        "absolute_error",
        "prediction_error",
        "high_error",
        "dangerous_overprediction",
        "severe_dangerous_overprediction",
    ]

    merged_df = merged_df.merge(
        v4_df[
            evaluation_columns
        ],
        on="unit",
        how="inner",
        validate="one_to_one",
    )

    review_priority = {
        "Not required": 0,
        "Recommended": 1,
        "Required": 2,
    }

    merged_df[
        "v3_review_priority"
    ] = merged_df[
        "v3_review_flag"
    ].map(
        review_priority
    )

    merged_df[
        "v4_review_priority"
    ] = merged_df[
        "v4_review_flag"
    ].map(
        review_priority
    )

    merged_df[
        "review_change"
    ] = (
        merged_df[
            "v4_review_priority"
        ]
        - merged_df[
            "v3_review_priority"
        ]
    )

    merged_df[
        "transition"
    ] = (
        merged_df[
            "v3_review_flag"
        ]
        + " → "
        + merged_df[
            "v4_review_flag"
        ]
    )

    return merged_df.sort_values(
        by=[
            "review_change",
            "absolute_error",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )


def build_transition_summary(
    transition_df,
):
    """Summarize review-category transitions."""

    transition_summary_df = (
        transition_df.groupby(
            [
                "v3_review_flag",
                "v4_review_flag",
            ],
            as_index=False,
        )
        .agg(
            engine_count=(
                "unit",
                "count",
            ),
            average_absolute_error=(
                "absolute_error",
                "mean",
            ),
            dangerous_overprediction_count=(
                "dangerous_overprediction",
                "sum",
            ),
            high_error_count=(
                "high_error",
                "sum",
            ),
        )
    )

    transition_summary_df[
        "percentage"
    ] = (
        transition_summary_df[
            "engine_count"
        ]
        / len(
            transition_df
        )
        * 100.0
    )

    return transition_summary_df.sort_values(
        by="engine_count",
        ascending=False,
    ).reset_index(
        drop=True
    )


def build_incremental_trajectory_analysis(
    transition_df,
):
    """Measure what the temporal layer added beyond V3."""

    upgraded_df = transition_df[
        transition_df[
            "review_change"
        ]
        > 0
    ].copy()

    newly_required_df = transition_df[
        (
            transition_df[
                "v4_review_flag"
            ]
            == "Required"
        )
        & (
            transition_df[
                "v3_review_flag"
            ]
            != "Required"
        )
    ].copy()

    rescued_dangerous_df = transition_df[
        (
            transition_df[
                "dangerous_overprediction"
            ]
        )
        & (
            transition_df[
                "v3_review_flag"
            ]
            == "Not required"
        )
        & (
            transition_df[
                "v4_review_flag"
            ]
            != "Not required"
        )
    ].copy()

    rescued_high_error_df = transition_df[
        (
            transition_df[
                "high_error"
            ]
        )
        & (
            transition_df[
                "v3_review_flag"
            ]
            == "Not required"
        )
        & (
            transition_df[
                "v4_review_flag"
            ]
            != "Not required"
        )
    ].copy()

    rows = [
        {
            "metric": "Engines upgraded by V4",
            "engine_count": len(
                upgraded_df
            ),
            "percentage_of_fleet": (
                len(
                    upgraded_df
                )
                / len(
                    transition_df
                )
                * 100.0
            ),
            "average_absolute_error": (
                float(
                    upgraded_df[
                        "absolute_error"
                    ].mean()
                )
                if not upgraded_df.empty
                else 0.0
            ),
        },
        {
            "metric": "New mandatory reviews",
            "engine_count": len(
                newly_required_df
            ),
            "percentage_of_fleet": (
                len(
                    newly_required_df
                )
                / len(
                    transition_df
                )
                * 100.0
            ),
            "average_absolute_error": (
                float(
                    newly_required_df[
                        "absolute_error"
                    ].mean()
                )
                if not newly_required_df.empty
                else 0.0
            ),
        },
        {
            "metric": (
                "Previously missed dangerous "
                "predictions now reviewed"
            ),
            "engine_count": len(
                rescued_dangerous_df
            ),
            "percentage_of_fleet": (
                len(
                    rescued_dangerous_df
                )
                / len(
                    transition_df
                )
                * 100.0
            ),
            "average_absolute_error": (
                float(
                    rescued_dangerous_df[
                        "absolute_error"
                    ].mean()
                )
                if not rescued_dangerous_df.empty
                else 0.0
            ),
        },
        {
            "metric": (
                "Previously missed high-error "
                "predictions now reviewed"
            ),
            "engine_count": len(
                rescued_high_error_df
            ),
            "percentage_of_fleet": (
                len(
                    rescued_high_error_df
                )
                / len(
                    transition_df
                )
                * 100.0
            ),
            "average_absolute_error": (
                float(
                    rescued_high_error_df[
                        "absolute_error"
                    ].mean()
                )
                if not rescued_high_error_df.empty
                else 0.0
            ),
        },
    ]

    return (
        pd.DataFrame(
            rows
        ),
        upgraded_df,
        newly_required_df,
        rescued_dangerous_df,
        rescued_high_error_df,
    )


def plot_review_workload(
    validation_metrics_df,
    model_name,
    dataset,
):
    """Plot V3 and V4 review workload."""

    plot_df = validation_metrics_df.copy()

    positions = np.arange(
        len(
            plot_df
        )
    )

    width = 0.35

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        positions - width / 2,
        plot_df[
            "required_review_rate"
        ]
        * 100.0,
        width=width,
        label="Required review",
    )

    plt.bar(
        positions + width / 2,
        plot_df[
            "any_review_rate"
        ]
        * 100.0,
        width=width,
        label="Any review",
    )

    plt.xticks(
        positions,
        plot_df[
            "version"
        ],
    )

    plt.ylabel(
        "Percentage of test engines"
    )

    plt.xlabel(
        "Decision-layer version"
    )

    plt.title(
        "Engineering Review Workload — "
        f"{model_name.upper()} {dataset}"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        VALIDATION_DIR,
        f"review_workload_"
        f"{model_name}_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_capture_comparison(
    validation_metrics_df,
    model_name,
    dataset,
):
    """Plot high-error and dangerous-error capture."""

    plot_df = validation_metrics_df.copy()

    positions = np.arange(
        len(
            plot_df
        )
    )

    width = 0.25

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        positions - width,
        plot_df[
            "high_error_required_recall"
        ]
        * 100.0,
        width=width,
        label="High-error recall",
    )

    plt.bar(
        positions,
        plot_df[
            "dangerous_required_recall"
        ]
        * 100.0,
        width=width,
        label="Dangerous-overprediction recall",
    )

    plt.bar(
        positions + width,
        plot_df[
            "severe_dangerous_required_recall"
        ]
        * 100.0,
        width=width,
        label="Severe-danger recall",
    )

    plt.xticks(
        positions,
        plot_df[
            "version"
        ],
    )

    plt.ylabel(
        "Captured by mandatory review (%)"
    )

    plt.xlabel(
        "Decision-layer version"
    )

    plt.title(
        "Mandatory Review Error Capture — "
        f"{model_name.upper()} {dataset}"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        VALIDATION_DIR,
        f"error_capture_"
        f"{model_name}_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_transition_matrix(
    transition_df,
    model_name,
    dataset,
):
    """Plot V3-to-V4 review transitions."""

    transition_matrix = pd.crosstab(
        transition_df[
            "v3_review_flag"
        ],
        transition_df[
            "v4_review_flag"
        ],
    ).reindex(
        index=REVIEW_LEVELS,
        columns=REVIEW_LEVELS,
        fill_value=0,
    )

    matrix_values = transition_matrix.to_numpy()

    plt.figure(
        figsize=(8, 7)
    )

    image = plt.imshow(
        matrix_values,
        aspect="auto",
    )

    plt.colorbar(
        image,
        label="Number of engines",
    )

    plt.xticks(
        np.arange(
            len(
                REVIEW_LEVELS
            )
        ),
        REVIEW_LEVELS,
        rotation=20,
    )

    plt.yticks(
        np.arange(
            len(
                REVIEW_LEVELS
            )
        ),
        REVIEW_LEVELS,
    )

    plt.xlabel(
        "V4 review decision"
    )

    plt.ylabel(
        "V3 review decision"
    )

    plt.title(
        "Review Decision Transitions — "
        f"{model_name.upper()} {dataset}"
    )

    for row_index in range(
        matrix_values.shape[
            0
        ]
    ):
        for column_index in range(
            matrix_values.shape[
                1
            ]
        ):
            plt.text(
                column_index,
                row_index,
                str(
                    matrix_values[
                        row_index,
                        column_index,
                    ]
                ),
                ha="center",
                va="center",
            )

    plt.tight_layout()

    output_path = os.path.join(
        VALIDATION_DIR,
        f"review_transitions_"
        f"{model_name}_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def print_selected_transitions(
    transition_df,
    selected_units,
):
    """Print V3-to-V4 changes for selected engines."""

    selected_df = transition_df[
        transition_df[
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
        "predicted_rul_mean",
        "absolute_error",
        "prediction_error",
        "v3_review_flag",
        "v4_review_flag",
        "trajectory_flag",
        "trajectory_trust",
        "recent_slope",
        "review_change",
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


def run_validation(
    model_name,
    dataset,
    selected_units,
    high_error_threshold,
    dangerous_threshold,
    severe_dangerous_threshold,
):
    """Validate V4 against V3 without changing any rules."""

    v3_df, v3_path = load_decision_results(
        version="v3",
        model_name=model_name,
        dataset=dataset,
    )

    v4_df, v4_path = load_decision_results(
        version="v4",
        model_name=model_name,
        dataset=dataset,
    )

    if set(
        v3_df[
            "unit"
        ]
    ) != set(
        v4_df[
            "unit"
        ]
    ):
        raise ValueError(
            "V3 and V4 do not contain the same test engines."
        )

    v3_evaluation_df = add_evaluation_labels(
        decision_df=v3_df,
        high_error_threshold=(
            high_error_threshold
        ),
        dangerous_threshold=(
            dangerous_threshold
        ),
        severe_dangerous_threshold=(
            severe_dangerous_threshold
        ),
    )

    v4_evaluation_df = add_evaluation_labels(
        decision_df=v4_df,
        high_error_threshold=(
            high_error_threshold
        ),
        dangerous_threshold=(
            dangerous_threshold
        ),
        severe_dangerous_threshold=(
            severe_dangerous_threshold
        ),
    )

    validation_metrics_df = pd.DataFrame(
        [
            calculate_validation_metrics(
                evaluation_df=(
                    v3_evaluation_df
                ),
                version="v3",
            ),
            calculate_validation_metrics(
                evaluation_df=(
                    v4_evaluation_df
                ),
                version="v4",
            ),
        ]
    )

    v3_review_performance_df = (
        build_review_performance_table(
            evaluation_df=(
                v3_evaluation_df
            ),
            version="v3",
        )
    )

    v4_review_performance_df = (
        build_review_performance_table(
            evaluation_df=(
                v4_evaluation_df
            ),
            version="v4",
        )
    )

    review_performance_df = pd.concat(
        [
            v3_review_performance_df,
            v4_review_performance_df,
        ],
        ignore_index=True,
    )

    region_performance_df = pd.concat(
        [
            build_rul_region_table(
                evaluation_df=(
                    v3_evaluation_df
                ),
                version="v3",
            ),
            build_rul_region_table(
                evaluation_df=(
                    v4_evaluation_df
                ),
                version="v4",
            ),
        ],
        ignore_index=True,
    )

    transition_df = build_transition_table(
        v3_df=v3_evaluation_df,
        v4_df=v4_evaluation_df,
    )

    transition_summary_df = (
        build_transition_summary(
            transition_df
        )
    )

    (
        incremental_summary_df,
        upgraded_df,
        newly_required_df,
        rescued_dangerous_df,
        rescued_high_error_df,
    ) = build_incremental_trajectory_analysis(
        transition_df
    )

    validation_metrics_path = os.path.join(
        VALIDATION_DIR,
        f"validation_metrics_"
        f"{model_name}_{dataset}.csv",
    )

    review_performance_path = os.path.join(
        VALIDATION_DIR,
        f"review_performance_"
        f"{model_name}_{dataset}.csv",
    )

    region_performance_path = os.path.join(
        VALIDATION_DIR,
        f"region_performance_"
        f"{model_name}_{dataset}.csv",
    )

    transition_path = os.path.join(
        VALIDATION_DIR,
        f"engine_transitions_"
        f"{model_name}_{dataset}.csv",
    )

    transition_summary_path = os.path.join(
        VALIDATION_DIR,
        f"transition_summary_"
        f"{model_name}_{dataset}.csv",
    )

    incremental_summary_path = os.path.join(
        VALIDATION_DIR,
        f"trajectory_incremental_summary_"
        f"{model_name}_{dataset}.csv",
    )

    upgraded_path = os.path.join(
        VALIDATION_DIR,
        f"upgraded_engines_"
        f"{model_name}_{dataset}.csv",
    )

    newly_required_path = os.path.join(
        VALIDATION_DIR,
        f"newly_required_engines_"
        f"{model_name}_{dataset}.csv",
    )

    rescued_dangerous_path = os.path.join(
        VALIDATION_DIR,
        f"rescued_dangerous_engines_"
        f"{model_name}_{dataset}.csv",
    )

    rescued_high_error_path = os.path.join(
        VALIDATION_DIR,
        f"rescued_high_error_engines_"
        f"{model_name}_{dataset}.csv",
    )

    validation_metrics_df.to_csv(
        validation_metrics_path,
        index=False,
    )

    review_performance_df.to_csv(
        review_performance_path,
        index=False,
    )

    region_performance_df.to_csv(
        region_performance_path,
        index=False,
    )

    transition_df.to_csv(
        transition_path,
        index=False,
    )

    transition_summary_df.to_csv(
        transition_summary_path,
        index=False,
    )

    incremental_summary_df.to_csv(
        incremental_summary_path,
        index=False,
    )

    upgraded_df.to_csv(
        upgraded_path,
        index=False,
    )

    newly_required_df.to_csv(
        newly_required_path,
        index=False,
    )

    rescued_dangerous_df.to_csv(
        rescued_dangerous_path,
        index=False,
    )

    rescued_high_error_df.to_csv(
        rescued_high_error_path,
        index=False,
    )

    workload_plot_path = plot_review_workload(
        validation_metrics_df=(
            validation_metrics_df
        ),
        model_name=model_name,
        dataset=dataset,
    )

    capture_plot_path = plot_capture_comparison(
        validation_metrics_df=(
            validation_metrics_df
        ),
        model_name=model_name,
        dataset=dataset,
    )

    transition_plot_path = plot_transition_matrix(
        transition_df=transition_df,
        model_name=model_name,
        dataset=dataset,
    )

    print(
        f"V3 source: {v3_path}"
    )

    print(
        f"V4 source: {v4_path}"
    )

    print(
        "\nEvaluation thresholds"
    )

    print(
        f"High absolute error:       "
        f">= {high_error_threshold:.1f} cycles"
    )

    print(
        f"Dangerous overprediction:  "
        f"> {dangerous_threshold:.1f} cycles"
    )

    print(
        f"Severe overprediction:     "
        f"> {severe_dangerous_threshold:.1f} cycles"
    )

    print(
        "\nV3 versus V4 validation metrics"
    )

    display_columns = [
        "version",
        "required_review_count",
        "required_review_rate",
        "any_review_rate",
        "high_error_required_recall",
        "dangerous_required_recall",
        "severe_dangerous_required_recall",
        "missed_high_error_count",
        "missed_dangerous_count",
        "low_error_required_count",
        "low_error_share_of_required_reviews",
        "not_required_average_absolute_error",
        "not_required_dangerous_rate",
    ]

    print(
        validation_metrics_df[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nReview category performance"
    )

    print(
        review_performance_df.to_string(
            index=False
        )
    )

    print(
        "\nPerformance by actual-RUL region"
    )

    print(
        region_performance_df.to_string(
            index=False
        )
    )

    print(
        "\nReview transition summary"
    )

    print(
        transition_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nIncremental contribution of trajectory diagnostics"
    )

    print(
        incremental_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nSelected engine transitions"
    )

    print_selected_transitions(
        transition_df=transition_df,
        selected_units=selected_units,
    )

    print(
        "\nSaved outputs"
    )

    print(
        f"Validation metrics:    "
        f"{validation_metrics_path}"
    )

    print(
        f"Review performance:    "
        f"{review_performance_path}"
    )

    print(
        f"Region performance:    "
        f"{region_performance_path}"
    )

    print(
        f"Engine transitions:    "
        f"{transition_path}"
    )

    print(
        f"Transition summary:    "
        f"{transition_summary_path}"
    )

    print(
        f"Incremental summary:   "
        f"{incremental_summary_path}"
    )

    print(
        f"Upgraded engines:      "
        f"{upgraded_path}"
    )

    print(
        f"New mandatory reviews: "
        f"{newly_required_path}"
    )

    print(
        f"Rescued dangerous:     "
        f"{rescued_dangerous_path}"
    )

    print(
        f"Rescued high-error:    "
        f"{rescued_high_error_path}"
    )

    print(
        f"Workload plot:         "
        f"{workload_plot_path}"
    )

    print(
        f"Capture plot:          "
        f"{capture_plot_path}"
    )

    print(
        f"Transition plot:       "
        f"{transition_plot_path}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        default="gru",
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

    parser.add_argument(
        "--high_error_threshold",
        type=float,
        default=20.0,
    )

    parser.add_argument(
        "--dangerous_threshold",
        type=float,
        default=10.0,
    )

    parser.add_argument(
        "--severe_dangerous_threshold",
        type=float,
        default=20.0,
    )

    arguments = parser.parse_args()

    if arguments.high_error_threshold <= 0:
        raise ValueError(
            "high_error_threshold must be positive."
        )

    if arguments.dangerous_threshold <= 0:
        raise ValueError(
            "dangerous_threshold must be positive."
        )

    if (
        arguments.severe_dangerous_threshold
        <= arguments.dangerous_threshold
    ):
        raise ValueError(
            "severe_dangerous_threshold must be greater "
            "than dangerous_threshold."
        )

    run_validation(
        model_name=arguments.model,
        dataset=arguments.dataset,
        selected_units=arguments.units,
        high_error_threshold=(
            arguments.high_error_threshold
        ),
        dangerous_threshold=(
            arguments.dangerous_threshold
        ),
        severe_dangerous_threshold=(
            arguments.severe_dangerous_threshold
        ),
    )