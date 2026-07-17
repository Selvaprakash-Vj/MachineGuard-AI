import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = "results_v2"

V4_DIR = os.path.join(
    RESULTS_DIR,
    "decisions_v4",
)

V5_DIR = os.path.join(
    RESULTS_DIR,
    "decisions_v5",
)

VALIDATION_DIR = os.path.join(
    RESULTS_DIR,
    "decision_validation_v5",
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

OPERATIONAL_PRIORITIES = [
    "Immediate",
    "High",
    "Elevated",
    "Watch",
    "Routine",
]


def safe_divide(
    numerator,
    denominator,
):
    """Safely divide two numbers."""

    if denominator == 0:
        return 0.0

    return float(
        numerator / denominator
    )


def load_decision_results(
    version,
    model_name,
    dataset,
):
    """Load V4 or V5 decision-layer results."""

    if version == "v4":
        source_directory = V4_DIR

    elif version == "v5":
        source_directory = V5_DIR

    else:
        raise ValueError(
            f"Unsupported decision-layer version: {version}"
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

    common_required_columns = [
        "unit",
        "actual_rul",
        "predicted_rul_mean",
        "absolute_error",
        "prediction_error",
        "engine_condition",
        "prediction_trust",
        "trajectory_flag",
        "trajectory_trust",
        "review_flag",
    ]

    version_specific_columns = {
        "v4": [
            "risk_score",
        ],
        "v5": [
            "condition_severity_score",
            "reliability_risk_score",
            "reliability_risk",
            "operational_priority_score",
            "operational_priority",
            "mc_deterministic_gap",
            "mc_gap_level",
        ],
    }

    required_columns = (
        common_required_columns
        + version_specific_columns[
            version
        ]
    )

    missing_columns = [
        column
        for column in required_columns
        if column not in decision_df.columns
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

    invalid_review_categories = (
        ~decision_df[
            "review_flag"
        ].isin(
            REVIEW_LEVELS
        )
    )

    if invalid_review_categories.any():
        unexpected_values = (
            decision_df.loc[
                invalid_review_categories,
                "review_flag",
            ]
            .drop_duplicates()
            .tolist()
        )

        raise ValueError(
            f"Unexpected review categories in "
            f"{version.upper()}: {unexpected_values}"
        )

    return (
        decision_df,
        result_path,
    )


def add_evaluation_labels(
    decision_df,
    high_error_threshold,
    dangerous_threshold,
    severe_dangerous_threshold,
):
    """
    Add retrospective evaluation labels.

    Actual RUL is used only for evaluation. It is not used by
    either decision layer when assigning decisions.
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
        "low_error"
    ] = (
        evaluation_df[
            "absolute_error"
        ]
        < 10.0
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
        "missed_severe_dangerous_prediction"
    ] = (
        evaluation_df[
            "severe_dangerous_overprediction"
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
    """Calculate workload and error-capture metrics."""

    total_engines = len(
        evaluation_df
    )

    required_mask = (
        evaluation_df[
            "review_flag"
        ]
        == "Required"
    )

    any_review_mask = (
        evaluation_df[
            "review_flag"
        ]
        != "Not required"
    )

    high_error_mask = evaluation_df[
        "high_error"
    ]

    dangerous_mask = evaluation_df[
        "dangerous_overprediction"
    ]

    severe_dangerous_mask = evaluation_df[
        "severe_dangerous_overprediction"
    ]

    required_count = int(
        required_mask.sum()
    )

    any_review_count = int(
        any_review_mask.sum()
    )

    high_error_count = int(
        high_error_mask.sum()
    )

    dangerous_count = int(
        dangerous_mask.sum()
    )

    severe_dangerous_count = int(
        severe_dangerous_mask.sum()
    )

    required_high_error_count = int(
        (
            required_mask
            & high_error_mask
        ).sum()
    )

    any_review_high_error_count = int(
        (
            any_review_mask
            & high_error_mask
        ).sum()
    )

    required_dangerous_count = int(
        (
            required_mask
            & dangerous_mask
        ).sum()
    )

    any_review_dangerous_count = int(
        (
            any_review_mask
            & dangerous_mask
        ).sum()
    )

    required_severe_count = int(
        (
            required_mask
            & severe_dangerous_mask
        ).sum()
    )

    any_review_severe_count = int(
        (
            any_review_mask
            & severe_dangerous_mask
        ).sum()
    )

    low_error_required_count = int(
        (
            required_mask
            & evaluation_df[
                "low_error"
            ]
        ).sum()
    )

    required_df = evaluation_df[
        required_mask
    ]

    recommended_df = evaluation_df[
        evaluation_df[
            "review_flag"
        ]
        == "Recommended"
    ]

    not_required_df = evaluation_df[
        evaluation_df[
            "review_flag"
        ]
        == "Not required"
    ]

    return {
        "version": version.upper(),
        "total_engines": total_engines,
        "required_review_count": (
            required_count
        ),
        "required_review_rate": safe_divide(
            required_count,
            total_engines,
        ),
        "any_review_count": (
            any_review_count
        ),
        "any_review_rate": safe_divide(
            any_review_count,
            total_engines,
        ),
        "high_error_count": (
            high_error_count
        ),
        "high_error_required_count": (
            required_high_error_count
        ),
        "high_error_required_recall": (
            safe_divide(
                required_high_error_count,
                high_error_count,
            )
        ),
        "high_error_any_review_recall": (
            safe_divide(
                any_review_high_error_count,
                high_error_count,
            )
        ),
        "high_error_required_precision": (
            safe_divide(
                required_high_error_count,
                required_count,
            )
        ),
        "missed_high_error_count": int(
            evaluation_df[
                "missed_high_error"
            ].sum()
        ),
        "dangerous_overprediction_count": (
            dangerous_count
        ),
        "dangerous_required_count": (
            required_dangerous_count
        ),
        "dangerous_required_recall": (
            safe_divide(
                required_dangerous_count,
                dangerous_count,
            )
        ),
        "dangerous_any_review_recall": (
            safe_divide(
                any_review_dangerous_count,
                dangerous_count,
            )
        ),
        "dangerous_required_precision": (
            safe_divide(
                required_dangerous_count,
                required_count,
            )
        ),
        "missed_dangerous_count": int(
            evaluation_df[
                "missed_dangerous_prediction"
            ].sum()
        ),
        "severe_dangerous_count": (
            severe_dangerous_count
        ),
        "severe_dangerous_required_count": (
            required_severe_count
        ),
        "severe_dangerous_required_recall": (
            safe_divide(
                required_severe_count,
                severe_dangerous_count,
            )
        ),
        "severe_dangerous_any_review_recall": (
            safe_divide(
                any_review_severe_count,
                severe_dangerous_count,
            )
        ),
        "missed_severe_dangerous_count": int(
            evaluation_df[
                "missed_severe_dangerous_prediction"
            ].sum()
        ),
        "low_error_required_count": (
            low_error_required_count
        ),
        "low_error_required_rate_of_fleet": (
            safe_divide(
                low_error_required_count,
                total_engines,
            )
        ),
        "low_error_share_of_required_reviews": (
            safe_divide(
                low_error_required_count,
                required_count,
            )
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
        "not_required_high_error_rate": (
            float(
                not_required_df[
                    "high_error"
                ].mean()
            )
            if not not_required_df.empty
            else 0.0
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
    }


def build_review_performance_table(
    evaluation_df,
    version,
):
    """Evaluate each review category."""

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
            }
        )

    return pd.DataFrame(
        rows
    )


def build_transition_table(
    v4_df,
    v5_df,
):
    """Build engine-level V4-to-V5 transitions."""

    v4_columns = [
        "unit",
        "review_flag",
        "engine_condition",
        "risk_score",
        "trajectory_flag",
        "trajectory_trust",
    ]

    v5_columns = [
        "unit",
        "review_flag",
        "engine_condition",
        "condition_severity_score",
        "reliability_risk_score",
        "reliability_risk",
        "operational_priority_score",
        "operational_priority",
        "trajectory_flag",
        "trajectory_trust",
        "mc_deterministic_gap",
        "mc_gap_level",
    ]

    transition_df = (
        v4_df[
            v4_columns
        ]
        .rename(
            columns={
                "review_flag": (
                    "v4_review_flag"
                ),
                "engine_condition": (
                    "v4_engine_condition"
                ),
                "risk_score": (
                    "v4_risk_score"
                ),
                "trajectory_flag": (
                    "v4_trajectory_flag"
                ),
                "trajectory_trust": (
                    "v4_trajectory_trust"
                ),
            }
        )
        .merge(
            v5_df[
                v5_columns
            ].rename(
                columns={
                    "review_flag": (
                        "v5_review_flag"
                    ),
                    "engine_condition": (
                        "v5_engine_condition"
                    ),
                    "trajectory_flag": (
                        "v5_trajectory_flag"
                    ),
                    "trajectory_trust": (
                        "v5_trajectory_trust"
                    ),
                }
            ),
            on="unit",
            how="inner",
            validate="one_to_one",
        )
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

    transition_df = transition_df.merge(
        v5_df[
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

    transition_df[
        "v4_review_priority"
    ] = transition_df[
        "v4_review_flag"
    ].map(
        review_priority
    )

    transition_df[
        "v5_review_priority"
    ] = transition_df[
        "v5_review_flag"
    ].map(
        review_priority
    )

    transition_df[
        "review_change"
    ] = (
        transition_df[
            "v5_review_priority"
        ]
        - transition_df[
            "v4_review_priority"
        ]
    )

    transition_df[
        "transition"
    ] = (
        transition_df[
            "v4_review_flag"
        ]
        + " → "
        + transition_df[
            "v5_review_flag"
        ]
    )

    transition_df[
        "mandatory_review_removed"
    ] = (
        (
            transition_df[
                "v4_review_flag"
            ]
            == "Required"
        )
        & (
            transition_df[
                "v5_review_flag"
            ]
            != "Required"
        )
    )

    transition_df[
        "mandatory_review_added"
    ] = (
        (
            transition_df[
                "v4_review_flag"
            ]
            != "Required"
        )
        & (
            transition_df[
                "v5_review_flag"
            ]
            == "Required"
        )
    )

    return transition_df.sort_values(
        by=[
            "review_change",
            "absolute_error",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )


def build_transition_summary(
    transition_df,
):
    """Summarize V4-to-V5 review transitions."""

    summary_df = (
        transition_df.groupby(
            [
                "v4_review_flag",
                "v5_review_flag",
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
            high_error_count=(
                "high_error",
                "sum",
            ),
            dangerous_overprediction_count=(
                "dangerous_overprediction",
                "sum",
            ),
            severe_dangerous_count=(
                "severe_dangerous_overprediction",
                "sum",
            ),
        )
    )

    summary_df[
        "percentage"
    ] = (
        summary_df[
            "engine_count"
        ]
        / len(
            transition_df
        )
        * 100.0
    )

    return summary_df.sort_values(
        by="engine_count",
        ascending=False,
    ).reset_index(
        drop=True
    )


def build_workload_reduction_analysis(
    transition_df,
):
    """Measure the consequences of reducing mandatory reviews."""

    removed_df = transition_df[
        transition_df[
            "mandatory_review_removed"
        ]
    ].copy()

    added_df = transition_df[
        transition_df[
            "mandatory_review_added"
        ]
    ].copy()

    removed_high_error_df = removed_df[
        removed_df[
            "high_error"
        ]
    ].copy()

    removed_dangerous_df = removed_df[
        removed_df[
            "dangerous_overprediction"
        ]
    ].copy()

    removed_severe_df = removed_df[
        removed_df[
            "severe_dangerous_overprediction"
        ]
    ].copy()

    removed_low_error_df = removed_df[
        removed_df[
            "absolute_error"
        ]
        < 10.0
    ].copy()

    downgraded_to_not_required_df = (
        transition_df[
            (
                transition_df[
                    "v4_review_flag"
                ]
                == "Required"
            )
            & (
                transition_df[
                    "v5_review_flag"
                ]
                == "Not required"
            )
        ].copy()
    )

    rows = [
        {
            "metric": (
                "Mandatory reviews removed"
            ),
            "engine_count": len(
                removed_df
            ),
            "percentage_of_fleet": (
                len(
                    removed_df
                )
                / len(
                    transition_df
                )
                * 100.0
            ),
            "average_absolute_error": (
                float(
                    removed_df[
                        "absolute_error"
                    ].mean()
                )
                if not removed_df.empty
                else 0.0
            ),
        },
        {
            "metric": (
                "Mandatory reviews added"
            ),
            "engine_count": len(
                added_df
            ),
            "percentage_of_fleet": (
                len(
                    added_df
                )
                / len(
                    transition_df
                )
                * 100.0
            ),
            "average_absolute_error": (
                float(
                    added_df[
                        "absolute_error"
                    ].mean()
                )
                if not added_df.empty
                else 0.0
            ),
        },
        {
            "metric": (
                "Low-error mandatory reviews removed"
            ),
            "engine_count": len(
                removed_low_error_df
            ),
            "percentage_of_fleet": (
                len(
                    removed_low_error_df
                )
                / len(
                    transition_df
                )
                * 100.0
            ),
            "average_absolute_error": (
                float(
                    removed_low_error_df[
                        "absolute_error"
                    ].mean()
                )
                if not removed_low_error_df.empty
                else 0.0
            ),
        },
        {
            "metric": (
                "High-error engines removed from "
                "mandatory review"
            ),
            "engine_count": len(
                removed_high_error_df
            ),
            "percentage_of_fleet": (
                len(
                    removed_high_error_df
                )
                / len(
                    transition_df
                )
                * 100.0
            ),
            "average_absolute_error": (
                float(
                    removed_high_error_df[
                        "absolute_error"
                    ].mean()
                )
                if not removed_high_error_df.empty
                else 0.0
            ),
        },
        {
            "metric": (
                "Dangerous overpredictions removed "
                "from mandatory review"
            ),
            "engine_count": len(
                removed_dangerous_df
            ),
            "percentage_of_fleet": (
                len(
                    removed_dangerous_df
                )
                / len(
                    transition_df
                )
                * 100.0
            ),
            "average_absolute_error": (
                float(
                    removed_dangerous_df[
                        "absolute_error"
                    ].mean()
                )
                if not removed_dangerous_df.empty
                else 0.0
            ),
        },
        {
            "metric": (
                "Severe overpredictions removed "
                "from mandatory review"
            ),
            "engine_count": len(
                removed_severe_df
            ),
            "percentage_of_fleet": (
                len(
                    removed_severe_df
                )
                / len(
                    transition_df
                )
                * 100.0
            ),
            "average_absolute_error": (
                float(
                    removed_severe_df[
                        "absolute_error"
                    ].mean()
                )
                if not removed_severe_df.empty
                else 0.0
            ),
        },
        {
            "metric": (
                "Required engines downgraded to "
                "Not required"
            ),
            "engine_count": len(
                downgraded_to_not_required_df
            ),
            "percentage_of_fleet": (
                len(
                    downgraded_to_not_required_df
                )
                / len(
                    transition_df
                )
                * 100.0
            ),
            "average_absolute_error": (
                float(
                    downgraded_to_not_required_df[
                        "absolute_error"
                    ].mean()
                )
                if not downgraded_to_not_required_df.empty
                else 0.0
            ),
        },
    ]

    return (
        pd.DataFrame(
            rows
        ),
        removed_df,
        added_df,
        removed_high_error_df,
        removed_dangerous_df,
        removed_severe_df,
        downgraded_to_not_required_df,
    )


def calculate_score_correlations(
    v5_df,
):
    """Evaluate whether V5 reliability risk tracks error."""

    reliability_scores = v5_df[
        "reliability_risk_score"
    ]

    absolute_errors = v5_df[
        "absolute_error"
    ]

    prediction_errors = v5_df[
        "prediction_error"
    ]

    pearson_absolute_error = (
        reliability_scores.corr(
            absolute_errors
        )
    )

    spearman_absolute_error = (
        reliability_scores.rank().corr(
            absolute_errors.rank()
        )
    )

    pearson_prediction_error = (
        reliability_scores.corr(
            prediction_errors
        )
    )

    spearman_prediction_error = (
        reliability_scores.rank().corr(
            prediction_errors.rank()
        )
    )

    return pd.DataFrame(
        [
            {
                "metric": (
                    "Pearson reliability risk "
                    "vs absolute error"
                ),
                "value": float(
                    pearson_absolute_error
                ),
            },
            {
                "metric": (
                    "Spearman reliability risk "
                    "vs absolute error"
                ),
                "value": float(
                    spearman_absolute_error
                ),
            },
            {
                "metric": (
                    "Pearson reliability risk "
                    "vs signed prediction error"
                ),
                "value": float(
                    pearson_prediction_error
                ),
            },
            {
                "metric": (
                    "Spearman reliability risk "
                    "vs signed prediction error"
                ),
                "value": float(
                    spearman_prediction_error
                ),
            },
        ]
    )


def build_reliability_ranking_curve(
    v5_df,
):
    """
    Evaluate error capture when reviewing the highest-risk engines.

    This is descriptive only. It does not change any thresholds.
    """

    total_high_error = int(
        v5_df[
            "high_error"
        ].sum()
    )

    total_dangerous = int(
        v5_df[
            "dangerous_overprediction"
        ].sum()
    )

    total_severe = int(
        v5_df[
            "severe_dangerous_overprediction"
        ].sum()
    )

    sorted_df = v5_df.sort_values(
        by="reliability_risk_score",
        ascending=False,
    ).reset_index(
        drop=True
    )

    rows = []

    for review_percentage in [
        10,
        20,
        30,
        40,
        50,
        60,
        70,
        80,
        90,
        100,
    ]:
        reviewed_count = max(
            1,
            int(
                np.ceil(
                    len(
                        sorted_df
                    )
                    * review_percentage
                    / 100.0
                )
            ),
        )

        reviewed_df = sorted_df.head(
            reviewed_count
        )

        rows.append(
            {
                "review_percentage": (
                    float(
                        review_percentage
                    )
                ),
                "reviewed_engines": (
                    reviewed_count
                ),
                "average_absolute_error": float(
                    reviewed_df[
                        "absolute_error"
                    ].mean()
                ),
                "high_error_recall": safe_divide(
                    int(
                        reviewed_df[
                            "high_error"
                        ].sum()
                    ),
                    total_high_error,
                ),
                "dangerous_overprediction_recall": (
                    safe_divide(
                        int(
                            reviewed_df[
                                "dangerous_overprediction"
                            ].sum()
                        ),
                        total_dangerous,
                    )
                ),
                "severe_dangerous_recall": (
                    safe_divide(
                        int(
                            reviewed_df[
                                "severe_dangerous_overprediction"
                            ].sum()
                        ),
                        total_severe,
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_operational_priority_table(
    v5_df,
):
    """Evaluate V5 operational-priority categories."""

    rows = []

    for priority in OPERATIONAL_PRIORITIES:
        group_df = v5_df[
            v5_df[
                "operational_priority"
            ]
            == priority
        ]

        if group_df.empty:
            continue

        rows.append(
            {
                "operational_priority": priority,
                "engine_count": len(
                    group_df
                ),
                "percentage": (
                    len(
                        group_df
                    )
                    / len(
                        v5_df
                    )
                    * 100.0
                ),
                "average_actual_rul": float(
                    group_df[
                        "actual_rul"
                    ].mean()
                ),
                "average_conservative_rul": (
                    float(
                        group_df[
                            "conservative_rul"
                        ].mean()
                    )
                    if (
                        "conservative_rul"
                        in group_df.columns
                    )
                    else np.nan
                ),
                "average_absolute_error": float(
                    group_df[
                        "absolute_error"
                    ].mean()
                ),
                "dangerous_overprediction_rate": float(
                    group_df[
                        "dangerous_overprediction"
                    ].mean()
                ),
                "average_condition_severity_score": float(
                    group_df[
                        "condition_severity_score"
                    ].mean()
                ),
                "average_reliability_risk_score": float(
                    group_df[
                        "reliability_risk_score"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def plot_review_workload(
    validation_metrics_df,
    model_name,
    dataset,
):
    """Plot V4 and V5 review workload."""

    positions = np.arange(
        len(
            validation_metrics_df
        )
    )

    width = 0.35

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        positions - width / 2,
        validation_metrics_df[
            "required_review_rate"
        ]
        * 100.0,
        width=width,
        label="Required review",
    )

    plt.bar(
        positions + width / 2,
        validation_metrics_df[
            "any_review_rate"
        ]
        * 100.0,
        width=width,
        label="Any review",
    )

    plt.xticks(
        positions,
        validation_metrics_df[
            "version"
        ],
    )

    plt.xlabel(
        "Decision-layer version"
    )

    plt.ylabel(
        "Percentage of test engines"
    )

    plt.title(
        "V4 vs V5 Engineering Review Workload — "
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
        f"review_workload_v4_v5_"
        f"{model_name}_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_error_capture(
    validation_metrics_df,
    model_name,
    dataset,
):
    """Plot mandatory-review error capture."""

    positions = np.arange(
        len(
            validation_metrics_df
        )
    )

    width = 0.25

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        positions - width,
        validation_metrics_df[
            "high_error_required_recall"
        ]
        * 100.0,
        width=width,
        label="High-error recall",
    )

    plt.bar(
        positions,
        validation_metrics_df[
            "dangerous_required_recall"
        ]
        * 100.0,
        width=width,
        label="Dangerous-overprediction recall",
    )

    plt.bar(
        positions + width,
        validation_metrics_df[
            "severe_dangerous_required_recall"
        ]
        * 100.0,
        width=width,
        label="Severe-overprediction recall",
    )

    plt.xticks(
        positions,
        validation_metrics_df[
            "version"
        ],
    )

    plt.xlabel(
        "Decision-layer version"
    )

    plt.ylabel(
        "Captured by mandatory review (%)"
    )

    plt.title(
        "V4 vs V5 Mandatory Review Error Capture — "
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
        f"error_capture_v4_v5_"
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
    """Plot V4-to-V5 review transitions."""

    transition_matrix = pd.crosstab(
        transition_df[
            "v4_review_flag"
        ],
        transition_df[
            "v5_review_flag"
        ],
    ).reindex(
        index=REVIEW_LEVELS,
        columns=REVIEW_LEVELS,
        fill_value=0,
    )

    matrix_values = (
        transition_matrix.to_numpy()
    )

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
        "V5 review decision"
    )

    plt.ylabel(
        "V4 review decision"
    )

    plt.title(
        "V4-to-V5 Review Transitions — "
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
        f"review_transitions_v4_v5_"
        f"{model_name}_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_reliability_ranking(
    ranking_df,
    model_name,
    dataset,
):
    """Plot error capture from reliability-risk ranking."""

    plt.figure(
        figsize=(10, 6)
    )

    plt.plot(
        ranking_df[
            "review_percentage"
        ],
        ranking_df[
            "high_error_recall"
        ]
        * 100.0,
        marker="o",
        label="High-error recall",
    )

    plt.plot(
        ranking_df[
            "review_percentage"
        ],
        ranking_df[
            "dangerous_overprediction_recall"
        ]
        * 100.0,
        marker="o",
        label="Dangerous-overprediction recall",
    )

    plt.plot(
        ranking_df[
            "review_percentage"
        ],
        ranking_df[
            "severe_dangerous_recall"
        ]
        * 100.0,
        marker="o",
        label="Severe-overprediction recall",
    )

    plt.xlabel(
        "Highest reliability-risk engines reviewed (%)"
    )

    plt.ylabel(
        "Error cases captured (%)"
    )

    plt.title(
        "Reliability-Risk Ranking Performance — "
        f"{model_name.upper()} {dataset}"
    )

    plt.legend()

    plt.grid(
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        VALIDATION_DIR,
        f"reliability_ranking_"
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
    """Print V4-to-V5 changes for selected engines."""

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
        "v4_review_flag",
        "v5_review_flag",
        "v5_trajectory_flag",
        "reliability_risk_score",
        "reliability_risk",
        "operational_priority",
        "mc_deterministic_gap",
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
    """Validate V5 against V4 without changing either system."""

    v4_df, v4_path = load_decision_results(
        version="v4",
        model_name=model_name,
        dataset=dataset,
    )

    v5_df, v5_path = load_decision_results(
        version="v5",
        model_name=model_name,
        dataset=dataset,
    )

    if set(
        v4_df[
            "unit"
        ]
    ) != set(
        v5_df[
            "unit"
        ]
    ):
        raise ValueError(
            "V4 and V5 do not contain the same test engines."
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

    v5_evaluation_df = add_evaluation_labels(
        decision_df=v5_df,
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
                    v4_evaluation_df
                ),
                version="v4",
            ),
            calculate_validation_metrics(
                evaluation_df=(
                    v5_evaluation_df
                ),
                version="v5",
            ),
        ]
    )

    review_performance_df = pd.concat(
        [
            build_review_performance_table(
                evaluation_df=(
                    v4_evaluation_df
                ),
                version="v4",
            ),
            build_review_performance_table(
                evaluation_df=(
                    v5_evaluation_df
                ),
                version="v5",
            ),
        ],
        ignore_index=True,
    )

    transition_df = build_transition_table(
        v4_df=v4_evaluation_df,
        v5_df=v5_evaluation_df,
    )

    transition_summary_df = (
        build_transition_summary(
            transition_df
        )
    )

    (
        workload_reduction_df,
        removed_df,
        added_df,
        removed_high_error_df,
        removed_dangerous_df,
        removed_severe_df,
        downgraded_to_not_required_df,
    ) = build_workload_reduction_analysis(
        transition_df
    )

    score_correlations_df = (
        calculate_score_correlations(
            v5_evaluation_df
        )
    )

    reliability_ranking_df = (
        build_reliability_ranking_curve(
            v5_evaluation_df
        )
    )

    operational_priority_df = (
        build_operational_priority_table(
            v5_evaluation_df
        )
    )

    validation_metrics_path = os.path.join(
        VALIDATION_DIR,
        f"validation_metrics_v4_v5_"
        f"{model_name}_{dataset}.csv",
    )

    review_performance_path = os.path.join(
        VALIDATION_DIR,
        f"review_performance_v4_v5_"
        f"{model_name}_{dataset}.csv",
    )

    transition_path = os.path.join(
        VALIDATION_DIR,
        f"engine_transitions_v4_v5_"
        f"{model_name}_{dataset}.csv",
    )

    transition_summary_path = os.path.join(
        VALIDATION_DIR,
        f"transition_summary_v4_v5_"
        f"{model_name}_{dataset}.csv",
    )

    workload_reduction_path = os.path.join(
        VALIDATION_DIR,
        f"workload_reduction_"
        f"{model_name}_{dataset}.csv",
    )

    removed_path = os.path.join(
        VALIDATION_DIR,
        f"mandatory_reviews_removed_"
        f"{model_name}_{dataset}.csv",
    )

    added_path = os.path.join(
        VALIDATION_DIR,
        f"mandatory_reviews_added_"
        f"{model_name}_{dataset}.csv",
    )

    removed_high_error_path = os.path.join(
        VALIDATION_DIR,
        f"removed_high_error_engines_"
        f"{model_name}_{dataset}.csv",
    )

    removed_dangerous_path = os.path.join(
        VALIDATION_DIR,
        f"removed_dangerous_engines_"
        f"{model_name}_{dataset}.csv",
    )

    removed_severe_path = os.path.join(
        VALIDATION_DIR,
        f"removed_severe_engines_"
        f"{model_name}_{dataset}.csv",
    )

    downgraded_path = os.path.join(
        VALIDATION_DIR,
        f"downgraded_to_not_required_"
        f"{model_name}_{dataset}.csv",
    )

    score_correlations_path = os.path.join(
        VALIDATION_DIR,
        f"score_correlations_"
        f"{model_name}_{dataset}.csv",
    )

    reliability_ranking_path = os.path.join(
        VALIDATION_DIR,
        f"reliability_ranking_"
        f"{model_name}_{dataset}.csv",
    )

    operational_priority_path = os.path.join(
        VALIDATION_DIR,
        f"operational_priority_validation_"
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

    transition_df.to_csv(
        transition_path,
        index=False,
    )

    transition_summary_df.to_csv(
        transition_summary_path,
        index=False,
    )

    workload_reduction_df.to_csv(
        workload_reduction_path,
        index=False,
    )

    removed_df.to_csv(
        removed_path,
        index=False,
    )

    added_df.to_csv(
        added_path,
        index=False,
    )

    removed_high_error_df.to_csv(
        removed_high_error_path,
        index=False,
    )

    removed_dangerous_df.to_csv(
        removed_dangerous_path,
        index=False,
    )

    removed_severe_df.to_csv(
        removed_severe_path,
        index=False,
    )

    downgraded_to_not_required_df.to_csv(
        downgraded_path,
        index=False,
    )

    score_correlations_df.to_csv(
        score_correlations_path,
        index=False,
    )

    reliability_ranking_df.to_csv(
        reliability_ranking_path,
        index=False,
    )

    operational_priority_df.to_csv(
        operational_priority_path,
        index=False,
    )

    workload_plot_path = plot_review_workload(
        validation_metrics_df=(
            validation_metrics_df
        ),
        model_name=model_name,
        dataset=dataset,
    )

    capture_plot_path = plot_error_capture(
        validation_metrics_df=(
            validation_metrics_df
        ),
        model_name=model_name,
        dataset=dataset,
    )

    transition_plot_path = (
        plot_transition_matrix(
            transition_df=transition_df,
            model_name=model_name,
            dataset=dataset,
        )
    )

    ranking_plot_path = (
        plot_reliability_ranking(
            ranking_df=(
                reliability_ranking_df
            ),
            model_name=model_name,
            dataset=dataset,
        )
    )

    print(
        f"V4 source: {v4_path}"
    )

    print(
        f"V5 source: {v5_path}"
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
        "\nV4 versus V5 validation metrics"
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
        "required_average_absolute_error",
        "not_required_average_absolute_error",
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
        "\nV4-to-V5 review transition summary"
    )

    print(
        transition_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nMandatory-review workload reduction"
    )

    print(
        workload_reduction_df.to_string(
            index=False
        )
    )

    print(
        "\nV5 reliability-score correlations"
    )

    print(
        score_correlations_df.to_string(
            index=False
        )
    )

    print(
        "\nReliability-risk ranking performance"
    )

    print(
        reliability_ranking_df.to_string(
            index=False
        )
    )

    print(
        "\nOperational-priority validation"
    )

    print(
        operational_priority_df.to_string(
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
        f"Validation metrics:   "
        f"{validation_metrics_path}"
    )

    print(
        f"Review performance:   "
        f"{review_performance_path}"
    )

    print(
        f"Engine transitions:   "
        f"{transition_path}"
    )

    print(
        f"Transition summary:   "
        f"{transition_summary_path}"
    )

    print(
        f"Workload reduction:   "
        f"{workload_reduction_path}"
    )

    print(
        f"Removed reviews:      "
        f"{removed_path}"
    )

    print(
        f"Added reviews:        "
        f"{added_path}"
    )

    print(
        f"Removed high-error:   "
        f"{removed_high_error_path}"
    )

    print(
        f"Removed dangerous:    "
        f"{removed_dangerous_path}"
    )

    print(
        f"Removed severe:       "
        f"{removed_severe_path}"
    )

    print(
        f"Downgraded engines:   "
        f"{downgraded_path}"
    )

    print(
        f"Score correlations:   "
        f"{score_correlations_path}"
    )

    print(
        f"Reliability ranking:  "
        f"{reliability_ranking_path}"
    )

    print(
        f"Priority validation:  "
        f"{operational_priority_path}"
    )

    print(
        f"Workload plot:        "
        f"{workload_plot_path}"
    )

    print(
        f"Capture plot:         "
        f"{capture_plot_path}"
    )

    print(
        f"Transition plot:      "
        f"{transition_plot_path}"
    )

    print(
        f"Ranking plot:         "
        f"{ranking_plot_path}"
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