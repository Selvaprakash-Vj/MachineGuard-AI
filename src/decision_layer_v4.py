import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


RESULTS_DIR = "results_v2"

UNCERTAINTY_DIR = os.path.join(
    RESULTS_DIR,
    "uncertainty",
)

TRAJECTORY_DIR = os.path.join(
    RESULTS_DIR,
    "trajectory_diagnostics",
)

DECISION_DIR = os.path.join(
    RESULTS_DIR,
    "decisions_v4",
)

RUL_CAP = 125.0

os.makedirs(
    DECISION_DIR,
    exist_ok=True,
)


def load_uncertainty_results(
    model_name,
    dataset,
):
    """Load calibrated Monte Carlo dropout results."""

    uncertainty_path = os.path.join(
        UNCERTAINTY_DIR,
        f"uncertainty_results_{model_name}_{dataset}.csv",
    )

    if not os.path.exists(
        uncertainty_path
    ):
        raise FileNotFoundError(
            "Uncertainty results not found: "
            f"{uncertainty_path}\n"
            "Run uncertainty_v2.py first."
        )

    uncertainty_df = pd.read_csv(
        uncertainty_path
    )

    required_columns = [
        "unit",
        "actual_rul",
        "predicted_rul_mean",
        "mc_standard_deviation",
        "calibrated_lower_bound",
        "calibrated_upper_bound",
        "interval_width",
        "absolute_error",
        "prediction_error",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column
        not in uncertainty_df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Uncertainty results are missing columns: "
            f"{missing_columns}"
        )

    uncertainty_df[
        "unit"
    ] = uncertainty_df[
        "unit"
    ].astype(int)

    return (
        uncertainty_df,
        uncertainty_path,
    )


def load_ridge_predictions(
    dataset,
):
    """Load independent Ridge baseline predictions."""

    ridge_path = os.path.join(
        RESULTS_DIR,
        f"predictions_baseline_ridge_{dataset}.csv",
    )

    if not os.path.exists(
        ridge_path
    ):
        raise FileNotFoundError(
            "Ridge predictions not found: "
            f"{ridge_path}\n"
            "Run baseline_v2.py first."
        )

    ridge_df = pd.read_csv(
        ridge_path
    )

    required_columns = [
        "unit",
        "predicted_rul",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column
        not in ridge_df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Ridge predictions are missing columns: "
            f"{missing_columns}"
        )

    ridge_df = ridge_df[
        required_columns
    ].rename(
        columns={
            "predicted_rul": (
                "ridge_prediction"
            ),
        }
    )

    ridge_df[
        "unit"
    ] = ridge_df[
        "unit"
    ].astype(int)

    ridge_df[
        "ridge_prediction"
    ] = np.clip(
        ridge_df[
            "ridge_prediction"
        ],
        0.0,
        RUL_CAP,
    )

    return ridge_df, ridge_path


def load_trajectory_results(
    model_name,
    dataset,
):
    """Load full-history temporal diagnostics."""

    trajectory_path = os.path.join(
        TRAJECTORY_DIR,
        f"trajectory_summary_{model_name}_{dataset}.csv",
    )

    if not os.path.exists(
        trajectory_path
    ):
        raise FileNotFoundError(
            "Trajectory summary not found: "
            f"{trajectory_path}\n"
            "Run trajectory_diagnostics_v2.py first."
        )

    trajectory_df = pd.read_csv(
        trajectory_path
    )

    required_columns = [
        "unit",
        "final_predicted_rul",
        "recent_slope",
        "recent_predicted_drop",
        "recent_decline_ratio",
        "recent_prediction_range",
        "recent_residual_std",
        "monotonicity_violation_rate",
        "large_jump_count",
        "largest_single_cycle_jump",
        "trajectory_flag",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column
        not in trajectory_df.columns
    ]

    if missing_columns:
        raise KeyError(
            "Trajectory summary is missing columns: "
            f"{missing_columns}"
        )

    trajectory_df[
        "unit"
    ] = trajectory_df[
        "unit"
    ].astype(int)

    selected_columns = [
        "unit",
        "final_predicted_rul",
        "recent_slope",
        "recent_predicted_drop",
        "recent_decline_ratio",
        "recent_prediction_range",
        "recent_residual_std",
        "monotonicity_violation_rate",
        "large_jump_count",
        "largest_single_cycle_jump",
        "trajectory_flag",
    ]

    return (
        trajectory_df[
            selected_columns
        ],
        trajectory_path,
    )


def classify_disagreement(
    disagreement,
):
    """Classify disagreement between GRU and Ridge."""

    if disagreement <= 10:
        return "Low"

    if disagreement <= 20:
        return "Medium"

    return "High"


def calculate_conservative_rul(
    calibrated_lower_bound,
    ridge_prediction,
):
    """
    Calculate a cautious RUL estimate.

    The lower value is selected from:
    - calibrated neural-network lower bound;
    - independent Ridge prediction.
    """

    return float(
        np.clip(
            min(
                calibrated_lower_bound,
                ridge_prediction,
            ),
            0.0,
            RUL_CAP,
        )
    )


def classify_engine_condition(
    conservative_rul,
):
    """Classify estimated engine condition."""

    if conservative_rul <= 20:
        return "Critical"

    if conservative_rul <= 50:
        return "Warning"

    if conservative_rul <= 80:
        return "Monitor"

    return "Healthy"


def classify_prediction_trust(
    interval_width,
    mc_standard_deviation,
    model_disagreement,
):
    """Classify static prediction trust."""

    if (
        interval_width <= 25
        and mc_standard_deviation <= 7.5
        and model_disagreement <= 10
    ):
        return "High"

    if (
        interval_width <= 45
        and mc_standard_deviation <= 15
        and model_disagreement <= 20
    ):
        return "Medium"

    return "Low"


def classify_trajectory_trust(
    trajectory_flag,
):
    """Convert temporal diagnostics into trajectory trust."""

    high_trust_flags = {
        "Consistent degradation",
    }

    medium_trust_flags = {
        "Rapid degradation signal",
        "Mixed trajectory",
    }

    if trajectory_flag in high_trust_flags:
        return "High"

    if trajectory_flag in medium_trust_flags:
        return "Medium"

    return "Low"


def assign_review_flag(
    engine_condition,
    prediction_trust,
    trajectory_trust,
    trajectory_flag,
    disagreement_level,
    interval_width,
):
    """
    Decide whether manual engineering review is needed.

    Static uncertainty and temporal behaviour are evaluated
    independently.
    """

    required_trajectory_flags = {
        "High-RUL plateau — review",
        "Unstable trajectory",
        "RUL increasing despite ageing",
    }

    recommended_trajectory_flags = {
        "Rapid degradation signal",
        "Mixed trajectory",
        "Weak degradation / plateau",
    }

    if (
        trajectory_flag
        in required_trajectory_flags
    ):
        return "Required"

    if disagreement_level == "High":
        return "Required"

    if interval_width > 50:
        return "Required"

    if (
        prediction_trust == "Low"
        and engine_condition
        in {
            "Critical",
            "Warning",
            "Monitor",
        }
    ):
        return "Required"

    if (
        trajectory_flag
        in recommended_trajectory_flags
    ):
        return "Recommended"

    if trajectory_trust == "Low":
        return "Recommended"

    if prediction_trust == "Low":
        return "Recommended"

    if disagreement_level == "Medium":
        return "Recommended"

    if interval_width > 35:
        return "Recommended"

    return "Not required"


def calculate_risk_score(
    conservative_rul,
    prediction_trust,
    trajectory_trust,
    trajectory_flag,
    disagreement_level,
    interval_width,
):
    """
    Calculate a transparent risk score from 0 to 100.

    Actual RUL is never used in score assignment.
    """

    condition_component = (
        100.0
        * (
            1.0
            - min(
                conservative_rul,
                RUL_CAP,
            )
            / RUL_CAP
        )
    )

    uncertainty_penalty = {
        "High": 0.0,
        "Medium": 7.0,
        "Low": 15.0,
    }[
        prediction_trust
    ]

    disagreement_penalty = {
        "Low": 0.0,
        "Medium": 7.0,
        "High": 15.0,
    }[
        disagreement_level
    ]

    interval_penalty = 0.0

    if interval_width > 50:
        interval_penalty = 10.0

    elif interval_width > 35:
        interval_penalty = 5.0

    trajectory_trust_penalty = {
        "High": 0.0,
        "Medium": 6.0,
        "Low": 12.0,
    }[
        trajectory_trust
    ]

    trajectory_flag_penalty = {
        "Consistent degradation": 0.0,
        "Rapid degradation signal": 5.0,
        "Mixed trajectory": 5.0,
        "Weak degradation / plateau": 8.0,
        "High-RUL plateau — review": 15.0,
        "RUL increasing despite ageing": 15.0,
        "Unstable trajectory": 15.0,
    }.get(
        trajectory_flag,
        5.0,
    )

    score = (
        condition_component
        + uncertainty_penalty
        + disagreement_penalty
        + interval_penalty
        + trajectory_trust_penalty
        + trajectory_flag_penalty
    )

    return float(
        np.clip(
            score,
            0.0,
            100.0,
        )
    )


def build_trajectory_message(
    trajectory_flag,
):
    """Translate the temporal flag into an operational explanation."""

    messages = {
        "Consistent degradation": (
            "The recent RUL trajectory shows a broadly "
            "consistent degradation pattern."
        ),
        "Rapid degradation signal": (
            "The predicted RUL is declining rapidly and "
            "should be verified against recent sensor trends."
        ),
        "Mixed trajectory": (
            "The temporal prediction pattern is mixed and "
            "does not provide a stable degradation trend."
        ),
        "Weak degradation / plateau": (
            "The predicted RUL shows only weak degradation "
            "despite continued engine ageing."
        ),
        "High-RUL plateau — review": (
            "The model remains on a high-RUL plateau despite "
            "continued ageing, indicating a possible shared "
            "model blind spot."
        ),
        "RUL increasing despite ageing": (
            "The predicted RUL is increasing despite additional "
            "engine cycles, indicating temporal inconsistency."
        ),
        "Unstable trajectory": (
            "The predicted RUL trajectory contains large jumps "
            "or excessive oscillations and should not be relied "
            "upon without engineering review."
        ),
    }

    return messages.get(
        trajectory_flag,
        "The temporal trajectory requires additional review.",
    )


def assign_operational_action(
    engine_condition,
    review_flag,
    trajectory_flag,
):
    """Generate a condition-aware and trajectory-aware action."""

    base_actions = {
        "Critical": (
            "Inspect immediately and restrict continued "
            "operation until the engine condition is verified."
        ),
        "Warning": (
            "Increase inspection frequency and prepare an "
            "early maintenance intervention."
        ),
        "Monitor": (
            "Continue operation with enhanced monitoring and "
            "repeat the RUL assessment after additional cycles."
        ),
        "Healthy": (
            "Continue routine operation and condition monitoring."
        ),
    }

    action = base_actions[
        engine_condition
    ]

    action += " " + build_trajectory_message(
        trajectory_flag
    )

    if review_flag == "Required":
        action += (
            " Manual engineering review is required before "
            "using the predicted RUL for an operational or "
            "maintenance decision."
        )

    elif review_flag == "Recommended":
        action += (
            " Engineering review is recommended before making "
            "a high-consequence decision."
        )

    return action


def build_decision_results(
    uncertainty_df,
    ridge_df,
    trajectory_df,
):
    """Combine prediction, uncertainty, disagreement and trajectory."""

    decision_df = uncertainty_df.merge(
        ridge_df,
        on="unit",
        how="inner",
        validate="one_to_one",
    )

    decision_df = decision_df.merge(
        trajectory_df,
        on="unit",
        how="inner",
        validate="one_to_one",
    )

    if len(
        decision_df
    ) != len(
        uncertainty_df
    ):
        raise ValueError(
            "Not every uncertainty result matched a Ridge "
            "prediction and trajectory result."
        )

    decision_df[
        "model_disagreement"
    ] = np.abs(
        decision_df[
            "predicted_rul_mean"
        ]
        - decision_df[
            "ridge_prediction"
        ]
    )

    decision_df[
        "disagreement_level"
    ] = decision_df[
        "model_disagreement"
    ].apply(
        classify_disagreement
    )

    decision_df[
        "conservative_rul"
    ] = decision_df.apply(
        lambda row: calculate_conservative_rul(
            calibrated_lower_bound=row[
                "calibrated_lower_bound"
            ],
            ridge_prediction=row[
                "ridge_prediction"
            ],
        ),
        axis=1,
    )

    decision_df[
        "engine_condition"
    ] = decision_df[
        "conservative_rul"
    ].apply(
        classify_engine_condition
    )

    decision_df[
        "prediction_trust"
    ] = decision_df.apply(
        lambda row: classify_prediction_trust(
            interval_width=row[
                "interval_width"
            ],
            mc_standard_deviation=row[
                "mc_standard_deviation"
            ],
            model_disagreement=row[
                "model_disagreement"
            ],
        ),
        axis=1,
    )

    decision_df[
        "trajectory_trust"
    ] = decision_df[
        "trajectory_flag"
    ].apply(
        classify_trajectory_trust
    )

    decision_df[
        "review_flag"
    ] = decision_df.apply(
        lambda row: assign_review_flag(
            engine_condition=row[
                "engine_condition"
            ],
            prediction_trust=row[
                "prediction_trust"
            ],
            trajectory_trust=row[
                "trajectory_trust"
            ],
            trajectory_flag=row[
                "trajectory_flag"
            ],
            disagreement_level=row[
                "disagreement_level"
            ],
            interval_width=row[
                "interval_width"
            ],
        ),
        axis=1,
    )

    decision_df[
        "risk_score"
    ] = decision_df.apply(
        lambda row: calculate_risk_score(
            conservative_rul=row[
                "conservative_rul"
            ],
            prediction_trust=row[
                "prediction_trust"
            ],
            trajectory_trust=row[
                "trajectory_trust"
            ],
            trajectory_flag=row[
                "trajectory_flag"
            ],
            disagreement_level=row[
                "disagreement_level"
            ],
            interval_width=row[
                "interval_width"
            ],
        ),
        axis=1,
    )

    decision_df[
        "operational_action"
    ] = decision_df.apply(
        lambda row: assign_operational_action(
            engine_condition=row[
                "engine_condition"
            ],
            review_flag=row[
                "review_flag"
            ],
            trajectory_flag=row[
                "trajectory_flag"
            ],
        ),
        axis=1,
    )

    condition_priority = {
        "Critical": 1,
        "Warning": 2,
        "Monitor": 3,
        "Healthy": 4,
    }

    review_priority = {
        "Required": 1,
        "Recommended": 2,
        "Not required": 3,
    }

    decision_df[
        "condition_priority"
    ] = decision_df[
        "engine_condition"
    ].map(
        condition_priority
    )

    decision_df[
        "review_priority"
    ] = decision_df[
        "review_flag"
    ].map(
        review_priority
    )

    decision_df = decision_df.sort_values(
        by=[
            "condition_priority",
            "review_priority",
            "risk_score",
        ],
        ascending=[
            True,
            True,
            False,
        ],
    ).reset_index(
        drop=True
    )

    return decision_df


def build_condition_summary(
    decision_df,
):
    """Summarize condition categories."""

    rows = []

    for condition in [
        "Critical",
        "Warning",
        "Monitor",
        "Healthy",
    ]:
        group_df = decision_df[
            decision_df[
                "engine_condition"
            ]
            == condition
        ]

        if group_df.empty:
            continue

        rows.append(
            {
                "engine_condition": condition,
                "engine_count": len(
                    group_df
                ),
                "percentage": (
                    len(
                        group_df
                    )
                    / len(
                        decision_df
                    )
                    * 100
                ),
                "average_predicted_rul": float(
                    group_df[
                        "predicted_rul_mean"
                    ].mean()
                ),
                "average_conservative_rul": float(
                    group_df[
                        "conservative_rul"
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


def build_trust_summary(
    decision_df,
):
    """Summarize static and temporal trust."""

    rows = []

    for prediction_trust in [
        "High",
        "Medium",
        "Low",
    ]:
        group_df = decision_df[
            decision_df[
                "prediction_trust"
            ]
            == prediction_trust
        ]

        if group_df.empty:
            continue

        rows.append(
            {
                "prediction_trust": prediction_trust,
                "engine_count": len(
                    group_df
                ),
                "percentage": (
                    len(
                        group_df
                    )
                    / len(
                        decision_df
                    )
                    * 100
                ),
                "average_interval_width": float(
                    group_df[
                        "interval_width"
                    ].mean()
                ),
                "average_absolute_error": float(
                    group_df[
                        "absolute_error"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_trajectory_summary(
    decision_df,
):
    """Summarize temporal diagnostic categories."""

    trajectory_summary_df = (
        decision_df.groupby(
            [
                "trajectory_flag",
                "trajectory_trust",
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
            average_recent_slope=(
                "recent_slope",
                "mean",
            ),
            average_risk_score=(
                "risk_score",
                "mean",
            ),
        )
    )

    trajectory_summary_df[
        "percentage"
    ] = (
        trajectory_summary_df[
            "engine_count"
        ]
        / len(
            decision_df
        )
        * 100
    )

    return trajectory_summary_df.sort_values(
        by="engine_count",
        ascending=False,
    ).reset_index(
        drop=True
    )


def build_review_summary(
    decision_df,
):
    """Evaluate review categories retrospectively."""

    rows = []

    for review_flag in [
        "Required",
        "Recommended",
        "Not required",
    ]:
        group_df = decision_df[
            decision_df[
                "review_flag"
            ]
            == review_flag
        ]

        if group_df.empty:
            continue

        rows.append(
            {
                "review_flag": review_flag,
                "engine_count": len(
                    group_df
                ),
                "percentage": (
                    len(
                        group_df
                    )
                    / len(
                        decision_df
                    )
                    * 100
                ),
                "average_absolute_error": float(
                    group_df[
                        "absolute_error"
                    ].mean()
                ),
                "dangerous_overprediction_rate": float(
                    (
                        group_df[
                            "prediction_error"
                        ]
                        > 10
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def evaluate_condition_categories(
    decision_df,
):
    """
    Retrospectively evaluate condition categories.

    Actual RUL is not used when assigning decisions.
    """

    rows = []

    for (
        condition,
        group_df,
    ) in decision_df.groupby(
        "engine_condition"
    ):
        rows.append(
            {
                "engine_condition": condition,
                "engine_count": len(
                    group_df
                ),
                "average_actual_rul": float(
                    group_df[
                        "actual_rul"
                    ].mean()
                ),
                "average_absolute_error": float(
                    group_df[
                        "absolute_error"
                    ].mean()
                ),
                "actual_rul_below_20_rate": float(
                    (
                        group_df[
                            "actual_rul"
                        ]
                        <= 20
                    ).mean()
                ),
                "actual_rul_below_50_rate": float(
                    (
                        group_df[
                            "actual_rul"
                        ]
                        <= 50
                    ).mean()
                ),
                "dangerous_overprediction_rate": float(
                    (
                        group_df[
                            "prediction_error"
                        ]
                        > 10
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def plot_condition_distribution(
    condition_summary_df,
    model_name,
    dataset,
):
    """Plot engine condition distribution."""

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        condition_summary_df[
            "engine_condition"
        ],
        condition_summary_df[
            "engine_count"
        ],
    )

    plt.xlabel(
        "Estimated engine condition"
    )

    plt.ylabel(
        "Number of test engines"
    )

    plt.title(
        "Integrated Engine Condition Distribution — "
        f"{model_name.upper()} {dataset}"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        DECISION_DIR,
        f"condition_distribution_v4_"
        f"{model_name}_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_review_distribution(
    review_summary_df,
    model_name,
    dataset,
):
    """Plot final engineering review categories."""

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        review_summary_df[
            "review_flag"
        ],
        review_summary_df[
            "engine_count"
        ],
    )

    plt.xlabel(
        "Engineering review"
    )

    plt.ylabel(
        "Number of test engines"
    )

    plt.title(
        "Integrated Review Decisions — "
        f"{model_name.upper()} {dataset}"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        DECISION_DIR,
        f"review_distribution_v4_"
        f"{model_name}_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_selected_units(
    decision_df,
    selected_units,
    model_name,
    dataset,
):
    """Plot evidence for selected engines."""

    if selected_units:
        plot_df = decision_df[
            decision_df[
                "unit"
            ].isin(
                selected_units
            )
        ].copy()

    else:
        plot_df = decision_df.nlargest(
            10,
            "risk_score",
        ).copy()

    if plot_df.empty:
        raise ValueError(
            "No matching engines found."
        )

    plot_df = plot_df.sort_values(
        by="unit"
    ).reset_index(
        drop=True
    )

    positions = np.arange(
        len(
            plot_df
        )
    )

    width = 0.25

    plt.figure(
        figsize=(12, 7)
    )

    plt.bar(
        positions - width,
        plot_df[
            "predicted_rul_mean"
        ],
        width=width,
        label=(
            f"{model_name.upper()} "
            "uncertainty mean"
        ),
    )

    plt.bar(
        positions,
        plot_df[
            "ridge_prediction"
        ],
        width=width,
        label="Ridge prediction",
    )

    plt.bar(
        positions + width,
        plot_df[
            "conservative_rul"
        ],
        width=width,
        label="Conservative RUL",
    )

    plt.scatter(
        positions,
        plot_df[
            "actual_rul"
        ],
        marker="x",
        s=100,
        label="Actual RUL",
    )

    plt.xticks(
        positions,
        [
            f"Unit {unit}"
            for unit in plot_df[
                "unit"
            ]
        ],
    )

    plt.xlabel(
        "Test engine"
    )

    plt.ylabel(
        "Remaining Useful Life"
    )

    plt.title(
        "Integrated Prediction and Trajectory Evidence — "
        f"{model_name.upper()} {dataset}"
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        DECISION_DIR,
        f"selected_decisions_v4_"
        f"{model_name}_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def print_selected_decisions(
    decision_df,
    selected_units,
):
    """Print integrated decisions for selected engines."""

    if selected_units:
        display_df = decision_df[
            decision_df[
                "unit"
            ].isin(
                selected_units
            )
        ].copy()

    else:
        display_df = decision_df.nlargest(
            10,
            "risk_score",
        ).copy()

    columns = [
        "unit",
        "actual_rul",
        "predicted_rul_mean",
        "ridge_prediction",
        "calibrated_lower_bound",
        "conservative_rul",
        "engine_condition",
        "prediction_trust",
        "trajectory_flag",
        "trajectory_trust",
        "review_flag",
        "recent_slope",
        "recent_predicted_drop",
        "model_disagreement",
        "risk_score",
        "operational_action",
    ]

    print(
        display_df[
            columns
        ]
        .sort_values(
            by="unit"
        )
        .to_string(
            index=False
        )
    )


def run_decision_layer(
    model_name,
    dataset,
    selected_units,
):
    """Run the integrated risk-aware decision layer."""

    (
        uncertainty_df,
        uncertainty_path,
    ) = load_uncertainty_results(
        model_name,
        dataset,
    )

    (
        ridge_df,
        ridge_path,
    ) = load_ridge_predictions(
        dataset
    )

    (
        trajectory_df,
        trajectory_path,
    ) = load_trajectory_results(
        model_name,
        dataset,
    )

    decision_df = build_decision_results(
        uncertainty_df=uncertainty_df,
        ridge_df=ridge_df,
        trajectory_df=trajectory_df,
    )

    condition_summary_df = (
        build_condition_summary(
            decision_df
        )
    )

    trust_summary_df = (
        build_trust_summary(
            decision_df
        )
    )

    trajectory_summary_df = (
        build_trajectory_summary(
            decision_df
        )
    )

    review_summary_df = (
        build_review_summary(
            decision_df
        )
    )

    condition_evaluation_df = (
        evaluate_condition_categories(
            decision_df
        )
    )

    decisions_path = os.path.join(
        DECISION_DIR,
        f"decision_results_v4_"
        f"{model_name}_{dataset}.csv",
    )

    condition_summary_path = os.path.join(
        DECISION_DIR,
        f"condition_summary_v4_"
        f"{model_name}_{dataset}.csv",
    )

    trust_summary_path = os.path.join(
        DECISION_DIR,
        f"trust_summary_v4_"
        f"{model_name}_{dataset}.csv",
    )

    trajectory_summary_path = os.path.join(
        DECISION_DIR,
        f"trajectory_summary_v4_"
        f"{model_name}_{dataset}.csv",
    )

    review_summary_path = os.path.join(
        DECISION_DIR,
        f"review_summary_v4_"
        f"{model_name}_{dataset}.csv",
    )

    evaluation_path = os.path.join(
        DECISION_DIR,
        f"condition_evaluation_v4_"
        f"{model_name}_{dataset}.csv",
    )

    decision_df.to_csv(
        decisions_path,
        index=False,
    )

    condition_summary_df.to_csv(
        condition_summary_path,
        index=False,
    )

    trust_summary_df.to_csv(
        trust_summary_path,
        index=False,
    )

    trajectory_summary_df.to_csv(
        trajectory_summary_path,
        index=False,
    )

    review_summary_df.to_csv(
        review_summary_path,
        index=False,
    )

    condition_evaluation_df.to_csv(
        evaluation_path,
        index=False,
    )

    condition_plot_path = (
        plot_condition_distribution(
            condition_summary_df,
            model_name,
            dataset,
        )
    )

    review_plot_path = (
        plot_review_distribution(
            review_summary_df,
            model_name,
            dataset,
        )
    )

    selected_plot_path = (
        plot_selected_units(
            decision_df,
            selected_units,
            model_name,
            dataset,
        )
    )

    print(
        f"Uncertainty source: {uncertainty_path}"
    )

    print(
        f"Ridge source:       {ridge_path}"
    )

    print(
        f"Trajectory source:  {trajectory_path}"
    )

    print(
        "\nEngine condition summary"
    )

    print(
        condition_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nPrediction trust summary"
    )

    print(
        trust_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nTrajectory diagnostic summary"
    )

    print(
        trajectory_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nEngineering review summary"
    )

    print(
        review_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nSelected integrated decisions"
    )

    print_selected_decisions(
        decision_df,
        selected_units,
    )

    print(
        "\nRetrospective condition evaluation"
    )

    print(
        condition_evaluation_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved outputs"
    )

    print(
        f"Decisions:          {decisions_path}"
    )

    print(
        f"Condition summary:  {condition_summary_path}"
    )

    print(
        f"Trust summary:      {trust_summary_path}"
    )

    print(
        f"Trajectory summary: {trajectory_summary_path}"
    )

    print(
        f"Review summary:     {review_summary_path}"
    )

    print(
        f"Evaluation:         {evaluation_path}"
    )

    print(
        f"Condition plot:     {condition_plot_path}"
    )

    print(
        f"Review plot:        {review_plot_path}"
    )

    print(
        f"Selected engines:   {selected_plot_path}"
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

    arguments = parser.parse_args()

    run_decision_layer(
        model_name=arguments.model,
        dataset=arguments.dataset,
        selected_units=arguments.units,
    )