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
    "decisions_v5",
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
        if column not in uncertainty_df.columns
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
    """Load independent Ridge predictions."""

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
        if column not in ridge_df.columns
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
            "predicted_rul": "ridge_prediction",
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

    return (
        ridge_df,
        ridge_path,
    )


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
        "recent_prediction_std",
        "recent_residual_std",
        "monotonicity_violation_rate",
        "large_jump_count",
        "largest_single_cycle_jump",
        "trajectory_flag",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in trajectory_df.columns
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

    return (
        trajectory_df[
            required_columns
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


def classify_mc_deterministic_gap(
    gap,
):
    """
    Classify disagreement between the MC-dropout mean
    and deterministic sequence-model prediction.
    """

    if gap <= 10:
        return "Low"

    if gap <= 20:
        return "Medium"

    return "High"


def calculate_conservative_rul(
    calibrated_lower_bound,
    ridge_prediction,
):
    """Select the more cautious available RUL estimate."""

    conservative_rul = min(
        calibrated_lower_bound,
        ridge_prediction,
    )

    return float(
        np.clip(
            conservative_rul,
            0.0,
            RUL_CAP,
        )
    )


def classify_engine_condition(
    conservative_rul,
):
    """Classify estimated physical engine condition."""

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
    """Convert temporal behaviour into a trust category."""

    if trajectory_flag == "Consistent degradation":
        return "High"

    if trajectory_flag in {
        "Rapid degradation signal",
        "Mixed trajectory",
    }:
        return "Medium"

    return "Low"


def calculate_condition_severity_score(
    conservative_rul,
):
    """
    Calculate condition severity from 0 to 100.

    A lower conservative RUL produces higher condition severity.
    No uncertainty or reliability evidence is included here.
    """

    severity = (
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

    return float(
        np.clip(
            severity,
            0.0,
            100.0,
        )
    )


def trust_category_score(
    trust_category,
):
    """Convert a trust category into reliability risk."""

    return {
        "High": 0.0,
        "Medium": 50.0,
        "Low": 100.0,
    }[
        trust_category
    ]


def disagreement_risk_score(
    disagreement_level,
):
    """Convert model disagreement into a risk score."""

    return {
        "Low": 0.0,
        "Medium": 50.0,
        "High": 100.0,
    }[
        disagreement_level
    ]


def interval_width_risk_score(
    interval_width,
):
    """
    Convert calibrated interval width into a continuous risk score.

    Widths at or below 20 cycles receive zero interval risk.
    Widths at or above 60 cycles receive maximum interval risk.
    """

    score = (
        (
            interval_width
            - 20.0
        )
        / 40.0
        * 100.0
    )

    return float(
        np.clip(
            score,
            0.0,
            100.0,
        )
    )


def mc_gap_risk_score(
    mc_deterministic_gap,
):
    """
    Convert MC-versus-deterministic disagreement into risk.

    A difference of 30 cycles or more receives maximum risk.
    """

    score = (
        mc_deterministic_gap
        / 30.0
        * 100.0
    )

    return float(
        np.clip(
            score,
            0.0,
            100.0,
        )
    )


def trajectory_flag_risk_score(
    trajectory_flag,
):
    """Convert the temporal diagnostic into reliability risk."""

    flag_scores = {
        "Consistent degradation": 0.0,
        "Rapid degradation signal": 40.0,
        "Mixed trajectory": 50.0,
        "Weak degradation / plateau": 60.0,
        "High-RUL plateau — review": 100.0,
        "RUL increasing despite ageing": 100.0,
        "Unstable trajectory": 100.0,
    }

    return float(
        flag_scores.get(
            trajectory_flag,
            50.0,
        )
    )


def calculate_reliability_risk_score(
    prediction_trust,
    trajectory_trust,
    disagreement_level,
    interval_width,
    mc_deterministic_gap,
    trajectory_flag,
):
    """
    Calculate prediction reliability risk from 0 to 100.

    This score contains no condition-severity component.

    Components:
    - static prediction trust:        25%
    - temporal trajectory trust:      25%
    - model disagreement:             15%
    - calibrated interval width:      15%
    - MC/deterministic disagreement:  10%
    - temporal diagnostic flag:       10%
    """

    static_component = trust_category_score(
        prediction_trust
    )

    temporal_component = trust_category_score(
        trajectory_trust
    )

    disagreement_component = disagreement_risk_score(
        disagreement_level
    )

    interval_component = interval_width_risk_score(
        interval_width
    )

    mc_gap_component = mc_gap_risk_score(
        mc_deterministic_gap
    )

    trajectory_flag_component = (
        trajectory_flag_risk_score(
            trajectory_flag
        )
    )

    score = (
        0.25
        * static_component
        + 0.25
        * temporal_component
        + 0.15
        * disagreement_component
        + 0.15
        * interval_component
        + 0.10
        * mc_gap_component
        + 0.10
        * trajectory_flag_component
    )

    return float(
        np.clip(
            score,
            0.0,
            100.0,
        )
    )


def classify_reliability_risk(
    reliability_risk_score,
):
    """Convert numerical reliability risk into a category."""

    if reliability_risk_score < 30:
        return "Low"

    if reliability_risk_score < 60:
        return "Medium"

    return "High"


def assign_review_flag(
    engine_condition,
    prediction_trust,
    trajectory_flag,
    reliability_risk_score,
    disagreement_level,
    mc_gap_level,
    interval_width,
):
    """
    Assign engineering review from reliability evidence.

    Estimated engine condition does not automatically create a
    reliability review. Condition urgency is handled separately.
    """

    mandatory_temporal_flags = {
        "High-RUL plateau — review",
        "RUL increasing despite ageing",
    }

    if trajectory_flag in mandatory_temporal_flags:
        return "Required"

    if disagreement_level == "High":
        return "Required"

    if mc_gap_level == "High":
        return "Required"

    if reliability_risk_score >= 70:
        return "Required"

    if (
        trajectory_flag == "Unstable trajectory"
        and engine_condition
        in {
            "Critical",
            "Warning",
            "Monitor",
        }
    ):
        return "Required"

    if (
        interval_width > 55
        and engine_condition
        != "Healthy"
    ):
        return "Required"

    if reliability_risk_score >= 35:
        return "Recommended"

    if trajectory_flag in {
        "Unstable trajectory",
        "Rapid degradation signal",
        "Mixed trajectory",
        "Weak degradation / plateau",
    }:
        return "Recommended"

    if prediction_trust == "Low":
        return "Recommended"

    if disagreement_level == "Medium":
        return "Recommended"

    if mc_gap_level == "Medium":
        return "Recommended"

    if interval_width > 35:
        return "Recommended"

    return "Not required"


def calculate_operational_priority_score(
    condition_severity_score,
    reliability_risk_score,
    review_flag,
):
    """
    Combine condition urgency and reliability risk.

    Condition severity receives greater weight because imminent
    failure remains operationally important even when the model
    prediction is reliable.
    """

    score = (
        0.70
        * condition_severity_score
        + 0.30
        * reliability_risk_score
    )

    minimum_scores = {
        "Required": 55.0,
        "Recommended": 35.0,
        "Not required": 0.0,
    }

    score = max(
        score,
        minimum_scores[
            review_flag
        ],
    )

    return float(
        np.clip(
            score,
            0.0,
            100.0,
        )
    )


def classify_operational_priority(
    engine_condition,
    review_flag,
    operational_priority_score,
):
    """Assign a human-readable operational priority."""

    if engine_condition == "Critical":
        return "Immediate"

    if (
        engine_condition == "Warning"
        or operational_priority_score >= 70
    ):
        return "High"

    if (
        review_flag == "Required"
        or engine_condition == "Monitor"
        or operational_priority_score >= 45
    ):
        return "Elevated"

    if review_flag == "Recommended":
        return "Watch"

    return "Routine"


def build_trajectory_message(
    trajectory_flag,
):
    """Translate the temporal flag into an explanation."""

    messages = {
        "Consistent degradation": (
            "The recent RUL trajectory shows a broadly "
            "consistent degradation pattern."
        ),
        "Rapid degradation signal": (
            "The predicted RUL is declining rapidly and "
            "should be checked against recent sensor trends."
        ),
        "Mixed trajectory": (
            "The temporal prediction pattern is mixed and "
            "does not provide a stable degradation trend."
        ),
        "Weak degradation / plateau": (
            "The predicted RUL shows weak degradation despite "
            "continued engine ageing."
        ),
        "High-RUL plateau — review": (
            "The model remains on a high-RUL plateau despite "
            "continued ageing, indicating a possible shared "
            "model blind spot."
        ),
        "RUL increasing despite ageing": (
            "The predicted RUL increases despite additional "
            "engine cycles, indicating temporal inconsistency."
        ),
        "Unstable trajectory": (
            "The predicted RUL contains large jumps or excessive "
            "oscillations and should not be used without review."
        ),
    }

    return messages.get(
        trajectory_flag,
        "The temporal prediction pattern requires review.",
    )


def assign_operational_action(
    engine_condition,
    review_flag,
    trajectory_flag,
):
    """
    Generate an operational recommendation.

    Mandatory review is stated before normal-operation guidance
    so that the message cannot contradict itself.
    """

    trajectory_message = build_trajectory_message(
        trajectory_flag
    )

    if engine_condition == "Critical":
        condition_action = (
            "Inspect immediately and restrict continued operation "
            "until the physical engine condition is verified."
        )

    elif engine_condition == "Warning":
        condition_action = (
            "Increase inspection frequency and prepare an early "
            "maintenance intervention."
        )

    elif engine_condition == "Monitor":
        condition_action = (
            "Use enhanced monitoring and repeat the RUL assessment "
            "after additional operating cycles."
        )

    else:
        condition_action = (
            "Continue routine condition monitoring."
        )

    if review_flag == "Required":
        return (
            "Manual engineering review is required before the "
            "predicted RUL is used for an operational or maintenance "
            "decision. "
            + condition_action
            + " "
            + trajectory_message
        )

    if review_flag == "Recommended":
        return (
            "Engineering review is recommended before making a "
            "high-consequence decision. "
            + condition_action
            + " "
            + trajectory_message
        )

    return (
        condition_action
        + " "
        + trajectory_message
    )


def build_decision_results(
    uncertainty_df,
    ridge_df,
    trajectory_df,
):
    """Build the integrated V5 decision results."""

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
        "mc_deterministic_gap"
    ] = np.abs(
        decision_df[
            "predicted_rul_mean"
        ]
        - decision_df[
            "final_predicted_rul"
        ]
    )

    decision_df[
        "mc_gap_level"
    ] = decision_df[
        "mc_deterministic_gap"
    ].apply(
        classify_mc_deterministic_gap
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
        "condition_severity_score"
    ] = decision_df[
        "conservative_rul"
    ].apply(
        calculate_condition_severity_score
    )

    decision_df[
        "reliability_risk_score"
    ] = decision_df.apply(
        lambda row: calculate_reliability_risk_score(
            prediction_trust=row[
                "prediction_trust"
            ],
            trajectory_trust=row[
                "trajectory_trust"
            ],
            disagreement_level=row[
                "disagreement_level"
            ],
            interval_width=row[
                "interval_width"
            ],
            mc_deterministic_gap=row[
                "mc_deterministic_gap"
            ],
            trajectory_flag=row[
                "trajectory_flag"
            ],
        ),
        axis=1,
    )

    decision_df[
        "reliability_risk"
    ] = decision_df[
        "reliability_risk_score"
    ].apply(
        classify_reliability_risk
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
            trajectory_flag=row[
                "trajectory_flag"
            ],
            reliability_risk_score=row[
                "reliability_risk_score"
            ],
            disagreement_level=row[
                "disagreement_level"
            ],
            mc_gap_level=row[
                "mc_gap_level"
            ],
            interval_width=row[
                "interval_width"
            ],
        ),
        axis=1,
    )

    decision_df[
        "operational_priority_score"
    ] = decision_df.apply(
        lambda row: calculate_operational_priority_score(
            condition_severity_score=row[
                "condition_severity_score"
            ],
            reliability_risk_score=row[
                "reliability_risk_score"
            ],
            review_flag=row[
                "review_flag"
            ],
        ),
        axis=1,
    )

    decision_df[
        "operational_priority"
    ] = decision_df.apply(
        lambda row: classify_operational_priority(
            engine_condition=row[
                "engine_condition"
            ],
            review_flag=row[
                "review_flag"
            ],
            operational_priority_score=row[
                "operational_priority_score"
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

    operational_priority_order = {
        "Immediate": 1,
        "High": 2,
        "Elevated": 3,
        "Watch": 4,
        "Routine": 5,
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

    decision_df[
        "operational_priority_order"
    ] = decision_df[
        "operational_priority"
    ].map(
        operational_priority_order
    )

    decision_df = decision_df.sort_values(
        by=[
            "operational_priority_order",
            "condition_priority",
            "review_priority",
            "operational_priority_score",
        ],
        ascending=[
            True,
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
    """Summarize condition severity."""

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
                    * 100.0
                ),
                "average_conservative_rul": float(
                    group_df[
                        "conservative_rul"
                    ].mean()
                ),
                "average_condition_severity_score": float(
                    group_df[
                        "condition_severity_score"
                    ].mean()
                ),
                "average_actual_rul": float(
                    group_df[
                        "actual_rul"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def build_reliability_summary(
    decision_df,
):
    """Summarize prediction reliability risk."""

    rows = []

    for reliability_risk in [
        "High",
        "Medium",
        "Low",
    ]:
        group_df = decision_df[
            decision_df[
                "reliability_risk"
            ]
            == reliability_risk
        ]

        if group_df.empty:
            continue

        rows.append(
            {
                "reliability_risk": reliability_risk,
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
                    * 100.0
                ),
                "average_reliability_risk_score": float(
                    group_df[
                        "reliability_risk_score"
                    ].mean()
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


def build_review_summary(
    decision_df,
):
    """Summarize engineering review decisions."""

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
                    * 100.0
                ),
                "average_reliability_risk_score": float(
                    group_df[
                        "reliability_risk_score"
                    ].mean()
                ),
                "average_condition_severity_score": float(
                    group_df[
                        "condition_severity_score"
                    ].mean()
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


def build_priority_summary(
    decision_df,
):
    """Summarize operational-priority categories."""

    rows = []

    for priority in [
        "Immediate",
        "High",
        "Elevated",
        "Watch",
        "Routine",
    ]:
        group_df = decision_df[
            decision_df[
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
                        decision_df
                    )
                    * 100.0
                ),
                "average_operational_priority_score": float(
                    group_df[
                        "operational_priority_score"
                    ].mean()
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
            }
        )

    return pd.DataFrame(
        rows
    )


def build_trajectory_summary(
    decision_df,
):
    """Summarize temporal diagnostics."""

    summary_df = (
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
            average_reliability_risk_score=(
                "reliability_risk_score",
                "mean",
            ),
            average_recent_slope=(
                "recent_slope",
                "mean",
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
            decision_df
        )
        * 100.0
    )

    return summary_df.sort_values(
        by="engine_count",
        ascending=False,
    ).reset_index(
        drop=True
    )


def plot_score_map(
    decision_df,
    model_name,
    dataset,
):
    """Plot condition severity against reliability risk."""

    plt.figure(
        figsize=(10, 7)
    )

    plt.scatter(
        decision_df[
            "condition_severity_score"
        ],
        decision_df[
            "reliability_risk_score"
        ],
        s=55,
        alpha=0.75,
    )

    for _, row in decision_df[
        decision_df[
            "unit"
        ].isin(
            [
                25,
                45,
                67,
                79,
            ]
        )
    ].iterrows():
        plt.annotate(
            f"Unit {int(row['unit'])}",
            (
                row[
                    "condition_severity_score"
                ],
                row[
                    "reliability_risk_score"
                ],
            ),
            xytext=(
                5,
                5,
            ),
            textcoords="offset points",
        )

    plt.axvline(
        60,
        linestyle="--",
        alpha=0.4,
    )

    plt.axhline(
        60,
        linestyle="--",
        alpha=0.4,
    )

    plt.xlabel(
        "Condition severity score"
    )

    plt.ylabel(
        "Prediction reliability risk"
    )

    plt.title(
        "Condition Severity vs Prediction Reliability — "
        f"{model_name.upper()} {dataset}"
    )

    plt.grid(
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        DECISION_DIR,
        f"condition_reliability_map_"
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
    """Plot engineering-review workload."""

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
        "Engineering review decision"
    )

    plt.ylabel(
        "Number of engines"
    )

    plt.title(
        "V5 Engineering Review Distribution — "
        f"{model_name.upper()} {dataset}"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        DECISION_DIR,
        f"review_distribution_v5_"
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
    """Plot selected engine predictions."""

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
            "operational_priority_score",
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
            "MC-dropout mean"
        ),
    )

    plt.bar(
        positions,
        plot_df[
            "final_predicted_rul"
        ],
        width=width,
        label=(
            f"{model_name.upper()} "
            "deterministic prediction"
        ),
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
        "V5 Integrated Decision Evidence — "
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
        f"selected_decisions_v5_"
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
    """Print selected V5 decisions."""

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
            "operational_priority_score",
        ).copy()

    columns = [
        "unit",
        "actual_rul",
        "predicted_rul_mean",
        "final_predicted_rul",
        "ridge_prediction",
        "conservative_rul",
        "engine_condition",
        "condition_severity_score",
        "prediction_trust",
        "trajectory_flag",
        "trajectory_trust",
        "mc_deterministic_gap",
        "mc_gap_level",
        "model_disagreement",
        "reliability_risk_score",
        "reliability_risk",
        "review_flag",
        "operational_priority_score",
        "operational_priority",
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
    """Run the V5 decision layer."""

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

    condition_summary_df = build_condition_summary(
        decision_df
    )

    reliability_summary_df = (
        build_reliability_summary(
            decision_df
        )
    )

    review_summary_df = build_review_summary(
        decision_df
    )

    priority_summary_df = build_priority_summary(
        decision_df
    )

    trajectory_summary_df = (
        build_trajectory_summary(
            decision_df
        )
    )

    decisions_path = os.path.join(
        DECISION_DIR,
        f"decision_results_v5_"
        f"{model_name}_{dataset}.csv",
    )

    condition_summary_path = os.path.join(
        DECISION_DIR,
        f"condition_summary_v5_"
        f"{model_name}_{dataset}.csv",
    )

    reliability_summary_path = os.path.join(
        DECISION_DIR,
        f"reliability_summary_v5_"
        f"{model_name}_{dataset}.csv",
    )

    review_summary_path = os.path.join(
        DECISION_DIR,
        f"review_summary_v5_"
        f"{model_name}_{dataset}.csv",
    )

    priority_summary_path = os.path.join(
        DECISION_DIR,
        f"priority_summary_v5_"
        f"{model_name}_{dataset}.csv",
    )

    trajectory_summary_path = os.path.join(
        DECISION_DIR,
        f"trajectory_summary_v5_"
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

    reliability_summary_df.to_csv(
        reliability_summary_path,
        index=False,
    )

    review_summary_df.to_csv(
        review_summary_path,
        index=False,
    )

    priority_summary_df.to_csv(
        priority_summary_path,
        index=False,
    )

    trajectory_summary_df.to_csv(
        trajectory_summary_path,
        index=False,
    )

    score_map_path = plot_score_map(
        decision_df,
        model_name,
        dataset,
    )

    review_plot_path = plot_review_distribution(
        review_summary_df,
        model_name,
        dataset,
    )

    selected_plot_path = plot_selected_units(
        decision_df,
        selected_units,
        model_name,
        dataset,
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
        "\nCondition severity summary"
    )

    print(
        condition_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nPrediction reliability summary"
    )

    print(
        reliability_summary_df.to_string(
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
        "\nOperational priority summary"
    )

    print(
        priority_summary_df.to_string(
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
        "\nSelected V5 decisions"
    )

    print_selected_decisions(
        decision_df,
        selected_units,
    )

    print(
        "\nSaved outputs"
    )

    print(
        f"Decisions:           {decisions_path}"
    )

    print(
        f"Condition summary:   {condition_summary_path}"
    )

    print(
        f"Reliability summary: {reliability_summary_path}"
    )

    print(
        f"Review summary:      {review_summary_path}"
    )

    print(
        f"Priority summary:    {priority_summary_path}"
    )

    print(
        f"Trajectory summary:  {trajectory_summary_path}"
    )

    print(
        f"Score map:           {score_map_path}"
    )

    print(
        f"Review plot:         {review_plot_path}"
    )

    print(
        f"Selected engines:    {selected_plot_path}"
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