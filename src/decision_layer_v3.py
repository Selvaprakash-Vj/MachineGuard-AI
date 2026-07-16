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
DECISION_DIR = os.path.join(
    RESULTS_DIR,
    "decisions_v3",
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
    """Load calibrated neural-network uncertainty results."""

    uncertainty_path = os.path.join(
        UNCERTAINTY_DIR,
        f"uncertainty_results_{model_name}_{dataset}.csv",
    )

    if not os.path.exists(uncertainty_path):
        raise FileNotFoundError(
            f"Uncertainty results not found: {uncertainty_path}\n"
            "Run uncertainty_v2.py first."
        )

    uncertainty_df = pd.read_csv(
        uncertainty_path
    )

    return uncertainty_df, uncertainty_path


def load_ridge_predictions(dataset):
    """Load independent Ridge baseline predictions."""

    ridge_path = os.path.join(
        RESULTS_DIR,
        f"predictions_baseline_ridge_{dataset}.csv",
    )

    if not os.path.exists(ridge_path):
        raise FileNotFoundError(
            f"Ridge predictions not found: {ridge_path}\n"
            "Run baseline_v2.py with --model ridge first."
        )

    ridge_df = pd.read_csv(
        ridge_path
    )

    ridge_df = ridge_df[
        [
            "unit",
            "predicted_rul",
        ]
    ].rename(
        columns={
            "predicted_rul": "ridge_prediction",
        }
    )

    return ridge_df, ridge_path


def classify_disagreement(
    disagreement,
):
    """Classify disagreement between LSTM and Ridge."""

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
    Calculate a conservative RUL estimate.

    The more cautious value is selected from:
    - the calibrated lower uncertainty bound;
    - the Ridge prediction.
    """

    return float(
        max(
            0.0,
            min(
                calibrated_lower_bound,
                ridge_prediction,
            ),
        )
    )


def classify_engine_condition(
    conservative_rul,
):
    """
    Classify estimated physical condition.

    This classification is independent of model confidence.
    """

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
    """
    Classify whether the RUL estimate itself is trustworthy.

    Trust depends on uncertainty and agreement between two
    independently structured models.
    """

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


def assign_review_flag(
    engine_condition,
    prediction_trust,
    disagreement_level,
    interval_width,
):
    """Decide whether an engineer should review the prediction."""

    if disagreement_level == "High":
        return "Required"

    if (
        prediction_trust == "Low"
        and engine_condition
        in [
            "Critical",
            "Warning",
            "Monitor",
        ]
    ):
        return "Required"

    if interval_width > 50:
        return "Required"

    if (
        prediction_trust == "Low"
        or disagreement_level == "Medium"
        or interval_width > 35
    ):
        return "Recommended"

    return "Not required"


def calculate_risk_score(
    conservative_rul,
    prediction_trust,
    disagreement_level,
    interval_width,
):
    """
    Calculate a transparent decision-risk score from 0 to 100.

    The score combines estimated condition severity with uncertainty.
    It does not use actual RUL.
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
    }[prediction_trust]

    disagreement_penalty = {
        "Low": 0.0,
        "Medium": 7.0,
        "High": 15.0,
    }[disagreement_level]

    if interval_width > 50:
        interval_penalty = 10.0

    elif interval_width > 35:
        interval_penalty = 5.0

    else:
        interval_penalty = 0.0

    score = (
        condition_component
        + uncertainty_penalty
        + disagreement_penalty
        + interval_penalty
    )

    return float(
        np.clip(
            score,
            0.0,
            100.0,
        )
    )


def assign_operational_action(
    engine_condition,
    review_flag,
):
    """Generate a clear operational recommendation."""

    base_actions = {
        "Critical": (
            "Inspect immediately and restrict continued operation "
            "until the engine condition is verified."
        ),
        "Warning": (
            "Schedule maintenance at the earliest practical "
            "opportunity and increase inspection frequency."
        ),
        "Monitor": (
            "Continue operation with enhanced monitoring and "
            "repeat the RUL assessment after additional cycles."
        ),
        "Healthy": (
            "Continue normal operation and routine condition "
            "monitoring."
        ),
    }

    action = base_actions[
        engine_condition
    ]

    if review_flag == "Required":
        action += (
            " Manual engineering review is required before "
            "relying on the predicted RUL."
        )

    elif review_flag == "Recommended":
        action += (
            " Engineering review is recommended because the "
            "prediction contains elevated uncertainty."
        )

    return action


def build_decision_results(
    uncertainty_df,
    ridge_df,
):
    """Combine prediction, uncertainty and model evidence."""

    decision_df = uncertainty_df.merge(
        ridge_df,
        on="unit",
        how="inner",
        validate="one_to_one",
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
        "review_flag"
    ] = decision_df.apply(
        lambda row: assign_review_flag(
            engine_condition=row[
                "engine_condition"
            ],
            prediction_trust=row[
                "prediction_trust"
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
    """Summarize estimated engine condition categories."""

    condition_order = [
        "Critical",
        "Warning",
        "Monitor",
        "Healthy",
    ]

    rows = []

    for condition in condition_order:
        condition_df = decision_df[
            decision_df[
                "engine_condition"
            ]
            == condition
        ]

        if condition_df.empty:
            continue

        rows.append(
            {
                "engine_condition": condition,
                "engine_count": len(
                    condition_df
                ),
                "percentage": (
                    len(condition_df)
                    / len(decision_df)
                    * 100
                ),
                "average_predicted_rul": float(
                    condition_df[
                        "predicted_rul_mean"
                    ].mean()
                ),
                "average_conservative_rul": float(
                    condition_df[
                        "conservative_rul"
                    ].mean()
                ),
                "average_risk_score": float(
                    condition_df[
                        "risk_score"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_trust_summary(
    decision_df,
):
    """Summarize prediction-trust and review categories."""

    rows = []

    for trust_level in [
        "High",
        "Medium",
        "Low",
    ]:
        trust_df = decision_df[
            decision_df[
                "prediction_trust"
            ]
            == trust_level
        ]

        if trust_df.empty:
            continue

        rows.append(
            {
                "prediction_trust": trust_level,
                "engine_count": len(
                    trust_df
                ),
                "percentage": (
                    len(trust_df)
                    / len(decision_df)
                    * 100
                ),
                "average_interval_width": float(
                    trust_df[
                        "interval_width"
                    ].mean()
                ),
                "average_model_disagreement": float(
                    trust_df[
                        "model_disagreement"
                    ].mean()
                ),
                "average_absolute_error": float(
                    trust_df[
                        "absolute_error"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def build_review_summary(
    decision_df,
):
    """Summarize required and recommended engineering reviews."""

    rows = []

    for review_flag in [
        "Required",
        "Recommended",
        "Not required",
    ]:
        review_df = decision_df[
            decision_df[
                "review_flag"
            ]
            == review_flag
        ]

        if review_df.empty:
            continue

        rows.append(
            {
                "review_flag": review_flag,
                "engine_count": len(
                    review_df
                ),
                "percentage": (
                    len(review_df)
                    / len(decision_df)
                    * 100
                ),
                "average_absolute_error": float(
                    review_df[
                        "absolute_error"
                    ].mean()
                ),
                "dangerous_overprediction_rate": float(
                    (
                        review_df[
                            "prediction_error"
                        ]
                        > 10
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def evaluate_condition_categories(
    decision_df,
):
    """
    Retrospectively evaluate condition categories.

    Actual RUL is used only for evaluation, never for assigning
    engine condition or operational action.
    """

    rows = []

    for condition, condition_df in decision_df.groupby(
        "engine_condition"
    ):
        rows.append(
            {
                "engine_condition": condition,
                "engine_count": len(
                    condition_df
                ),
                "average_actual_rul": float(
                    condition_df[
                        "actual_rul"
                    ].mean()
                ),
                "average_absolute_error": float(
                    condition_df[
                        "absolute_error"
                    ].mean()
                ),
                "actual_rul_below_20_rate": float(
                    (
                        condition_df[
                            "actual_rul"
                        ]
                        <= 20
                    ).mean()
                ),
                "actual_rul_below_50_rate": float(
                    (
                        condition_df[
                            "actual_rul"
                        ]
                        <= 50
                    ).mean()
                ),
                "dangerous_overprediction_rate": float(
                    (
                        condition_df[
                            "prediction_error"
                        ]
                        > 10
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def plot_condition_distribution(
    condition_summary_df,
    model_name,
    dataset,
):
    """Plot the estimated engine-condition distribution."""

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
        f"Engine Condition Distribution — "
        f"{model_name.upper()} {dataset}"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        DECISION_DIR,
        f"condition_distribution_"
        f"{model_name}_{dataset}.png",
    )

    plt.savefig(
        output_path,
        dpi=300,
    )

    plt.close()

    return output_path


def plot_trust_distribution(
    trust_summary_df,
    model_name,
    dataset,
):
    """Plot prediction-trust categories."""

    plt.figure(
        figsize=(9, 6)
    )

    plt.bar(
        trust_summary_df[
            "prediction_trust"
        ],
        trust_summary_df[
            "engine_count"
        ],
    )

    plt.xlabel(
        "Prediction trust"
    )

    plt.ylabel(
        "Number of test engines"
    )

    plt.title(
        f"Prediction Trust Distribution — "
        f"{model_name.upper()} {dataset}"
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        DECISION_DIR,
        f"trust_distribution_"
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
    """Plot decision evidence for selected engines."""

    if selected_units:
        plot_df = decision_df[
            decision_df["unit"].isin(
                selected_units
            )
        ].copy()

    else:
        plot_df = (
            decision_df
            .nlargest(
                10,
                "risk_score",
            )
            .copy()
        )

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
        len(plot_df)
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
        label="LSTM uncertainty mean",
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
        s=90,
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
        f"Condition and Trust Evidence — "
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
        f"selected_decisions_"
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
    """Print decisions for selected engines."""

    if selected_units:
        display_df = decision_df[
            decision_df[
                "unit"
            ].isin(
                selected_units
            )
        ].copy()

    else:
        display_df = (
            decision_df
            .nlargest(
                10,
                "risk_score",
            )
            .copy()
        )

    columns = [
        "unit",
        "actual_rul",
        "predicted_rul_mean",
        "ridge_prediction",
        "calibrated_lower_bound",
        "conservative_rul",
        "engine_condition",
        "prediction_trust",
        "review_flag",
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
    """Run the separated condition-and-trust decision layer."""

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

    decision_df = build_decision_results(
        uncertainty_df,
        ridge_df,
    )

    condition_summary_df = (
        build_condition_summary(
            decision_df
        )
    )

    trust_summary_df = build_trust_summary(
        decision_df
    )

    review_summary_df = build_review_summary(
        decision_df
    )

    condition_evaluation_df = (
        evaluate_condition_categories(
            decision_df
        )
    )

    decisions_path = os.path.join(
        DECISION_DIR,
        f"decision_results_v3_"
        f"{model_name}_{dataset}.csv",
    )

    condition_summary_path = os.path.join(
        DECISION_DIR,
        f"condition_summary_v3_"
        f"{model_name}_{dataset}.csv",
    )

    trust_summary_path = os.path.join(
        DECISION_DIR,
        f"trust_summary_v3_"
        f"{model_name}_{dataset}.csv",
    )

    review_summary_path = os.path.join(
        DECISION_DIR,
        f"review_summary_v3_"
        f"{model_name}_{dataset}.csv",
    )

    evaluation_path = os.path.join(
        DECISION_DIR,
        f"condition_evaluation_v3_"
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

    trust_plot_path = (
        plot_trust_distribution(
            trust_summary_df,
            model_name,
            dataset,
        )
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
        "\nEngineering review summary"
    )

    print(
        review_summary_df.to_string(
            index=False
        )
    )

    print(
        "\nSelected engine decisions"
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
        f"Review summary:     {review_summary_path}"
    )

    print(
        f"Evaluation:         {evaluation_path}"
    )

    print(
        f"Condition plot:     {condition_plot_path}"
    )

    print(
        f"Trust plot:         {trust_plot_path}"
    )

    print(
        f"Selected engines:   {selected_plot_path}"
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
        "--units",
        type=int,
        nargs="*",
        default=None,
    )

    arguments = parser.parse_args()

    run_decision_layer(
        model_name=arguments.model,
        dataset=arguments.dataset,
        selected_units=arguments.units,
    )