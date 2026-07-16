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
    "decisions",
)

os.makedirs(
    DECISION_DIR,
    exist_ok=True,
)


def load_uncertainty_results(
    model_name,
    dataset,
):
    """Load calibrated LSTM uncertainty results."""

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
    """Load Ridge baseline predictions for model comparison."""

    ridge_path = os.path.join(
        RESULTS_DIR,
        f"predictions_baseline_ridge_{dataset}.csv",
    )

    if not os.path.exists(ridge_path):
        raise FileNotFoundError(
            f"Ridge predictions not found: {ridge_path}\n"
            "Run baseline_v2.py with the Ridge model first."
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
    lower_bound,
    ridge_prediction,
):
    """
    Create a conservative RUL proxy.

    The decision layer uses the lower LSTM confidence bound
    and the Ridge prediction, selecting the more cautious value.
    """

    return float(
        min(
            lower_bound,
            ridge_prediction,
        )
    )


def calculate_risk_score(
    predicted_rul,
    conservative_rul,
    interval_width,
    confidence,
    disagreement,
):
    """
    Calculate a transparent rule-based risk score from 0 to 100.

    Higher scores indicate greater maintenance concern.
    """

    score = 0.0

    # Conservative remaining-life contribution.
    if conservative_rul <= 10:
        score += 50

    elif conservative_rul <= 20:
        score += 42

    elif conservative_rul <= 35:
        score += 32

    elif conservative_rul <= 50:
        score += 22

    elif conservative_rul <= 80:
        score += 12

    # Mean predicted RUL contribution.
    if predicted_rul <= 20:
        score += 25

    elif predicted_rul <= 50:
        score += 18

    elif predicted_rul <= 80:
        score += 10

    # Uncertainty contribution.
    if confidence == "Low":
        score += 15

    elif confidence == "Medium":
        score += 7

    # Model disagreement contribution.
    if disagreement > 30:
        score += 15

    elif disagreement > 20:
        score += 10

    elif disagreement > 10:
        score += 5

    # Very broad intervals are operationally difficult.
    if interval_width > 50:
        score += 10

    elif interval_width > 35:
        score += 5

    return float(
        min(
            score,
            100.0,
        )
    )


def assign_decision_status(
    conservative_rul,
    predicted_rul,
    confidence,
    disagreement_level,
    risk_score,
):
    """Assign a risk-aware operational status."""

    if (
        conservative_rul <= 20
        or predicted_rul <= 15
        or risk_score >= 70
    ):
        return "Critical"

    if (
        conservative_rul <= 40
        or predicted_rul <= 35
        or risk_score >= 50
    ):
        return "Maintenance required"

    if (
        confidence == "Low"
        or disagreement_level == "High"
        or risk_score >= 30
    ):
        return "Engineering review"

    if (
        conservative_rul <= 80
        or predicted_rul <= 80
        or disagreement_level == "Medium"
    ):
        return "Enhanced monitoring"

    return "Normal monitoring"


def assign_recommendation(
    decision_status,
):
    """Generate a practical maintenance recommendation."""

    recommendations = {
        "Critical": (
            "Inspect immediately and restrict continued operation "
            "until the engine condition is verified."
        ),
        "Maintenance required": (
            "Schedule maintenance at the earliest practical opportunity "
            "and increase inspection frequency."
        ),
        "Engineering review": (
            "Review sensor trends and model explanations before making "
            "a maintenance or continued-operation decision."
        ),
        "Enhanced monitoring": (
            "Continue operation with increased sensor monitoring and "
            "repeat the RUL assessment after additional cycles."
        ),
        "Normal monitoring": (
            "Continue normal operation and routine condition monitoring."
        ),
    }

    return recommendations[
        decision_status
    ]


def build_decision_results(
    uncertainty_df,
    ridge_df,
):
    """Combine prediction, uncertainty and model-disagreement evidence."""

    decision_df = uncertainty_df.merge(
        ridge_df,
        on="unit",
        how="inner",
        validate="one_to_one",
    )

    decision_df["model_disagreement"] = np.abs(
        decision_df["predicted_rul_mean"]
        - decision_df["ridge_prediction"]
    )

    decision_df["disagreement_level"] = (
        decision_df["model_disagreement"]
        .apply(
            classify_disagreement
        )
    )

    decision_df["conservative_rul"] = (
        decision_df.apply(
            lambda row: calculate_conservative_rul(
                lower_bound=row[
                    "calibrated_lower_bound"
                ],
                ridge_prediction=row[
                    "ridge_prediction"
                ],
            ),
            axis=1,
        )
    )

    decision_df["risk_score"] = (
        decision_df.apply(
            lambda row: calculate_risk_score(
                predicted_rul=row[
                    "predicted_rul_mean"
                ],
                conservative_rul=row[
                    "conservative_rul"
                ],
                interval_width=row[
                    "interval_width"
                ],
                confidence=row[
                    "confidence"
                ],
                disagreement=row[
                    "model_disagreement"
                ],
            ),
            axis=1,
        )
    )

    decision_df["decision_status"] = (
        decision_df.apply(
            lambda row: assign_decision_status(
                conservative_rul=row[
                    "conservative_rul"
                ],
                predicted_rul=row[
                    "predicted_rul_mean"
                ],
                confidence=row[
                    "confidence"
                ],
                disagreement_level=row[
                    "disagreement_level"
                ],
                risk_score=row[
                    "risk_score"
                ],
            ),
            axis=1,
        )
    )

    decision_df["recommendation"] = (
        decision_df[
            "decision_status"
        ].apply(
            assign_recommendation
        )
    )

    status_priority = {
        "Critical": 1,
        "Maintenance required": 2,
        "Engineering review": 3,
        "Enhanced monitoring": 4,
        "Normal monitoring": 5,
    }

    decision_df["status_priority"] = (
        decision_df[
            "decision_status"
        ].map(
            status_priority
        )
    )

    decision_df = decision_df.sort_values(
        by=[
            "status_priority",
            "risk_score",
        ],
        ascending=[
            True,
            False,
        ],
    ).reset_index(drop=True)

    return decision_df


def build_decision_summary(
    decision_df,
):
    """Summarize the number of engines in each decision category."""

    status_order = [
        "Critical",
        "Maintenance required",
        "Engineering review",
        "Enhanced monitoring",
        "Normal monitoring",
    ]

    rows = []

    for status in status_order:
        status_df = decision_df[
            decision_df["decision_status"]
            == status
        ]

        if status_df.empty:
            continue

        rows.append(
            {
                "decision_status": status,
                "engine_count": len(
                    status_df
                ),
                "percentage": (
                    len(status_df)
                    / len(decision_df)
                    * 100
                ),
                "average_predicted_rul": float(
                    status_df[
                        "predicted_rul_mean"
                    ].mean()
                ),
                "average_conservative_rul": float(
                    status_df[
                        "conservative_rul"
                    ].mean()
                ),
                "average_risk_score": float(
                    status_df[
                        "risk_score"
                    ].mean()
                ),
                "average_model_disagreement": float(
                    status_df[
                        "model_disagreement"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(rows)


def evaluate_decisions(
    decision_df,
):
    """
    Evaluate the decision layer retrospectively.

    Actual RUL is used only here for evaluation, never for assigning
    the operational status.
    """

    evaluation_rows = []

    for status, status_df in decision_df.groupby(
        "decision_status"
    ):
        evaluation_rows.append(
            {
                "decision_status": status,
                "engine_count": len(
                    status_df
                ),
                "average_actual_rul": float(
                    status_df[
                        "actual_rul"
                    ].mean()
                ),
                "average_absolute_error": float(
                    status_df[
                        "absolute_error"
                    ].mean()
                ),
                "actual_rul_below_20_rate": float(
                    (
                        status_df[
                            "actual_rul"
                        ]
                        <= 20
                    ).mean()
                ),
                "actual_rul_below_50_rate": float(
                    (
                        status_df[
                            "actual_rul"
                        ]
                        <= 50
                    ).mean()
                ),
                "dangerous_overprediction_rate": float(
                    (
                        status_df[
                            "prediction_error"
                        ]
                        > 10
                    ).mean()
                ),
            }
        )

    return pd.DataFrame(
        evaluation_rows
    )


def plot_decision_distribution(
    summary_df,
    model_name,
    dataset,
):
    """Plot the number of engines in each decision category."""

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        summary_df[
            "decision_status"
        ],
        summary_df[
            "engine_count"
        ],
    )

    plt.ylabel(
        "Number of test engines"
    )

    plt.xlabel(
        "Decision status"
    )

    plt.title(
        f"Risk-Aware Maintenance Decisions — "
        f"{model_name.upper()} {dataset}"
    )

    plt.xticks(
        rotation=15,
        ha="right",
    )

    plt.grid(
        axis="y",
        alpha=0.25,
    )

    plt.tight_layout()

    output_path = os.path.join(
        DECISION_DIR,
        f"decision_distribution_"
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
    """Plot prediction evidence for selected engines."""

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
            "No matching engines found for plotting."
        )

    plot_df = plot_df.sort_values(
        by="unit"
    ).reset_index(drop=True)

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
            for unit in plot_df["unit"]
        ],
    )

    plt.ylabel(
        "Remaining Useful Life"
    )

    plt.xlabel(
        "Test engine"
    )

    plt.title(
        f"Decision Evidence — "
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
    """Print decision information for selected engines."""

    if selected_units:
        display_df = decision_df[
            decision_df["unit"].isin(
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

    display_columns = [
        "unit",
        "actual_rul",
        "predicted_rul_mean",
        "ridge_prediction",
        "calibrated_lower_bound",
        "conservative_rul",
        "confidence",
        "model_disagreement",
        "disagreement_level",
        "risk_score",
        "decision_status",
        "recommendation",
    ]

    print(
        display_df[
            display_columns
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
    """Run the complete risk-aware decision pipeline."""

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

    summary_df = build_decision_summary(
        decision_df
    )

    evaluation_df = evaluate_decisions(
        decision_df
    )

    decisions_path = os.path.join(
        DECISION_DIR,
        f"decision_results_"
        f"{model_name}_{dataset}.csv",
    )

    summary_path = os.path.join(
        DECISION_DIR,
        f"decision_summary_"
        f"{model_name}_{dataset}.csv",
    )

    evaluation_path = os.path.join(
        DECISION_DIR,
        f"decision_evaluation_"
        f"{model_name}_{dataset}.csv",
    )

    decision_df.to_csv(
        decisions_path,
        index=False,
    )

    summary_df.to_csv(
        summary_path,
        index=False,
    )

    evaluation_df.to_csv(
        evaluation_path,
        index=False,
    )

    distribution_plot_path = (
        plot_decision_distribution(
            summary_df,
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
        "\nDecision summary"
    )

    print(
        summary_df.to_string(
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
        "\nRetrospective decision evaluation"
    )

    print(
        evaluation_df.to_string(
            index=False
        )
    )

    print(
        "\nSaved outputs"
    )

    print(
        f"Decisions:       {decisions_path}"
    )

    print(
        f"Summary:         {summary_path}"
    )

    print(
        f"Evaluation:      {evaluation_path}"
    )

    print(
        f"Distribution:    {distribution_plot_path}"
    )

    print(
        f"Selected units:  {selected_plot_path}"
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