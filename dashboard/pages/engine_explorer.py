import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import RESULTS_DIR


DATASET_NAME = "FD001"
MODEL_NAME = "gru"


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

@st.cache_data
def load_csv(file_path):
    """Load and cache a CSV file."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{file_path}"
        )

    return pd.read_csv(file_path)


def validate_decision_data(
    decision_df: pd.DataFrame,
) -> None:
    """Validate columns required by the Engine Explorer."""

    required_columns = {
        "unit",
        "actual_rul",
        "predicted_rul_mean",
        "final_predicted_rul",
        "ridge_prediction",
        "calibrated_lower_bound",
        "calibrated_upper_bound",
        "interval_width",
        "mc_standard_deviation",
        "conservative_rul",
        "absolute_error",
        "prediction_error",
        "engine_condition",
        "condition_severity_score",
        "prediction_trust",
        "trajectory_flag",
        "trajectory_trust",
        "recent_slope",
        "recent_predicted_drop",
        "recent_prediction_range",
        "monotonicity_violation_rate",
        "large_jump_count",
        "model_disagreement",
        "disagreement_level",
        "mc_deterministic_gap",
        "mc_gap_level",
        "reliability_risk_score",
        "reliability_risk",
        "review_flag",
        "operational_priority_score",
        "operational_priority",
        "operational_action",
    }

    missing_columns = sorted(
        required_columns.difference(
            decision_df.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "The V5 decision file is missing columns: "
            f"{missing_columns}"
        )

    if decision_df["unit"].duplicated().any():
        raise ValueError(
            "Duplicate engine IDs were found."
        )


decision_path = (
    RESULTS_DIR
    / "decisions_v5"
    / (
        f"decision_results_v5_"
        f"{MODEL_NAME}_{DATASET_NAME}.csv"
    )
)


try:
    decision_df = load_csv(
        decision_path
    )

    validate_decision_data(
        decision_df
    )

except (
    FileNotFoundError,
    KeyError,
    ValueError,
) as error:
    st.title("Engine Explorer")

    st.error(
        "Engine-level decision results could not be loaded."
    )

    st.code(
        str(error)
    )

    st.stop()


decision_df["unit"] = decision_df[
    "unit"
].astype(int)


# ---------------------------------------------------------
# Engine selection
# ---------------------------------------------------------

engine_ids = sorted(
    decision_df[
        "unit"
    ].tolist()
)

default_unit = (
    67
    if 67 in engine_ids
    else engine_ids[0]
)

default_index = engine_ids.index(
    default_unit
)


with st.sidebar:
    st.markdown("---")
    st.markdown("### Engine selection")

    selected_unit = st.selectbox(
        "Test engine",
        options=engine_ids,
        index=default_index,
        format_func=lambda unit: f"Engine {unit}",
        help=(
            "Select one of the 100 unseen FD001 test engines."
        ),
        key="engine_explorer_unit",
    )

    st.caption(
        "Engine 67 is the shared-model blind-spot case."
    )


selected_engine = (
    decision_df.loc[
        decision_df["unit"] == selected_unit
    ]
    .iloc[0]
)


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">Engine Explorer</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="subtitle">
        Detailed prediction evidence, uncertainty and engineering
        decision for FD001 test Engine {selected_unit}
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Headline decision metrics
# ---------------------------------------------------------

metric_columns = st.columns(
    6
)

metric_columns[0].metric(
    "Predicted RUL",
    (
        f"{selected_engine['predicted_rul_mean']:.1f}"
    ),
    help="GRU Monte Carlo dropout mean, in cycles.",
)

metric_columns[1].metric(
    "Conservative RUL",
    (
        f"{selected_engine['conservative_rul']:.1f}"
    ),
    help=(
        "More cautious estimate based on the calibrated "
        "lower bound and Ridge prediction."
    ),
)

metric_columns[2].metric(
    "Condition",
    selected_engine[
        "engine_condition"
    ],
)

metric_columns[3].metric(
    "Reliability risk",
    selected_engine[
        "reliability_risk"
    ],
)

metric_columns[4].metric(
    "Engineering review",
    selected_engine[
        "review_flag"
    ],
)

metric_columns[5].metric(
    "Operational priority",
    selected_engine[
        "operational_priority"
    ],
)


# ---------------------------------------------------------
# Operational recommendation
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Operational Recommendation'
    '</div>',
    unsafe_allow_html=True,
)

review_flag = selected_engine[
    "review_flag"
]

operational_action = selected_engine[
    "operational_action"
]

if review_flag == "Required":
    st.error(
        operational_action
    )

elif review_flag == "Recommended":
    st.warning(
        operational_action
    )

else:
    st.success(
        operational_action
    )


# ---------------------------------------------------------
# Prediction evidence
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Prediction Evidence'
    '</div>',
    unsafe_allow_html=True,
)


evidence_df = pd.DataFrame(
    [
        {
            "Estimate": "GRU MC-dropout mean",
            "RUL": selected_engine[
                "predicted_rul_mean"
            ],
        },
        {
            "Estimate": "GRU deterministic",
            "RUL": selected_engine[
                "final_predicted_rul"
            ],
        },
        {
            "Estimate": "Ridge baseline",
            "RUL": selected_engine[
                "ridge_prediction"
            ],
        },
        {
            "Estimate": "Conservative RUL",
            "RUL": selected_engine[
                "conservative_rul"
            ],
        },
        {
            "Estimate": "Actual RUL",
            "RUL": selected_engine[
                "actual_rul"
            ],
        },
    ]
)


left_column, right_column = st.columns(
    [1.1, 0.9]
)


with left_column:
    evidence_figure = px.bar(
        evidence_df,
        x="Estimate",
        y="RUL",
        text="RUL",
        title=(
            f"RUL Estimates — Engine {selected_unit}"
        ),
    )

    evidence_figure.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
    )

    evidence_figure.update_layout(
        xaxis_title="",
        yaxis_title="Remaining Useful Life (cycles)",
        showlegend=False,
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        evidence_figure,
        use_container_width=True,
    )


with right_column:
    st.markdown("#### Decision evidence")

    interpretation_df = pd.DataFrame(
        {
            "Evidence": [
                "Prediction trust",
                "Trajectory trust",
                "Trajectory diagnostic",
                "Model disagreement",
                "Disagreement level",
                "MC vs deterministic gap",
                "MC-gap level",
                "Condition severity score",
                "Reliability-risk score",
                "Operational-priority score",
            ],
            "Result": [
                selected_engine[
                    "prediction_trust"
                ],
                selected_engine[
                    "trajectory_trust"
                ],
                selected_engine[
                    "trajectory_flag"
                ],
                (
                    f"{selected_engine['model_disagreement']:.1f} cycles"
                ),
                selected_engine[
                    "disagreement_level"
                ],
                (
                    f"{selected_engine['mc_deterministic_gap']:.1f} cycles"
                ),
                selected_engine[
                    "mc_gap_level"
                ],
                (
                    f"{selected_engine['condition_severity_score']:.1f}"
                ),
                (
                    f"{selected_engine['reliability_risk_score']:.1f}"
                ),
                (
                    f"{selected_engine['operational_priority_score']:.1f}"
                ),
            ],
        }
    )

    st.dataframe(
        interpretation_df,
        use_container_width=True,
        hide_index=True,
        height=420,
    )


# ---------------------------------------------------------
# Calibrated uncertainty interval
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Calibrated Prediction Interval'
    '</div>',
    unsafe_allow_html=True,
)


lower_bound = float(
    selected_engine[
        "calibrated_lower_bound"
    ]
)

upper_bound = float(
    selected_engine[
        "calibrated_upper_bound"
    ]
)

prediction_mean = float(
    selected_engine[
        "predicted_rul_mean"
    ]
)

actual_rul = float(
    selected_engine[
        "actual_rul"
    ]
)


interval_metrics = st.columns(
    5
)

interval_metrics[0].metric(
    "Lower bound",
    f"{lower_bound:.1f}",
    help="Cycles",
)

interval_metrics[1].metric(
    "Prediction mean",
    f"{prediction_mean:.1f}",
    help="Cycles",
)

interval_metrics[2].metric(
    "Upper bound",
    f"{upper_bound:.1f}",
    help="Cycles",
)

interval_metrics[3].metric(
    "Interval width",
    (
        f"{selected_engine['interval_width']:.1f}"
    ),
    help="Cycles",
)

interval_metrics[4].metric(
    "MC standard deviation",
    (
        f"{selected_engine['mc_standard_deviation']:.1f}"
    ),
    help="Cycles",
)


interval_figure = go.Figure()

interval_figure.add_trace(
    go.Scatter(
        x=[
            lower_bound,
            upper_bound,
        ],
        y=[
            "Calibrated interval",
            "Calibrated interval",
        ],
        mode="lines",
        name="Calibrated interval",
        line=dict(
            width=14,
        ),
        hovertemplate=(
            f"Interval: {lower_bound:.1f}–"
            f"{upper_bound:.1f} cycles"
            "<extra></extra>"
        ),
    )
)

interval_figure.add_trace(
    go.Scatter(
        x=[
            prediction_mean
        ],
        y=[
            "Calibrated interval"
        ],
        mode="markers",
        name="Predicted mean",
        marker=dict(
            size=15,
            symbol="circle",
        ),
        hovertemplate=(
            "Predicted mean: %{x:.1f} cycles"
            "<extra></extra>"
        ),
    )
)

interval_figure.add_trace(
    go.Scatter(
        x=[
            actual_rul
        ],
        y=[
            "Calibrated interval"
        ],
        mode="markers",
        name="Actual RUL",
        marker=dict(
            size=16,
            symbol="x",
        ),
        hovertemplate=(
            "Actual RUL: %{x:.1f} cycles"
            "<extra></extra>"
        ),
    )
)

interval_figure.update_layout(
    title=(
        f"GRU Prediction Interval — Engine {selected_unit}"
    ),
    xaxis_title="Remaining Useful Life (cycles)",
    yaxis_title="",
    height=330,
    margin=dict(
        l=20,
        r=20,
        t=55,
        b=20,
    ),
)

st.plotly_chart(
    interval_figure,
    use_container_width=True,
)


actual_is_covered = (
    lower_bound
    <= actual_rul
    <= upper_bound
)

if actual_is_covered:
    st.success(
        "The calibrated interval contains the actual RUL "
        "for this retrospective test engine."
    )

else:
    st.error(
        "The actual RUL falls outside the calibrated "
        "prediction interval for this engine."
    )


# ---------------------------------------------------------
# Temporal behaviour
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Temporal Diagnostic Evidence'
    '</div>',
    unsafe_allow_html=True,
)


trajectory_metrics = st.columns(
    6
)

trajectory_metrics[0].metric(
    "Trajectory diagnostic",
    selected_engine[
        "trajectory_flag"
    ],
)

trajectory_metrics[1].metric(
    "Recent slope",
    (
        f"{selected_engine['recent_slope']:.3f}"
    ),
    help="Predicted RUL change per operating cycle.",
)

trajectory_metrics[2].metric(
    "Recent predicted drop",
    (
        f"{selected_engine['recent_predicted_drop']:.1f}"
    ),
    help="Cycles",
)

trajectory_metrics[3].metric(
    "Recent prediction range",
    (
        f"{selected_engine['recent_prediction_range']:.1f}"
    ),
    help="Cycles",
)

trajectory_metrics[4].metric(
    "Monotonicity violations",
    (
        f"{selected_engine['monotonicity_violation_rate'] * 100:.1f}%"
    ),
)

trajectory_metrics[5].metric(
    "Large jumps",
    int(
        selected_engine[
            "large_jump_count"
        ]
    ),
)


trajectory_explanations = {
    "Consistent degradation": (
        "The predicted RUL generally declines as the engine "
        "accumulates operating cycles."
    ),
    "Rapid degradation signal": (
        "The predicted RUL is declining unusually quickly and "
        "should be checked against recent sensor trends."
    ),
    "Mixed trajectory": (
        "The trajectory contains both stable and degrading "
        "behaviour, reducing confidence in the temporal pattern."
    ),
    "Weak degradation / plateau": (
        "The estimated RUL changes very little despite "
        "continued engine ageing."
    ),
    "High-RUL plateau — review": (
        "The model remains at a high RUL despite continued "
        "ageing, indicating a possible shared model blind spot."
    ),
    "RUL increasing despite ageing": (
        "The model predicts increasing life while the engine "
        "gets older, which is physically suspicious."
    ),
    "Unstable trajectory": (
        "The predicted RUL contains large jumps or excessive "
        "oscillations."
    ),
}


trajectory_flag = selected_engine[
    "trajectory_flag"
]

trajectory_message = trajectory_explanations.get(
    trajectory_flag,
    "The prediction history requires engineering examination.",
)


if trajectory_flag in {
    "High-RUL plateau — review",
    "RUL increasing despite ageing",
    "Unstable trajectory",
}:
    st.error(
        f"**{trajectory_flag}** — {trajectory_message}"
    )

elif trajectory_flag in {
    "Rapid degradation signal",
    "Mixed trajectory",
    "Weak degradation / plateau",
}:
    st.warning(
        f"**{trajectory_flag}** — {trajectory_message}"
    )

else:
    st.success(
        f"**{trajectory_flag}** — {trajectory_message}"
    )


# ---------------------------------------------------------
# Retrospective prediction error
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Retrospective Evaluation'
    '</div>',
    unsafe_allow_html=True,
)


error_columns = st.columns(
    4
)

error_columns[0].metric(
    "Actual RUL",
    (
        f"{selected_engine['actual_rul']:.1f}"
    ),
    help="Cycles",
)

error_columns[1].metric(
    "Signed prediction error",
    (
        f"{selected_engine['prediction_error']:+.1f}"
    ),
    help=(
        "Positive means the model overpredicted "
        "remaining life."
    ),
)

error_columns[2].metric(
    "Absolute error",
    (
        f"{selected_engine['absolute_error']:.1f}"
    ),
    help="Cycles",
)

dangerous_overprediction = (
    selected_engine[
        "prediction_error"
    ]
    > 10.0
)

error_columns[3].metric(
    "Dangerous overprediction",
    (
        "Yes"
        if dangerous_overprediction
        else "No"
    ),
    help=(
        "Defined retrospectively as overprediction "
        "greater than 10 cycles."
    ),
)


st.warning(
    """
    Actual RUL and prediction error are shown only for retrospective
    evaluation of the NASA test set. In real deployment, the future
    failure point would be unknown.
    """
)


# ---------------------------------------------------------
# Complete selected-engine record
# ---------------------------------------------------------

with st.expander(
    "Inspect the complete Engine Explorer record"
):
    selected_record = (
        selected_engine
        .to_frame(
            name="Value"
        )
        .reset_index()
        .rename(
            columns={
                "index": "Field",
            }
        )
    )

    st.dataframe(
        selected_record,
        use_container_width=True,
        hide_index=True,
        height=520,
    )


st.caption(
    f"Decision source: {decision_path}"
)