import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import RESULTS_DIR


DATASET_NAME = "FD001"
MODEL_NAME = "gru"
TARGET_COVERAGE = 0.90
RECENT_WINDOW_LENGTH = 20


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
    """Validate columns required for uncertainty analysis."""

    required_columns = {
        "unit",
        "actual_rul",
        "predicted_rul_mean",
        "final_predicted_rul",
        "calibrated_lower_bound",
        "calibrated_upper_bound",
        "interval_width",
        "mc_standard_deviation",
        "absolute_error",
        "prediction_error",
        "engine_condition",
        "prediction_trust",
        "trajectory_flag",
        "trajectory_trust",
        "recent_slope",
        "recent_predicted_drop",
        "recent_decline_ratio",
        "recent_prediction_range",
        "recent_prediction_std",
        "recent_residual_std",
        "monotonicity_violation_rate",
        "large_jump_count",
        "largest_single_cycle_jump",
        "reliability_risk",
        "reliability_risk_score",
        "review_flag",
        "operational_priority",
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


def validate_trajectory_data(
    trajectory_df: pd.DataFrame,
) -> None:
    """Validate columns required for trajectory analysis."""

    required_columns = {
        "unit",
        "cycle",
        "predicted_rul",
        "actual_rul_capped",
    }

    missing_columns = sorted(
        required_columns.difference(
            trajectory_df.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "The trajectory file is missing columns: "
            f"{missing_columns}"
        )


decision_path = (
    RESULTS_DIR
    / "decisions_v5"
    / (
        f"decision_results_v5_"
        f"{MODEL_NAME}_{DATASET_NAME}.csv"
    )
)

trajectory_path = (
    RESULTS_DIR
    / "trajectory_diagnostics"
    / (
        f"trajectory_predictions_"
        f"{MODEL_NAME}_{DATASET_NAME}.csv"
    )
)


try:
    decision_df = load_csv(
        decision_path
    )

    trajectory_df = load_csv(
        trajectory_path
    )

    validate_decision_data(
        decision_df
    )

    validate_trajectory_data(
        trajectory_df
    )

except (
    FileNotFoundError,
    KeyError,
    ValueError,
) as error:
    st.title("Trajectory & Uncertainty")

    st.error(
        "Trajectory or uncertainty results could not be loaded."
    )

    st.code(
        str(error)
    )

    st.stop()


decision_df["unit"] = decision_df[
    "unit"
].astype(int)

trajectory_df["unit"] = trajectory_df[
    "unit"
].astype(int)


# ---------------------------------------------------------
# Engine selection
# ---------------------------------------------------------

engine_ids = sorted(
    decision_df["unit"].tolist()
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
    st.markdown("### Diagnostic selection")

    selected_unit = st.selectbox(
        "Test engine",
        options=engine_ids,
        index=default_index,
        format_func=lambda unit: f"Engine {unit}",
        key="trajectory_uncertainty_unit",
    )

    show_actual_curve = st.checkbox(
        "Show retrospective actual RUL",
        value=True,
        help=(
            "Actual RUL would not be known during "
            "real deployment."
        ),
    )

    recent_window_length = st.slider(
        "Recent diagnostic window",
        min_value=10,
        max_value=40,
        value=RECENT_WINDOW_LENGTH,
        step=5,
    )


selected_engine = (
    decision_df.loc[
        decision_df["unit"] == selected_unit
    ]
    .iloc[0]
)

engine_trajectory_df = (
    trajectory_df.loc[
        trajectory_df["unit"] == selected_unit
    ]
    .copy()
    .sort_values("cycle")
    .reset_index(drop=True)
)


if engine_trajectory_df.empty:
    st.error(
        f"No trajectory values were found for Engine {selected_unit}."
    )

    st.stop()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">Trajectory & Uncertainty</div>',
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="subtitle">
        Prediction evolution, temporal diagnostics and calibrated
        uncertainty for FD001 test Engine {selected_unit}
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Selected-engine headline metrics
# ---------------------------------------------------------

headline_columns = st.columns(
    6
)

headline_columns[0].metric(
    "Observed cycles",
    int(
        engine_trajectory_df[
            "cycle"
        ].max()
    ),
)

headline_columns[1].metric(
    "Final predicted RUL",
    (
        f"{selected_engine['final_predicted_rul']:.1f}"
    ),
    help="Cycles",
)

headline_columns[2].metric(
    "MC prediction mean",
    (
        f"{selected_engine['predicted_rul_mean']:.1f}"
    ),
    help="Cycles",
)

headline_columns[3].metric(
    "Trajectory diagnostic",
    selected_engine[
        "trajectory_flag"
    ],
)

headline_columns[4].metric(
    "Reliability risk",
    selected_engine[
        "reliability_risk"
    ],
)

headline_columns[5].metric(
    "Engineering review",
    selected_engine[
        "review_flag"
    ],
)


# ---------------------------------------------------------
# Full trajectory chart
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Remaining Useful Life Prediction History'
    '</div>',
    unsafe_allow_html=True,
)


final_cycle = int(
    engine_trajectory_df[
        "cycle"
    ].max()
)

effective_recent_window = min(
    recent_window_length,
    len(
        engine_trajectory_df
    ),
)

recent_start_index = (
    len(
        engine_trajectory_df
    )
    - effective_recent_window
)

recent_start_cycle = int(
    engine_trajectory_df.iloc[
        recent_start_index
    ][
        "cycle"
    ]
)


trajectory_figure = go.Figure()

trajectory_figure.add_trace(
    go.Scatter(
        x=engine_trajectory_df[
            "cycle"
        ],
        y=engine_trajectory_df[
            "predicted_rul"
        ],
        mode="lines",
        name="GRU predicted RUL",
        line=dict(
            width=3,
        ),
        hovertemplate=(
            "Cycle: %{x}<br>"
            "Predicted RUL: %{y:.1f}"
            "<extra></extra>"
        ),
    )
)


if show_actual_curve:
    trajectory_figure.add_trace(
        go.Scatter(
            x=engine_trajectory_df[
                "cycle"
            ],
            y=engine_trajectory_df[
                "actual_rul_capped"
            ],
            mode="lines",
            name="Actual capped RUL",
            line=dict(
                width=2,
                dash="dash",
            ),
            hovertemplate=(
                "Cycle: %{x}<br>"
                "Actual RUL: %{y:.1f}"
                "<extra></extra>"
            ),
        )
    )


trajectory_figure.add_trace(
    go.Scatter(
        x=[
            final_cycle
        ],
        y=[
            selected_engine[
                "final_predicted_rul"
            ]
        ],
        mode="markers",
        name="Final deterministic prediction",
        marker=dict(
            size=13,
            symbol="circle",
        ),
    )
)


if show_actual_curve:
    trajectory_figure.add_trace(
        go.Scatter(
            x=[
                final_cycle
            ],
            y=[
                selected_engine[
                    "actual_rul"
                ]
            ],
            mode="markers",
            name="Final actual RUL",
            marker=dict(
                size=15,
                symbol="x",
            ),
        )
    )


trajectory_figure.add_vrect(
    x0=recent_start_cycle,
    x1=final_cycle,
    opacity=0.10,
    line_width=0,
    annotation_text="Recent diagnostic window",
    annotation_position="top left",
)


trajectory_figure.update_layout(
    title=(
        f"RUL Prediction History — Engine {selected_unit}"
    ),
    xaxis_title="Observed operating cycle",
    yaxis_title="Remaining Useful Life (cycles)",
    hovermode="x unified",
    legend_title="Trajectory evidence",
    margin=dict(
        l=20,
        r=20,
        t=65,
        b=20,
    ),
)

st.plotly_chart(
    trajectory_figure,
    use_container_width=True,
)


if show_actual_curve:
    st.info(
        """
        The actual RUL curve is displayed only for retrospective
        NASA test evaluation. In real deployment, the future failure
        point and true remaining life would be unknown.
        """
    )


# ---------------------------------------------------------
# Temporal diagnostics
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Temporal Diagnostic Evidence'
    '</div>',
    unsafe_allow_html=True,
)


temporal_columns = st.columns(
    6
)

temporal_columns[0].metric(
    "Recent slope",
    (
        f"{selected_engine['recent_slope']:.3f}"
    ),
    help="Predicted RUL change per observed cycle.",
)

temporal_columns[1].metric(
    "Recent predicted drop",
    (
        f"{selected_engine['recent_predicted_drop']:.1f}"
    ),
    help="Cycles",
)

temporal_columns[2].metric(
    "Recent decline ratio",
    (
        f"{selected_engine['recent_decline_ratio']:.2f}"
    ),
)

temporal_columns[3].metric(
    "Prediction range",
    (
        f"{selected_engine['recent_prediction_range']:.1f}"
    ),
    help="Cycles",
)

temporal_columns[4].metric(
    "Monotonicity violations",
    (
        f"{selected_engine['monotonicity_violation_rate'] * 100:.1f}%"
    ),
)

temporal_columns[5].metric(
    "Large jumps",
    int(
        selected_engine[
            "large_jump_count"
        ]
    ),
)


secondary_temporal_columns = st.columns(
    4
)

secondary_temporal_columns[0].metric(
    "Recent prediction std.",
    (
        f"{selected_engine['recent_prediction_std']:.2f}"
    ),
)

secondary_temporal_columns[1].metric(
    "Recent residual std.",
    (
        f"{selected_engine['recent_residual_std']:.2f}"
    ),
)

secondary_temporal_columns[2].metric(
    "Largest single-cycle jump",
    (
        f"{selected_engine['largest_single_cycle_jump']:.1f}"
    ),
    help="Cycles",
)

secondary_temporal_columns[3].metric(
    "Trajectory trust",
    selected_engine[
        "trajectory_trust"
    ],
)


trajectory_explanations = {
    "Consistent degradation": (
        "The estimated RUL generally decreases as additional "
        "operating cycles are observed."
    ),
    "Rapid degradation signal": (
        "The predicted RUL is declining unusually quickly. "
        "Recent sensor evidence may indicate accelerating degradation."
    ),
    "Mixed trajectory": (
        "The prediction history combines stable and degrading "
        "behaviour, reducing confidence in the temporal trend."
    ),
    "Weak degradation / plateau": (
        "The model shows only weak decline despite continued "
        "engine ageing."
    ),
    "High-RUL plateau — review": (
        "The model remains on a high-RUL plateau while the engine "
        "continues ageing. This can indicate a shared model blind spot."
    ),
    "RUL increasing despite ageing": (
        "The estimated remaining life increases while the engine "
        "gets older, which is physically suspicious."
    ),
    "Unstable trajectory": (
        "The predicted RUL contains large jumps or excessive "
        "oscillations and should not be interpreted from the final "
        "point alone."
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
# Recent trajectory inspection
# ---------------------------------------------------------

with st.expander(
    "Inspect recent cycle-level predictions"
):
    recent_display_columns = [
        "cycle",
        "predicted_rul",
        "actual_rul_capped",
    ]

    optional_columns = [
        "prediction_error",
        "absolute_error",
    ]

    for column in optional_columns:
        if column in engine_trajectory_df.columns:
            recent_display_columns.append(
                column
            )

    recent_table = (
        engine_trajectory_df[
            recent_display_columns
        ]
        .tail(
            effective_recent_window
        )
        .copy()
        .rename(
            columns={
                "cycle": "Cycle",
                "predicted_rul": "Predicted RUL",
                "actual_rul_capped": "Actual capped RUL",
                "prediction_error": "Prediction error",
                "absolute_error": "Absolute error",
            }
        )
    )

    numeric_columns = [
        column
        for column in recent_table.columns
        if column != "Cycle"
    ]

    recent_table[
        numeric_columns
    ] = recent_table[
        numeric_columns
    ].round(
        2
    )

    st.dataframe(
        recent_table,
        use_container_width=True,
        hide_index=True,
        height=450,
    )


# ---------------------------------------------------------
# Selected-engine calibrated uncertainty
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Selected-Engine Uncertainty'
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


uncertainty_columns = st.columns(
    5
)

uncertainty_columns[0].metric(
    "Lower bound",
    f"{lower_bound:.1f}",
    help="Cycles",
)

uncertainty_columns[1].metric(
    "Prediction mean",
    f"{prediction_mean:.1f}",
    help="Cycles",
)

uncertainty_columns[2].metric(
    "Upper bound",
    f"{upper_bound:.1f}",
    help="Cycles",
)

uncertainty_columns[3].metric(
    "Interval width",
    (
        f"{selected_engine['interval_width']:.1f}"
    ),
    help="Cycles",
)

uncertainty_columns[4].metric(
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
        name="Prediction interval",
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
    )
)

interval_figure.update_layout(
    title=(
        f"Calibrated RUL Interval — Engine {selected_unit}"
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
# Fleet-wide uncertainty analysis
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Fleet-Wide Uncertainty Calibration'
    '</div>',
    unsafe_allow_html=True,
)


uncertainty_df = decision_df.copy()

uncertainty_df[
    "interval_contains_actual"
] = (
    (
        uncertainty_df[
            "actual_rul"
        ]
        >= uncertainty_df[
            "calibrated_lower_bound"
        ]
    )
    & (
        uncertainty_df[
            "actual_rul"
        ]
        <= uncertainty_df[
            "calibrated_upper_bound"
        ]
    )
)


observed_coverage = float(
    uncertainty_df[
        "interval_contains_actual"
    ].mean()
)

average_interval_width = float(
    uncertainty_df[
        "interval_width"
    ].mean()
)

average_mc_std = float(
    uncertainty_df[
        "mc_standard_deviation"
    ].mean()
)

pearson_width_error = float(
    uncertainty_df[
        "interval_width"
    ].corr(
        uncertainty_df[
            "absolute_error"
        ]
    )
)

spearman_width_error = float(
    uncertainty_df[
        "interval_width"
    ].rank().corr(
        uncertainty_df[
            "absolute_error"
        ].rank()
    )
)


fleet_uncertainty_columns = st.columns(
    5
)

fleet_uncertainty_columns[0].metric(
    "Observed coverage",
    f"{observed_coverage * 100:.1f}%",
)

fleet_uncertainty_columns[1].metric(
    "Target coverage",
    f"{TARGET_COVERAGE * 100:.1f}%",
)

fleet_uncertainty_columns[2].metric(
    "Average interval width",
    f"{average_interval_width:.1f}",
    help="Cycles",
)

fleet_uncertainty_columns[3].metric(
    "Average MC deviation",
    f"{average_mc_std:.1f}",
    help="Cycles",
)

fleet_uncertainty_columns[4].metric(
    "Width–error correlation",
    f"{pearson_width_error:.3f}",
    help="Pearson correlation",
)


coverage_gap = (
    observed_coverage
    - TARGET_COVERAGE
)

if abs(
    coverage_gap
) <= 0.03:
    st.success(
        "Observed coverage is close to the requested "
        "90% calibration target."
    )

elif coverage_gap < 0:
    st.warning(
        "Observed coverage is below the requested target. "
        "The calibrated intervals may be slightly too narrow."
    )

else:
    st.info(
        "Observed coverage exceeds the requested target. "
        "The intervals may be conservative."
    )


# ---------------------------------------------------------
# Uncertainty versus error
# ---------------------------------------------------------

left_column, right_column = st.columns(
    2
)


with left_column:
    uncertainty_error_figure = px.scatter(
        uncertainty_df,
        x="interval_width",
        y="absolute_error",
        color="interval_contains_actual",
        hover_name="unit",
        hover_data={
            "actual_rul": ":.1f",
            "predicted_rul_mean": ":.1f",
            "mc_standard_deviation": ":.2f",
            "trajectory_flag": True,
            "review_flag": True,
        },
        labels={
            "interval_width": (
                "Calibrated interval width (cycles)"
            ),
            "absolute_error": (
                "Absolute prediction error (cycles)"
            ),
            "interval_contains_actual": (
                "Interval covered actual"
            ),
        },
        title=(
            "Interval Width vs Absolute Error"
        ),
    )

    uncertainty_error_figure.update_layout(
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        uncertainty_error_figure,
        use_container_width=True,
    )


with right_column:
    mc_error_figure = px.scatter(
        uncertainty_df,
        x="mc_standard_deviation",
        y="absolute_error",
        color="reliability_risk",
        hover_name="unit",
        hover_data={
            "actual_rul": ":.1f",
            "predicted_rul_mean": ":.1f",
            "interval_width": ":.1f",
            "trajectory_flag": True,
            "review_flag": True,
        },
        labels={
            "mc_standard_deviation": (
                "MC standard deviation (cycles)"
            ),
            "absolute_error": (
                "Absolute prediction error (cycles)"
            ),
            "reliability_risk": (
                "Reliability risk"
            ),
        },
        title=(
            "MC Dropout Variation vs Absolute Error"
        ),
    )

    mc_error_figure.update_layout(
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        mc_error_figure,
        use_container_width=True,
    )


st.info(
    f"""
    Interval width has a Pearson correlation of
    **{pearson_width_error:.3f}** and a Spearman correlation of
    **{spearman_width_error:.3f}** with absolute prediction error.
    Positive values indicate that wider intervals tend to occur on
    more difficult engines, although uncertainty is not a perfect
    error detector.
    """
)


# ---------------------------------------------------------
# Calibration by actual-RUL region
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Calibration Across Engine-Life Regions'
    '</div>',
    unsafe_allow_html=True,
)


uncertainty_df[
    "actual_rul_region"
] = pd.cut(
    uncertainty_df[
        "actual_rul"
    ],
    bins=[
        float("-inf"),
        20,
        50,
        80,
        float("inf"),
    ],
    labels=[
        "Critical: 0–20",
        "Warning: 21–50",
        "Monitor: 51–80",
        "Healthy: 81+",
    ],
)


region_uncertainty_df = (
    uncertainty_df.groupby(
        "actual_rul_region",
        observed=True,
        as_index=False,
    )
    .agg(
        engine_count=(
            "unit",
            "count",
        ),
        coverage=(
            "interval_contains_actual",
            "mean",
        ),
        average_interval_width=(
            "interval_width",
            "mean",
        ),
        average_absolute_error=(
            "absolute_error",
            "mean",
        ),
    )
)


region_uncertainty_df[
    "coverage_percentage"
] = (
    region_uncertainty_df[
        "coverage"
    ]
    * 100.0
)


left_column, right_column = st.columns(
    2
)


with left_column:
    coverage_figure = px.bar(
        region_uncertainty_df,
        x="actual_rul_region",
        y="coverage_percentage",
        text="coverage_percentage",
        hover_data={
            "engine_count": True,
            "average_interval_width": ":.1f",
            "average_absolute_error": ":.1f",
        },
        labels={
            "actual_rul_region": (
                "Actual RUL region"
            ),
            "coverage_percentage": (
                "Observed coverage (%)"
            ),
        },
        title=(
            "Prediction-Interval Coverage by RUL Region"
        ),
    )

    coverage_figure.add_hline(
        y=90,
        line_dash="dash",
        annotation_text="90% target",
        opacity=0.6,
    )

    coverage_figure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    coverage_figure.update_layout(
        xaxis_title="",
        showlegend=False,
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        coverage_figure,
        use_container_width=True,
    )


with right_column:
    width_figure = px.bar(
        region_uncertainty_df,
        x="actual_rul_region",
        y="average_interval_width",
        text="average_interval_width",
        hover_data={
            "engine_count": True,
            "coverage_percentage": ":.1f",
            "average_absolute_error": ":.1f",
        },
        labels={
            "actual_rul_region": (
                "Actual RUL region"
            ),
            "average_interval_width": (
                "Average interval width (cycles)"
            ),
        },
        title=(
            "Average Interval Width by RUL Region"
        ),
    )

    width_figure.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
    )

    width_figure.update_layout(
        xaxis_title="",
        showlegend=False,
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        width_figure,
        use_container_width=True,
    )


# ---------------------------------------------------------
# Fleet uncertainty table
# ---------------------------------------------------------

with st.expander(
    "Inspect complete fleet uncertainty results"
):
    uncertainty_table = (
        uncertainty_df[
            [
                "unit",
                "actual_rul",
                "predicted_rul_mean",
                "mc_standard_deviation",
                "calibrated_lower_bound",
                "calibrated_upper_bound",
                "interval_width",
                "absolute_error",
                "interval_contains_actual",
                "trajectory_flag",
                "review_flag",
            ]
        ]
        .copy()
        .rename(
            columns={
                "unit": "Engine",
                "actual_rul": "Actual RUL",
                "predicted_rul_mean": "Predicted RUL",
                "mc_standard_deviation": (
                    "MC standard deviation"
                ),
                "calibrated_lower_bound": (
                    "Lower bound"
                ),
                "calibrated_upper_bound": (
                    "Upper bound"
                ),
                "interval_width": (
                    "Interval width"
                ),
                "absolute_error": (
                    "Absolute error"
                ),
                "interval_contains_actual": (
                    "Interval covered actual"
                ),
                "trajectory_flag": (
                    "Trajectory diagnostic"
                ),
                "review_flag": (
                    "Engineering review"
                ),
            }
        )
    )

    numeric_columns = [
        "Actual RUL",
        "Predicted RUL",
        "MC standard deviation",
        "Lower bound",
        "Upper bound",
        "Interval width",
        "Absolute error",
    ]

    uncertainty_table[
        numeric_columns
    ] = uncertainty_table[
        numeric_columns
    ].round(
        2
    )

    st.dataframe(
        uncertainty_table,
        use_container_width=True,
        hide_index=True,
        height=500,
    )


st.warning(
    """
    Monte Carlo dropout intervals represent uncertainty within the
    trained model. They do not capture every real-world risk, including
    unseen fault modes, sensor failures, maintenance history, domain
    shift or differences between simulated and physical engines.
    """
)

st.caption(
    f"Trajectory source: {trajectory_path}"
)

st.caption(
    f"Decision and uncertainty source: {decision_path}"
)