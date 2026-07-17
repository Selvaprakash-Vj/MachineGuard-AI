from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------

st.set_page_config(
    page_title="MachineGuard AI",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_ROOT / "results_v2"


# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.4rem;
            font-weight: 750;
            margin-bottom: 0.15rem;
        }

        .subtitle {
            font-size: 1rem;
            color: #6b7280;
            margin-bottom: 1.5rem;
        }

        .section-heading {
            font-size: 1.35rem;
            font-weight: 700;
            margin-top: 1rem;
            margin-bottom: 0.8rem;
        }

        div[data-testid="stMetric"] {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 12px;
            padding: 14px;
        }

        .info-box {
            border: 1px solid rgba(128, 128, 128, 0.25);
            border-radius: 12px;
            padding: 16px;
            margin-bottom: 16px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Data loading
# ---------------------------------------------------------

@st.cache_data
def load_csv(file_path: Path) -> pd.DataFrame:
    """Load a CSV file and cache it for dashboard use."""

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file was not found:\n{file_path}"
        )

    return pd.read_csv(file_path)


def get_decision_results_path(
    model_name: str,
    dataset_name: str,
) -> Path:
    """Return the V5 decision-result path."""

    return (
        RESULTS_DIR
        / "decisions_v5"
        / (
            f"decision_results_v5_"
            f"{model_name}_{dataset_name}.csv"
        )
    )


def validate_decision_data(
    decision_df: pd.DataFrame,
) -> None:
    """Validate the minimum columns needed by the dashboard."""

    required_columns = {
        "unit",
        "actual_rul",
        "predicted_rul_mean",
        "conservative_rul",
        "engine_condition",
        "reliability_risk",
        "reliability_risk_score",
        "review_flag",
        "operational_priority",
        "operational_priority_score",
        "trajectory_flag",
    }

    missing_columns = sorted(
        required_columns.difference(
            decision_df.columns
        )
    )

    if missing_columns:
        raise KeyError(
            "The V5 decision file is missing these columns: "
            f"{missing_columns}"
        )

    if decision_df["unit"].duplicated().any():
        raise ValueError(
            "Duplicate engine IDs were found in the V5 decision file."
        )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.title("⚙️ MachineGuard AI")

    st.caption(
        "NASA C-MAPSS predictive-maintenance platform"
    )

    st.divider()

    dataset_name = st.selectbox(
        "Dataset",
        options=["FD001"],
        index=0,
        help=(
            "FD001 contains one operating condition "
            "and one fault mode."
        ),
    )

    model_name = st.selectbox(
        "Primary model",
        options=["gru"],
        format_func=lambda value: value.upper(),
        index=0,
    )

    st.divider()

    st.markdown("### Current system")

    st.markdown(
        """
        - Primary model: **GRU**
        - Independent baseline: **Ridge**
        - Decision layer: **V5**
        - Sequence length: **30 cycles**
        - RUL cap: **125 cycles**
        """
    )


# ---------------------------------------------------------
# Load selected dataset
# ---------------------------------------------------------

decision_path = get_decision_results_path(
    model_name=model_name,
    dataset_name=dataset_name,
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
    st.error(
        "MachineGuard AI could not load the required results."
    )

    st.code(
        str(error)
    )

    st.info(
        "Run the V5 decision layer before launching the dashboard."
    )

    st.stop()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">MachineGuard AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Explainable, uncertainty-aware Remaining Useful Life
        prediction and engineering decision support
    </div>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# Fleet overview metrics
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">Fleet Overview</div>',
    unsafe_allow_html=True,
)

total_engines = len(
    decision_df
)

critical_engines = int(
    (
        decision_df["engine_condition"]
        == "Critical"
    ).sum()
)

required_reviews = int(
    (
        decision_df["review_flag"]
        == "Required"
    ).sum()
)

high_reliability_risk = int(
    (
        decision_df["reliability_risk"]
        == "High"
    ).sum()
)

average_predicted_rul = float(
    decision_df[
        "predicted_rul_mean"
    ].mean()
)

metric_columns = st.columns(
    5
)

metric_columns[0].metric(
    label="Test engines",
    value=total_engines,
)

metric_columns[1].metric(
    label="Critical condition",
    value=critical_engines,
)

metric_columns[2].metric(
    label="Mandatory reviews",
    value=required_reviews,
)

metric_columns[3].metric(
    label="High reliability risk",
    value=high_reliability_risk,
)

metric_columns[4].metric(
    label="Average predicted RUL",
    value=f"{average_predicted_rul:.1f} cycles",
)


# ---------------------------------------------------------
# Fleet distribution charts
# ---------------------------------------------------------

left_column, right_column = st.columns(
    2
)

condition_order = [
    "Critical",
    "Warning",
    "Monitor",
    "Healthy",
]

review_order = [
    "Required",
    "Recommended",
    "Not required",
]

condition_counts = (
    decision_df[
        "engine_condition"
    ]
    .value_counts()
    .reindex(
        condition_order,
        fill_value=0,
    )
    .rename_axis(
        "Engine condition"
    )
    .reset_index(
        name="Engine count"
    )
)

review_counts = (
    decision_df[
        "review_flag"
    ]
    .value_counts()
    .reindex(
        review_order,
        fill_value=0,
    )
    .rename_axis(
        "Review decision"
    )
    .reset_index(
        name="Engine count"
    )
)

with left_column:
    condition_figure = px.bar(
        condition_counts,
        x="Engine condition",
        y="Engine count",
        title="Estimated Engine Condition",
        text="Engine count",
    )

    condition_figure.update_layout(
        showlegend=False,
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    condition_figure.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        condition_figure,
        use_container_width=True,
    )

with right_column:
    review_figure = px.bar(
        review_counts,
        x="Review decision",
        y="Engine count",
        title="Engineering Review Workload",
        text="Engine count",
    )

    review_figure.update_layout(
        showlegend=False,
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    review_figure.update_traces(
        textposition="outside"
    )

    st.plotly_chart(
        review_figure,
        use_container_width=True,
    )


# ---------------------------------------------------------
# Reliability and condition map
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Condition Severity vs Prediction Reliability'
    '</div>',
    unsafe_allow_html=True,
)

if "condition_severity_score" in decision_df.columns:
    score_map = px.scatter(
        decision_df,
        x="condition_severity_score",
        y="reliability_risk_score",
        hover_name="unit",
        hover_data={
            "unit": True,
            "engine_condition": True,
            "review_flag": True,
            "operational_priority": True,
            "trajectory_flag": True,
            "condition_severity_score": ":.1f",
            "reliability_risk_score": ":.1f",
        },
        labels={
            "condition_severity_score": (
                "Condition severity score"
            ),
            "reliability_risk_score": (
                "Prediction reliability risk"
            ),
        },
        title=(
            "Each point represents one FD001 test engine"
        ),
    )

    score_map.add_vline(
        x=60,
        line_dash="dash",
        opacity=0.4,
    )

    score_map.add_hline(
        y=60,
        line_dash="dash",
        opacity=0.4,
    )

    score_map.update_layout(
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        score_map,
        use_container_width=True,
    )

else:
    st.warning(
        "Condition severity scores are not available."
    )


# ---------------------------------------------------------
# Fleet table
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">Fleet Decision Table</div>',
    unsafe_allow_html=True,
)

display_columns = [
    "unit",
    "predicted_rul_mean",
    "conservative_rul",
    "engine_condition",
    "reliability_risk",
    "review_flag",
    "operational_priority",
    "trajectory_flag",
]

fleet_table = (
    decision_df[
        display_columns
    ]
    .copy()
    .rename(
        columns={
            "unit": "Engine",
            "predicted_rul_mean": "Predicted RUL",
            "conservative_rul": "Conservative RUL",
            "engine_condition": "Condition",
            "reliability_risk": "Reliability risk",
            "review_flag": "Engineering review",
            "operational_priority": "Operational priority",
            "trajectory_flag": "Trajectory diagnostic",
        }
    )
)

fleet_table[
    "Predicted RUL"
] = fleet_table[
    "Predicted RUL"
].round(
    1
)

fleet_table[
    "Conservative RUL"
] = fleet_table[
    "Conservative RUL"
].round(
    1
)

st.dataframe(
    fleet_table,
    use_container_width=True,
    hide_index=True,
    height=480,
)

st.caption(
    f"Loaded from: {decision_path}"
)
# ---------------------------------------------------------
# Engine Explorer
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-heading">Engine Explorer</div>',
    unsafe_allow_html=True,
)

selected_unit = st.selectbox(
    "Select a test engine",
    options=sorted(
        decision_df["unit"].astype(int).tolist()
    ),
    index=66,
    format_func=lambda unit: f"Engine {unit}",
)

selected_engine = (
    decision_df.loc[
        decision_df["unit"] == selected_unit
    ]
    .iloc[0]
)

st.markdown(
    f"### Engine {selected_unit} decision summary"
)

metric_columns = st.columns(5)

metric_columns[0].metric(
    "Predicted RUL",
    f"{selected_engine['predicted_rul_mean']:.1f} cycles",
)

metric_columns[1].metric(
    "Conservative RUL",
    f"{selected_engine['conservative_rul']:.1f} cycles",
)

metric_columns[2].metric(
    "Condition",
    selected_engine["engine_condition"],
)

metric_columns[3].metric(
    "Reliability risk",
    selected_engine["reliability_risk"],
)

metric_columns[4].metric(
    "Operational priority",
    selected_engine["operational_priority"],
)


# ---------------------------------------------------------
# Selected-engine decision evidence
# ---------------------------------------------------------

left_column, right_column = st.columns(
    [1.15, 0.85]
)

with left_column:
    evidence_rows = [
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
    ]

    if "actual_rul" in selected_engine.index:
        evidence_rows.append(
            {
                "Estimate": "Actual RUL",
                "RUL": selected_engine[
                    "actual_rul"
                ],
            }
        )

    evidence_df = pd.DataFrame(
        evidence_rows
    )

    evidence_figure = px.bar(
        evidence_df,
        x="Estimate",
        y="RUL",
        text="RUL",
        title=(
            f"Prediction Evidence — Engine {selected_unit}"
        ),
    )

    evidence_figure.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
    )

    evidence_figure.update_layout(
        showlegend=False,
        yaxis_title="Remaining Useful Life (cycles)",
        xaxis_title="",
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
    st.markdown("#### Engineering interpretation")

    interpretation_table = pd.DataFrame(
        {
            "Decision element": [
                "Engineering review",
                "Trajectory diagnostic",
                "Prediction trust",
                "Trajectory trust",
                "Reliability score",
                "Condition severity",
                "Model disagreement",
                "MC vs deterministic gap",
            ],
            "Result": [
                selected_engine[
                    "review_flag"
                ],
                selected_engine[
                    "trajectory_flag"
                ],
                selected_engine[
                    "prediction_trust"
                ],
                selected_engine[
                    "trajectory_trust"
                ],
                (
                    f"{selected_engine['reliability_risk_score']:.1f}"
                ),
                (
                    f"{selected_engine['condition_severity_score']:.1f}"
                ),
                (
                    f"{selected_engine['model_disagreement']:.1f} cycles"
                ),
                (
                    f"{selected_engine['mc_deterministic_gap']:.1f} cycles"
                ),
            ],
        }
    )

    st.dataframe(
        interpretation_table,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# Prediction interval
# ---------------------------------------------------------

st.markdown("#### Calibrated uncertainty interval")

interval_columns = st.columns(4)

interval_columns[0].metric(
    "Lower bound",
    (
        f"{selected_engine['calibrated_lower_bound']:.1f} cycles"
    ),
)

interval_columns[1].metric(
    "Prediction mean",
    (
        f"{selected_engine['predicted_rul_mean']:.1f} cycles"
    ),
)

interval_columns[2].metric(
    "Upper bound",
    (
        f"{selected_engine['calibrated_upper_bound']:.1f} cycles"
    ),
)

interval_columns[3].metric(
    "Interval width",
    (
        f"{selected_engine['interval_width']:.1f} cycles"
    ),
)


# ---------------------------------------------------------
# Recommended operational action
# ---------------------------------------------------------

st.markdown("#### Recommended action")

review_flag = selected_engine[
    "review_flag"
]

if review_flag == "Required":
    st.error(
        selected_engine[
            "operational_action"
        ]
    )

elif review_flag == "Recommended":
    st.warning(
        selected_engine[
            "operational_action"
        ]
    )

else:
    st.success(
        selected_engine[
            "operational_action"
        ]
    )
    # ---------------------------------------------------------
# Trajectory Diagnostics
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-heading">Trajectory Diagnostics</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Tracks how the GRU Remaining Useful Life prediction changes "
    "as additional operating cycles become available."
)

trajectory_path = (
    RESULTS_DIR
    / "trajectory_diagnostics"
    / (
        f"trajectory_predictions_"
        f"{model_name}_{dataset_name}.csv"
    )
)

try:
    trajectory_df = load_csv(
        trajectory_path
    )

    required_trajectory_columns = {
        "unit",
        "cycle",
        "predicted_rul",
        "actual_rul_capped",
    }

    missing_trajectory_columns = sorted(
        required_trajectory_columns.difference(
            trajectory_df.columns
        )
    )

    if missing_trajectory_columns:
        raise KeyError(
            "Trajectory data is missing these columns: "
            f"{missing_trajectory_columns}"
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
        raise ValueError(
            f"No trajectory was found for Engine {selected_unit}."
        )

except (
    FileNotFoundError,
    KeyError,
    ValueError,
) as error:
    st.error(
        "Trajectory diagnostics could not be loaded."
    )

    st.code(
        str(error)
    )

else:
    final_cycle = int(
        engine_trajectory_df["cycle"].max()
    )

    recent_window_length = min(
        20,
        len(engine_trajectory_df),
    )

    recent_start_index = (
        len(engine_trajectory_df)
        - recent_window_length
    )

    recent_start_cycle = int(
        engine_trajectory_df.iloc[
            recent_start_index
        ]["cycle"]
    )

    final_prediction = float(
        engine_trajectory_df.iloc[-1][
            "predicted_rul"
        ]
    )

    final_actual = float(
        engine_trajectory_df.iloc[-1][
            "actual_rul_capped"
        ]
    )

    trajectory_figure = go.Figure()

    trajectory_figure.add_trace(
        go.Scatter(
            x=engine_trajectory_df["cycle"],
            y=engine_trajectory_df["predicted_rul"],
            mode="lines",
            name="GRU predicted RUL",
            line=dict(
                width=3,
            ),
            hovertemplate=(
                "Cycle: %{x}<br>"
                "Predicted RUL: %{y:.1f}<extra></extra>"
            ),
        )
    )

    trajectory_figure.add_trace(
        go.Scatter(
            x=engine_trajectory_df["cycle"],
            y=engine_trajectory_df["actual_rul_capped"],
            mode="lines",
            name="Actual capped RUL",
            line=dict(
                width=2,
                dash="dash",
            ),
            hovertemplate=(
                "Cycle: %{x}<br>"
                "Actual RUL: %{y:.1f}<extra></extra>"
            ),
        )
    )

    trajectory_figure.add_trace(
        go.Scatter(
            x=[final_cycle],
            y=[final_prediction],
            mode="markers",
            name="Final prediction",
            marker=dict(
                size=12,
                symbol="circle",
            ),
            hovertemplate=(
                "Final cycle: %{x}<br>"
                "Final prediction: %{y:.1f}<extra></extra>"
            ),
        )
    )

    trajectory_figure.add_trace(
        go.Scatter(
            x=[final_cycle],
            y=[final_actual],
            mode="markers",
            name="Final actual RUL",
            marker=dict(
                size=13,
                symbol="x",
            ),
            hovertemplate=(
                "Final cycle: %{x}<br>"
                "Final actual RUL: %{y:.1f}<extra></extra>"
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
        legend_title="Trajectory",
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

    st.info(
        "The actual RUL curve is shown only for retrospective "
        "test evaluation. In a real deployment, the future failure "
        "point and actual RUL would be unknown."
    )


    # -----------------------------------------------------
    # Temporal diagnostic metrics
    # -----------------------------------------------------

    st.markdown("#### Temporal behaviour")

    trajectory_metric_columns = st.columns(
        6
    )

    trajectory_metric_columns[0].metric(
        "Observed cycles",
        final_cycle,
    )

    trajectory_metric_columns[1].metric(
        "Recent slope",
        (
            f"{selected_engine['recent_slope']:.2f}"
            " RUL/cycle"
        ),
    )

    trajectory_metric_columns[2].metric(
        "Recent RUL drop",
        (
            f"{selected_engine['recent_predicted_drop']:.1f}"
            " cycles"
        ),
    )

    trajectory_metric_columns[3].metric(
        "Prediction range",
        (
            f"{selected_engine['recent_prediction_range']:.1f}"
            " cycles"
        ),
    )

    trajectory_metric_columns[4].metric(
        "Monotonicity violations",
        (
            f"{selected_engine['monotonicity_violation_rate'] * 100:.1f}%"
        ),
    )

    trajectory_metric_columns[5].metric(
        "Large jumps",
        int(
            selected_engine["large_jump_count"]
        ),
    )


    # -----------------------------------------------------
    # Trajectory interpretation
    # -----------------------------------------------------

    trajectory_flag = selected_engine[
        "trajectory_flag"
    ]

    trajectory_explanations = {
        "Consistent degradation": (
            "The predicted RUL generally decreases as the engine "
            "accumulates operating cycles. This is the expected "
            "behaviour for a degrading system."
        ),
        "Rapid degradation signal": (
            "The predicted RUL is declining unusually quickly. "
            "Recent sensor behaviour may indicate accelerating "
            "degradation."
        ),
        "Mixed trajectory": (
            "The prediction contains both degrading and stable "
            "behaviour, reducing confidence in the temporal trend."
        ),
        "Weak degradation / plateau": (
            "The RUL estimate changes very little despite continued "
            "engine ageing."
        ),
        "High-RUL plateau — review": (
            "The model continues predicting a high RUL while the "
            "engine accumulates cycles. This may indicate a shared "
            "model blind spot even when multiple models agree."
        ),
        "RUL increasing despite ageing": (
            "The predicted RUL increases while the engine gets older. "
            "This behaviour is physically suspicious and requires "
            "engineering review."
        ),
        "Unstable trajectory": (
            "The predicted RUL contains large jumps or oscillations. "
            "The final point should not be interpreted without the "
            "full prediction history."
        ),
    }

    st.markdown("#### Trajectory interpretation")

    trajectory_message = trajectory_explanations.get(
        trajectory_flag,
        "The prediction history requires further examination.",
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


    # -----------------------------------------------------
    # Recent-cycle inspection table
    # -----------------------------------------------------

    with st.expander(
        "Inspect the most recent trajectory values"
    ):
        recent_display_columns = [
            "cycle",
            "predicted_rul",
            "actual_rul_capped",
        ]

        if "prediction_error" in engine_trajectory_df.columns:
            recent_display_columns.append(
                "prediction_error"
            )

        recent_trajectory_table = (
            engine_trajectory_df[
                recent_display_columns
            ]
            .tail(recent_window_length)
            .copy()
            .rename(
                columns={
                    "cycle": "Cycle",
                    "predicted_rul": "Predicted RUL",
                    "actual_rul_capped": "Actual capped RUL",
                    "prediction_error": "Prediction error",
                }
            )
        )

        numeric_columns = [
            column
            for column in recent_trajectory_table.columns
            if column != "Cycle"
        ]

        recent_trajectory_table[
            numeric_columns
        ] = recent_trajectory_table[
            numeric_columns
        ].round(2)

        st.dataframe(
            recent_trajectory_table,
            use_container_width=True,
            hide_index=True,
        )

    st.caption(
        f"Trajectory source: {trajectory_path}"
    )
    # ---------------------------------------------------------
# Model Performance and Benchmarking
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-heading">'
    'Model Performance & Benchmarking'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Deployment-style evaluation using one final prediction "
    "for each of the 100 unseen FD001 test engines."
)


# ---------------------------------------------------------
# Benchmark results
# ---------------------------------------------------------

model_performance_df = pd.DataFrame(
    [
        {
            "Model": "Mean baseline",
            "RMSE": 38.00,
            "MAE": 31.00,
            "R²": 0.000,
            "NASA score": None,
            "Role": "Naive reference",
        },
        {
            "Model": "Ridge",
            "RMSE": 14.9931,
            "MAE": 12.2316,
            "R²": 0.8600,
            "NASA score": 374.9761,
            "Role": "Interpretable baseline",
        },
        {
            "Model": "LSTM",
            "RMSE": 14.2865,
            "MAE": 10.9647,
            "R²": 0.8729,
            "NASA score": 443.0913,
            "Role": "Secondary sequence model",
        },
        {
            "Model": "GRU",
            "RMSE": 13.5849,
            "MAE": 9.8770,
            "R²": 0.8851,
            "NASA score": 317.7112,
            "Role": "Primary standalone model",
        },
        {
            "Model": "Transformer",
            "RMSE": 21.7041,
            "MAE": 15.4325,
            "R²": 0.7060,
            "NASA score": 1647.0,
            "Role": "Rejected for FD001",
        },
        {
            "Model": "GRU–LSTM ensemble",
            "RMSE": 13.2122,
            "MAE": 10.0868,
            "R²": 0.8913,
            "NASA score": 291.3001,
            "Role": "Best aggregate accuracy",
        },
    ]
)


# ---------------------------------------------------------
# Best-model metrics
# ---------------------------------------------------------

best_rmse_row = model_performance_df.loc[
    model_performance_df["RMSE"].idxmin()
]

best_mae_row = model_performance_df.loc[
    model_performance_df["MAE"].idxmin()
]

best_r2_row = model_performance_df.loc[
    model_performance_df["R²"].idxmax()
]

valid_nasa_df = model_performance_df.dropna(
    subset=["NASA score"]
)

best_nasa_row = valid_nasa_df.loc[
    valid_nasa_df["NASA score"].idxmin()
]

performance_metrics = st.columns(4)

performance_metrics[0].metric(
    "Lowest RMSE",
    f"{best_rmse_row['RMSE']:.2f}",
    help=f"Model: {best_rmse_row['Model']}",
)

performance_metrics[1].metric(
    "Lowest MAE",
    f"{best_mae_row['MAE']:.2f}",
    help=f"Model: {best_mae_row['Model']}",
)

performance_metrics[2].metric(
    "Highest R²",
    f"{best_r2_row['R²']:.3f}",
    help=f"Model: {best_r2_row['Model']}",
)

performance_metrics[3].metric(
    "Lowest NASA score",
    f"{best_nasa_row['NASA score']:.1f}",
    help=f"Model: {best_nasa_row['Model']}",
)


# ---------------------------------------------------------
# Performance charts
# ---------------------------------------------------------

left_column, right_column = st.columns(2)

with left_column:
    rmse_figure = px.bar(
        model_performance_df,
        x="Model",
        y="RMSE",
        text="RMSE",
        title="Root Mean Squared Error",
        hover_data={
            "Role": True,
            "MAE": ":.2f",
            "R²": ":.3f",
        },
    )

    rmse_figure.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    rmse_figure.update_layout(
        xaxis_title="",
        yaxis_title="RMSE (cycles)",
        showlegend=False,
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        rmse_figure,
        use_container_width=True,
    )

with right_column:
    mae_figure = px.bar(
        model_performance_df,
        x="Model",
        y="MAE",
        text="MAE",
        title="Mean Absolute Error",
        hover_data={
            "Role": True,
            "RMSE": ":.2f",
            "R²": ":.3f",
        },
    )

    mae_figure.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    mae_figure.update_layout(
        xaxis_title="",
        yaxis_title="MAE (cycles)",
        showlegend=False,
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        mae_figure,
        use_container_width=True,
    )


# ---------------------------------------------------------
# Accuracy versus asymmetric maintenance penalty
# ---------------------------------------------------------

left_column, right_column = st.columns(2)

with left_column:
    r2_figure = px.bar(
        model_performance_df,
        x="Model",
        y="R²",
        text="R²",
        title="Explained Variance",
        hover_data={
            "Role": True,
            "RMSE": ":.2f",
            "MAE": ":.2f",
        },
    )

    r2_figure.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    r2_figure.update_layout(
        xaxis_title="",
        yaxis_title="R²",
        showlegend=False,
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        r2_figure,
        use_container_width=True,
    )

with right_column:
    nasa_plot_df = model_performance_df.dropna(
        subset=["NASA score"]
    )

    nasa_figure = px.bar(
        nasa_plot_df,
        x="Model",
        y="NASA score",
        text="NASA score",
        title="NASA Asymmetric RUL Score",
        hover_data={
            "Role": True,
            "RMSE": ":.2f",
            "MAE": ":.2f",
        },
    )

    nasa_figure.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
    )

    nasa_figure.update_layout(
        xaxis_title="",
        yaxis_title="NASA score — lower is better",
        showlegend=False,
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        nasa_figure,
        use_container_width=True,
    )


# ---------------------------------------------------------
# Benchmark table
# ---------------------------------------------------------

st.markdown("#### Complete benchmark table")

benchmark_table = model_performance_df.copy()

benchmark_table["RMSE"] = benchmark_table["RMSE"].round(2)
benchmark_table["MAE"] = benchmark_table["MAE"].round(2)
benchmark_table["R²"] = benchmark_table["R²"].round(3)
benchmark_table["NASA score"] = benchmark_table[
    "NASA score"
].round(1)

st.dataframe(
    benchmark_table,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# Model-selection interpretation
# ---------------------------------------------------------

st.markdown("#### Model-selection decision")

selection_left, selection_right = st.columns(2)

with selection_left:
    st.success(
        """
        **Primary model — GRU**

        The GRU produced the lowest standalone MAE, strong RMSE,
        strong NASA score and fewer dangerous overpredictions than
        the accuracy ensemble. It is therefore the main model used
        by MachineGuard AI.
        """
    )

    st.info(
        """
        **Accuracy model — GRU–LSTM ensemble**

        Validation-only weight selection produced a 60% GRU and
        40% LSTM ensemble. It achieved the best overall RMSE,
        R² and NASA score.
        """
    )

with selection_right:
    st.warning(
        """
        **Independent baseline — Ridge**

        Ridge remained surprisingly competitive and provides an
        independent comparison against the neural-network output.
        Large disagreement can indicate unreliable behaviour.
        """
    )

    st.error(
        """
        **Rejected model — Transformer**

        The Transformer added substantial complexity but performed
        considerably worse on FD001. More sophisticated architecture
        did not automatically produce a better prognostic model.
        """
    )


# ---------------------------------------------------------
# Original repository comparison
# ---------------------------------------------------------

st.markdown("#### Improvement from the original implementation")

comparison_df = pd.DataFrame(
    [
        {
            "Pipeline": "Original repository LSTM",
            "RMSE": 63.7928,
            "MAE": 50.8283,
            "R²": -0.3658,
        },
        {
            "Pipeline": "MachineGuard AI GRU",
            "RMSE": 13.5849,
            "MAE": 9.8770,
            "R²": 0.8851,
        },
        {
            "Pipeline": "MachineGuard AI ensemble",
            "RMSE": 13.2122,
            "MAE": 10.0868,
            "R²": 0.8913,
        },
    ]
)

comparison_figure = px.bar(
    comparison_df,
    x="Pipeline",
    y="RMSE",
    text="RMSE",
    title="Original Pipeline vs Leakage-Safe MachineGuard AI",
    hover_data={
        "MAE": ":.2f",
        "R²": ":.3f",
    },
)

comparison_figure.update_traces(
    texttemplate="%{text:.2f}",
    textposition="outside",
)

comparison_figure.update_layout(
    xaxis_title="",
    yaxis_title="RMSE (cycles)",
    showlegend=False,
    margin=dict(
        l=20,
        r=20,
        t=55,
        b=20,
    ),
)

st.plotly_chart(
    comparison_figure,
    use_container_width=True,
)

st.info(
    """
    The improvement is not merely the result of changing the neural
    network. MachineGuard AI rebuilt the full experimental pipeline:
    engine-level validation, train-only scaling, deployment-aligned
    test evaluation, proper baselines and leakage prevention.
    """
)

st.caption(
    "Benchmark values are rounded from the preserved FD001 "
    "evaluation outputs."
)
# ---------------------------------------------------------
# Global Sensor Explainability
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-heading">'
    'Global Sensor Explainability'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Occlusion analysis measures how much model performance "
    "degrades when one input feature is neutralized."
)


# ---------------------------------------------------------
# Load global importance results
# ---------------------------------------------------------

explanation_dir = (
    RESULTS_DIR
    / "explanations"
)

gru_importance_path = (
    explanation_dir
    / (
        f"global_importance_gru_"
        f"{dataset_name}.csv"
    )
)

lstm_importance_path = (
    explanation_dir
    / (
        f"global_importance_lstm_"
        f"{dataset_name}.csv"
    )
)

try:
    gru_importance_df = load_csv(
        gru_importance_path
    )

    required_importance_columns = {
        "feature",
        "baseline_rmse",
        "occluded_rmse",
        "rmse_increase",
        "baseline_mae",
        "occluded_mae",
        "mae_increase",
        "mean_absolute_prediction_change",
        "mean_signed_prediction_change",
    }

    missing_importance_columns = sorted(
        required_importance_columns.difference(
            gru_importance_df.columns
        )
    )

    if missing_importance_columns:
        raise KeyError(
            "GRU explainability results are missing columns: "
            f"{missing_importance_columns}"
        )

except (
    FileNotFoundError,
    KeyError,
) as error:
    st.error(
        "Global explainability results could not be loaded."
    )

    st.code(
        str(error)
    )

    st.info(
        "Run explain_model_v2.py for the GRU model "
        "before using this section."
    )

else:
    gru_importance_df = (
        gru_importance_df
        .copy()
        .sort_values(
            by="rmse_increase",
            ascending=False,
        )
        .reset_index(
            drop=True
        )
    )


    # -----------------------------------------------------
    # Explainability summary metrics
    # -----------------------------------------------------

    most_important_feature = (
        gru_importance_df.iloc[0]
    )

    largest_prediction_change_row = (
        gru_importance_df.loc[
            gru_importance_df[
                "mean_absolute_prediction_change"
            ].idxmax()
        ]
    )

    positive_effect_row = (
        gru_importance_df.loc[
            gru_importance_df[
                "mean_signed_prediction_change"
            ].idxmax()
        ]
    )

    negative_effect_row = (
        gru_importance_df.loc[
            gru_importance_df[
                "mean_signed_prediction_change"
            ].idxmin()
        ]
    )

    explanation_metrics = st.columns(
        4
    )

    explanation_metrics[0].metric(
        "Most important sensor",
        most_important_feature[
            "feature"
        ],
        help=(
            "Ranked by increase in RMSE after occlusion."
        ),
    )

    explanation_metrics[1].metric(
        "Largest RMSE increase",
        (
            f"{most_important_feature['rmse_increase']:.3f}"
        ),
        help=(
            "Higher values indicate greater global importance."
        ),
    )

    explanation_metrics[2].metric(
        "Largest prediction change",
        (
            f"{largest_prediction_change_row['mean_absolute_prediction_change']:.2f}"
            " cycles"
        ),
        help=(
            f"Feature: {largest_prediction_change_row['feature']}"
        ),
    )

    explanation_metrics[3].metric(
        "Baseline explainability RMSE",
        (
            f"{gru_importance_df['baseline_rmse'].iloc[0]:.2f}"
            " cycles"
        ),
    )


    # -----------------------------------------------------
    # Top sensor importance charts
    # -----------------------------------------------------

    top_feature_count = st.slider(
        "Number of sensors to display",
        min_value=5,
        max_value=min(
            24,
            len(
                gru_importance_df
            ),
        ),
        value=min(
            12,
            len(
                gru_importance_df
            ),
        ),
        step=1,
        key="global_sensor_count",
    )

    top_gru_importance_df = (
        gru_importance_df
        .head(
            top_feature_count
        )
        .sort_values(
            by="rmse_increase",
            ascending=True,
        )
    )

    left_column, right_column = st.columns(
        2
    )

    with left_column:
        rmse_importance_figure = px.bar(
            top_gru_importance_df,
            x="rmse_increase",
            y="feature",
            orientation="h",
            text="rmse_increase",
            title=(
                "GRU Feature Importance — RMSE Increase"
            ),
            hover_data={
                "baseline_rmse": ":.3f",
                "occluded_rmse": ":.3f",
                "mae_increase": ":.3f",
                "mean_absolute_prediction_change": ":.3f",
            },
            labels={
                "rmse_increase": (
                    "Increase in RMSE after occlusion"
                ),
                "feature": "Input feature",
            },
        )

        rmse_importance_figure.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
        )

        rmse_importance_figure.update_layout(
            showlegend=False,
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20,
            ),
        )

        st.plotly_chart(
            rmse_importance_figure,
            use_container_width=True,
        )

    with right_column:
        mae_importance_df = (
            gru_importance_df
            .head(
                top_feature_count
            )
            .sort_values(
                by="mae_increase",
                ascending=True,
            )
        )

        mae_importance_figure = px.bar(
            mae_importance_df,
            x="mae_increase",
            y="feature",
            orientation="h",
            text="mae_increase",
            title=(
                "GRU Feature Importance — MAE Increase"
            ),
            hover_data={
                "baseline_mae": ":.3f",
                "occluded_mae": ":.3f",
                "rmse_increase": ":.3f",
                "mean_absolute_prediction_change": ":.3f",
            },
            labels={
                "mae_increase": (
                    "Increase in MAE after occlusion"
                ),
                "feature": "Input feature",
            },
        )

        mae_importance_figure.update_traces(
            texttemplate="%{text:.3f}",
            textposition="outside",
        )

        mae_importance_figure.update_layout(
            showlegend=False,
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20,
            ),
        )

        st.plotly_chart(
            mae_importance_figure,
            use_container_width=True,
        )


    # -----------------------------------------------------
    # Prediction sensitivity
    # -----------------------------------------------------

    st.markdown(
        "#### Prediction sensitivity after sensor occlusion"
    )

    sensitivity_df = (
        gru_importance_df
        .sort_values(
            by="mean_absolute_prediction_change",
            ascending=False,
        )
        .head(
            top_feature_count
        )
        .sort_values(
            by="mean_absolute_prediction_change",
            ascending=True,
        )
    )

    sensitivity_figure = px.bar(
        sensitivity_df,
        x="mean_absolute_prediction_change",
        y="feature",
        orientation="h",
        text="mean_absolute_prediction_change",
        title=(
            "Average Change in Predicted RUL"
        ),
        hover_data={
            "mean_signed_prediction_change": ":.3f",
            "rmse_increase": ":.3f",
            "mae_increase": ":.3f",
        },
        labels={
            "mean_absolute_prediction_change": (
                "Mean absolute prediction change (cycles)"
            ),
            "feature": "Input feature",
        },
    )

    sensitivity_figure.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    sensitivity_figure.update_layout(
        showlegend=False,
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        sensitivity_figure,
        use_container_width=True,
    )


    # -----------------------------------------------------
    # Signed prediction influence
    # -----------------------------------------------------

    st.markdown(
        "#### Direction of average prediction change"
    )

    signed_change_df = (
        gru_importance_df
        .sort_values(
            by="mean_absolute_prediction_change",
            ascending=False,
        )
        .head(
            top_feature_count
        )
        .sort_values(
            by="mean_signed_prediction_change",
            ascending=True,
        )
    )

    signed_change_figure = px.bar(
        signed_change_df,
        x="mean_signed_prediction_change",
        y="feature",
        orientation="h",
        text="mean_signed_prediction_change",
        title=(
            "Signed RUL Change When a Feature Is Occluded"
        ),
        hover_data={
            "mean_absolute_prediction_change": ":.3f",
            "rmse_increase": ":.3f",
            "mae_increase": ":.3f",
        },
        labels={
            "mean_signed_prediction_change": (
                "Mean signed prediction change (cycles)"
            ),
            "feature": "Input feature",
        },
    )

    signed_change_figure.add_vline(
        x=0,
        line_dash="dash",
        opacity=0.5,
    )

    signed_change_figure.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    signed_change_figure.update_layout(
        showlegend=False,
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        signed_change_figure,
        use_container_width=True,
    )

    st.info(
        """
        A positive signed change means that neutralizing the feature
        caused the predicted RUL to increase on average. This suggests
        that the original feature values were generally contributing
        evidence of degradation.

        A negative signed change means that neutralizing the feature
        caused the predicted RUL to decrease on average.
        """
    )


    # -----------------------------------------------------
    # GRU and LSTM explainability comparison
    # -----------------------------------------------------

    if lstm_importance_path.exists():
        try:
            lstm_importance_df = load_csv(
                lstm_importance_path
            )

            comparison_required_columns = {
                "feature",
                "rmse_increase",
                "mae_increase",
                "mean_absolute_prediction_change",
            }

            missing_lstm_columns = sorted(
                comparison_required_columns.difference(
                    lstm_importance_df.columns
                )
            )

            if missing_lstm_columns:
                raise KeyError(
                    "LSTM explainability results are missing columns: "
                    f"{missing_lstm_columns}"
                )

        except KeyError as error:
            st.warning(
                "The LSTM explainability file could not be compared."
            )

            st.code(
                str(error)
            )

        else:
            st.markdown(
                "#### GRU vs LSTM sensor reliance"
            )

            model_comparison_df = (
                gru_importance_df[
                    [
                        "feature",
                        "rmse_increase",
                    ]
                ]
                .rename(
                    columns={
                        "rmse_increase": (
                            "GRU RMSE increase"
                        ),
                    }
                )
                .merge(
                    lstm_importance_df[
                        [
                            "feature",
                            "rmse_increase",
                        ]
                    ].rename(
                        columns={
                            "rmse_increase": (
                                "LSTM RMSE increase"
                            ),
                        }
                    ),
                    on="feature",
                    how="inner",
                    validate="one_to_one",
                )
            )

            model_comparison_df[
                "Combined importance"
            ] = (
                model_comparison_df[
                    "GRU RMSE increase"
                ].abs()
                + model_comparison_df[
                    "LSTM RMSE increase"
                ].abs()
            )

            model_comparison_df = (
                model_comparison_df
                .sort_values(
                    by="Combined importance",
                    ascending=False,
                )
                .head(
                    top_feature_count
                )
            )

            comparison_long_df = (
                model_comparison_df[
                    [
                        "feature",
                        "GRU RMSE increase",
                        "LSTM RMSE increase",
                    ]
                ]
                .melt(
                    id_vars="feature",
                    var_name="Model",
                    value_name="RMSE increase",
                )
            )

            comparison_figure = px.bar(
                comparison_long_df,
                x="feature",
                y="RMSE increase",
                color="Model",
                barmode="group",
                title=(
                    "Sensor Importance Across Sequence Models"
                ),
                labels={
                    "feature": "Input feature",
                    "RMSE increase": (
                        "RMSE increase after occlusion"
                    ),
                },
            )

            comparison_figure.update_layout(
                margin=dict(
                    l=20,
                    r=20,
                    t=55,
                    b=20,
                ),
            )

            st.plotly_chart(
                comparison_figure,
                use_container_width=True,
            )

            st.caption(
                "Features that are important to both GRU and LSTM "
                "provide stronger evidence of shared sensor reliance."
            )


    # -----------------------------------------------------
    # Explainability table
    # -----------------------------------------------------

    with st.expander(
        "Inspect complete global feature-importance results"
    ):
        importance_table = (
            gru_importance_df
            .copy()
            .rename(
                columns={
                    "feature": "Feature",
                    "baseline_rmse": "Baseline RMSE",
                    "occluded_rmse": "Occluded RMSE",
                    "rmse_increase": "RMSE increase",
                    "baseline_mae": "Baseline MAE",
                    "occluded_mae": "Occluded MAE",
                    "mae_increase": "MAE increase",
                    "mean_absolute_prediction_change": (
                        "Mean absolute prediction change"
                    ),
                    "mean_signed_prediction_change": (
                        "Mean signed prediction change"
                    ),
                }
            )
        )

        numeric_columns = [
            column
            for column in importance_table.columns
            if column != "Feature"
        ]

        importance_table[
            numeric_columns
        ] = importance_table[
            numeric_columns
        ].round(
            4
        )

        st.dataframe(
            importance_table,
            use_container_width=True,
            hide_index=True,
        )


    # -----------------------------------------------------
    # Engineering interpretation
    # -----------------------------------------------------

    st.markdown(
        "#### Explainability interpretation"
    )

    top_sensor_names = (
        gru_importance_df
        .head(5)[
            "feature"
        ]
        .tolist()
    )

    st.success(
        "The GRU relies most strongly on "
        f"**{', '.join(top_sensor_names)}**. "
        "Neutralizing these sensors produced the largest "
        "increase in prediction error."
    )

    st.warning(
        """
        Occlusion importance describes how much the trained model
        depends on a feature. It does not prove that the sensor is
        physically causal, nor does it replace engineering knowledge
        about turbofan degradation.
        """
    )

    st.caption(
        f"GRU explainability source: {gru_importance_path}"
    )
# ---------------------------------------------------------
# Unit 67 Blind-Spot Case Study
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-heading">'
    'Case Study: Shared Model Blind Spot'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Engine 67 demonstrates why static prediction agreement "
    "does not necessarily guarantee a trustworthy result."
)

case_unit = 67

case_engine_rows = decision_df.loc[
    decision_df["unit"] == case_unit
]

if case_engine_rows.empty:
    st.warning(
        "Engine 67 is not available in the selected dataset."
    )

else:
    case_engine = case_engine_rows.iloc[0]

    case_metric_columns = st.columns(5)

    case_metric_columns[0].metric(
        "Actual RUL",
        f"{case_engine['actual_rul']:.1f} cycles",
    )

    case_metric_columns[1].metric(
        "GRU prediction",
        f"{case_engine['predicted_rul_mean']:.1f} cycles",
    )

    case_metric_columns[2].metric(
        "Ridge prediction",
        f"{case_engine['ridge_prediction']:.1f} cycles",
    )

    case_metric_columns[3].metric(
        "Prediction error",
        f"{case_engine['prediction_error']:+.1f} cycles",
    )

    case_metric_columns[4].metric(
        "Engineering review",
        case_engine["review_flag"],
    )


    # -----------------------------------------------------
    # Static evidence
    # -----------------------------------------------------

    static_left, static_right = st.columns(
        [1.1, 0.9]
    )

    with static_left:
        case_prediction_df = pd.DataFrame(
            [
                {
                    "Estimate": "GRU MC mean",
                    "RUL": case_engine[
                        "predicted_rul_mean"
                    ],
                },
                {
                    "Estimate": "GRU deterministic",
                    "RUL": case_engine[
                        "final_predicted_rul"
                    ],
                },
                {
                    "Estimate": "Ridge baseline",
                    "RUL": case_engine[
                        "ridge_prediction"
                    ],
                },
                {
                    "Estimate": "Conservative estimate",
                    "RUL": case_engine[
                        "conservative_rul"
                    ],
                },
                {
                    "Estimate": "Actual RUL",
                    "RUL": case_engine[
                        "actual_rul"
                    ],
                },
            ]
        )

        case_prediction_figure = px.bar(
            case_prediction_df,
            x="Estimate",
            y="RUL",
            text="RUL",
            title="Static Prediction Evidence — Engine 67",
        )

        case_prediction_figure.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside",
        )

        case_prediction_figure.update_layout(
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
            case_prediction_figure,
            use_container_width=True,
        )

    with static_right:
        st.markdown("#### Why the result initially looked safe")

        apparent_safety_df = pd.DataFrame(
            {
                "Evidence": [
                    "GRU vs Ridge disagreement",
                    "GRU MC vs deterministic gap",
                    "Static prediction trust",
                    "Predicted engine condition",
                    "Reliability-risk category",
                ],
                "Result": [
                    (
                        f"{case_engine['model_disagreement']:.1f} "
                        "cycles"
                    ),
                    (
                        f"{case_engine['mc_deterministic_gap']:.1f} "
                        "cycles"
                    ),
                    case_engine[
                        "prediction_trust"
                    ],
                    case_engine[
                        "engine_condition"
                    ],
                    case_engine[
                        "reliability_risk"
                    ],
                ],
            }
        )

        st.dataframe(
            apparent_safety_df,
            use_container_width=True,
            hide_index=True,
        )

        st.info(
            """
            The static models produced similar high-RUL estimates.
            Model disagreement was therefore low, and the engine
            appeared healthy based on the final prediction alone.
            """
        )


    # -----------------------------------------------------
    # Load Engine 67 trajectory
    # -----------------------------------------------------

    case_trajectory_path = (
        RESULTS_DIR
        / "trajectory_diagnostics"
        / (
            f"trajectory_predictions_"
            f"{model_name}_{dataset_name}.csv"
        )
    )

    try:
        case_trajectory_all_df = load_csv(
            case_trajectory_path
        )

        case_trajectory_df = (
            case_trajectory_all_df.loc[
                case_trajectory_all_df["unit"]
                == case_unit
            ]
            .copy()
            .sort_values("cycle")
            .reset_index(drop=True)
        )

        if case_trajectory_df.empty:
            raise ValueError(
                "No trajectory values were found for Engine 67."
            )

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
    ) as error:
        st.error(
            "Engine 67 trajectory data could not be loaded."
        )

        st.code(
            str(error)
        )

    else:
        st.markdown(
            "#### Temporal evidence that exposed the blind spot"
        )

        case_trajectory_figure = go.Figure()

        case_trajectory_figure.add_trace(
            go.Scatter(
                x=case_trajectory_df["cycle"],
                y=case_trajectory_df[
                    "predicted_rul"
                ],
                mode="lines",
                name="GRU predicted RUL",
                line=dict(
                    width=3,
                ),
            )
        )

        case_trajectory_figure.add_trace(
            go.Scatter(
                x=case_trajectory_df["cycle"],
                y=case_trajectory_df[
                    "actual_rul_capped"
                ],
                mode="lines",
                name="Actual capped RUL",
                line=dict(
                    width=2,
                    dash="dash",
                ),
            )
        )

        case_trajectory_figure.update_layout(
            title=(
                "Engine 67 Remained on a High-RUL Plateau"
            ),
            xaxis_title="Observed operating cycle",
            yaxis_title="Remaining Useful Life (cycles)",
            hovermode="x unified",
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20,
            ),
        )

        st.plotly_chart(
            case_trajectory_figure,
            use_container_width=True,
        )


    # -----------------------------------------------------
    # Diagnostic conclusion
    # -----------------------------------------------------

    conclusion_columns = st.columns(4)

    conclusion_columns[0].metric(
        "Recent slope",
        f"{case_engine['recent_slope']:.3f}",
        help="RUL change per additional operating cycle.",
    )

    conclusion_columns[1].metric(
        "Recent predicted drop",
        (
            f"{case_engine['recent_predicted_drop']:.1f} "
            "cycles"
        ),
    )

    conclusion_columns[2].metric(
        "Trajectory diagnostic",
        case_engine["trajectory_flag"],
    )

    conclusion_columns[3].metric(
        "Operational priority",
        case_engine["operational_priority"],
    )

    st.error(
        """
        **MachineGuard AI conclusion**

        Engine 67 accumulated operating cycles while its predicted
        RUL remained almost unchanged at a high level. The GRU and
        Ridge agreed because both models were affected by a similar
        blind spot.

        The temporal diagnostic therefore overruled the apparently
        safe static evidence and assigned a mandatory engineering
        review.
        """
    )

    st.success(
        """
        **Portfolio significance**

        This case demonstrates that model agreement is not the same
        as model correctness. MachineGuard AI evaluates prediction
        behaviour over time instead of relying only on a single
        final RUL estimate.
        """
    )
    # ---------------------------------------------------------
# Uncertainty and Calibration
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-heading">'
    'Uncertainty & Calibration'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Monte Carlo dropout produces a distribution of RUL estimates. "
    "The calibrated interval communicates how uncertain the model "
    "is about each engine prediction."
)


# ---------------------------------------------------------
# Validate uncertainty columns
# ---------------------------------------------------------

required_uncertainty_columns = {
    "actual_rul",
    "predicted_rul_mean",
    "mc_standard_deviation",
    "calibrated_lower_bound",
    "calibrated_upper_bound",
    "interval_width",
    "absolute_error",
}

missing_uncertainty_columns = sorted(
    required_uncertainty_columns.difference(
        decision_df.columns
    )
)

if missing_uncertainty_columns:
    st.warning(
        "Fleet-wide uncertainty analysis is unavailable."
    )

    st.code(
        "Missing columns: "
        f"{missing_uncertainty_columns}"
    )

else:
    uncertainty_df = decision_df.copy()

    uncertainty_df[
        "interval_contains_actual"
    ] = (
        (
            uncertainty_df["actual_rul"]
            >= uncertainty_df[
                "calibrated_lower_bound"
            ]
        )
        & (
            uncertainty_df["actual_rul"]
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

    uncertainty_error_correlation = float(
        uncertainty_df[
            "interval_width"
        ].corr(
            uncertainty_df[
                "absolute_error"
            ]
        )
    )

    widest_interval_engine = int(
        uncertainty_df.loc[
            uncertainty_df[
                "interval_width"
            ].idxmax(),
            "unit",
        ]
    )


    # -----------------------------------------------------
    # Fleet uncertainty metrics
    # -----------------------------------------------------

    uncertainty_metrics = st.columns(5)

    uncertainty_metrics[0].metric(
        "Observed coverage",
        f"{observed_coverage * 100:.1f}%",
        help=(
            "Percentage of actual RUL values contained "
            "inside the calibrated prediction interval."
        ),
    )

    uncertainty_metrics[1].metric(
        "Target coverage",
        "90.0%",
    )

    uncertainty_metrics[2].metric(
        "Average interval width",
        f"{average_interval_width:.1f} cycles",
    )

    uncertainty_metrics[3].metric(
        "Average MC deviation",
        f"{average_mc_std:.1f} cycles",
    )

    uncertainty_metrics[4].metric(
        "Widest interval",
        f"Engine {widest_interval_engine}",
    )


    # -----------------------------------------------------
    # Coverage interpretation
    # -----------------------------------------------------

    coverage_gap = (
        observed_coverage
        - 0.90
    )

    if abs(
        coverage_gap
    ) <= 0.03:
        st.success(
            "Observed interval coverage is close to the "
            "requested 90% calibration target."
        )

    elif coverage_gap < 0:
        st.warning(
            "Observed coverage is below the 90% target. "
            "The intervals may be slightly too narrow."
        )

    else:
        st.info(
            "Observed coverage exceeds the 90% target. "
            "The intervals may be conservative."
        )


    # -----------------------------------------------------
    # Uncertainty versus prediction error
    # -----------------------------------------------------

    st.markdown(
        "#### Does uncertainty identify difficult predictions?"
    )

    uncertainty_error_figure = px.scatter(
        uncertainty_df,
        x="interval_width",
        y="absolute_error",
        hover_name="unit",
        hover_data={
            "actual_rul": ":.1f",
            "predicted_rul_mean": ":.1f",
            "mc_standard_deviation": ":.2f",
            "interval_contains_actual": True,
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
        },
        title=(
            "Prediction Interval Width vs Absolute Error"
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

    st.info(
        f"Pearson correlation between interval width and "
        f"absolute error: "
        f"**{uncertainty_error_correlation:.3f}**. "
        "A positive value means wider uncertainty intervals "
        "tend to occur on more difficult predictions."
    )


    # -----------------------------------------------------
    # Coverage by actual-RUL operating region
    # -----------------------------------------------------

    st.markdown(
        "#### Calibration across engine-life regions"
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
        region_coverage_figure = px.bar(
            region_uncertainty_df,
            x="actual_rul_region",
            y="coverage_percentage",
            text="coverage_percentage",
            title="Prediction-Interval Coverage by RUL Region",
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
        )

        region_coverage_figure.add_hline(
            y=90,
            line_dash="dash",
            annotation_text="90% target",
            opacity=0.6,
        )

        region_coverage_figure.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        region_coverage_figure.update_layout(
            showlegend=False,
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20,
            ),
        )

        st.plotly_chart(
            region_coverage_figure,
            use_container_width=True,
        )

    with right_column:
        region_width_figure = px.bar(
            region_uncertainty_df,
            x="actual_rul_region",
            y="average_interval_width",
            text="average_interval_width",
            title="Average Interval Width by RUL Region",
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
        )

        region_width_figure.update_traces(
            texttemplate="%{text:.1f}",
            textposition="outside",
        )

        region_width_figure.update_layout(
            showlegend=False,
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20,
            ),
        )

        st.plotly_chart(
            region_width_figure,
            use_container_width=True,
        )


    # -----------------------------------------------------
    # Engine-level interval visualization
    # -----------------------------------------------------

    st.markdown(
        f"#### Engine {selected_unit} uncertainty interval"
    )

    selected_lower = float(
        selected_engine[
            "calibrated_lower_bound"
        ]
    )

    selected_upper = float(
        selected_engine[
            "calibrated_upper_bound"
        ]
    )

    selected_mean = float(
        selected_engine[
            "predicted_rul_mean"
        ]
    )

    selected_actual = float(
        selected_engine[
            "actual_rul"
        ]
    )

    selected_interval_figure = go.Figure()

    selected_interval_figure.add_trace(
        go.Scatter(
            x=[
                selected_lower,
                selected_upper,
            ],
            y=[
                "Calibrated interval",
                "Calibrated interval",
            ],
            mode="lines",
            name="Prediction interval",
            line=dict(
                width=12,
            ),
            hovertemplate=(
                "Interval: "
                f"{selected_lower:.1f}–"
                f"{selected_upper:.1f} cycles"
                "<extra></extra>"
            ),
        )
    )

    selected_interval_figure.add_trace(
        go.Scatter(
            x=[
                selected_mean
            ],
            y=[
                "Calibrated interval"
            ],
            mode="markers",
            name="Predicted mean",
            marker=dict(
                size=14,
                symbol="circle",
            ),
        )
    )

    selected_interval_figure.add_trace(
        go.Scatter(
            x=[
                selected_actual
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

    selected_interval_figure.update_layout(
        title=(
            f"Calibrated RUL Interval — "
            f"Engine {selected_unit}"
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
        selected_interval_figure,
        use_container_width=True,
    )

    selected_is_covered = bool(
        (
            selected_actual
            >= selected_lower
        )
        and (
            selected_actual
            <= selected_upper
        )
    )

    if selected_is_covered:
        st.success(
            "The calibrated interval contains the actual "
            "RUL for this retrospective test engine."
        )

    else:
        st.error(
            "The actual RUL falls outside the calibrated "
            "prediction interval for this engine."
        )


    # -----------------------------------------------------
    # Uncertainty inspection table
    # -----------------------------------------------------

    with st.expander(
        "Inspect fleet uncertainty results"
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
                ]
            ]
            .copy()
            .rename(
                columns={
                    "unit": "Engine",
                    "actual_rul": "Actual RUL",
                    "predicted_rul_mean": "Predicted RUL",
                    "mc_standard_deviation": "MC standard deviation",
                    "calibrated_lower_bound": "Lower bound",
                    "calibrated_upper_bound": "Upper bound",
                    "interval_width": "Interval width",
                    "absolute_error": "Absolute error",
                    "interval_contains_actual": "Interval covered actual",
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
            height=450,
        )

    st.warning(
        """
        Prediction intervals quantify model uncertainty, not every
        source of real-world engineering uncertainty. They do not
        account for unobserved faults, sensor failures, maintenance
        history, domain shift or differences between simulated and
        real engines.
        """
    )
    # ---------------------------------------------------------
# Engineering Decision-Layer Validation
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-heading">'
    'Engineering Decision-Layer Validation'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    "Retrospective validation of review workload and error capture. "
    "Actual RUL labels are used only for evaluation, never to assign "
    "the operational decisions."
)


# ---------------------------------------------------------
# Load V4 versus V5 validation results
# ---------------------------------------------------------

decision_validation_dir = (
    RESULTS_DIR
    / "decision_validation_v5"
)

validation_metrics_path = (
    decision_validation_dir
    / (
        f"validation_metrics_v4_v5_"
        f"{model_name}_{dataset_name}.csv"
    )
)

transition_summary_path = (
    decision_validation_dir
    / (
        f"transition_summary_v4_v5_"
        f"{model_name}_{dataset_name}.csv"
    )
)

workload_reduction_path = (
    decision_validation_dir
    / (
        f"workload_reduction_"
        f"{model_name}_{dataset_name}.csv"
    )
)

reliability_ranking_path = (
    decision_validation_dir
    / (
        f"reliability_ranking_"
        f"{model_name}_{dataset_name}.csv"
    )
)

try:
    validation_metrics_df = load_csv(
        validation_metrics_path
    )

    transition_summary_df = load_csv(
        transition_summary_path
    )

    workload_reduction_df = load_csv(
        workload_reduction_path
    )

    reliability_ranking_df = load_csv(
        reliability_ranking_path
    )

    required_validation_columns = {
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
    }

    missing_validation_columns = sorted(
        required_validation_columns.difference(
            validation_metrics_df.columns
        )
    )

    if missing_validation_columns:
        raise KeyError(
            "Decision validation is missing columns: "
            f"{missing_validation_columns}"
        )

except (
    FileNotFoundError,
    KeyError,
) as error:
    st.warning(
        "Decision-layer validation results could not be loaded."
    )

    st.code(
        str(error)
    )

else:
    v4_row = (
        validation_metrics_df.loc[
            validation_metrics_df["version"] == "V4"
        ]
        .iloc[0]
    )

    v5_row = (
        validation_metrics_df.loc[
            validation_metrics_df["version"] == "V5"
        ]
        .iloc[0]
    )


    # -----------------------------------------------------
    # V4 to V5 headline comparison
    # -----------------------------------------------------

    st.markdown("#### V4 to V5 operational improvement")

    validation_metric_columns = st.columns(5)

    validation_metric_columns[0].metric(
        "V5 mandatory reviews",
        int(
            v5_row["required_review_count"]
        ),
        delta=(
            int(
                v5_row["required_review_count"]
            )
            - int(
                v4_row["required_review_count"]
            )
        ),
        help=(
            "Negative delta means fewer mandatory "
            "engineering reviews."
        ),
    )

    validation_metric_columns[1].metric(
        "Any-review coverage",
        (
            f"{v5_row['any_review_rate'] * 100:.0f}%"
        ),
        delta=(
            f"{(
                v5_row['any_review_rate']
                - v4_row['any_review_rate']
            ) * 100:+.0f}%"
        ),
    )

    validation_metric_columns[2].metric(
        "High-error mandatory recall",
        (
            f"{v5_row['high_error_required_recall'] * 100:.1f}%"
        ),
        delta=(
            f"{(
                v5_row['high_error_required_recall']
                - v4_row['high_error_required_recall']
            ) * 100:+.1f}%"
        ),
    )

    validation_metric_columns[3].metric(
        "Dangerous-error mandatory recall",
        (
            f"{v5_row['dangerous_required_recall'] * 100:.1f}%"
        ),
        delta=(
            f"{(
                v5_row['dangerous_required_recall']
                - v4_row['dangerous_required_recall']
            ) * 100:+.1f}%"
        ),
    )

    validation_metric_columns[4].metric(
        "Missed high-error engines",
        int(
            v5_row["missed_high_error_count"]
        ),
    )


    # -----------------------------------------------------
    # Workload and capture charts
    # -----------------------------------------------------

    chart_df = validation_metrics_df.copy()

    chart_df[
        "Mandatory review rate (%)"
    ] = (
        chart_df[
            "required_review_rate"
        ]
        * 100.0
    )

    chart_df[
        "Any review rate (%)"
    ] = (
        chart_df[
            "any_review_rate"
        ]
        * 100.0
    )

    workload_long_df = chart_df.melt(
        id_vars="version",
        value_vars=[
            "Mandatory review rate (%)",
            "Any review rate (%)",
        ],
        var_name="Review category",
        value_name="Percentage",
    )

    capture_df = validation_metrics_df[
        [
            "version",
            "high_error_required_recall",
            "dangerous_required_recall",
            "severe_dangerous_required_recall",
        ]
    ].copy()

    capture_long_df = capture_df.melt(
        id_vars="version",
        var_name="Error type",
        value_name="Recall",
    )

    capture_long_df[
        "Recall (%)"
    ] = (
        capture_long_df["Recall"]
        * 100.0
    )

    capture_name_map = {
        "high_error_required_recall": (
            "High absolute error"
        ),
        "dangerous_required_recall": (
            "Dangerous overprediction"
        ),
        "severe_dangerous_required_recall": (
            "Severe overprediction"
        ),
    }

    capture_long_df[
        "Error type"
    ] = capture_long_df[
        "Error type"
    ].map(
        capture_name_map
    )

    left_column, right_column = st.columns(2)

    with left_column:
        workload_figure = px.bar(
            workload_long_df,
            x="version",
            y="Percentage",
            color="Review category",
            barmode="group",
            text="Percentage",
            title="Engineering Review Workload",
            labels={
                "version": "Decision layer",
            },
        )

        workload_figure.update_traces(
            texttemplate="%{text:.0f}%",
            textposition="outside",
        )

        workload_figure.update_layout(
            yaxis_title="Fleet percentage (%)",
            xaxis_title="",
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20,
            ),
        )

        st.plotly_chart(
            workload_figure,
            use_container_width=True,
        )

    with right_column:
        capture_figure = px.bar(
            capture_long_df,
            x="version",
            y="Recall (%)",
            color="Error type",
            barmode="group",
            text="Recall (%)",
            title="Mandatory Review Error Capture",
            labels={
                "version": "Decision layer",
            },
        )

        capture_figure.update_traces(
            texttemplate="%{text:.1f}%",
            textposition="outside",
        )

        capture_figure.update_layout(
            xaxis_title="",
            margin=dict(
                l=20,
                r=20,
                t=55,
                b=20,
            ),
        )

        st.plotly_chart(
            capture_figure,
            use_container_width=True,
        )


    # -----------------------------------------------------
    # Workload reduction details
    # -----------------------------------------------------

    st.markdown(
        "#### What changed when moving from V4 to V5?"
    )

    workload_display_df = (
        workload_reduction_df
        .copy()
        .rename(
            columns={
                "metric": "Validation outcome",
                "engine_count": "Engines",
                "percentage_of_fleet": (
                    "Fleet percentage"
                ),
                "average_absolute_error": (
                    "Average absolute error"
                ),
            }
        )
    )

    workload_display_df[
        "Fleet percentage"
    ] = workload_display_df[
        "Fleet percentage"
    ].round(1)

    workload_display_df[
        "Average absolute error"
    ] = workload_display_df[
        "Average absolute error"
    ].round(2)

    st.dataframe(
        workload_display_df,
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        """
        V5 removed 13 mandatory reviews while keeping all of those
        engines inside the **Recommended** review category. No engine
        moved directly from **Required** to **Not required**.
        """
    )

    st.warning(
        """
        Mandatory-review recall decreased because V5 distinguishes
        between a hard engineering block and a recommended human
        review. Recommended does not mean that the prediction should
        be ignored for high-consequence decisions.
        """
    )


    # -----------------------------------------------------
    # Review transition summary
    # -----------------------------------------------------

    st.markdown(
        "#### Review-decision transitions"
    )

    transition_display_df = (
        transition_summary_df
        .copy()
        .rename(
            columns={
                "v4_review_flag": "V4 decision",
                "v5_review_flag": "V5 decision",
                "engine_count": "Engines",
                "average_absolute_error": (
                    "Average absolute error"
                ),
                "high_error_count": (
                    "High-error engines"
                ),
                "dangerous_overprediction_count": (
                    "Dangerous overpredictions"
                ),
                "severe_dangerous_count": (
                    "Severe overpredictions"
                ),
                "percentage": "Fleet percentage",
            }
        )
    )

    numeric_round_columns = [
        "Average absolute error",
        "Fleet percentage",
    ]

    transition_display_df[
        numeric_round_columns
    ] = transition_display_df[
        numeric_round_columns
    ].round(2)

    st.dataframe(
        transition_display_df,
        use_container_width=True,
        hide_index=True,
    )


    # -----------------------------------------------------
    # Reliability-risk ranking curve
    # -----------------------------------------------------

    st.markdown(
        "#### Reliability-risk ranking performance"
    )

    ranking_plot_df = (
        reliability_ranking_df
        .copy()
    )

    ranking_plot_df[
        "High-error recall (%)"
    ] = (
        ranking_plot_df[
            "high_error_recall"
        ]
        * 100.0
    )

    ranking_plot_df[
        "Dangerous-overprediction recall (%)"
    ] = (
        ranking_plot_df[
            "dangerous_overprediction_recall"
        ]
        * 100.0
    )

    ranking_plot_df[
        "Severe-overprediction recall (%)"
    ] = (
        ranking_plot_df[
            "severe_dangerous_recall"
        ]
        * 100.0
    )

    ranking_long_df = ranking_plot_df.melt(
        id_vars="review_percentage",
        value_vars=[
            "High-error recall (%)",
            "Dangerous-overprediction recall (%)",
            "Severe-overprediction recall (%)",
        ],
        var_name="Error category",
        value_name="Captured cases (%)",
    )

    ranking_figure = px.line(
        ranking_long_df,
        x="review_percentage",
        y="Captured cases (%)",
        color="Error category",
        markers=True,
        title=(
            "Error Capture When Reviewing the "
            "Highest-Risk Engines"
        ),
        labels={
            "review_percentage": (
                "Highest reliability-risk engines reviewed (%)"
            ),
        },
    )

    ranking_figure.update_layout(
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        ranking_figure,
        use_container_width=True,
    )

    st.info(
        """
        The reliability score is a prioritisation aid rather than a
        guarantee of correctness. It should be combined with engine
        condition, trajectory behaviour and engineering judgement.
        """
    )

    st.caption(
        f"Decision validation source: {validation_metrics_path}"
    )
    # ---------------------------------------------------------
# Project Architecture and Methodology
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-heading">'
    'Project Architecture & Methodology'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    "MachineGuard AI transforms raw NASA C-MAPSS sensor histories "
    "into Remaining Useful Life predictions, uncertainty estimates, "
    "temporal diagnostics and engineering decisions."
)


# ---------------------------------------------------------
# Project objective
# ---------------------------------------------------------

st.markdown("#### Engineering objective")

objective_left, objective_right = st.columns(
    [1.15, 0.85]
)

with objective_left:
    st.info(
        """
        **Prediction task**

        Given the most recent operating history of a turbofan engine,
        estimate how many operational cycles remain before the
        simulated failure threshold is reached.

        The model uses a sequence of **30 operating cycles**, with
        **3 operational settings** and **21 sensor measurements**
        recorded at each cycle.
        """
    )

with objective_right:
    objective_metrics = pd.DataFrame(
        {
            "Project element": [
                "Dataset",
                "Primary model",
                "Input sequence",
                "Input variables",
                "Target",
                "Test engines",
            ],
            "Configuration": [
                "NASA C-MAPSS FD001",
                "GRU",
                "30 cycles",
                "24 features per cycle",
                "Remaining Useful Life",
                "100 unseen engines",
            ],
        }
    )

    st.dataframe(
        objective_metrics,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# End-to-end architecture diagram
# ---------------------------------------------------------

st.markdown("#### End-to-end system architecture")

architecture_labels = [
    "NASA C-MAPSS\nRaw Data",
    "Leakage-Safe\nPreprocessing",
    "Ridge\nBaseline",
    "LSTM",
    "GRU",
    "Transformer",
    "Model\nBenchmarking",
    "Uncertainty\nEstimation",
    "Global Sensor\nExplainability",
    "Trajectory\nDiagnostics",
    "Condition\nSeverity",
    "Reliability\nRisk",
    "Engineering\nReview",
    "Operational\nPriority",
    "Streamlit\nDashboard",
]

architecture_sources = [
    0,
    1,
    1,
    1,
    1,
    2,
    3,
    4,
    5,
    4,
    4,
    4,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
]

architecture_targets = [
    1,
    2,
    3,
    4,
    5,
    6,
    6,
    6,
    6,
    7,
    8,
    9,
    11,
    11,
    11,
    12,
    12,
    13,
    14,
]

architecture_values = [
    4,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
    1,
]

architecture_figure = go.Figure(
    data=[
        go.Sankey(
            arrangement="snap",
            node=dict(
                pad=18,
                thickness=20,
                line=dict(
                    width=0.5,
                ),
                label=architecture_labels,
            ),
            link=dict(
                source=architecture_sources,
                target=architecture_targets,
                value=architecture_values,
            ),
        )
    ]
)

architecture_figure.update_layout(
    title="MachineGuard AI Data and Decision Flow",
    height=680,
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20,
    ),
)

st.plotly_chart(
    architecture_figure,
    use_container_width=True,
)


# ---------------------------------------------------------
# Methodology stages
# ---------------------------------------------------------

st.markdown("#### Methodology stages")

methodology_tabs = st.tabs(
    [
        "1. Data",
        "2. Preprocessing",
        "3. Modelling",
        "4. Evaluation",
        "5. Safety layer",
    ]
)

with methodology_tabs[0]:
    st.markdown(
        """
        **NASA C-MAPSS FD001**

        The dataset contains simulated run-to-failure histories
        for commercial turbofan engines.

        Each row represents one operating cycle and contains:

        - Engine identifier
        - Cycle number
        - Three operational settings
        - Twenty-one sensor measurements

        Training engines are observed until failure. Test engines
        stop before failure, and NASA provides one true Remaining
        Useful Life value for each final test observation.
        """
    )

    data_structure_df = pd.DataFrame(
        {
            "Dataset component": [
                "Training engines",
                "Test engines",
                "Operating conditions",
                "Fault modes",
                "Sensor channels",
            ],
            "FD001 value": [
                100,
                100,
                1,
                1,
                21,
            ],
        }
    )

    st.dataframe(
        data_structure_df,
        use_container_width=True,
        hide_index=True,
    )

with methodology_tabs[1]:
    st.markdown(
        """
        **Leakage-safe preprocessing**

        The original experimental pipeline was rebuilt to prevent
        test contamination and unrealistic evaluation.

        The corrected process:

        1. Splits complete engine IDs into training and validation.
        2. Fits the feature scaler only on training engines.
        3. Creates overlapping 30-cycle windows inside each engine.
        4. Caps training RUL at 125 cycles.
        5. Uses edge padding for engines with short histories.
        6. Creates exactly one final evaluation window per test engine.
        """
    )

    preprocessing_df = pd.DataFrame(
        {
            "Processed split": [
                "Training",
                "Validation",
                "Test",
            ],
            "Samples": [
                14459,
                3272,
                100,
            ],
            "Evaluation role": [
                "Model fitting",
                "Model selection",
                "Final unseen evaluation",
            ],
        }
    )

    st.dataframe(
        preprocessing_df,
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        """
        Entire engines—not individual overlapping windows—are assigned
        to either training or validation. This prevents near-identical
        histories from the same engine appearing on both sides.
        """
    )

with methodology_tabs[2]:
    st.markdown(
        """
        **Model benchmarking**

        Multiple modelling approaches were evaluated rather than
        assuming that a complex neural network would be best.

        - Mean prediction baseline
        - Ridge Regression
        - LSTM
        - GRU
        - Transformer
        - Validation-selected GRU–LSTM ensemble

        The GRU was selected as the primary standalone model because
        it achieved the lowest MAE and strong asymmetric NASA scoring.

        The ensemble was selected using validation engines only:
        **60% GRU + 40% LSTM**.
        """
    )

    model_role_df = pd.DataFrame(
        {
            "Model": [
                "Ridge",
                "LSTM",
                "GRU",
                "Transformer",
                "GRU–LSTM ensemble",
            ],
            "Final role": [
                "Independent interpretable baseline",
                "Secondary sequence model",
                "Primary operational model",
                "Rejected for FD001",
                "Best aggregate accuracy",
            ],
        }
    )

    st.dataframe(
        model_role_df,
        use_container_width=True,
        hide_index=True,
    )

with methodology_tabs[3]:
    st.markdown(
        """
        **Deployment-aligned evaluation**

        Each test engine contributes exactly one final prediction.

        The system reports:

        - Root Mean Squared Error
        - Mean Absolute Error
        - Coefficient of determination
        - NASA asymmetric RUL score
        - Signed prediction error
        - Dangerous overprediction rate

        Overprediction is especially important because predicting
        more remaining life than an engine truly has may delay
        maintenance.
        """
    )

    evaluation_equations_df = pd.DataFrame(
        {
            "Metric": [
                "Prediction error",
                "Absolute error",
                "RMSE",
                "MAE",
            ],
            "Meaning": [
                "Predicted RUL − actual RUL",
                "Magnitude of engine-level error",
                "Penalises large errors strongly",
                "Average prediction error magnitude",
            ],
        }
    )

    st.dataframe(
        evaluation_equations_df,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        """
        Actual test RUL values are used only after prediction for
        retrospective evaluation. They are never passed into the
        trained model or the operational decision rules.
        """
    )

with methodology_tabs[4]:
    st.markdown(
        """
        **Safety and decision-support layer**

        The final prediction is not used alone.

        MachineGuard AI also evaluates:

        - Monte Carlo dropout uncertainty
        - Calibrated prediction intervals
        - GRU versus Ridge disagreement
        - Monte Carlo versus deterministic GRU disagreement
        - Global sensor reliance
        - Prediction evolution across the engine history
        - High-RUL plateaus
        - Prediction jumps and monotonicity violations

        The V5 layer separates physical condition severity from
        prediction reliability.
        """
    )

    safety_layer_df = pd.DataFrame(
        {
            "Output": [
                "Engine condition",
                "Reliability risk",
                "Engineering review",
                "Operational priority",
            ],
            "Possible categories": [
                "Critical / Warning / Monitor / Healthy",
                "High / Medium / Low",
                "Required / Recommended / Not required",
                "Immediate / High / Elevated / Watch / Routine",
            ],
        }
    )

    st.dataframe(
        safety_layer_df,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# Original pipeline audit
# ---------------------------------------------------------

st.markdown(
    "#### What was corrected from the original implementation?"
)

audit_df = pd.DataFrame(
    [
        {
            "Original issue": "Test data used during training",
            "Engineering risk": (
                "Test-set leakage and optimistic model selection"
            ),
            "MachineGuard AI correction": (
                "Separate engine-level validation set"
            ),
        },
        {
            "Original issue": "No train-only feature scaling",
            "Engineering risk": (
                "Inconsistent optimization and possible leakage"
            ),
            "MachineGuard AI correction": (
                "StandardScaler fitted only on training engines"
            ),
        },
        {
            "Original issue": (
                "Overlapping test windows treated as observations"
            ),
            "Engineering risk": (
                "Long engines receive disproportionate evaluation weight"
            ),
            "MachineGuard AI correction": (
                "One final prediction per unseen test engine"
            ),
        },
        {
            "Original issue": "No classical baseline",
            "Engineering risk": (
                "Neural-network complexity could not be justified"
            ),
            "MachineGuard AI correction": (
                "Mean and Ridge benchmarks"
            ),
        },
        {
            "Original issue": "Single deterministic RUL output",
            "Engineering risk": (
                "No confidence or uncertainty information"
            ),
            "MachineGuard AI correction": (
                "MC dropout and calibrated intervals"
            ),
        },
        {
            "Original issue": "Only final prediction evaluated",
            "Engineering risk": (
                "Temporal model blind spots remain hidden"
            ),
            "MachineGuard AI correction": (
                "Full-history trajectory diagnostics"
            ),
        },
        {
            "Original issue": "No operational decision framework",
            "Engineering risk": (
                "Prediction cannot directly support engineering review"
            ),
            "MachineGuard AI correction": (
                "Condition, reliability, review and priority layers"
            ),
        },
    ]
)

st.dataframe(
    audit_df,
    use_container_width=True,
    hide_index=True,
    height=390,
)


# ---------------------------------------------------------
# Key technical contribution
# ---------------------------------------------------------

st.markdown("#### Key technical contribution")

contribution_left, contribution_right = st.columns(2)

with contribution_left:
    st.success(
        """
        **Beyond predictive accuracy**

        MachineGuard AI does not stop after calculating an RUL value.
        It examines whether the prediction is uncertain, whether
        independent models agree and whether the estimate behaves
        sensibly as the engine continues to age.
        """
    )

with contribution_right:
    st.error(
        """
        **Shared blind-spot detection**

        Engine 67 showed that GRU, Ridge and static uncertainty
        evidence could all agree on an incorrect high-RUL estimate.
        The temporal plateau diagnostic exposed the failure and
        triggered mandatory engineering review.
        """
    )


# ---------------------------------------------------------
# Reproducibility summary
# ---------------------------------------------------------

st.markdown("#### Reproducibility and preserved artefacts")

reproducibility_df = pd.DataFrame(
    {
        "Artefact": [
            "Processed sequences",
            "Feature scaler",
            "Trained neural networks",
            "Test predictions",
            "Uncertainty outputs",
            "Explainability outputs",
            "Trajectory histories",
            "Decision-layer outputs",
            "Validation reports",
        ],
        "Purpose": [
            "Leakage-safe model inputs",
            "Consistent feature transformation",
            "Reusable LSTM, GRU and Transformer models",
            "One deployment-style prediction per engine",
            "Calibrated confidence evidence",
            "Global sensor reliance",
            "Prediction evolution across operating cycles",
            "Condition and engineering decisions",
            "Retrospective workload and safety evaluation",
        ],
    }
)

st.dataframe(
    reproducibility_df,
    use_container_width=True,
    hide_index=True,
)

st.info(
    """
    The current portfolio release is comprehensively developed and
    evaluated on **FD001**. FD002–FD004 remain future extensions for
    multiple operating conditions and fault modes.
    """
)
# ---------------------------------------------------------
# Limitations, Responsible Use and Release Readiness
# ---------------------------------------------------------

st.divider()

st.markdown(
    '<div class="section-heading">'
    'Limitations & Responsible Use'
    '</div>',
    unsafe_allow_html=True,
)

st.caption(
    "MachineGuard AI is an engineering research prototype developed "
    "using simulated turbofan degradation data. Its outputs support "
    "analysis and human review; they do not replace certified "
    "maintenance procedures."
)


# ---------------------------------------------------------
# Responsible-use headline
# ---------------------------------------------------------

st.error(
    """
    **Research prototype — not approved for real aircraft maintenance**

    MachineGuard AI has been trained and evaluated on the simulated
    NASA C-MAPSS FD001 benchmark. It must not be used to release,
    ground, maintain or certify a real aircraft engine without
    extensive validation using representative operational data,
    physical inspections, safety analysis and regulatory approval.
    """
)


# ---------------------------------------------------------
# Retrospective versus deployment information
# ---------------------------------------------------------

st.markdown(
    "#### Retrospective evaluation versus real deployment"
)

evaluation_left, evaluation_right = st.columns(2)

with evaluation_left:
    st.info(
        """
        **What this dashboard shows during retrospective testing**

        - Predicted Remaining Useful Life
        - Actual test RUL
        - Prediction error
        - Whether the calibrated interval covered the truth
        - Review-rule recall and workload
        - Error behaviour across engine-life regions

        Actual RUL is displayed because NASA provides test labels
        for model evaluation.
        """
    )

with evaluation_right:
    st.warning(
        """
        **What would be available in a real deployment**

        - Current sensor history
        - Model RUL estimate
        - Prediction interval
        - Model disagreement
        - Trajectory behaviour
        - Condition classification
        - Reliability and review recommendation

        The future failure point and actual RUL would be unknown.
        """
    )


# ---------------------------------------------------------
# Main technical limitations
# ---------------------------------------------------------

st.markdown("#### Current technical limitations")

limitations_df = pd.DataFrame(
    [
        {
            "Limitation": "Simulated source data",
            "Why it matters": (
                "C-MAPSS represents physics-informed simulated "
                "degradation, not measurements from an operating fleet."
            ),
            "Required future work": (
                "Validate on representative real-world engine data."
            ),
        },
        {
            "Limitation": "FD001 only",
            "Why it matters": (
                "The current release contains one operating condition "
                "and one fault mode."
            ),
            "Required future work": (
                "Extend and independently validate on FD002–FD004."
            ),
        },
        {
            "Limitation": "RUL capped at 125 cycles",
            "Why it matters": (
                "Predictions in the early healthy phase describe "
                "at least approximately 125 remaining cycles rather "
                "than exact long-horizon life."
            ),
            "Required future work": (
                "Evaluate alternative target formulations and caps."
            ),
        },
        {
            "Limitation": "Thirty-cycle input window",
            "Why it matters": (
                "The model sees only a recent sequence rather than "
                "the complete maintenance and operating history."
            ),
            "Required future work": (
                "Study longer histories and multi-scale temporal inputs."
            ),
        },
        {
            "Limitation": "Monte Carlo dropout uncertainty",
            "Why it matters": (
                "The interval mainly represents model uncertainty and "
                "does not capture every operational risk."
            ),
            "Required future work": (
                "Compare conformal prediction, ensembles and Bayesian "
                "approaches."
            ),
        },
        {
            "Limitation": "Rule-based decision thresholds",
            "Why it matters": (
                "Review categories and reliability thresholds are "
                "engineering heuristics, not certified safety limits."
            ),
            "Required future work": (
                "Calibrate rules on independent validation datasets."
            ),
        },
        {
            "Limitation": "Global occlusion explainability",
            "Why it matters": (
                "Feature importance describes model reliance and does "
                "not establish physical causality."
            ),
            "Required future work": (
                "Add local explanations and domain-expert validation."
            ),
        },
        {
            "Limitation": "Unknown domain shift",
            "Why it matters": (
                "Different engine types, sensors, environments or "
                "maintenance practices may change model behaviour."
            ),
            "Required future work": (
                "Add drift detection and out-of-distribution monitoring."
            ),
        },
        {
            "Limitation": "No sensor-fault model",
            "Why it matters": (
                "A failed or biased sensor could produce misleading "
                "RUL estimates."
            ),
            "Required future work": (
                "Add sensor-quality checks and fault-tolerant inference."
            ),
        },
    ]
)

st.dataframe(
    limitations_df,
    use_container_width=True,
    hide_index=True,
    height=500,
)


# ---------------------------------------------------------
# Uncertainty sources not fully represented
# ---------------------------------------------------------

st.markdown(
    "#### Sources of uncertainty outside the current model"
)

uncertainty_source_columns = st.columns(3)

with uncertainty_source_columns[0]:
    st.markdown(
        """
        **Data uncertainty**

        - Sensor noise
        - Missing measurements
        - Sensor calibration drift
        - Faulty instrumentation
        - Unseen operating conditions
        """
    )

with uncertainty_source_columns[1]:
    st.markdown(
        """
        **Model uncertainty**

        - Limited training engines
        - Architecture assumptions
        - Hyperparameter selection
        - Shared model blind spots
        - Dataset shift
        """
    )

with uncertainty_source_columns[2]:
    st.markdown(
        """
        **Operational uncertainty**

        - Maintenance history
        - Component replacement
        - Mission severity
        - Environmental exposure
        - Human inspection findings
        """
    )


# ---------------------------------------------------------
# Safe engineering workflow
# ---------------------------------------------------------

st.markdown("#### Recommended engineering workflow")

workflow_df = pd.DataFrame(
    [
        {
            "Step": 1,
            "Engineering activity": "Validate incoming data",
            "Purpose": (
                "Check sensor availability, units, ranges and quality."
            ),
        },
        {
            "Step": 2,
            "Engineering activity": "Generate RUL evidence",
            "Purpose": (
                "Calculate model prediction, uncertainty interval "
                "and independent-baseline estimate."
            ),
        },
        {
            "Step": 3,
            "Engineering activity": "Inspect temporal behaviour",
            "Purpose": (
                "Review degradation slope, plateaus, jumps and "
                "monotonicity violations."
            ),
        },
        {
            "Step": 4,
            "Engineering activity": "Review system recommendation",
            "Purpose": (
                "Consider condition severity, reliability risk and "
                "operational priority separately."
            ),
        },
        {
            "Step": 5,
            "Engineering activity": "Combine with physical evidence",
            "Purpose": (
                "Use inspections, maintenance records and expert "
                "knowledge before taking action."
            ),
        },
        {
            "Step": 6,
            "Engineering activity": "Record the final decision",
            "Purpose": (
                "Preserve the model version, input data, evidence "
                "and responsible engineer approval."
            ),
        },
    ]
)

st.dataframe(
    workflow_df,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# Interpretation of decision categories
# ---------------------------------------------------------

st.markdown("#### How to interpret the decision outputs")

decision_meaning_tabs = st.tabs(
    [
        "Condition",
        "Reliability",
        "Review",
        "Priority",
    ]
)

with decision_meaning_tabs[0]:
    st.markdown(
        """
        **Engine condition** describes the estimated physical urgency
        based primarily on conservative Remaining Useful Life.

        - **Critical:** immediate condition concern
        - **Warning:** early maintenance preparation
        - **Monitor:** enhanced monitoring
        - **Healthy:** routine condition monitoring

        This category does not describe whether the prediction itself
        is trustworthy.
        """
    )

with decision_meaning_tabs[1]:
    st.markdown(
        """
        **Reliability risk** describes how suspicious or unstable the
        prediction evidence appears.

        - **High:** substantial uncertainty or conflicting evidence
        - **Medium:** caution is needed
        - **Low:** evidence is comparatively stable

        Low reliability risk does not guarantee that the prediction
        is correct. Engine 67 demonstrates a shared blind spot that
        required an explicit temporal safety rule.
        """
    )

with decision_meaning_tabs[2]:
    st.markdown(
        """
        **Engineering review** determines the strength of the human
        review recommendation.

        - **Required:** do not use the RUL for a consequential decision
          before manual engineering review
        - **Recommended:** human review is advised before a
          high-consequence decision
        - **Not required:** no additional model-triggered review,
          while normal engineering processes still apply
        """
    )

with decision_meaning_tabs[3]:
    st.markdown(
        """
        **Operational priority** combines estimated condition urgency
        and prediction reliability evidence.

        - **Immediate**
        - **High**
        - **Elevated**
        - **Watch**
        - **Routine**

        It is a prioritisation aid, not an automatic maintenance order.
        """
    )


# ---------------------------------------------------------
# Release readiness
# ---------------------------------------------------------

st.markdown("#### FD001 release readiness")

release_readiness_df = pd.DataFrame(
    [
        {
            "Component": "Leakage-safe preprocessing",
            "Status": "Complete",
            "Evidence": (
                "Engine-level split and train-only scaling"
            ),
        },
        {
            "Component": "Baseline benchmarking",
            "Status": "Complete",
            "Evidence": (
                "Mean and Ridge baselines"
            ),
        },
        {
            "Component": "Sequence-model training",
            "Status": "Complete",
            "Evidence": (
                "LSTM, GRU and Transformer evaluated"
            ),
        },
        {
            "Component": "Deployment-style testing",
            "Status": "Complete",
            "Evidence": (
                "One final prediction per test engine"
            ),
        },
        {
            "Component": "Uncertainty estimation",
            "Status": "Complete",
            "Evidence": (
                "MC dropout and calibrated intervals"
            ),
        },
        {
            "Component": "Global explainability",
            "Status": "Complete",
            "Evidence": (
                "Occlusion-based feature reliance"
            ),
        },
        {
            "Component": "Trajectory diagnostics",
            "Status": "Complete",
            "Evidence": (
                "13,096 historical prediction windows"
            ),
        },
        {
            "Component": "Decision-support layer",
            "Status": "Complete",
            "Evidence": (
                "V5 condition, reliability, review and priority"
            ),
        },
        {
            "Component": "Decision-layer validation",
            "Status": "Complete",
            "Evidence": (
                "V4–V5 safety and workload comparison"
            ),
        },
        {
            "Component": "Interactive dashboard",
            "Status": "In progress",
            "Evidence": (
                "Core analytical sections implemented"
            ),
        },
        {
            "Component": "FD002–FD004 validation",
            "Status": "Future work",
            "Evidence": (
                "Not included in the current release"
            ),
        },
        {
            "Component": "Real-engine certification",
            "Status": "Out of scope",
            "Evidence": (
                "Requires operational data and regulatory processes"
            ),
        },
    ]
)

st.dataframe(
    release_readiness_df,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# Future roadmap
# ---------------------------------------------------------

st.markdown("#### Development roadmap")

roadmap_columns = st.columns(3)

with roadmap_columns[0]:
    st.success(
        """
        **Portfolio Release 1**

        - Complete FD001 dashboard
        - Add navigation
        - Clean requirements
        - Write professional README
        - Add screenshots
        - Publish repository
        """
    )

with roadmap_columns[1]:
    st.info(
        """
        **Research Release 2**

        - Extend to FD002–FD004
        - Add operating-condition normalization
        - Add fault-mode analysis
        - Compare cross-dataset generalization
        - Validate decision rules independently
        """
    )

with roadmap_columns[2]:
    st.warning(
        """
        **Industrialization Path**

        - Real sensor ingestion
        - Data-quality monitoring
        - Drift detection
        - Model registry
        - Audit logging
        - Human approval workflow
        - Safety and regulatory validation
        """
    )


# ---------------------------------------------------------
# Final project statement
# ---------------------------------------------------------

st.markdown("#### Final project statement")

st.success(
    """
    MachineGuard AI is an end-to-end predictive-maintenance research
    platform for NASA C-MAPSS FD001. It combines leakage-safe RUL
    prediction, model benchmarking, calibrated uncertainty, global
    sensor explainability, temporal blind-spot detection and
    engineering decision support.

    The system is designed to demonstrate responsible ML engineering:
    predictive performance is treated as only one part of a broader
    evidence and human-review process.
    """
)

st.caption(
    "MachineGuard AI — FD001 research and portfolio release"
)