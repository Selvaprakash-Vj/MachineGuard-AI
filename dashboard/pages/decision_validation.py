import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import RESULTS_DIR


DATASET_NAME = "FD001"
MODEL_NAME = "gru"

SELECTED_CASE_UNITS = [
    25,
    45,
    67,
    79,
]


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


def validate_columns(
    dataframe,
    required_columns,
    file_description,
):
    """Validate required dataframe columns."""

    missing_columns = sorted(
        set(required_columns).difference(
            dataframe.columns
        )
    )

    if missing_columns:
        raise KeyError(
            f"{file_description} is missing columns: "
            f"{missing_columns}"
        )


validation_dir = (
    RESULTS_DIR
    / "decision_validation_v5"
)

validation_metrics_path = (
    validation_dir
    / (
        f"validation_metrics_v4_v5_"
        f"{MODEL_NAME}_{DATASET_NAME}.csv"
    )
)

review_performance_path = (
    validation_dir
    / (
        f"review_performance_v4_v5_"
        f"{MODEL_NAME}_{DATASET_NAME}.csv"
    )
)

transition_summary_path = (
    validation_dir
    / (
        f"transition_summary_v4_v5_"
        f"{MODEL_NAME}_{DATASET_NAME}.csv"
    )
)

engine_transitions_path = (
    validation_dir
    / (
        f"engine_transitions_v4_v5_"
        f"{MODEL_NAME}_{DATASET_NAME}.csv"
    )
)

workload_reduction_path = (
    validation_dir
    / (
        f"workload_reduction_"
        f"{MODEL_NAME}_{DATASET_NAME}.csv"
    )
)

reliability_ranking_path = (
    validation_dir
    / (
        f"reliability_ranking_"
        f"{MODEL_NAME}_{DATASET_NAME}.csv"
    )
)

score_correlations_path = (
    validation_dir
    / (
        f"score_correlations_"
        f"{MODEL_NAME}_{DATASET_NAME}.csv"
    )
)

priority_validation_path = (
    validation_dir
    / (
        f"operational_priority_validation_"
        f"{MODEL_NAME}_{DATASET_NAME}.csv"
    )
)


try:
    validation_metrics_df = load_csv(
        validation_metrics_path
    )

    review_performance_df = load_csv(
        review_performance_path
    )

    transition_summary_df = load_csv(
        transition_summary_path
    )

    engine_transitions_df = load_csv(
        engine_transitions_path
    )

    workload_reduction_df = load_csv(
        workload_reduction_path
    )

    reliability_ranking_df = load_csv(
        reliability_ranking_path
    )

    score_correlations_df = load_csv(
        score_correlations_path
    )

    priority_validation_df = load_csv(
        priority_validation_path
    )


    validate_columns(
        validation_metrics_df,
        {
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
        },
        "Validation metrics",
    )

    validate_columns(
        review_performance_df,
        {
            "version",
            "review_flag",
            "engine_count",
            "percentage",
            "average_absolute_error",
            "median_absolute_error",
            "high_error_rate",
            "dangerous_overprediction_rate",
            "severe_dangerous_rate",
        },
        "Review performance",
    )

    validate_columns(
        transition_summary_df,
        {
            "v4_review_flag",
            "v5_review_flag",
            "engine_count",
            "average_absolute_error",
            "high_error_count",
            "dangerous_overprediction_count",
            "severe_dangerous_count",
            "percentage",
        },
        "Transition summary",
    )

    validate_columns(
        engine_transitions_df,
        {
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
        },
        "Engine transitions",
    )

    validate_columns(
        workload_reduction_df,
        {
            "metric",
            "engine_count",
            "percentage_of_fleet",
            "average_absolute_error",
        },
        "Workload reduction",
    )

    validate_columns(
        reliability_ranking_df,
        {
            "review_percentage",
            "reviewed_engines",
            "average_absolute_error",
            "high_error_recall",
            "dangerous_overprediction_recall",
            "severe_dangerous_recall",
        },
        "Reliability ranking",
    )

except (
    FileNotFoundError,
    KeyError,
) as error:
    st.title("Decision Validation")

    st.error(
        "Decision-validation results could not be loaded."
    )

    st.code(
        str(error)
    )

    st.info(
        "Run validate_decision_layer_v5.py before "
        "opening this page."
    )

    st.stop()


# ---------------------------------------------------------
# Prepare version rows
# ---------------------------------------------------------

v4_rows = validation_metrics_df.loc[
    validation_metrics_df[
        "version"
    ]
    == "V4"
]

v5_rows = validation_metrics_df.loc[
    validation_metrics_df[
        "version"
    ]
    == "V5"
]


if v4_rows.empty or v5_rows.empty:
    st.error(
        "The validation file must contain both V4 and V5 results."
    )

    st.stop()


v4_row = v4_rows.iloc[0]
v5_row = v5_rows.iloc[0]


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("---")
    st.markdown("### Validation controls")

    recall_scope = st.radio(
        "Recall display",
        options=[
            "Mandatory review",
            "Any review",
        ],
        index=0,
        help=(
            "Mandatory review shows hard engineering blocks. "
            "Any review includes Required and Recommended."
        ),
    )

    selected_transition_units = st.multiselect(
        "Case-study engines",
        options=sorted(
            engine_transitions_df[
                "unit"
            ]
            .astype(int)
            .unique()
            .tolist()
        ),
        default=[
            unit
            for unit in SELECTED_CASE_UNITS
            if unit
            in engine_transitions_df[
                "unit"
            ].astype(int).tolist()
        ],
    )

    st.caption(
        "Actual RUL is used retrospectively only for "
        "validation of the decision rules."
    )


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">Decision Validation</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Retrospective evaluation of engineering-review workload,
        error capture and the transition from decision layer V4 to V5
    </div>
    """,
    unsafe_allow_html=True,
)

st.info(
    """
    The decision rules do not use actual RUL when assigning review
    categories. Actual test labels are used afterward to measure
    whether the rules captured difficult or dangerous predictions.
    """
)


# ---------------------------------------------------------
# Headline V4 versus V5 results
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'V4 vs V5 Headline Results'
    '</div>',
    unsafe_allow_html=True,
)


mandatory_review_delta = int(
    v5_row[
        "required_review_count"
    ]
    - v4_row[
        "required_review_count"
    ]
)

high_error_recall_delta = (
    v5_row[
        "high_error_required_recall"
    ]
    - v4_row[
        "high_error_required_recall"
    ]
)

dangerous_recall_delta = (
    v5_row[
        "dangerous_required_recall"
    ]
    - v4_row[
        "dangerous_required_recall"
    ]
)

severe_recall_delta = (
    v5_row[
        "severe_dangerous_required_recall"
    ]
    - v4_row[
        "severe_dangerous_required_recall"
    ]
)


headline_columns = st.columns(
    6
)

headline_columns[0].metric(
    "V4 mandatory reviews",
    int(
        v4_row[
            "required_review_count"
        ]
    ),
)

headline_columns[1].metric(
    "V5 mandatory reviews",
    int(
        v5_row[
            "required_review_count"
        ]
    ),
    delta=mandatory_review_delta,
    delta_color="inverse",
)

headline_columns[2].metric(
    "V5 any-review coverage",
    (
        f"{v5_row['any_review_rate'] * 100:.0f}%"
    ),
)

headline_columns[3].metric(
    "High-error mandatory recall",
    (
        f"{v5_row['high_error_required_recall'] * 100:.1f}%"
    ),
    delta=(
        f"{high_error_recall_delta * 100:+.1f}%"
    ),
)

headline_columns[4].metric(
    "Dangerous mandatory recall",
    (
        f"{v5_row['dangerous_required_recall'] * 100:.1f}%"
    ),
    delta=(
        f"{dangerous_recall_delta * 100:+.1f}%"
    ),
)

headline_columns[5].metric(
    "Severe mandatory recall",
    (
        f"{v5_row['severe_dangerous_required_recall'] * 100:.1f}%"
    ),
    delta=(
        f"{severe_recall_delta * 100:+.1f}%"
    ),
)


st.success(
    """
    V5 reduced mandatory engineering reviews from **61 engines to
    48 engines** while keeping the same **87% any-review coverage**.
    No engine moved directly from Required to Not required.
    """
)


# ---------------------------------------------------------
# Workload comparison
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Engineering Review Workload'
    '</div>',
    unsafe_allow_html=True,
)


workload_chart_df = (
    validation_metrics_df[
        [
            "version",
            "required_review_rate",
            "any_review_rate",
        ]
    ]
    .copy()
)

workload_chart_df[
    "Mandatory review (%)"
] = (
    workload_chart_df[
        "required_review_rate"
    ]
    * 100.0
)

workload_chart_df[
    "Any review (%)"
] = (
    workload_chart_df[
        "any_review_rate"
    ]
    * 100.0
)


workload_long_df = workload_chart_df.melt(
    id_vars="version",
    value_vars=[
        "Mandatory review (%)",
        "Any review (%)",
    ],
    var_name="Review scope",
    value_name="Fleet percentage",
)


workload_figure = px.bar(
    workload_long_df,
    x="version",
    y="Fleet percentage",
    color="Review scope",
    barmode="group",
    text="Fleet percentage",
    title="Review Workload Across Decision Layers",
    labels={
        "version": "Decision layer",
    },
)

workload_figure.update_traces(
    texttemplate="%{text:.0f}%",
    textposition="outside",
)

workload_figure.update_layout(
    xaxis_title="",
    yaxis_title="Fleet percentage (%)",
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20,
    ),
)

st.plotly_chart(
    workload_figure,
    use_container_width=True,
)


# ---------------------------------------------------------
# Error-capture comparison
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Error Capture'
    '</div>',
    unsafe_allow_html=True,
)


if recall_scope == "Mandatory review":
    recall_columns = {
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

    recall_title = (
        "Error Cases Captured by Mandatory Review"
    )

else:
    recall_columns = {
        "high_error_any_review_recall": (
            "High absolute error"
        ),
        "dangerous_any_review_recall": (
            "Dangerous overprediction"
        ),
        "severe_dangerous_any_review_recall": (
            "Severe overprediction"
        ),
    }

    recall_title = (
        "Error Cases Captured by Any Review"
    )


available_recall_columns = [
    column
    for column in recall_columns
    if column
    in validation_metrics_df.columns
]


capture_df = validation_metrics_df[
    [
        "version",
        *available_recall_columns,
    ]
].copy()


capture_long_df = capture_df.melt(
    id_vars="version",
    value_vars=available_recall_columns,
    var_name="Error type",
    value_name="Recall",
)

capture_long_df[
    "Error type"
] = capture_long_df[
    "Error type"
].map(
    recall_columns
)

capture_long_df[
    "Recall (%)"
] = (
    capture_long_df[
        "Recall"
    ]
    * 100.0
)


capture_figure = px.bar(
    capture_long_df,
    x="version",
    y="Recall (%)",
    color="Error type",
    barmode="group",
    text="Recall (%)",
    title=recall_title,
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
        t=60,
        b=20,
    ),
)

st.plotly_chart(
    capture_figure,
    use_container_width=True,
)


if recall_scope == "Mandatory review":
    st.warning(
        """
        V5 deliberately reduced hard mandatory-review recall because
        13 engines were downgraded from Required to Recommended.
        These engines still remained inside the human-review workflow.
        """
    )

else:
    st.info(
        """
        Any-review recall includes both Required and Recommended
        categories. This better represents whether a prediction remains
        inside the human engineering-review process.
        """
    )


# ---------------------------------------------------------
# Review-category performance
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Performance by Review Category'
    '</div>',
    unsafe_allow_html=True,
)


review_plot_df = review_performance_df.copy()

review_plot_df[
    "High-error rate (%)"
] = (
    review_plot_df[
        "high_error_rate"
    ]
    * 100.0
)

review_plot_df[
    "Dangerous overprediction (%)"
] = (
    review_plot_df[
        "dangerous_overprediction_rate"
    ]
    * 100.0
)

review_plot_df[
    "Severe overprediction (%)"
] = (
    review_plot_df[
        "severe_dangerous_rate"
    ]
    * 100.0
)


left_column, right_column = st.columns(
    2
)


with left_column:
    review_error_figure = px.bar(
        review_plot_df,
        x="review_flag",
        y="average_absolute_error",
        color="version",
        barmode="group",
        text="average_absolute_error",
        title="Average Absolute Error by Review Category",
        labels={
            "review_flag": "Review decision",
            "average_absolute_error": (
                "Average absolute error (cycles)"
            ),
            "version": "Decision layer",
        },
    )

    review_error_figure.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    review_error_figure.update_layout(
        xaxis_title="",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    st.plotly_chart(
        review_error_figure,
        use_container_width=True,
    )


with right_column:
    review_danger_figure = px.bar(
        review_plot_df,
        x="review_flag",
        y="Dangerous overprediction (%)",
        color="version",
        barmode="group",
        text="Dangerous overprediction (%)",
        title="Dangerous Overprediction Rate",
        labels={
            "review_flag": "Review decision",
            "version": "Decision layer",
        },
    )

    review_danger_figure.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
    )

    review_danger_figure.update_layout(
        xaxis_title="",
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=20,
        ),
    )

    st.plotly_chart(
        review_danger_figure,
        use_container_width=True,
    )


review_display_df = (
    review_performance_df
    .copy()
    .rename(
        columns={
            "version": "Version",
            "review_flag": "Review decision",
            "engine_count": "Engines",
            "percentage": "Fleet percentage",
            "average_absolute_error": (
                "Average absolute error"
            ),
            "median_absolute_error": (
                "Median absolute error"
            ),
            "high_error_rate": (
                "High-error rate"
            ),
            "dangerous_overprediction_rate": (
                "Dangerous-overprediction rate"
            ),
            "severe_dangerous_rate": (
                "Severe-overprediction rate"
            ),
        }
    )
)


for percentage_column in [
    "High-error rate",
    "Dangerous-overprediction rate",
    "Severe-overprediction rate",
]:
    review_display_df[
        percentage_column
    ] = (
        review_display_df[
            percentage_column
        ]
        * 100.0
    ).round(
        1
    )


for numeric_column in [
    "Fleet percentage",
    "Average absolute error",
    "Median absolute error",
]:
    review_display_df[
        numeric_column
    ] = review_display_df[
        numeric_column
    ].round(
        2
    )


st.dataframe(
    review_display_df,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# Review transition matrix
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'V4 to V5 Review Transitions'
    '</div>',
    unsafe_allow_html=True,
)


review_levels = [
    "Required",
    "Recommended",
    "Not required",
]


transition_matrix = pd.crosstab(
    engine_transitions_df[
        "v4_review_flag"
    ],
    engine_transitions_df[
        "v5_review_flag"
    ],
).reindex(
    index=review_levels,
    columns=review_levels,
    fill_value=0,
)


matrix_values = (
    transition_matrix.to_numpy()
)


transition_figure = go.Figure(
    data=go.Heatmap(
        z=matrix_values,
        x=review_levels,
        y=review_levels,
        text=matrix_values,
        texttemplate="%{text}",
        hovertemplate=(
            "V4: %{y}<br>"
            "V5: %{x}<br>"
            "Engines: %{z}"
            "<extra></extra>"
        ),
        colorbar=dict(
            title="Engines"
        ),
    )
)


transition_figure.update_layout(
    title="Engineering Review Transition Matrix",
    xaxis_title="V5 decision",
    yaxis_title="V4 decision",
    height=520,
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20,
    ),
)

st.plotly_chart(
    transition_figure,
    use_container_width=True,
)


st.success(
    """
    The only V4-to-V5 downgrade was **Required → Recommended**.
    Thirteen engines changed category, while none moved directly to
    Not required.
    """
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


transition_display_df[
    "Average absolute error"
] = transition_display_df[
    "Average absolute error"
].round(
    2
)

transition_display_df[
    "Fleet percentage"
] = transition_display_df[
    "Fleet percentage"
].round(
    1
)


st.dataframe(
    transition_display_df,
    use_container_width=True,
    hide_index=True,
)


# ---------------------------------------------------------
# Workload reduction analysis
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Mandatory-Review Workload Reduction'
    '</div>',
    unsafe_allow_html=True,
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
].round(
    1
)

workload_display_df[
    "Average absolute error"
] = workload_display_df[
    "Average absolute error"
].round(
    2
)


st.dataframe(
    workload_display_df,
    use_container_width=True,
    hide_index=True,
)


reduction_columns = st.columns(
    4
)


mandatory_removed_row = (
    workload_reduction_df.loc[
        workload_reduction_df[
            "metric"
        ]
        == "Mandatory reviews removed"
    ]
)

low_error_removed_row = (
    workload_reduction_df.loc[
        workload_reduction_df[
            "metric"
        ]
        == "Low-error mandatory reviews removed"
    ]
)

high_error_removed_row = (
    workload_reduction_df.loc[
        workload_reduction_df[
            "metric"
        ]
        == (
            "High-error engines removed from "
            "mandatory review"
        )
    ]
)

severe_removed_row = (
    workload_reduction_df.loc[
        workload_reduction_df[
            "metric"
        ]
        == (
            "Severe overpredictions removed "
            "from mandatory review"
        )
    ]
)


if not mandatory_removed_row.empty:
    reduction_columns[0].metric(
        "Mandatory reviews removed",
        int(
            mandatory_removed_row.iloc[0][
                "engine_count"
            ]
        ),
    )

if not low_error_removed_row.empty:
    reduction_columns[1].metric(
        "Low-error reviews removed",
        int(
            low_error_removed_row.iloc[0][
                "engine_count"
            ]
        ),
    )

if not high_error_removed_row.empty:
    reduction_columns[2].metric(
        "High-error cases downgraded",
        int(
            high_error_removed_row.iloc[0][
                "engine_count"
            ]
        ),
    )

if not severe_removed_row.empty:
    reduction_columns[3].metric(
        "Severe cases downgraded",
        int(
            severe_removed_row.iloc[0][
                "engine_count"
            ]
        ),
    )


st.warning(
    """
    Four dangerous overpredictions moved from Required to Recommended,
    but no severe overprediction moved out of mandatory review.
    Recommended review must therefore remain meaningful for
    high-consequence maintenance decisions.
    """
)


# ---------------------------------------------------------
# Reliability-risk ranking
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Reliability-Risk Ranking Performance'
    '</div>',
    unsafe_allow_html=True,
)


ranking_plot_df = reliability_ranking_df.copy()

ranking_plot_df[
    "High-error recall (%)"
] = (
    ranking_plot_df[
        "high_error_recall"
    ]
    * 100.0
)

ranking_plot_df[
    "Dangerous recall (%)"
] = (
    ranking_plot_df[
        "dangerous_overprediction_recall"
    ]
    * 100.0
)

ranking_plot_df[
    "Severe recall (%)"
] = (
    ranking_plot_df[
        "severe_dangerous_recall"
    ]
    * 100.0
)


ranking_long_df = ranking_plot_df.melt(
    id_vars=[
        "review_percentage",
        "reviewed_engines",
        "average_absolute_error",
    ],
    value_vars=[
        "High-error recall (%)",
        "Dangerous recall (%)",
        "Severe recall (%)",
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
        "Highest Reliability-Risk Engines"
    ),
    labels={
        "review_percentage": (
            "Highest-risk engines reviewed (%)"
        ),
    },
    hover_data={
        "reviewed_engines": True,
        "average_absolute_error": ":.2f",
    },
)

ranking_figure.update_layout(
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20,
    ),
)

st.plotly_chart(
    ranking_figure,
    use_container_width=True,
)


st.info(
    """
    Reviewing the highest 70% of engines by reliability-risk score
    captured all high-error and severe-overprediction cases in the
    retrospective FD001 evaluation. The score is useful for ranking,
    but it is not a guarantee of correctness.
    """
)


# ---------------------------------------------------------
# Reliability score correlations
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Reliability Score Behaviour'
    '</div>',
    unsafe_allow_html=True,
)


if not score_correlations_df.empty:
    correlation_display_df = (
        score_correlations_df
        .copy()
        .rename(
            columns={
                "metric": "Correlation",
                "value": "Value",
            }
        )
    )

    correlation_display_df[
        "Value"
    ] = correlation_display_df[
        "Value"
    ].round(
        3
    )

    st.dataframe(
        correlation_display_df,
        use_container_width=True,
        hide_index=True,
    )


    absolute_error_correlation_row = (
        score_correlations_df.loc[
            score_correlations_df[
                "metric"
            ]
            == (
                "Pearson reliability risk "
                "vs absolute error"
            )
        ]
    )

    if not absolute_error_correlation_row.empty:
        correlation_value = float(
            absolute_error_correlation_row.iloc[0][
                "value"
            ]
        )

        st.info(
            f"""
            The Pearson correlation between reliability-risk score
            and absolute prediction error is **{correlation_value:.3f}**.
            This is positive but moderate, so the score should remain
            supporting evidence rather than an automated truth measure.
            """
        )


# ---------------------------------------------------------
# Operational-priority validation
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Operational Priority Validation'
    '</div>',
    unsafe_allow_html=True,
)


if not priority_validation_df.empty:
    priority_display_df = (
        priority_validation_df
        .copy()
        .rename(
            columns={
                "operational_priority": (
                    "Operational priority"
                ),
                "engine_count": "Engines",
                "percentage": "Fleet percentage",
                "average_actual_rul": (
                    "Average actual RUL"
                ),
                "average_conservative_rul": (
                    "Average conservative RUL"
                ),
                "average_absolute_error": (
                    "Average absolute error"
                ),
                "dangerous_overprediction_rate": (
                    "Dangerous-overprediction rate"
                ),
                "average_condition_severity_score": (
                    "Average condition severity"
                ),
                "average_reliability_risk_score": (
                    "Average reliability risk"
                ),
            }
        )
    )


    if (
        "Dangerous-overprediction rate"
        in priority_display_df.columns
    ):
        priority_display_df[
            "Dangerous-overprediction rate"
        ] = (
            priority_display_df[
                "Dangerous-overprediction rate"
            ]
            * 100.0
        ).round(
            1
        )


    numeric_priority_columns = [
        "Fleet percentage",
        "Average actual RUL",
        "Average conservative RUL",
        "Average absolute error",
        "Average condition severity",
        "Average reliability risk",
    ]

    for column in numeric_priority_columns:
        if column in priority_display_df.columns:
            priority_display_df[
                column
            ] = priority_display_df[
                column
            ].round(
                2
            )


    st.dataframe(
        priority_display_df,
        use_container_width=True,
        hide_index=True,
    )


# ---------------------------------------------------------
# Selected engine transitions
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Selected Engine Case Studies'
    '</div>',
    unsafe_allow_html=True,
)


selected_transition_df = (
    engine_transitions_df.loc[
        engine_transitions_df[
            "unit"
        ]
        .astype(int)
        .isin(
            selected_transition_units
        )
    ]
    .copy()
    .sort_values(
        by="unit"
    )
)


if selected_transition_df.empty:
    st.warning(
        "No case-study engines are selected."
    )

else:
    case_display_df = (
        selected_transition_df[
            [
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
        ]
        .copy()
        .rename(
            columns={
                "unit": "Engine",
                "actual_rul": "Actual RUL",
                "predicted_rul_mean": "Predicted RUL",
                "absolute_error": "Absolute error",
                "prediction_error": "Signed error",
                "v4_review_flag": "V4 review",
                "v5_review_flag": "V5 review",
                "v5_trajectory_flag": (
                    "Trajectory diagnostic"
                ),
                "reliability_risk_score": (
                    "Reliability-risk score"
                ),
                "reliability_risk": (
                    "Reliability risk"
                ),
                "operational_priority": (
                    "Operational priority"
                ),
                "mc_deterministic_gap": (
                    "MC–deterministic gap"
                ),
                "review_change": (
                    "Review-level change"
                ),
            }
        )
    )


    numeric_case_columns = [
        "Actual RUL",
        "Predicted RUL",
        "Absolute error",
        "Signed error",
        "Reliability-risk score",
        "MC–deterministic gap",
    ]

    case_display_df[
        numeric_case_columns
    ] = case_display_df[
        numeric_case_columns
    ].round(
        2
    )


    st.dataframe(
        case_display_df,
        use_container_width=True,
        hide_index=True,
    )


    for _, case_row in selected_transition_df.iterrows():
        unit = int(
            case_row[
                "unit"
            ]
        )

        with st.expander(
            f"Engine {unit} validation interpretation"
        ):
            st.markdown(
                f"""
                **Actual RUL:** {case_row['actual_rul']:.1f} cycles

                **Predicted RUL:** {case_row['predicted_rul_mean']:.1f}
                cycles

                **Absolute error:** {case_row['absolute_error']:.1f}
                cycles

                **V4 review:** {case_row['v4_review_flag']}

                **V5 review:** {case_row['v5_review_flag']}

                **Trajectory diagnostic:**
                {case_row['v5_trajectory_flag']}

                **Reliability risk:**
                {case_row['reliability_risk']}
                ({case_row['reliability_risk_score']:.1f})

                **Operational priority:**
                {case_row['operational_priority']}
                """
            )


            if unit == 67:
                st.error(
                    """
                    Engine 67 is the critical shared-model blind-spot
                    example. Static models agreed on an incorrectly high
                    RUL, but the high-RUL temporal plateau triggered
                    mandatory review in both V4 and V5.
                    """
                )


# ---------------------------------------------------------
# Validation summary
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Validation Conclusion'
    '</div>',
    unsafe_allow_html=True,
)


conclusion_left, conclusion_right = st.columns(
    2
)


with conclusion_left:
    st.success(
        """
        **What V5 improved**

        - Reduced mandatory review workload by 13 engines
        - Preserved 87% any-review coverage
        - Kept Unit 67 under mandatory review
        - Removed seven low-error mandatory reviews
        - Separated condition severity from prediction reliability
        """
    )


with conclusion_right:
    st.warning(
        """
        **What remains imperfect**

        - Mandatory dangerous-error recall fell from 65.2% to 47.8%
        - Four dangerous cases moved to Recommended
        - Reliability risk has only moderate correlation with error
        - Thresholds remain engineering heuristics
        - Further calibration requires independent datasets
        """
    )


st.info(
    """
    V5 is the preferred portfolio decision layer because it provides a
    clearer and more practical distinction between mandatory and
    recommended review. V4 remains a useful conservative safety
    benchmark.
    """
)


# ---------------------------------------------------------
# Downloads and source information
# ---------------------------------------------------------

validation_download = validation_metrics_df.to_csv(
    index=False
).encode(
    "utf-8"
)

transition_download = engine_transitions_df.to_csv(
    index=False
).encode(
    "utf-8"
)


download_columns = st.columns(
    2
)

download_columns[0].download_button(
    label="Download V4–V5 validation metrics",
    data=validation_download,
    file_name=(
        "machineguard_validation_v4_v5_"
        "gru_FD001.csv"
    ),
    mime="text/csv",
)

download_columns[1].download_button(
    label="Download engine transitions",
    data=transition_download,
    file_name=(
        "machineguard_engine_transitions_"
        "gru_FD001.csv"
    ),
    mime="text/csv",
)


st.caption(
    f"Validation metrics source: {validation_metrics_path}"
)

st.caption(
    f"Engine transition source: {engine_transitions_path}"
)