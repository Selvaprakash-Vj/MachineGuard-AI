import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.shared import RESULTS_DIR


DATASET_NAME = "FD001"
MODEL_NAME = "gru"


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
    """Validate columns required by the fleet page."""

    required_columns = {
        "unit",
        "actual_rul",
        "predicted_rul_mean",
        "conservative_rul",
        "engine_condition",
        "condition_severity_score",
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
    st.title("Fleet Overview")

    st.error(
        "Fleet results could not be loaded."
    )

    st.code(
        str(error)
    )

    st.stop()


# ---------------------------------------------------------
# Sidebar filters
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("---")
    st.markdown("### Fleet filters")

    selected_conditions = st.multiselect(
        "Engine condition",
        options=[
            "Critical",
            "Warning",
            "Monitor",
            "Healthy",
        ],
        default=[
            "Critical",
            "Warning",
            "Monitor",
            "Healthy",
        ],
    )

    selected_reviews = st.multiselect(
        "Engineering review",
        options=[
            "Required",
            "Recommended",
            "Not required",
        ],
        default=[
            "Required",
            "Recommended",
            "Not required",
        ],
    )

    selected_priorities = st.multiselect(
        "Operational priority",
        options=[
            "Immediate",
            "High",
            "Elevated",
            "Watch",
            "Routine",
        ],
        default=[
            "Immediate",
            "High",
            "Elevated",
            "Watch",
            "Routine",
        ],
    )

    selected_reliability = st.multiselect(
        "Reliability risk",
        options=[
            "High",
            "Medium",
            "Low",
        ],
        default=[
            "High",
            "Medium",
            "Low",
        ],
    )


filtered_df = decision_df.loc[
    decision_df[
        "engine_condition"
    ].isin(
        selected_conditions
    )
    & decision_df[
        "review_flag"
    ].isin(
        selected_reviews
    )
    & decision_df[
        "operational_priority"
    ].isin(
        selected_priorities
    )
    & decision_df[
        "reliability_risk"
    ].isin(
        selected_reliability
    )
].copy()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">Fleet Overview</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Fleet-level engine condition, prediction reliability,
        review workload and operational prioritisation
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    f"Showing {len(filtered_df)} of "
    f"{len(decision_df)} FD001 test engines."
)


if filtered_df.empty:
    st.warning(
        "No engines match the selected filters."
    )

    st.stop()


# ---------------------------------------------------------
# Fleet headline metrics
# ---------------------------------------------------------

total_engines = len(
    filtered_df
)

critical_engines = int(
    (
        filtered_df[
            "engine_condition"
        ]
        == "Critical"
    ).sum()
)

required_reviews = int(
    (
        filtered_df[
            "review_flag"
        ]
        == "Required"
    ).sum()
)

high_reliability_risk = int(
    (
        filtered_df[
            "reliability_risk"
        ]
        == "High"
    ).sum()
)

average_predicted_rul = float(
    filtered_df[
        "predicted_rul_mean"
    ].mean()
)

average_conservative_rul = float(
    filtered_df[
        "conservative_rul"
    ].mean()
)


metric_columns = st.columns(
    6
)

metric_columns[0].metric(
    "Visible engines",
    total_engines,
)

metric_columns[1].metric(
    "Critical condition",
    critical_engines,
)

metric_columns[2].metric(
    "Mandatory reviews",
    required_reviews,
)

metric_columns[3].metric(
    "High reliability risk",
    high_reliability_risk,
)

metric_columns[4].metric(
    "Average predicted RUL",
    f"{average_predicted_rul:.1f}",
    help="Cycles",
)

metric_columns[5].metric(
    "Average conservative RUL",
    f"{average_conservative_rul:.1f}",
    help="Cycles",
)


# ---------------------------------------------------------
# Fleet distribution charts
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Fleet Condition and Review Workload'
    '</div>',
    unsafe_allow_html=True,
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

priority_order = [
    "Immediate",
    "High",
    "Elevated",
    "Watch",
    "Routine",
]

reliability_order = [
    "High",
    "Medium",
    "Low",
]


condition_counts = (
    filtered_df[
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
    filtered_df[
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


left_column, right_column = st.columns(
    2
)

with left_column:
    condition_figure = px.bar(
        condition_counts,
        x="Engine condition",
        y="Engine count",
        text="Engine count",
        title="Estimated Engine Condition",
    )

    condition_figure.update_traces(
        textposition="outside"
    )

    condition_figure.update_layout(
        showlegend=False,
        xaxis_title="",
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
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
        text="Engine count",
        title="Engineering Review Workload",
    )

    review_figure.update_traces(
        textposition="outside"
    )

    review_figure.update_layout(
        showlegend=False,
        xaxis_title="",
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        review_figure,
        use_container_width=True,
    )


# ---------------------------------------------------------
# Priority and reliability distributions
# ---------------------------------------------------------

priority_counts = (
    filtered_df[
        "operational_priority"
    ]
    .value_counts()
    .reindex(
        priority_order,
        fill_value=0,
    )
    .rename_axis(
        "Operational priority"
    )
    .reset_index(
        name="Engine count"
    )
)

reliability_counts = (
    filtered_df[
        "reliability_risk"
    ]
    .value_counts()
    .reindex(
        reliability_order,
        fill_value=0,
    )
    .rename_axis(
        "Reliability risk"
    )
    .reset_index(
        name="Engine count"
    )
)


left_column, right_column = st.columns(
    2
)

with left_column:
    priority_figure = px.bar(
        priority_counts,
        x="Operational priority",
        y="Engine count",
        text="Engine count",
        title="Operational Priority Distribution",
    )

    priority_figure.update_traces(
        textposition="outside"
    )

    priority_figure.update_layout(
        showlegend=False,
        xaxis_title="",
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        priority_figure,
        use_container_width=True,
    )


with right_column:
    reliability_figure = px.bar(
        reliability_counts,
        x="Reliability risk",
        y="Engine count",
        text="Engine count",
        title="Prediction Reliability Distribution",
    )

    reliability_figure.update_traces(
        textposition="outside"
    )

    reliability_figure.update_layout(
        showlegend=False,
        xaxis_title="",
        margin=dict(
            l=20,
            r=20,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        reliability_figure,
        use_container_width=True,
    )


# ---------------------------------------------------------
# Condition severity versus reliability risk
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Condition Severity vs Prediction Reliability'
    '</div>',
    unsafe_allow_html=True,
)

score_map = px.scatter(
    filtered_df,
    x="condition_severity_score",
    y="reliability_risk_score",
    color="review_flag",
    symbol="engine_condition",
    hover_name="unit",
    hover_data={
        "unit": True,
        "actual_rul": ":.1f",
        "predicted_rul_mean": ":.1f",
        "conservative_rul": ":.1f",
        "engine_condition": True,
        "reliability_risk": True,
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
        "review_flag": (
            "Engineering review"
        ),
    },
    title=(
        "Each point represents one unseen FD001 test engine"
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

st.info(
    """
    Condition severity and prediction reliability represent different
    questions. An engine may require urgent maintenance even when its
    prediction is trustworthy, or appear healthy while its prediction
    behaviour remains suspicious.
    """
)


# ---------------------------------------------------------
# Fleet decision table
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Fleet Decision Table'
    '</div>',
    unsafe_allow_html=True,
)


priority_rank = {
    "Immediate": 1,
    "High": 2,
    "Elevated": 3,
    "Watch": 4,
    "Routine": 5,
}

review_rank = {
    "Required": 1,
    "Recommended": 2,
    "Not required": 3,
}


table_df = filtered_df.copy()

table_df[
    "_priority_rank"
] = table_df[
    "operational_priority"
].map(
    priority_rank
)

table_df[
    "_review_rank"
] = table_df[
    "review_flag"
].map(
    review_rank
)

table_df = table_df.sort_values(
    by=[
        "_priority_rank",
        "_review_rank",
        "operational_priority_score",
    ],
    ascending=[
        True,
        True,
        False,
    ],
)


display_columns = [
    "unit",
    "actual_rul",
    "predicted_rul_mean",
    "conservative_rul",
    "engine_condition",
    "reliability_risk",
    "review_flag",
    "operational_priority",
    "trajectory_flag",
]

fleet_table = (
    table_df[
        display_columns
    ]
    .copy()
    .rename(
        columns={
            "unit": "Engine",
            "actual_rul": "Actual RUL",
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

for column in [
    "Actual RUL",
    "Predicted RUL",
    "Conservative RUL",
]:
    fleet_table[
        column
    ] = fleet_table[
        column
    ].round(
        1
    )


st.dataframe(
    fleet_table,
    use_container_width=True,
    hide_index=True,
    height=520,
)


csv_data = fleet_table.to_csv(
    index=False
).encode(
    "utf-8"
)

st.download_button(
    label="Download filtered fleet table",
    data=csv_data,
    file_name=(
        "machineguard_fleet_overview_"
        "gru_FD001.csv"
    ),
    mime="text/csv",
)

st.caption(
    """
    Actual RUL is displayed only because this is retrospective NASA
    test evaluation. It would not be known during real deployment.
    """
)