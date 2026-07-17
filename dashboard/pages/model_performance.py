import pandas as pd
import plotly.express as px
import streamlit as st


DATASET_NAME = "FD001"


# ---------------------------------------------------------
# Preserved benchmark results
# ---------------------------------------------------------

model_performance_df = pd.DataFrame(
    [
        {
            "Model": "Mean baseline",
            "RMSE": 38.00,
            "MAE": 31.00,
            "R²": 0.0000,
            "NASA score": None,
            "Mean error": None,
            "Dangerous overprediction rate": None,
            "Role": "Naive reference",
            "Selected": "No",
        },
        {
            "Model": "Ridge",
            "RMSE": 14.9931,
            "MAE": 12.2316,
            "R²": 0.8600,
            "NASA score": 374.9761,
            "Mean error": 2.7584,
            "Dangerous overprediction rate": None,
            "Role": "Independent interpretable baseline",
            "Selected": "Supporting model",
        },
        {
            "Model": "LSTM",
            "RMSE": 14.2865,
            "MAE": 10.9647,
            "R²": 0.8729,
            "NASA score": 443.0913,
            "Mean error": 4.5149,
            "Dangerous overprediction rate": None,
            "Role": "Secondary sequence model",
            "Selected": "Ensemble component",
        },
        {
            "Model": "GRU",
            "RMSE": 13.5849,
            "MAE": 9.8770,
            "R²": 0.8851,
            "NASA score": 317.7112,
            "Mean error": 1.4729,
            "Dangerous overprediction rate": 0.23,
            "Role": "Primary standalone model",
            "Selected": "Primary",
        },
        {
            "Model": "Transformer",
            "RMSE": 21.7041,
            "MAE": 15.4325,
            "R²": 0.7060,
            "NASA score": 1647.0,
            "Mean error": 3.36,
            "Dangerous overprediction rate": None,
            "Role": "Rejected for FD001",
            "Selected": "No",
        },
        {
            "Model": "GRU–LSTM ensemble",
            "RMSE": 13.2122,
            "MAE": 10.0868,
            "R²": 0.8913,
            "NASA score": 291.3001,
            "Mean error": 2.6897,
            "Dangerous overprediction rate": 0.29,
            "Role": "Best aggregate accuracy",
            "Selected": "Accuracy benchmark",
        },
    ]
)


original_comparison_df = pd.DataFrame(
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


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------

with st.sidebar:
    st.markdown("---")
    st.markdown("### Benchmark controls")

    selected_models = st.multiselect(
        "Models to display",
        options=model_performance_df[
            "Model"
        ].tolist(),
        default=model_performance_df[
            "Model"
        ].tolist(),
    )

    selected_metric = st.selectbox(
        "Primary comparison metric",
        options=[
            "RMSE",
            "MAE",
            "R²",
            "NASA score",
        ],
        index=0,
    )

    st.caption(
        "Lower is better for RMSE, MAE and NASA score. "
        "Higher is better for R²."
    )


filtered_performance_df = model_performance_df.loc[
    model_performance_df[
        "Model"
    ].isin(
        selected_models
    )
].copy()


if filtered_performance_df.empty:
    st.warning(
        "Select at least one model from the sidebar."
    )

    st.stop()


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">Model Performance</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Deployment-style comparison of classical, recurrent and
        attention-based Remaining Useful Life models
    </div>
    """,
    unsafe_allow_html=True,
)

st.caption(
    "Each FD001 test engine contributes exactly one final "
    "Remaining Useful Life prediction."
)


# ---------------------------------------------------------
# Best-model headline metrics
# ---------------------------------------------------------

best_rmse_row = model_performance_df.loc[
    model_performance_df[
        "RMSE"
    ].idxmin()
]

best_mae_row = model_performance_df.loc[
    model_performance_df[
        "MAE"
    ].idxmin()
]

best_r2_row = model_performance_df.loc[
    model_performance_df[
        "R²"
    ].idxmax()
]

valid_nasa_df = model_performance_df.dropna(
    subset=[
        "NASA score"
    ]
)

best_nasa_row = valid_nasa_df.loc[
    valid_nasa_df[
        "NASA score"
    ].idxmin()
]


metric_columns = st.columns(
    4
)

metric_columns[0].metric(
    "Lowest RMSE",
    f"{best_rmse_row['RMSE']:.2f}",
    help=f"Model: {best_rmse_row['Model']}",
)

metric_columns[1].metric(
    "Lowest MAE",
    f"{best_mae_row['MAE']:.2f}",
    help=f"Model: {best_mae_row['Model']}",
)

metric_columns[2].metric(
    "Highest R²",
    f"{best_r2_row['R²']:.3f}",
    help=f"Model: {best_r2_row['Model']}",
)

metric_columns[3].metric(
    "Lowest NASA score",
    f"{best_nasa_row['NASA score']:.1f}",
    help=f"Model: {best_nasa_row['Model']}",
)


# ---------------------------------------------------------
# Primary selected comparison
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Selected Metric Comparison'
    '</div>',
    unsafe_allow_html=True,
)


primary_plot_df = filtered_performance_df.copy()

if selected_metric == "NASA score":
    primary_plot_df = primary_plot_df.dropna(
        subset=[
            "NASA score"
        ]
    )


ascending_order = (
    selected_metric
    != "R²"
)

primary_plot_df = primary_plot_df.sort_values(
    by=selected_metric,
    ascending=ascending_order,
)


primary_figure = px.bar(
    primary_plot_df,
    x="Model",
    y=selected_metric,
    text=selected_metric,
    hover_data={
        "RMSE": ":.3f",
        "MAE": ":.3f",
        "R²": ":.4f",
        "NASA score": ":.1f",
        "Role": True,
        "Selected": True,
    },
    title=(
        f"{selected_metric} Comparison — {DATASET_NAME}"
    ),
)

if selected_metric == "R²":
    primary_figure.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

elif selected_metric == "NASA score":
    primary_figure.update_traces(
        texttemplate="%{text:.1f}",
        textposition="outside",
    )

else:
    primary_figure.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )


primary_figure.update_layout(
    xaxis_title="",
    showlegend=False,
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20,
    ),
)

st.plotly_chart(
    primary_figure,
    use_container_width=True,
)


# ---------------------------------------------------------
# RMSE and MAE charts
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Prediction Error'
    '</div>',
    unsafe_allow_html=True,
)


left_column, right_column = st.columns(
    2
)


with left_column:
    rmse_df = filtered_performance_df.sort_values(
        by="RMSE",
        ascending=True,
    )

    rmse_figure = px.bar(
        rmse_df,
        x="Model",
        y="RMSE",
        text="RMSE",
        title="Root Mean Squared Error",
        hover_data={
            "Role": True,
            "MAE": ":.3f",
            "R²": ":.4f",
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
    mae_df = filtered_performance_df.sort_values(
        by="MAE",
        ascending=True,
    )

    mae_figure = px.bar(
        mae_df,
        x="Model",
        y="MAE",
        text="MAE",
        title="Mean Absolute Error",
        hover_data={
            "Role": True,
            "RMSE": ":.3f",
            "R²": ":.4f",
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
# R-squared and NASA score
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Explained Variance and Asymmetric Penalty'
    '</div>',
    unsafe_allow_html=True,
)


left_column, right_column = st.columns(
    2
)


with left_column:
    r2_df = filtered_performance_df.sort_values(
        by="R²",
        ascending=False,
    )

    r2_figure = px.bar(
        r2_df,
        x="Model",
        y="R²",
        text="R²",
        title="Coefficient of Determination",
        hover_data={
            "Role": True,
            "RMSE": ":.3f",
            "MAE": ":.3f",
        },
    )

    r2_figure.add_hline(
        y=0,
        line_dash="dash",
        opacity=0.5,
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
    nasa_df = (
        filtered_performance_df
        .dropna(
            subset=[
                "NASA score"
            ]
        )
        .sort_values(
            by="NASA score",
            ascending=True,
        )
    )

    nasa_figure = px.bar(
        nasa_df,
        x="Model",
        y="NASA score",
        text="NASA score",
        title="NASA Asymmetric RUL Score",
        hover_data={
            "Role": True,
            "RMSE": ":.3f",
            "MAE": ":.3f",
            "Mean error": ":.3f",
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


st.info(
    """
    The NASA scoring function applies an asymmetric penalty to RUL
    errors. Overpredicting remaining life is generally penalized more
    strongly because it may delay maintenance intervention.
    """
)


# ---------------------------------------------------------
# Accuracy versus dangerous overprediction
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Accuracy and Safety Trade-Off'
    '</div>',
    unsafe_allow_html=True,
)


safety_comparison_df = (
    model_performance_df[
        model_performance_df[
            "Dangerous overprediction rate"
        ].notna()
    ]
    .copy()
)


safety_comparison_df[
    "Dangerous overprediction (%)"
] = (
    safety_comparison_df[
        "Dangerous overprediction rate"
    ]
    * 100.0
)


safety_figure = px.scatter(
    safety_comparison_df,
    x="RMSE",
    y="Dangerous overprediction (%)",
    text="Model",
    size="R²",
    hover_name="Model",
    hover_data={
        "MAE": ":.3f",
        "R²": ":.4f",
        "NASA score": ":.1f",
        "Role": True,
    },
    title=(
        "Aggregate Accuracy vs Dangerous Overprediction Rate"
    ),
    labels={
        "RMSE": "RMSE (cycles)",
    },
)

safety_figure.update_traces(
    textposition="top center",
)

safety_figure.update_layout(
    margin=dict(
        l=20,
        r=20,
        t=60,
        b=20,
    ),
)

st.plotly_chart(
    safety_figure,
    use_container_width=True,
)


st.warning(
    """
    The GRU–LSTM ensemble achieved the best RMSE and NASA score,
    but its dangerous-overprediction rate was higher than the GRU.
    MachineGuard AI therefore uses the GRU as the primary operational
    model and preserves the ensemble as an accuracy benchmark.
    """
)


# ---------------------------------------------------------
# Model-selection reasoning
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Final Model Selection'
    '</div>',
    unsafe_allow_html=True,
)


selection_left, selection_right = st.columns(
    2
)


with selection_left:
    st.success(
        """
        **Primary operational model — GRU**

        - Lowest standalone MAE
        - Strong standalone RMSE
        - Strong NASA score
        - Lower dangerous-overprediction rate than the ensemble
        - Efficient recurrent architecture
        - Used for uncertainty and trajectory diagnostics
        """
    )

    st.info(
        """
        **Accuracy benchmark — GRU–LSTM ensemble**

        Validation-only weight search selected:

        - 60% GRU
        - 40% LSTM
        - 0% Ridge
        - 0% Transformer

        The weights were fixed before evaluating the test engines.
        """
    )


with selection_right:
    st.warning(
        """
        **Independent baseline — Ridge**

        Ridge remained surprisingly competitive and provides a model
        based on a fundamentally different learning approach.

        It is used as independent evidence inside the decision layer.
        """
    )

    st.error(
        """
        **Rejected model — Transformer**

        The Transformer added considerable architectural complexity
        but performed substantially worse than GRU, LSTM and Ridge on
        FD001.

        Greater model complexity did not produce better prognostics.
        """
    )


# ---------------------------------------------------------
# Complete benchmark table
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Complete Benchmark Table'
    '</div>',
    unsafe_allow_html=True,
)


benchmark_table = model_performance_df.copy()

benchmark_table[
    "RMSE"
] = benchmark_table[
    "RMSE"
].round(
    3
)

benchmark_table[
    "MAE"
] = benchmark_table[
    "MAE"
].round(
    3
)

benchmark_table[
    "R²"
] = benchmark_table[
    "R²"
].round(
    4
)

benchmark_table[
    "NASA score"
] = benchmark_table[
    "NASA score"
].round(
    1
)

benchmark_table[
    "Mean error"
] = benchmark_table[
    "Mean error"
].round(
    3
)

benchmark_table[
    "Dangerous overprediction rate"
] = (
    benchmark_table[
        "Dangerous overprediction rate"
    ]
    * 100.0
).round(
    1
)


benchmark_table = benchmark_table.rename(
    columns={
        "Dangerous overprediction rate": (
            "Dangerous overprediction (%)"
        ),
    }
)


st.dataframe(
    benchmark_table,
    use_container_width=True,
    hide_index=True,
)


csv_data = benchmark_table.to_csv(
    index=False
).encode(
    "utf-8"
)

st.download_button(
    label="Download benchmark table",
    data=csv_data,
    file_name=(
        "machineguard_model_benchmarks_FD001.csv"
    ),
    mime="text/csv",
)


# ---------------------------------------------------------
# Original repository comparison
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Original Repository vs MachineGuard AI'
    '</div>',
    unsafe_allow_html=True,
)


comparison_left, comparison_right = st.columns(
    2
)


with comparison_left:
    comparison_rmse_figure = px.bar(
        original_comparison_df,
        x="Pipeline",
        y="RMSE",
        text="RMSE",
        title="RMSE Improvement",
        hover_data={
            "MAE": ":.3f",
            "R²": ":.4f",
        },
    )

    comparison_rmse_figure.update_traces(
        texttemplate="%{text:.2f}",
        textposition="outside",
    )

    comparison_rmse_figure.update_layout(
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
        comparison_rmse_figure,
        use_container_width=True,
    )


with comparison_right:
    comparison_r2_figure = px.bar(
        original_comparison_df,
        x="Pipeline",
        y="R²",
        text="R²",
        title="Explained Variance Improvement",
        hover_data={
            "RMSE": ":.3f",
            "MAE": ":.3f",
        },
    )

    comparison_r2_figure.add_hline(
        y=0,
        line_dash="dash",
        opacity=0.5,
    )

    comparison_r2_figure.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    comparison_r2_figure.update_layout(
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
        comparison_r2_figure,
        use_container_width=True,
    )


rmse_reduction_percentage = (
    (
        original_comparison_df.loc[
            original_comparison_df[
                "Pipeline"
            ]
            == "Original repository LSTM",
            "RMSE",
        ].iloc[0]
        - original_comparison_df.loc[
            original_comparison_df[
                "Pipeline"
            ]
            == "MachineGuard AI GRU",
            "RMSE",
        ].iloc[0]
    )
    / original_comparison_df.loc[
        original_comparison_df[
            "Pipeline"
        ]
        == "Original repository LSTM",
        "RMSE",
    ].iloc[0]
    * 100.0
)


st.success(
    f"""
    MachineGuard AI reduced reproduced LSTM-era RMSE from approximately
    **63.79 cycles** to **13.58 cycles** with the GRU—an approximate
    reduction of **{rmse_reduction_percentage:.1f}%**.
    """
)


st.warning(
    """
    This is not a pure architecture-to-architecture comparison.
    MachineGuard AI corrected the full experimental methodology:
    engine-level validation, train-only scaling, deployment-aligned
    evaluation, proper baselines and leakage prevention.
    """
)


# ---------------------------------------------------------
# Evaluation methodology
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Evaluation Methodology'
    '</div>',
    unsafe_allow_html=True,
)


methodology_df = pd.DataFrame(
    [
        {
            "Evaluation principle": (
                "One prediction per test engine"
            ),
            "Implementation": (
                "Only the final available 30-cycle window is evaluated."
            ),
        },
        {
            "Evaluation principle": (
                "Unseen engine testing"
            ),
            "Implementation": (
                "The 100 FD001 test engines are not used for training."
            ),
        },
        {
            "Evaluation principle": (
                "Validation-only model selection"
            ),
            "Implementation": (
                "Early stopping and ensemble weights use validation "
                "engines only."
            ),
        },
        {
            "Evaluation principle": (
                "Consistent RUL target"
            ),
            "Implementation": (
                "Training and evaluation use the 125-cycle cap."
            ),
        },
        {
            "Evaluation principle": (
                "Multiple performance measures"
            ),
            "Implementation": (
                "RMSE, MAE, R², NASA score and signed error are tracked."
            ),
        },
    ]
)


st.dataframe(
    methodology_df,
    use_container_width=True,
    hide_index=True,
)


st.info(
    """
    The benchmark demonstrates that model choice should be based on
    several engineering criteria. The lowest RMSE does not automatically
    identify the safest operational model.
    """
)


st.caption(
    "Benchmark values are preserved from the completed MachineGuard AI "
    "FD001 evaluation pipeline."
)