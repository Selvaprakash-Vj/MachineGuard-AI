import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.shared import RESULTS_DIR


DATASET_NAME = "FD001"
PRIMARY_MODEL = "gru"
SECONDARY_MODEL = "lstm"


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


def validate_importance_data(
    importance_df: pd.DataFrame,
    model_name: str,
) -> None:
    """Validate columns required for explainability analysis."""

    required_columns = {
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

    missing_columns = sorted(
        required_columns.difference(
            importance_df.columns
        )
    )

    if missing_columns:
        raise KeyError(
            f"{model_name.upper()} explainability file "
            f"is missing columns: {missing_columns}"
        )

    if importance_df["feature"].duplicated().any():
        raise ValueError(
            f"Duplicate features were found in the "
            f"{model_name.upper()} explainability file."
        )


explanation_dir = (
    RESULTS_DIR
    / "explanations"
)

gru_importance_path = (
    explanation_dir
    / (
        f"global_importance_"
        f"{PRIMARY_MODEL}_{DATASET_NAME}.csv"
    )
)

lstm_importance_path = (
    explanation_dir
    / (
        f"global_importance_"
        f"{SECONDARY_MODEL}_{DATASET_NAME}.csv"
    )
)


try:
    gru_importance_df = load_csv(
        gru_importance_path
    )

    validate_importance_data(
        importance_df=gru_importance_df,
        model_name=PRIMARY_MODEL,
    )

except (
    FileNotFoundError,
    KeyError,
    ValueError,
) as error:
    st.title("Explainability")

    st.error(
        "GRU explainability results could not be loaded."
    )

    st.code(
        str(error)
    )

    st.info(
        "Run the explainability pipeline for the GRU "
        "before opening this page."
    )

    st.stop()


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


# ---------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------

maximum_feature_count = min(
    24,
    len(
        gru_importance_df
    ),
)


with st.sidebar:
    st.markdown("---")
    st.markdown("### Explainability controls")

    importance_metric = st.selectbox(
        "Primary ranking metric",
        options=[
            "RMSE increase",
            "MAE increase",
            "Prediction sensitivity",
        ],
        index=0,
        help=(
            "Features are ranked by the selected "
            "occlusion-based importance measure."
        ),
    )

    top_feature_count = st.slider(
        "Features to display",
        min_value=5,
        max_value=maximum_feature_count,
        value=min(
            12,
            maximum_feature_count,
        ),
        step=1,
    )

    st.caption(
        "Occlusion analysis measures model reliance, "
        "not physical causality."
    )


ranking_column_map = {
    "RMSE increase": "rmse_increase",
    "MAE increase": "mae_increase",
    "Prediction sensitivity": (
        "mean_absolute_prediction_change"
    ),
}

ranking_column = ranking_column_map[
    importance_metric
]


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">Explainability</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Global occlusion-based analysis of the sensor features
        used by the GRU Remaining Useful Life model
    </div>
    """,
    unsafe_allow_html=True,
)


st.info(
    """
    Each input feature is neutralized one at a time and the trained
    GRU is evaluated again. If prediction error rises substantially,
    the model depends strongly on that feature across the FD001
    test fleet.
    """
)


# ---------------------------------------------------------
# Explainability headline metrics
# ---------------------------------------------------------

most_important_rmse = (
    gru_importance_df.loc[
        gru_importance_df[
            "rmse_increase"
        ].idxmax()
    ]
)

most_important_mae = (
    gru_importance_df.loc[
        gru_importance_df[
            "mae_increase"
        ].idxmax()
    ]
)

largest_prediction_change = (
    gru_importance_df.loc[
        gru_importance_df[
            "mean_absolute_prediction_change"
        ].idxmax()
    ]
)

largest_positive_signed_change = (
    gru_importance_df.loc[
        gru_importance_df[
            "mean_signed_prediction_change"
        ].idxmax()
    ]
)

largest_negative_signed_change = (
    gru_importance_df.loc[
        gru_importance_df[
            "mean_signed_prediction_change"
        ].idxmin()
    ]
)


metric_columns = st.columns(
    5
)

metric_columns[0].metric(
    "Top RMSE feature",
    most_important_rmse[
        "feature"
    ],
    help=(
        f"RMSE increase: "
        f"{most_important_rmse['rmse_increase']:.3f}"
    ),
)

metric_columns[1].metric(
    "Top MAE feature",
    most_important_mae[
        "feature"
    ],
    help=(
        f"MAE increase: "
        f"{most_important_mae['mae_increase']:.3f}"
    ),
)

metric_columns[2].metric(
    "Largest prediction sensitivity",
    largest_prediction_change[
        "feature"
    ],
    help=(
        f"Mean absolute change: "
        f"{largest_prediction_change['mean_absolute_prediction_change']:.2f} "
        "cycles"
    ),
)

metric_columns[3].metric(
    "Baseline RMSE",
    (
        f"{gru_importance_df['baseline_rmse'].iloc[0]:.2f}"
    ),
    help="Cycles",
)

metric_columns[4].metric(
    "Baseline MAE",
    (
        f"{gru_importance_df['baseline_mae'].iloc[0]:.2f}"
    ),
    help="Cycles",
)


# ---------------------------------------------------------
# Ranked feature importance
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Global Sensor Importance'
    '</div>',
    unsafe_allow_html=True,
)


ranked_importance_df = (
    gru_importance_df
    .sort_values(
        by=ranking_column,
        ascending=False,
    )
    .head(
        top_feature_count
    )
    .sort_values(
        by=ranking_column,
        ascending=True,
    )
    .copy()
)


ranking_title_map = {
    "RMSE increase": (
        "Increase in RMSE After Feature Occlusion"
    ),
    "MAE increase": (
        "Increase in MAE After Feature Occlusion"
    ),
    "Prediction sensitivity": (
        "Average Absolute Change in Predicted RUL"
    ),
}

ranking_axis_map = {
    "RMSE increase": (
        "RMSE increase after occlusion"
    ),
    "MAE increase": (
        "MAE increase after occlusion"
    ),
    "Prediction sensitivity": (
        "Mean absolute prediction change (cycles)"
    ),
}


importance_figure = px.bar(
    ranked_importance_df,
    x=ranking_column,
    y="feature",
    orientation="h",
    text=ranking_column,
    title=ranking_title_map[
        importance_metric
    ],
    hover_data={
        "baseline_rmse": ":.3f",
        "occluded_rmse": ":.3f",
        "rmse_increase": ":.3f",
        "baseline_mae": ":.3f",
        "occluded_mae": ":.3f",
        "mae_increase": ":.3f",
        "mean_absolute_prediction_change": ":.3f",
        "mean_signed_prediction_change": ":.3f",
    },
    labels={
        ranking_column: ranking_axis_map[
            importance_metric
        ],
        "feature": "Input feature",
    },
)

importance_figure.update_traces(
    texttemplate="%{text:.3f}",
    textposition="outside",
)

importance_figure.update_layout(
    showlegend=False,
    margin=dict(
        l=20,
        r=30,
        t=60,
        b=20,
    ),
)

st.plotly_chart(
    importance_figure,
    use_container_width=True,
)


top_feature_names = (
    gru_importance_df
    .sort_values(
        by="rmse_increase",
        ascending=False,
    )
    .head(5)[
        "feature"
    ]
    .tolist()
)

st.success(
    "The GRU relies most strongly on "
    f"**{', '.join(top_feature_names)}** when ranked by "
    "the increase in RMSE after feature occlusion."
)


# ---------------------------------------------------------
# RMSE and MAE importance comparison
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Error Degradation After Occlusion'
    '</div>',
    unsafe_allow_html=True,
)


top_rmse_df = (
    gru_importance_df
    .sort_values(
        by="rmse_increase",
        ascending=False,
    )
    .head(
        top_feature_count
    )
    .sort_values(
        by="rmse_increase",
        ascending=True,
    )
)

top_mae_df = (
    gru_importance_df
    .sort_values(
        by="mae_increase",
        ascending=False,
    )
    .head(
        top_feature_count
    )
    .sort_values(
        by="mae_increase",
        ascending=True,
    )
)


left_column, right_column = st.columns(
    2
)


with left_column:
    rmse_figure = px.bar(
        top_rmse_df,
        x="rmse_increase",
        y="feature",
        orientation="h",
        text="rmse_increase",
        title="RMSE Increase",
        hover_data={
            "baseline_rmse": ":.3f",
            "occluded_rmse": ":.3f",
            "mean_absolute_prediction_change": ":.3f",
        },
        labels={
            "rmse_increase": (
                "RMSE increase after occlusion"
            ),
            "feature": "Feature",
        },
    )

    rmse_figure.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    rmse_figure.update_layout(
        showlegend=False,
        margin=dict(
            l=20,
            r=30,
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
        top_mae_df,
        x="mae_increase",
        y="feature",
        orientation="h",
        text="mae_increase",
        title="MAE Increase",
        hover_data={
            "baseline_mae": ":.3f",
            "occluded_mae": ":.3f",
            "mean_absolute_prediction_change": ":.3f",
        },
        labels={
            "mae_increase": (
                "MAE increase after occlusion"
            ),
            "feature": "Feature",
        },
    )

    mae_figure.update_traces(
        texttemplate="%{text:.3f}",
        textposition="outside",
    )

    mae_figure.update_layout(
        showlegend=False,
        margin=dict(
            l=20,
            r=30,
            t=55,
            b=20,
        ),
    )

    st.plotly_chart(
        mae_figure,
        use_container_width=True,
    )


# ---------------------------------------------------------
# Prediction sensitivity
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Prediction Sensitivity'
    '</div>',
    unsafe_allow_html=True,
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
        "Average Change in Predicted RUL "
        "When a Feature Is Occluded"
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
        r=30,
        t=60,
        b=20,
    ),
)

st.plotly_chart(
    sensitivity_figure,
    use_container_width=True,
)


st.info(
    """
    Prediction sensitivity measures how much the RUL estimate changes,
    on average, when a feature is neutralized. A sensor may produce a
    noticeable prediction change even when its effect on fleet-level
    RMSE is more modest.
    """
)


# ---------------------------------------------------------
# Signed prediction change
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Direction of Prediction Influence'
    '</div>',
    unsafe_allow_html=True,
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
        "Signed RUL Change After Feature Occlusion"
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
        r=30,
        t=60,
        b=20,
    ),
)

st.plotly_chart(
    signed_change_figure,
    use_container_width=True,
)


direction_columns = st.columns(
    2
)


with direction_columns[0]:
    st.info(
        f"""
        **Largest positive signed change**

        **{largest_positive_signed_change['feature']}**

        Neutralizing this feature increased predicted RUL by
        approximately
        **{largest_positive_signed_change['mean_signed_prediction_change']:.2f}
        cycles** on average.

        This suggests that its original values generally supplied
        evidence that pushed RUL downward.
        """
    )


with direction_columns[1]:
    st.info(
        f"""
        **Largest negative signed change**

        **{largest_negative_signed_change['feature']}**

        Neutralizing this feature changed predicted RUL by
        approximately
        **{largest_negative_signed_change['mean_signed_prediction_change']:.2f}
        cycles** on average.

        This suggests that its original values generally supplied
        evidence that pushed RUL upward.
        """
    )


# ---------------------------------------------------------
# GRU versus LSTM sensor reliance
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'GRU vs LSTM Sensor Reliance'
    '</div>',
    unsafe_allow_html=True,
)


if not lstm_importance_path.exists():
    st.warning(
        "LSTM explainability results were not found, so "
        "cross-model comparison is unavailable."
    )

else:
    try:
        lstm_importance_df = load_csv(
            lstm_importance_path
        )

        validate_importance_data(
            importance_df=lstm_importance_df,
            model_name=SECONDARY_MODEL,
        )

    except (
        FileNotFoundError,
        KeyError,
        ValueError,
    ) as error:
        st.warning(
            "LSTM explainability results could not be loaded."
        )

        st.code(
            str(error)
        )

    else:
        model_comparison_df = (
            gru_importance_df[
                [
                    "feature",
                    "rmse_increase",
                    "mae_increase",
                    "mean_absolute_prediction_change",
                ]
            ]
            .rename(
                columns={
                    "rmse_increase": (
                        "GRU RMSE increase"
                    ),
                    "mae_increase": (
                        "GRU MAE increase"
                    ),
                    "mean_absolute_prediction_change": (
                        "GRU prediction sensitivity"
                    ),
                }
            )
            .merge(
                lstm_importance_df[
                    [
                        "feature",
                        "rmse_increase",
                        "mae_increase",
                        "mean_absolute_prediction_change",
                    ]
                ].rename(
                    columns={
                        "rmse_increase": (
                            "LSTM RMSE increase"
                        ),
                        "mae_increase": (
                            "LSTM MAE increase"
                        ),
                        "mean_absolute_prediction_change": (
                            "LSTM prediction sensitivity"
                        ),
                    }
                ),
                on="feature",
                how="inner",
                validate="one_to_one",
            )
        )

        model_comparison_df[
            "Combined RMSE importance"
        ] = (
            model_comparison_df[
                "GRU RMSE increase"
            ].abs()
            + model_comparison_df[
                "LSTM RMSE increase"
            ].abs()
        )

        top_comparison_df = (
            model_comparison_df
            .sort_values(
                by="Combined RMSE importance",
                ascending=False,
            )
            .head(
                top_feature_count
            )
        )

        comparison_long_df = (
            top_comparison_df[
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
                t=60,
                b=20,
            ),
        )

        st.plotly_chart(
            comparison_figure,
            use_container_width=True,
        )


        gru_rank_df = (
            gru_importance_df[
                [
                    "feature",
                    "rmse_increase",
                ]
            ]
            .copy()
        )

        gru_rank_df[
            "GRU rank"
        ] = (
            gru_rank_df[
                "rmse_increase"
            ]
            .rank(
                ascending=False,
                method="min",
            )
            .astype(int)
        )

        lstm_rank_df = (
            lstm_importance_df[
                [
                    "feature",
                    "rmse_increase",
                ]
            ]
            .copy()
        )

        lstm_rank_df[
            "LSTM rank"
        ] = (
            lstm_rank_df[
                "rmse_increase"
            ]
            .rank(
                ascending=False,
                method="min",
            )
            .astype(int)
        )

        rank_comparison_df = (
            gru_rank_df[
                [
                    "feature",
                    "GRU rank",
                ]
            ]
            .merge(
                lstm_rank_df[
                    [
                        "feature",
                        "LSTM rank",
                    ]
                ],
                on="feature",
                how="inner",
                validate="one_to_one",
            )
        )

        rank_comparison_df[
            "Average rank"
        ] = (
            rank_comparison_df[
                [
                    "GRU rank",
                    "LSTM rank",
                ]
            ]
            .mean(
                axis=1
            )
        )

        shared_top_features = (
            rank_comparison_df
            .sort_values(
                by="Average rank",
                ascending=True,
            )
            .head(5)[
                "feature"
            ]
            .tolist()
        )

        st.success(
            "The strongest shared sensor reliance across GRU and "
            f"LSTM includes **{', '.join(shared_top_features)}**."
        )

        st.caption(
            """
            Agreement between GRU and LSTM importance rankings gives
            evidence of shared model reliance. It does not guarantee
            that both models are correct; shared reliance can also
            contribute to shared blind spots.
            """
        )


# ---------------------------------------------------------
# Complete explainability table
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Complete Explainability Results'
    '</div>',
    unsafe_allow_html=True,
)


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
    height=520,
)


csv_data = importance_table.to_csv(
    index=False
).encode(
    "utf-8"
)

st.download_button(
    label="Download GRU explainability results",
    data=csv_data,
    file_name=(
        "machineguard_global_importance_"
        "gru_FD001.csv"
    ),
    mime="text/csv",
)


# ---------------------------------------------------------
# Interpretation and limitations
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Engineering Interpretation'
    '</div>',
    unsafe_allow_html=True,
)


interpretation_columns = st.columns(
    2
)


with interpretation_columns[0]:
    st.success(
        """
        **What this analysis supports**

        - Identifying which inputs the GRU relies upon
        - Comparing feature reliance across models
        - Detecting unusually dominant model inputs
        - Guiding further sensor and failure-case analysis
        - Improving model transparency
        """
    )


with interpretation_columns[1]:
    st.warning(
        """
        **What this analysis does not prove**

        - Physical causality
        - Sensor health or calibration quality
        - That a high-ranked sensor directly causes degradation
        - That shared GRU–LSTM importance guarantees correctness
        - Certification-grade interpretability
        """
    )


st.info(
    """
    This release contains global fleet-level explainability.
    Engine-specific feature contributions were explored for selected
    difficult engines but were not yet preserved as reusable CSV
    artefacts. Local explanations remain a future dashboard extension.
    """
)


st.caption(
    f"GRU explainability source: {gru_importance_path}"
)

if lstm_importance_path.exists():
    st.caption(
        f"LSTM explainability source: {lstm_importance_path}"
    )