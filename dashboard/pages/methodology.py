from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard.shared import PROJECT_ROOT, RESULTS_DIR


DATASET_NAME = "FD001"
PRIMARY_MODEL = "GRU"
SEQUENCE_LENGTH = 30
RUL_CAP = 125
MC_DROPOUT_PASSES = 100
TARGET_COVERAGE = 0.90
OBSERVED_COVERAGE = 0.88
CALIBRATION_MULTIPLIER = 1.3725


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------

def relative_path(path: Path) -> str:
    """Return a clean project-relative path where possible."""

    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def artifact_status(path: Path) -> str:
    """Return a human-readable artifact status."""

    return "Available" if path.exists() else "Missing"


# ---------------------------------------------------------
# Header
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">'
    'Methodology & Responsible Use'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
        Technical architecture, leakage-safe evaluation methodology,
        engineering assumptions, limitations and intended use of
        MachineGuard AI
    </div>
    """,
    unsafe_allow_html=True,
)


st.info(
    """
    MachineGuard AI is a portfolio-grade predictive-maintenance
    research platform built around the NASA C-MAPSS FD001 dataset.
    It estimates turbofan Remaining Useful Life while also exposing
    uncertainty, temporal behaviour, model disagreement and
    engineering-review requirements.
    """
)


# ---------------------------------------------------------
# Release summary
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Release Summary'
    '</div>',
    unsafe_allow_html=True,
)


summary_columns = st.columns(6)

summary_columns[0].metric(
    "Dataset",
    DATASET_NAME,
)

summary_columns[1].metric(
    "Primary model",
    PRIMARY_MODEL,
)

summary_columns[2].metric(
    "Sequence length",
    SEQUENCE_LENGTH,
    help="Operating cycles per model input window.",
)

summary_columns[3].metric(
    "RUL cap",
    RUL_CAP,
    help="Maximum training and evaluation target in cycles.",
)

summary_columns[4].metric(
    "Test engines",
    100,
)

summary_columns[5].metric(
    "Evaluation unit",
    "1 per engine",
    help="One final prediction for each unseen test engine.",
)


# ---------------------------------------------------------
# Main methodology tabs
# ---------------------------------------------------------

architecture_tab, data_tab, model_tab, responsibility_tab, reproducibility_tab = (
    st.tabs(
        [
            "System Architecture",
            "Data & Evaluation",
            "Models & Decision Layer",
            "Responsible Use",
            "Reproducibility",
        ]
    )
)


# =========================================================
# SYSTEM ARCHITECTURE
# =========================================================

with architecture_tab:
    st.markdown(
        '<div class="section-heading">'
        'End-to-End System Architecture'
        '</div>',
        unsafe_allow_html=True,
    )

    architecture_labels = [
        "NASA C-MAPSS\nFD001",
        "Leakage-safe\npreprocessing",
        "30-cycle\nsequences",
        "Ridge",
        "LSTM",
        "GRU",
        "Transformer",
        "Validation-only\nmodel selection",
        "MC-dropout\nuncertainty",
        "Trajectory\ndiagnostics",
        "Global sensor\nexplainability",
        "V5 engineering\ndecision layer",
        "Streamlit\nportfolio dashboard",
    ]

    architecture_sources = [
        0,
        1,
        2,
        2,
        2,
        2,
        3,
        4,
        5,
        6,
        5,
        5,
        5,
        7,
        8,
        9,
        10,
        11,
    ]

    architecture_targets = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        7,
        7,
        7,
        8,
        9,
        10,
        11,
        11,
        11,
        11,
        12,
    ]

    architecture_values = [
        4,
        4,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        1,
        1,
        1,
        4,
    ]

    architecture_figure = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=22,
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
        title="MachineGuard AI Information Flow",
        height=650,
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

    st.markdown("#### Architecture layers")

    architecture_df = pd.DataFrame(
        [
            {
                "Layer": "Data layer",
                "Purpose": (
                    "Loads FD001 train, test and test-RUL files and "
                    "constructs engine-level sequences."
                ),
                "Primary output": (
                    "Scaled 30-cycle tensors with unit metadata."
                ),
            },
            {
                "Layer": "Prediction layer",
                "Purpose": (
                    "Compares classical, recurrent and attention-based "
                    "RUL regression models."
                ),
                "Primary output": (
                    "One final RUL estimate per unseen test engine."
                ),
            },
            {
                "Layer": "Reliability layer",
                "Purpose": (
                    "Measures uncertainty, model disagreement and "
                    "temporal prediction behaviour."
                ),
                "Primary output": (
                    "Reliability-risk evidence and trajectory flags."
                ),
            },
            {
                "Layer": "Explainability layer",
                "Purpose": (
                    "Measures global model reliance through feature "
                    "occlusion."
                ),
                "Primary output": (
                    "Sensor rankings and prediction sensitivity."
                ),
            },
            {
                "Layer": "Decision layer",
                "Purpose": (
                    "Separates engine condition, prediction reliability, "
                    "human review and operational priority."
                ),
                "Primary output": (
                    "V5 engineering decision record for each engine."
                ),
            },
            {
                "Layer": "Presentation layer",
                "Purpose": (
                    "Provides fleet, engine, uncertainty, explainability "
                    "and validation views."
                ),
                "Primary output": (
                    "Interactive multipage Streamlit dashboard."
                ),
            },
        ]
    )

    st.dataframe(
        architecture_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Repository structure")

    st.code(
        """
MachineGuard-AI/
│
├── app.py
├── app_single_page_backup.py
│
├── dashboard/
│   ├── shared.py
│   └── pages/
│       ├── complete_dashboard.py
│       ├── fleet_overview.py
│       ├── engine_explorer.py
│       ├── trajectory_uncertainty.py
│       ├── explainability.py
│       ├── model_performance.py
│       ├── decision_validation.py
│       └── methodology.py
│
├── src/
│   ├── preprocessing_v2.py
│   ├── trajectory_diagnostics_v2.py
│   └── decision_layer_v5.py
│
└── results_v2/
    ├── decisions_v5/
    ├── trajectory_diagnostics/
    ├── explanations/
    └── decision_validation_v5/
        """.strip(),
        language="text",
    )


# =========================================================
# DATA AND EVALUATION
# =========================================================

with data_tab:
    st.markdown(
        '<div class="section-heading">'
        'Problem Definition'
        '</div>',
        unsafe_allow_html=True,
    )

    problem_df = pd.DataFrame(
        [
            {
                "Property": "Task",
                "MachineGuard AI design": (
                    "Supervised Remaining Useful Life regression."
                ),
            },
            {
                "Property": "Dataset",
                "MachineGuard AI design": (
                    "NASA C-MAPSS FD001 turbofan degradation dataset."
                ),
            },
            {
                "Property": "Input",
                "MachineGuard AI design": (
                    "A sequence of 30 operating cycles containing "
                    "operational settings and sensor measurements."
                ),
            },
            {
                "Property": "Target",
                "MachineGuard AI design": (
                    "Remaining Useful Life in operating cycles."
                ),
            },
            {
                "Property": "Target transformation",
                "MachineGuard AI design": (
                    "Piecewise-linear RUL capped at 125 cycles."
                ),
            },
            {
                "Property": "Deployment-style output",
                "MachineGuard AI design": (
                    "One final RUL estimate for each test engine."
                ),
            },
        ]
    )

    st.dataframe(
        problem_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-heading">'
        'Leakage-Safe Preprocessing'
        '</div>',
        unsafe_allow_html=True,
    )

    preprocessing_df = pd.DataFrame(
        [
            {
                "Stage": "Engine-level split",
                "Implementation": (
                    "Complete engines are assigned to either training "
                    "or validation. Windows from the same engine cannot "
                    "appear in both groups."
                ),
                "Risk prevented": (
                    "Sequence overlap and engine-identity leakage."
                ),
            },
            {
                "Stage": "Train-only scaling",
                "Implementation": (
                    "The StandardScaler is fitted only on training-engine "
                    "measurements and then applied to validation and test."
                ),
                "Risk prevented": (
                    "Validation and test distribution leakage."
                ),
            },
            {
                "Stage": "Sliding training windows",
                "Implementation": (
                    "Thirty-cycle sequences are generated from training "
                    "and validation engines."
                ),
                "Risk prevented": (
                    "Loss of temporal information."
                ),
            },
            {
                "Stage": "Short-engine handling",
                "Implementation": (
                    "Edge padding is used when fewer than 30 observed "
                    "cycles are available."
                ),
                "Risk prevented": (
                    "Dropping valid short-history engines."
                ),
            },
            {
                "Stage": "Deployment-aligned test evaluation",
                "Implementation": (
                    "Only the final available sequence is evaluated for "
                    "each of the 100 test engines."
                ),
                "Risk prevented": (
                    "Repeated weighting of engines through overlapping "
                    "test windows."
                ),
            },
            {
                "Stage": "Consistent target cap",
                "Implementation": (
                    "The 125-cycle cap is applied consistently during "
                    "training, validation and reported test error."
                ),
                "Risk prevented": (
                    "Mismatch between optimization and evaluation targets."
                ),
            },
        ]
    )

    st.dataframe(
        preprocessing_df,
        use_container_width=True,
        hide_index=True,
    )

    st.success(
        """
        The revised preprocessing produced approximately
        **14,459 training windows**, **3,272 validation windows**
        and exactly **100 final test windows** for FD001.
        """
    )

    st.markdown(
        '<div class="section-heading">'
        'Evaluation Protocol'
        '</div>',
        unsafe_allow_html=True,
    )

    evaluation_df = pd.DataFrame(
        [
            {
                "Principle": "Validation-only tuning",
                "Application": (
                    "Early stopping, model comparison and ensemble "
                    "weight selection use validation engines."
                ),
            },
            {
                "Principle": "Frozen test set",
                "Application": (
                    "The FD001 test engines are used only after model "
                    "and ensemble decisions are fixed."
                ),
            },
            {
                "Principle": "One engine, one vote",
                "Application": (
                    "Each unseen test engine contributes exactly one "
                    "final prediction to headline metrics."
                ),
            },
            {
                "Principle": "Multiple metrics",
                "Application": (
                    "RMSE, MAE, R², NASA score and signed error are "
                    "evaluated together."
                ),
            },
            {
                "Principle": "Safety-aware analysis",
                "Application": (
                    "Overprediction is examined separately because "
                    "optimistic RUL estimates can delay maintenance."
                ),
            },
            {
                "Principle": "Retrospective decision validation",
                "Application": (
                    "Actual RUL is used only after decision assignment "
                    "to measure error capture and review effectiveness."
                ),
            },
        ]
    )

    st.dataframe(
        evaluation_df,
        use_container_width=True,
        hide_index=True,
    )

    with st.expander(
        "Why overlapping test-window evaluation is misleading"
    ):
        st.markdown(
            """
            Evaluating every overlapping sequence from every test engine
            can give engines with longer histories disproportionately
            more influence over the reported metrics.

            MachineGuard AI instead evaluates the final observed
            sequence from each engine. This reflects the real operational
            question:

            **Given all data available today, what is the engine's
            estimated remaining useful life?**
            """
        )

    with st.expander(
        "Why the scaler must be fitted on training engines only"
    ):
        st.markdown(
            """
            A scaler learns population statistics such as feature means
            and standard deviations. Fitting it using validation or test
            measurements transfers information from unseen engines into
            the training pipeline.

            Even though no RUL labels are used during scaling, this still
            changes the representation available to the model and can
            create optimistic evaluation.
            """
        )


# =========================================================
# MODELS AND DECISION LAYER
# =========================================================

with model_tab:
    st.markdown(
        '<div class="section-heading">'
        'Model Benchmark'
        '</div>',
        unsafe_allow_html=True,
    )

    model_df = pd.DataFrame(
        [
            {
                "Model": "Ridge",
                "RMSE": 14.9931,
                "MAE": 12.2316,
                "R²": 0.8600,
                "NASA score": 374.9761,
                "Final role": "Independent baseline",
            },
            {
                "Model": "LSTM",
                "RMSE": 14.2865,
                "MAE": 10.9647,
                "R²": 0.8729,
                "NASA score": 443.0913,
                "Final role": "Secondary sequence model",
            },
            {
                "Model": "GRU",
                "RMSE": 13.5849,
                "MAE": 9.8770,
                "R²": 0.8851,
                "NASA score": 317.7112,
                "Final role": "Primary operational model",
            },
            {
                "Model": "Transformer",
                "RMSE": 21.7041,
                "MAE": 15.4325,
                "R²": 0.7060,
                "NASA score": 1647.0,
                "Final role": "Rejected for FD001",
            },
            {
                "Model": "60% GRU + 40% LSTM",
                "RMSE": 13.2122,
                "MAE": 10.0868,
                "R²": 0.8913,
                "NASA score": 291.3001,
                "Final role": "Best aggregate accuracy",
            },
        ]
    )

    st.dataframe(
        model_df,
        use_container_width=True,
        hide_index=True,
    )

    selection_columns = st.columns(3)

    with selection_columns[0]:
        st.success(
            """
            **Primary model: GRU**

            The GRU provides the best standalone balance of RMSE,
            MAE, NASA score, computational efficiency and lower
            dangerous-overprediction rate.
            """
        )

    with selection_columns[1]:
        st.info(
            """
            **Accuracy model: Ensemble**

            Validation-only selection produced a 60% GRU and 40% LSTM
            ensemble. It achieved the best aggregate RMSE and NASA score.
            """
        )

    with selection_columns[2]:
        st.warning(
            """
            **Supporting model: Ridge**

            Ridge remained competitive despite its simplicity and
            supplies independent evidence from a non-sequential model.
            """
        )

    st.error(
        """
        **Transformer decision:** The Transformer was retained as an
        honest negative result. It added complexity but performed
        substantially worse than GRU, LSTM and Ridge on FD001.
        """
    )

    st.markdown(
        '<div class="section-heading">'
        'Uncertainty and Temporal Diagnostics'
        '</div>',
        unsafe_allow_html=True,
    )

    reliability_df = pd.DataFrame(
        [
            {
                "Mechanism": "Monte Carlo dropout",
                "Purpose": (
                    "Runs the GRU repeatedly with dropout active to "
                    "measure prediction variation."
                ),
                "Release result": (
                    f"{MC_DROPOUT_PASSES} passes per engine."
                ),
            },
            {
                "Mechanism": "Interval calibration",
                "Purpose": (
                    "Scales the raw MC-dropout spread using validation "
                    "residual behaviour."
                ),
                "Release result": (
                    f"Multiplier {CALIBRATION_MULTIPLIER:.4f}."
                ),
            },
            {
                "Mechanism": "Coverage evaluation",
                "Purpose": (
                    "Checks how often the calibrated interval contains "
                    "the retrospective test RUL."
                ),
                "Release result": (
                    f"{OBSERVED_COVERAGE * 100:.0f}% observed versus "
                    f"{TARGET_COVERAGE * 100:.0f}% requested."
                ),
            },
            {
                "Mechanism": "Model disagreement",
                "Purpose": (
                    "Compares GRU evidence with the independent Ridge "
                    "baseline."
                ),
                "Release result": (
                    "Used as one reliability-risk input."
                ),
            },
            {
                "Mechanism": "Trajectory diagnostics",
                "Purpose": (
                    "Inspects slope, plateau behaviour, jumps and "
                    "monotonicity across historical windows."
                ),
                "Release result": (
                    "Detects failures hidden by final-point agreement."
                ),
            },
        ]
    )

    st.dataframe(
        reliability_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-heading">'
        'V5 Engineering Decision Layer'
        '</div>',
        unsafe_allow_html=True,
    )

    decision_layer_df = pd.DataFrame(
        [
            {
                "Output": "Engine condition",
                "Question answered": (
                    "How urgent does the estimated physical condition "
                    "appear?"
                ),
                "Examples": (
                    "Critical, Warning, Monitor, Healthy"
                ),
            },
            {
                "Output": "Reliability risk",
                "Question answered": (
                    "How suspicious or uncertain is the model evidence?"
                ),
                "Examples": "High, Medium, Low",
            },
            {
                "Output": "Engineering review",
                "Question answered": (
                    "How strongly should a human engineer inspect the "
                    "prediction?"
                ),
                "Examples": (
                    "Required, Recommended, Not required"
                ),
            },
            {
                "Output": "Operational priority",
                "Question answered": (
                    "How urgently should this engine enter the "
                    "maintenance workflow?"
                ),
                "Examples": (
                    "Immediate, High, Elevated, Watch, Routine"
                ),
            },
        ]
    )

    st.dataframe(
        decision_layer_df,
        use_container_width=True,
        hide_index=True,
    )

    st.info(
        """
        Condition and reliability are intentionally separated.
        A healthy-looking engine can still require review when its
        trajectory is suspicious, while a critical engine can have a
        comparatively trustworthy prediction.
        """
    )

    st.markdown("#### Why Engine 67 matters")

    st.error(
        """
        Engine 67 exposed a shared static-model blind spot. The GRU and
        Ridge both produced similarly high final RUL estimates even
        though the retrospective actual RUL was much lower.

        Final-point disagreement alone therefore looked reassuring.

        Historical trajectory analysis revealed that the prediction
        remained on an unrealistically high plateau while the engine
        continued ageing. The V5 decision layer preserved a mandatory
        engineering review because of this temporal evidence.
        """
    )

    st.markdown(
        '<div class="section-heading">'
        'Global Explainability'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        Global sensor importance is measured using **feature occlusion**.
        Each input feature is neutralized separately and the trained
        model is evaluated again.

        The analysis records:

        - increase in RMSE
        - increase in MAE
        - mean absolute change in predicted RUL
        - mean signed change in predicted RUL
        """
    )

    top_sensor_df = pd.DataFrame(
        {
            "GRU importance rank": list(range(1, 11)),
            "Feature": [
                "sensor11",
                "sensor9",
                "sensor14",
                "sensor15",
                "sensor20",
                "sensor2",
                "sensor12",
                "sensor13",
                "sensor3",
                "sensor17",
            ],
        }
    )

    st.dataframe(
        top_sensor_df,
        use_container_width=True,
        hide_index=True,
    )

    st.warning(
        """
        Occlusion importance measures model reliance, not physical
        causality. A highly ranked sensor may be correlated with a
        degradation process without directly causing it.
        """
    )


# =========================================================
# RESPONSIBLE USE
# =========================================================

with responsibility_tab:
    st.markdown(
        '<div class="section-heading">'
        'Intended Use'
        '</div>',
        unsafe_allow_html=True,
    )

    intended_use_df = pd.DataFrame(
        [
            {
                "Suitable use": (
                    "Research and engineering portfolio demonstration"
                ),
                "Reason": (
                    "Shows a complete ML workflow from data processing "
                    "through reliability-aware decision support."
                ),
            },
            {
                "Suitable use": (
                    "Retrospective prognostic analysis"
                ),
                "Reason": (
                    "Compares predictions against known NASA test labels."
                ),
            },
            {
                "Suitable use": (
                    "Human-in-the-loop maintenance prioritisation"
                ),
                "Reason": (
                    "Review flags and operational priorities support "
                    "rather than replace engineering judgement."
                ),
            },
            {
                "Suitable use": (
                    "Model-risk exploration"
                ),
                "Reason": (
                    "Uncertainty, disagreement and temporal diagnostics "
                    "make failure behaviour visible."
                ),
            },
        ]
    )

    st.dataframe(
        intended_use_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-heading">'
        'Prohibited Interpretation'
        '</div>',
        unsafe_allow_html=True,
    )

    st.error(
        """
        MachineGuard AI must not be treated as:

        - a certified aircraft-maintenance system
        - an autonomous maintenance-authorisation tool
        - a substitute for inspection, diagnostics or engineering review
        - proof that a specific sensor physically caused degradation
        - a guarantee of future failure time
        """
    )

    st.markdown(
        '<div class="section-heading">'
        'Current Limitations'
        '</div>',
        unsafe_allow_html=True,
    )

    limitations_df = pd.DataFrame(
        [
            {
                "Limitation": "Simulated benchmark data",
                "Consequence": (
                    "C-MAPSS does not reproduce every physical, "
                    "operational and maintenance factor present in "
                    "real turbofan fleets."
                ),
                "Mitigation": (
                    "Present results as research evidence rather than "
                    "deployment certification."
                ),
            },
            {
                "Limitation": "FD001 operating scope",
                "Consequence": (
                    "FD001 represents one operating-condition regime "
                    "and one fault-mode regime."
                ),
                "Mitigation": (
                    "Extend testing to FD002, FD003, FD004 and external "
                    "datasets."
                ),
            },
            {
                "Limitation": "Heuristic decision thresholds",
                "Consequence": (
                    "Condition, risk and review cutoffs are engineering "
                    "rules rather than certified maintenance standards."
                ),
                "Mitigation": (
                    "Recalibrate thresholds with domain experts and "
                    "independent operational evidence."
                ),
            },
            {
                "Limitation": "Incomplete uncertainty coverage",
                "Consequence": (
                    "MC dropout captures model variation but not all "
                    "forms of data, sensor or domain uncertainty."
                ),
                "Mitigation": (
                    "Add conformal prediction, ensembles, OOD detection "
                    "and sensor-health monitoring."
                ),
            },
            {
                "Limitation": "Shared model blind spots",
                "Consequence": (
                    "Different models may agree and still be wrong."
                ),
                "Mitigation": (
                    "Use trajectory diagnostics and independent physical "
                    "evidence rather than disagreement alone."
                ),
            },
            {
                "Limitation": "Global explanations only",
                "Consequence": (
                    "The current reusable artifacts do not preserve a "
                    "full local sensor attribution for every engine."
                ),
                "Mitigation": (
                    "Add engine-specific attribution artifacts in a "
                    "future release."
                ),
            },
            {
                "Limitation": "No maintenance-history context",
                "Consequence": (
                    "The model does not know component replacements, "
                    "inspection findings or repair quality."
                ),
                "Mitigation": (
                    "Integrate maintenance records in a real application."
                ),
            },
            {
                "Limitation": "No direct sensor-fault diagnosis",
                "Consequence": (
                    "Suspicious predictions may originate from sensor "
                    "failure rather than engine degradation."
                ),
                "Mitigation": (
                    "Add data-quality and sensor-plausibility checks."
                ),
            },
        ]
    )

    st.dataframe(
        limitations_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown(
        '<div class="section-heading">'
        'Human Oversight Principles'
        '</div>',
        unsafe_allow_html=True,
    )

    oversight_columns = st.columns(2)

    with oversight_columns[0]:
        st.success(
            """
            **The model may support**

            - fleet screening
            - maintenance prioritisation
            - difficult-case identification
            - comparison of multiple predictive signals
            - structured engineering review
            """
        )

    with oversight_columns[1]:
        st.warning(
            """
            **The engineer remains responsible for**

            - confirming sensor validity
            - interpreting operating context
            - reviewing maintenance history
            - determining inspection actions
            - approving operational decisions
            """
        )

    st.markdown(
        '<div class="section-heading">'
        'Future Technical Extensions'
        '</div>',
        unsafe_allow_html=True,
    )

    future_work_df = pd.DataFrame(
        [
            {
                "Priority": "High",
                "Extension": "Conformal prediction",
                "Value": (
                    "Provides distribution-free empirical coverage "
                    "under defined assumptions."
                ),
            },
            {
                "Priority": "High",
                "Extension": "Out-of-distribution detection",
                "Value": (
                    "Identifies engines operating outside the training "
                    "data manifold."
                ),
            },
            {
                "Priority": "High",
                "Extension": "FD002–FD004 validation",
                "Value": (
                    "Tests generalisation across operating conditions "
                    "and fault-mode complexity."
                ),
            },
            {
                "Priority": "Medium",
                "Extension": "Engine-level local explanations",
                "Value": (
                    "Shows which recent sensor behaviour influenced a "
                    "specific decision."
                ),
            },
            {
                "Priority": "Medium",
                "Extension": "Sensor-quality monitoring",
                "Value": (
                    "Separates potential sensor faults from physical "
                    "degradation signals."
                ),
            },
            {
                "Priority": "Medium",
                "Extension": "Threshold optimisation",
                "Value": (
                    "Balances review workload, missed-risk cost and "
                    "maintenance consequences."
                ),
            },
            {
                "Priority": "Long term",
                "Extension": "Real fleet validation",
                "Value": (
                    "Evaluates domain shift, maintenance actions and "
                    "actual operational constraints."
                ),
            },
        ]
    )

    st.dataframe(
        future_work_df,
        use_container_width=True,
        hide_index=True,
    )


# =========================================================
# REPRODUCIBILITY
# =========================================================

with reproducibility_tab:
    st.markdown(
        '<div class="section-heading">'
        'Release Artifact Inventory'
        '</div>',
        unsafe_allow_html=True,
    )

    artifact_paths = [
        {
            "Artifact": "V5 engine decisions",
            "Path": (
                RESULTS_DIR
                / "decisions_v5"
                / "decision_results_v5_gru_FD001.csv"
            ),
            "Purpose": (
                "Condition, reliability, review and operational priority."
            ),
        },
        {
            "Artifact": "GRU trajectory predictions",
            "Path": (
                RESULTS_DIR
                / "trajectory_diagnostics"
                / "trajectory_predictions_gru_FD001.csv"
            ),
            "Purpose": (
                "Cycle-level historical predicted-RUL trajectories."
            ),
        },
        {
            "Artifact": "GRU global importance",
            "Path": (
                RESULTS_DIR
                / "explanations"
                / "global_importance_gru_FD001.csv"
            ),
            "Purpose": (
                "Occlusion-based global sensor importance."
            ),
        },
        {
            "Artifact": "LSTM global importance",
            "Path": (
                RESULTS_DIR
                / "explanations"
                / "global_importance_lstm_FD001.csv"
            ),
            "Purpose": (
                "Cross-model explainability comparison."
            ),
        },
        {
            "Artifact": "V4–V5 validation metrics",
            "Path": (
                RESULTS_DIR
                / "decision_validation_v5"
                / "validation_metrics_v4_v5_gru_FD001.csv"
            ),
            "Purpose": (
                "Review workload and error-capture validation."
            ),
        },
        {
            "Artifact": "Single-page dashboard backup",
            "Path": (
                PROJECT_ROOT
                / "app_single_page_backup.py"
            ),
            "Purpose": (
                "Preserved dashboard before multipage migration."
            ),
        },
    ]

    artifact_inventory_df = pd.DataFrame(
        [
            {
                "Artifact": artifact["Artifact"],
                "Status": artifact_status(
                    artifact["Path"]
                ),
                "Project path": relative_path(
                    artifact["Path"]
                ),
                "Purpose": artifact["Purpose"],
            }
            for artifact in artifact_paths
        ]
    )

    st.dataframe(
        artifact_inventory_df,
        use_container_width=True,
        hide_index=True,
    )

    available_artifacts = int(
        (
            artifact_inventory_df[
                "Status"
            ]
            == "Available"
        ).sum()
    )

    total_artifacts = len(
        artifact_inventory_df
    )

    artifact_columns = st.columns(3)

    artifact_columns[0].metric(
        "Artifacts checked",
        total_artifacts,
    )

    artifact_columns[1].metric(
        "Available",
        available_artifacts,
    )

    artifact_columns[2].metric(
        "Missing",
        total_artifacts - available_artifacts,
    )

    if available_artifacts == total_artifacts:
        st.success(
            "All dashboard release artifacts checked on this page "
            "are currently available."
        )
    else:
        st.warning(
            "One or more release artifacts are missing. The affected "
            "dashboard page may not load until the generating pipeline "
            "is rerun."
        )

    st.markdown(
        '<div class="section-heading">'
        'Reproducibility Principles'
        '</div>',
        unsafe_allow_html=True,
    )

    reproducibility_df = pd.DataFrame(
        [
            {
                "Principle": "Preserved outputs",
                "Implementation": (
                    "Metrics, predictions, explanations and validation "
                    "results are stored as versioned CSV artifacts."
                ),
            },
            {
                "Principle": "Frozen release logic",
                "Implementation": (
                    "The completed FD001 pipeline is preserved before "
                    "dashboard restructuring."
                ),
            },
            {
                "Principle": "Validation-only selection",
                "Implementation": (
                    "Test labels do not determine early stopping, model "
                    "selection or ensemble weights."
                ),
            },
            {
                "Principle": "Explicit rejected experiments",
                "Implementation": (
                    "The underperforming Transformer result is retained "
                    "rather than omitted."
                ),
            },
            {
                "Principle": "Retrospective-label separation",
                "Implementation": (
                    "Actual RUL is clearly marked whenever displayed in "
                    "the dashboard."
                ),
            },
            {
                "Principle": "Backup before migration",
                "Implementation": (
                    "The original working single-page dashboard remains "
                    "available during the multipage transition."
                ),
            },
        ]
    )

    st.dataframe(
        reproducibility_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Known release scripts")

    script_df = pd.DataFrame(
        [
            {
                "Script": "src/preprocessing_v2.py",
                "Responsibility": (
                    "Leakage-safe sequence creation, train-only scaling "
                    "and final-window test preparation."
                ),
            },
            {
                "Script": "src/trajectory_diagnostics_v2.py",
                "Responsibility": (
                    "Historical prediction trajectories and temporal "
                    "diagnostic features."
                ),
            },
            {
                "Script": "src/decision_layer_v5.py",
                "Responsibility": (
                    "Condition, reliability, review and operational "
                    "priority decisions."
                ),
            },
            {
                "Script": "app.py",
                "Responsibility": (
                    "Multipage Streamlit routing and navigation."
                ),
            },
            {
                "Script": "dashboard/shared.py",
                "Responsibility": (
                    "Shared project paths, styling and sidebar branding."
                ),
            },
        ]
    )

    st.dataframe(
        script_df,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Dashboard launch command")

    st.code(
        "streamlit run app.py",
        language="powershell",
    )

    st.caption(
        f"Project root resolved by the dashboard: {PROJECT_ROOT}"
    )


# ---------------------------------------------------------
# Final project statement
# ---------------------------------------------------------

st.markdown(
    '<div class="section-heading">'
    'Project Positioning'
    '</div>',
    unsafe_allow_html=True,
)

st.success(
    """
    MachineGuard AI demonstrates more than training a high-performing
    neural network. The project shows how to design a complete
    engineering ML workflow around leakage prevention, honest
    benchmarking, uncertainty, explainability, temporal failure
    detection, human review and responsible operational use.
    """
)

st.warning(
    """
    The project remains a research and portfolio system. Its outputs
    require engineering interpretation and must not be used as
    autonomous aircraft-maintenance instructions.
    """
)