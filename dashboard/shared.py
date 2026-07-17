from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results_v2"


def configure_page() -> None:
    """Configure the common Streamlit application frame."""

    st.set_page_config(
        page_title="MachineGuard AI",
        page_icon="⚙️",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def apply_global_styles() -> None:
    """Apply common dashboard styling."""

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


def render_sidebar_branding() -> None:
    """Render shared sidebar project information."""

    with st.sidebar:
        st.markdown("## ⚙️ MachineGuard AI")

        st.caption(
            "Explainable and uncertainty-aware "
            "predictive maintenance"
        )

        st.markdown("---")

        st.markdown(
            """
            **Current release**

            - Dataset: NASA C-MAPSS FD001
            - Primary model: GRU
            - Decision layer: V5
            - Test engines: 100
            - RUL cap: 125 cycles
            """
        )