import streamlit as st

from dashboard.shared import (
    apply_global_styles,
    configure_page,
    render_sidebar_branding,
)


configure_page()
apply_global_styles()


pages = {
    "MachineGuard AI": [
        st.Page(
            "dashboard/pages/complete_dashboard.py",
            title="Complete Dashboard — Full Project Story",
            icon="⚙️",
            default=True,
        ),
    ],
    "Operations": [
        st.Page(
            "dashboard/pages/fleet_overview.py",
            title="Fleet Overview",
            icon="📊",
        ),
        st.Page(
            "dashboard/pages/engine_explorer.py",
            title="Engine Explorer",
            icon="🔍",
        ),
        st.Page(
            "dashboard/pages/trajectory_uncertainty.py",
            title="Trajectory & Uncertainty",
            icon="📈",
        ),
    ],
    "Model Intelligence": [
        st.Page(
            "dashboard/pages/explainability.py",
            title="Explainability",
            icon="🧠",
        ),
        st.Page(
            "dashboard/pages/model_performance.py",
            title="Model Performance",
            icon="🏁",
        ),
        st.Page(
            "dashboard/pages/decision_validation.py",
            title="Decision Validation",
            icon="🛡️",
        ),
    ],
    "Project": [
        st.Page(
            "dashboard/pages/methodology.py",
            title="Methodology & Responsible Use",
            icon="📘",
        ),
    ],
}


navigation = st.navigation(
    pages,
    position="sidebar",
    expanded=True,
)

render_sidebar_branding()

navigation.run()