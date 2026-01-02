import streamlit as st
from interface.investor_profile_view import render_investor_questionnaire, show_investor_profile
from interface.main_interface import apply_global_styles, render_sidebar
from interface.render_initial_portfolio import render_constraints_portfolio
from interface.landing_page import render as render_landing
from interface.render_portfolio_results import render_results
from interface.render_analysis import render_analysis


def run_app():
    """
    Computes run app.

    Parameters
    ----------

    Returns
    -------
    Any: run app output.
    """

    # Basic Streamlit page setup (title, icon and full-width layout)
    st.set_page_config(page_title="UOC - Robo Advisor", page_icon="📈", layout="wide")

    # Apply global CSS styles so everything looks consistent
    apply_global_styles()

    # Default route when the app starts
    st.session_state.setdefault("route", "landing")

    # Check if a route is passed via URL (useful for refreshes or links)
    qp = st.query_params
    if qp.get("route"):
        st.session_state["route"] = qp.get("route")
        st.query_params.clear()
        st.rerun()

    # Current route controls what page is rendered
    route = st.session_state["route"]

    # Landing / welcome page
    if route == "landing":
        render_landing()

    # Questionnaire page (risk profile)
    elif route == "questionnaire":
        render_sidebar()
        answers = render_investor_questionnaire()
        show_investor_profile(answers)

    # Portfolio setup and constraints page
    elif route == "portfolio":
        render_sidebar()
        render_constraints_portfolio()

        # If data is not ready yet, stop here
        if not st.session_state.get("data_ready", False):
            return

    # Results page (portfolio evolution)
    elif route == "results":
        render_sidebar()
        render_results()

    # Data analysis section
    elif route == "analysis":
        render_sidebar()
        render_analysis()

    # Fallback: if something weird happens, go back to landing
    else:
        st.session_state["route"] = "landing"
        st.rerun()