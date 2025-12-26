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
    st.set_page_config(page_title="UOC - Robo Advisor", page_icon="📈", layout="wide")
    apply_global_styles()

    st.session_state.setdefault("route", "landing")

    qp = st.query_params
    if qp.get("route"):
        st.session_state["route"] = qp.get("route")
        st.query_params.clear()
        st.rerun()

    route = st.session_state["route"]

    if route == "landing":
        render_landing()

    elif route == "questionnaire":
        page = render_sidebar()
        answers = render_investor_questionnaire()
        show_investor_profile(answers)

    elif route == "portfolio":
        page = render_sidebar()
        render_constraints_portfolio()
        if not st.session_state.get("data_ready", False):
            return

    elif route == "results":
        page = render_sidebar()
        render_results()

    elif route == "analysis":
        page = render_sidebar()
        render_analysis()

    else:
        st.session_state["route"] = "landing"
        st.rerun()