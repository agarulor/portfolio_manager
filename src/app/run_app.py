import streamlit as st
from interface.investor_profile_view import render_investor_questionnaire, show_investor_profile, render_sidebar_profile_summary
from interface.main_interface import apply_global_styles, render_sidebar, render_portfolio


def run_app():
    st.set_page_config(page_title="Gestor de carteras", layout="centered")

    apply_global_styles()
    page = render_sidebar()

    if page == "Perfil de riesgo":
        st.header("Perfil de riesgo del inversor")
        answers = render_investor_questionnaire()
        show_investor_profile(answers)

    elif page == "Cartera de inversión":
        render_portfolio()

    render_sidebar_profile_summary()