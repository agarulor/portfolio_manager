import streamlit as st
from interface.investor_profile_view import (
    render_investor_questionnaire,
    show_investor_profile,
    render_sidebar_profile_summary,
)
from interface.main_interface import apply_global_styles, render_sidebar
from interface.render_portfolio import render_portfolio
from interface.landing_page import render as render_landing

def run_app():
    st.set_page_config(page_title="UOC - Robo Advisor", page_icon="📈", layout="wide")
    apply_global_styles()  # estilos globales 1 vez

    if "route" not in st.session_state:
        st.session_state["route"] = "landing"

    # capturar query params (click del botón HTML)
    qp = st.query_params
    if qp.get("route"):
        st.session_state["route"] = qp.get("route")
        st.query_params.clear()
        st.rerun()

    # router
    route = st.session_state["route"]

    if route == "landing":
        render_landing()

    elif route == "questionnaire":
        page = render_sidebar()
        st.header("Perfil de riesgo del inversor")
        answers = render_investor_questionnaire()
        show_investor_profile(answers)
        render_sidebar_profile_summary()

    elif route == "portfolio":
        page = render_sidebar()
        render_portfolio()

    else:
        st.session_state["route"] = "landing"
        st.rerun()

""" st.set_page_config(page_title="Gestor de carteras", layout="centered")

  apply_global_styles()
  page = render_sidebar()

  if page == "Perfil de riesgo":
      st.header("Perfil de riesgo del inversor")
      answers = render_investor_questionnaire()
      show_investor_profile(answers)

  elif page == "Cartera de inversión":
      render_portfolio()

  render_sidebar_profile_summary()"""