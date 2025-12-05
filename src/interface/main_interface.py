import streamlit as st
from interface.questionnaire import render_investor_questionnaire
from investor_information.investor_profile import investor_target_volatility


def get_investor_profile(answers):

    (sigma_min, sigma_max), RA, RC, RT = investor_target_volatility(**answers)
    st.subheader("Resultados del perfil de riesgo")
    col1, col2, col3 = st.columns(3)
    col1.metric("Apetito de riesgo (RA)", RA)
    col2.metric("Capacidad de riesgo (RC)", RC)
    col3.metric("Tolerancia final (RT)", RT)

    st.subheader("Volatilidad objetivo de la cartera")
    st.write(
        f"Rango recomendado de volatilidad anualizada: "
        f"**{sigma_min * 100:.1f}% – {sigma_max * 100:.1f}%**"
    )




def render_app():
    tab_profile, tab_portfolio = st.tabs(["Perfil de riesgo", "Cartera de inversión"])
    with tab_profile:
        answers = render_investor_questionnaire()

    if answers is None:
        st.write("Por favor, completa el cuestionario")
        return

    get_investor_profile(answers)





