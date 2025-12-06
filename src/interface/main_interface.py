import streamlit as st
from interface.questionnaire import render_investor_questionnaire
from investor_information.investor_profile import investor_target_volatility
from types import MappingProxyType
RISK_COLOR = MappingProxyType({1: "#2ecc71", 2: "#2ecc71", 3: "#f39c12", 4: "#f39c12", 5: "#e74c3c", 6: "#e74c3c"})
RISK_PROFILE_DICTIONARY = MappingProxyType({
    1: "Perfil bajo de riesgo",
    2: "Perfil medio-bajo de riesgo",
    3: "Perfil medio de riesgo",
    4: "Perfil medio-alto de riesgo",
    5: "Perfil alto de riesgo",
    6: "Perfil agresivo de riesgo"
})

def get_investor_profile(answers):

    (sigma_min, sigma_max), RA, RC, RT = investor_target_volatility(**answers)

    st.subheader("Resultados del perfil de riesgo")

    col1, col2 = st.columns(2)
    col1.markdown(
        f"""
        <div style="text-align: center;">
            <div style="font-size: 18px; font-weight: 500; color: #555;">
                Apetito de riesgo:
            </div>
            <div style="font-size: 36px; font-weight: 800; margin-top: 4px;">
                {RA}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    col2.markdown(
        f"""
        <div style="text-align: center;">
            <div style="font-size: 18px; font-weight: 500; color: #555;">
                Capacidad de asumir riesgo
            </div>
            <div style="font-size: 36px; font-weight: 800; margin-top: 4px;">
                {RC}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    color_tolerance = RISK_COLOR[RT]
    st.markdown(
        f"""
        <div style="text-align: center; font-size: 26px; font-weight: 600; margin-top: 40px">
            Tolerancia final (RT)
            <div style= "font-size: 48px;font-weight: 800;color: {color_tolerance};margin-top: 10px">{RT} </div>
            <div style="font-size: 48px;font-weight: 800; color: {color_tolerance};margin-top: 10px"> {RISK_PROFILE_DICTIONARY[RT]} </div>
        </div>
        """,
        unsafe_allow_html=True
    )

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

    with tab_portfolio:
        print("hola")



