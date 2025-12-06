import streamlit as st
from interface.investor_profile_view import render_investor_questionnaire, show_investor_profile
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

def apply_global_styles():
    st.markdown(
        """
        <style>
        /* Ajustes globales opcionales */
        .main > div {
            padding-top: 1rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


def render_sidebar_profile_summary():
    if "risk_result" not in st.session_state:
        return

    res = st.session_state["risk_result"]
    RT = res["RT"]
    color = RISK_COLOR[RT]

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Perfil del inversor")

    st.sidebar.markdown(
        f"""
        <div style="text-align: center; font-size: 14px; font-weight: 500; color: #555;">
            Tolerancia final (RT)
            <div style="font-size: 32px; font-weight: 800; color: {color}; margin-top: 6px;">
                {RT}
            </div>
            <div style="font-size: 14px; font-weight: 600; color: {color}; margin-top: 4px;">
                {RISK_PROFILE_DICTIONARY[RT]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_sidebar():
    st.sidebar.title("Menú")

    page = st.sidebar.radio(
        "Opciones",
        options=["Perfil de riesgo", "Cartera de inversión"],
        index=0
    )
    render_sidebar_profile_summary()
    return page


def render_portfolio():
    """Contenido de la pestaña 'Cartera de inversión'."""
    if "risk_result" not in st.session_state:
        st.warning("Primero completa el cuestionario de perfil de riesgo.")
        return

    res = st.session_state["risk_result"]
    RT = res["RT"]
    sigma_min = res["sigma_min"]
    sigma_max = res["sigma_max"]

    st.header("Cartera de inversión recomendada")

    st.write(
        f"Perfil de riesgo final: **{RT} – {RISK_PROFILE_DICTIONARY[RT]}**"
    )
    st.write(
        f"Volatilidad objetivo: **{sigma_min*100:.1f}% – {sigma_max*100:.1f}%**"
    )

    # aquí luego puedes meter tu optimización y mostrar pesos
    st.info("Aquí iría la construcción de la cartera (pesos por activo, gráficos, etc.).")

def render_app():
    apply_global_styles()
    page = render_sidebar()

    if page == "Perfil de riesgo":
        st.header("Perfil de riesgo del inversor")
        answers = render_investor_questionnaire()
        show_investor_profile(answers)



    elif page == "Cartera de inversión":
        render_portfolio()