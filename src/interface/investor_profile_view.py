import streamlit as st
from types import MappingProxyType # to avoid changes to a dictionary
from investor_information.investor_profile import investor_target_volatility

from interface.main_interface import render_sidebar_profile_summary

RISK_COLOR = MappingProxyType({1: "#2ecc71", 2: "#2ecc71", 3: "#f39c12", 4: "#f39c12", 5: "#e74c3c", 6: "#e74c3c"})
RISK_PROFILE_DICTIONARY = MappingProxyType({
    1: "Perfil bajo de riesgo",
    2: "Perfil medio-bajo de riesgo",
    3: "Perfil medio de riesgo",
    4: "Perfil medio-alto de riesgo",
    5: "Perfil alto de riesgo",
    6: "Perfil agresivo de riesgo"
})


def radio_question(
    number,
    text,
    options_dict,
    key,
    default_index=2,
):
    # Título
    st.markdown(
        f"""
        <div style="
        font-size: 1.1rem;
        font-weight: 700;
        color: #000078;
        margin-bottom: -5.0rem;
        margin-top: 1.0rem;
        ">
        {number}) {text}
        </div>
        """,
        unsafe_allow_html=True
    )

    # Radio
    return st.radio(
        label=" ",
        options=list(options_dict.keys()),
        format_func=lambda x: options_dict[x],
        index=default_index,
        key=key,
    )

def render_investor_questionnaire():

    st.title("Cuestionario sobre la tolerancia al riesgo del inversor para su clasificación")

    st.write("""Por favor, responda a las siguientes preguntas para poder determinar su perfil de riesgo. """
             """De esta forma podremos recomendarle una cartera de inversión acorde a sus necesidades""")

    st.header("Información referente sobre su apetito de riesgo")

    # ----------------------------------------------------------
    # RISK APPETITE
    # ----------------------------------------------------------

    # We first create the dictionaries with the answers for each dropdown menu
    knowledge_options = {
        1: "1 - Conocimiento bajo",
        2: "2 - Conocimiento medio-bajo",
        3: "3 - Conocimiento medio",
        4: "4 - Conocimiento medio-alto",
        5: "5 - Conocimiento alto",
        6: "6 - Conocimiento muy alto"
    }

    risk_level_options = {
        1: "1 - Bajo",
        2: "2 - Medio-bajo",
        3: "3 - Medio",
        4: "4 - Medio-alto",
        5: "5 - Alto",
        6: "6 - Muy alto"
    }

    downside_reaction_options = {
        1: "1 - Vendería toda la inversión",
        2: "2 - Vendería una parte de la inversión",
        3: "3 - Mantendría la inversión",
        4: "4 - Compraría más para aprovechar la caída"
    }


    knowledge = radio_question(
        number=1,
        text="Conocimiento financiero y experiencia",
        options_dict=knowledge_options,
        key="knowledge",
        default_index=2,
    )

    risk_level = radio_question(
        number=2,
        text="Nivel de riesgo dispuesto a asumir",
        options_dict=risk_level_options,
        key="risk_level",
        default_index=2,
    )

    downside_reaction = radio_question(
        number=3,
        text="Reacción ante caídas fuertes del precio de los activos",
        options_dict=downside_reaction_options,
        key="downside_reaction",
        default_index=2,
    )

    #----------------------------------------------------------
    # RISK CAPACITY
    #----------------------------------------------------------

    st.header("Información referente a la capacidad de asumir riesgo")

    liquidity_options = {
        1: "1 - Liquidez necesaria inmediata (muy alta necesidad de liquidez)",
        2: "2 - Alta necesidad de liquidez",
        3: "3 - Necesidad de liquidez media",
        4: "4 - Baja necesidad de liquidez",
        5: "5 - Muy baja necesidad de liquidez"
    }


    income_options = {
        1: "1 - Ingresos anuales muy bajos",
        2: "2 - Ingresos anuales bajos",
        3: "3 - Ingresos anuales medios",
        4: "4 - Ingresos anuales altos",
        5: "5 - Ingresos anuales muy altos"
    }

    net_worth_options = {
        1: "1 - Ahorros y patrimonio muy bajos",
        2: "2 - Ahorros y patrimonio bajos",
        3: "3 - Ahorros y patrimonio medios",
        4: "4 - Ahorros y patrimonio altos",
        5: "5 - Ahorros y patrimonio muy altos"
    }

    horizon_options = {
        1: "1 - Muy corto plazo",
        2: "2 - Corto plazo",
        3: "3 - Medio plazo",
        4: "4 - Largo plazo",
        5: "5 - Muy largo plazo"
    }

    goal_importance_options = {
        1: "1 - Objetivo financiero crítico",
        2: "2 - Objetivo financiero de importancia media",
        3: "3 - Objetivo financiero flexible"
    }

    liquidity_need = st.selectbox(
        "4) Liquidez necesaria:",
        options=list(liquidity_options.keys()),
        format_func=lambda x: liquidity_options[x],
        index=2
    )

    annual_income = st.selectbox(
        "5) Ingresos anuales:",
        options=list(income_options.keys()),
        format_func=lambda x: income_options[x],
        index=2
    )

    net_worth = st.selectbox(
        "6) Ahorros y patrimonio:",
        options=list(net_worth_options.keys()),
        format_func=lambda x: net_worth_options[x],
        index=2
    )

    investment_horizon = st.selectbox(
        "7) Horizonte temporal:",
        options=list(horizon_options.keys()),
        format_func=lambda x: horizon_options[x],
        index=2
    )

    financial_goal_importance = st.selectbox(
        "8) Importancia del objetivo financiero:",
        options=list(goal_importance_options.keys()),
        format_func=lambda x: goal_importance_options[x],
        index=1,
    )

    st.markdown("---")

    submitted = st.button("Obtener perfil de riesgo")

    if not submitted:
        # Button not yet pushed
        return None

    # We build the dictionary
    answers = {
        "knowledge": knowledge,
        "risk_level": risk_level,
        "downside_reaction": downside_reaction,
        "liquidity_need": liquidity_need,
        "annual_income": annual_income,
        "net_worth": net_worth,
        "investment_horizon": investment_horizon,
        "financial_goal_importance": financial_goal_importance,
    }

    return answers

def render_investor_profile_view(RA, RC, RT, sigma_min, sigma_max):
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
            <div style="font-size: 48px;font-weight: 800;color: {color_tolerance};margin-top: 10px">
                {RT}
            </div>
            <div style="font-size: 24px;font-weight: 800; color: {color_tolerance};margin-top: 10px">
                {RISK_PROFILE_DICTIONARY[RT]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.subheader("Volatilidad objetivo de la cartera")
    st.write(
        f"Rango recomendado de volatilidad anualizada: "
        f"**{sigma_min * 100:.1f}% – {sigma_max * 100:.1f}%**"
    )


def show_investor_profile(answers):
    if answers is not None:
        (sigma_min, sigma_max), RA, RC, RT = investor_target_volatility(**answers)

        # Save session state
        st.session_state["risk_result"] = {
            "RA": RA,
            "RC": RC,
            "RT": RT,
            "sigma_min": sigma_min,
            "sigma_max": sigma_max,
        }
        # We show the window
        render_investor_profile_view(RA, RC, RT, sigma_min, sigma_max)

    elif "risk_result" in st.session_state:
        # No update of the questionnaire,
        # But we can show a past resulta
        res = st.session_state["risk_result"]
        render_investor_profile_view(
            RA=res["RA"],
            RC=res["RC"],
            RT=res["RT"],
            sigma_min=res["sigma_min"],
            sigma_max=res["sigma_max"],
        )

    else:
        # No previous answers. We show info panel
        st.info("Por favor, completa el cuestionario para calcular tu perfil.")
