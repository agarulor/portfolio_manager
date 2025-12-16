import streamlit as st
from types import MappingProxyType # to avoid changes to a dictionary
from investor_information.investor_profile import investor_target_volatility

from interface.main_interface import render_sidebar_profile_summary

RISK_COLOR = MappingProxyType({   1: "#2ecc71", 2: "#6bdc8b", 3: "#f1c40f", 4: "#f39c12", 5: "#e67e22", 6: "#e74c3c",})
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


def render_risk_scale(rt_value: int):
    # colors from clearer to red

    blocks_html = ""
    labels_html = ""
    for i in range(1, 7):
        active = i == rt_value
        blocks_html += f"""
        <div style="
        flex:1;
        height:{'52px' if active else '38px'};
        background:{RISK_COLOR[i]};
        border-radius:8px;
        display:flex;
        align-items:center;
        justify-content:center;
        font-weight:800;
        color:white;
        font-size:1.2rem;
        box-shadow:{'0 0 0 3px #000078' if active else 'none'};
        opacity:{'1' if active else '0.6'};
        transition:all 0.2s ease;
        ">
            {i}
        </div>
        """

        labels_html += f"""
        <div style="
        flex:1;
        text-align:center;
        font-size:0.85rem;
        font-weight:{'700' if active else '500'};
        color:{RISK_COLOR[i] if active else '#64748B'};
        margin-top:0.4rem;
        ">
        {RISK_PROFILE_DICTIONARY[i]}
        </div>
        """
    st.markdown(
        f"""
        <div style="margin-top:2.5rem;">
        <div style="
        text-align:center;
        font-size:1.5rem;
        font-weight:700;
        color:#000078;
        margin-bottom:1rem;
        ">
        Su perfil de riesgo es el siguiente
        </div>
        
        <div style="
        display:flex;
        gap:0.5rem;
        max-width:700px;
        margin:0 auto;
        ">
        {blocks_html}
        </div>
        
        <div style="
        display:flex;
        gap:0.5rem;
        max-width:700px;
        margin:0 auto;
        ">
        {labels_html}
        </div>
        </div>
        """,
        unsafe_allow_html=True
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
        1: "1 - Inmediata (muy alta necesidad de liquidez)",
        2: "2 - Alta",
        3: "3 - Media",
        4: "4 - Baja",
        5: "5 - Muy baja"
    }

    income_options = {
        1: "1 - Muy bajos",
        2: "2 - Bajos",
        3: "3 - Medios",
        4: "4 - Altos",
        5: "5 - Muy altos"
    }

    net_worth_options = {
        1: "1 - Muy bajos",
        2: "2 - Bajos",
        3: "3 - Medios",
        4: "4 - Altos",
        5: "5 - Muy altos"
    }

    horizon_options = {
        1: "1 - Muy corto plazo",
        2: "2 - Corto plazo",
        3: "3 - Medio plazo",
        4: "4 - Largo plazo",
        5: "5 - Muy largo plazo"
    }

    goal_importance_options = {
        1: "1 - Crítico",
        2: "2 - Importancia media",
        3: "3 - Flexible"
    }

    liquidity_need = radio_question(
        number=1,
        text="Liquidez necesaria",
        options_dict=liquidity_options,
        key="liquidity_need",
        default_index=2,
    )

    annual_income = radio_question(
        number=2,
        text="Ingresos anuales",
        options_dict=income_options,
        key="annual_income",
        default_index=2,
    )

    net_worth = radio_question(
        number=3,
        text="Ahorros y patrimonio",
        options_dict=net_worth_options,
        key="net_worth",
        default_index=2,
    )

    investment_horizon = radio_question(
        number=4,
        text="Horizonte temporal",
        options_dict=horizon_options,
        key="investment_horizon",
        default_index=2,
    )

    financial_goal_importance = radio_question(
        number=5,
        text="Importancia del objetivo financiero",
        options_dict=goal_importance_options,
        key="financial_goal",
        default_index=2,
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


    render_risk_scale(RT)
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
