import streamlit as st
from types import MappingProxyType # to avoid changes to a dictionary

from investor_information.investor_profile import investor_target_volatility

RISK_PROFILE_DICTIONARY = MappingProxyType({
    1: "Perfil bajo de riesgo",
    2: "Perfil medio-bajo de riesgo",
    3: "Perfil medio de riesgo",
    4: "Perfil medio-alto de riesgo",
    5: "Perfil alto de riesgo",
    6: "Perfil agresivo de riesgo"
})

def investor_questionnaire():
    st.set_page_config(page_title="Cuestionario sobre la tolerancia de riesgo del invesor", layout="centered")

    st.title("Cuestionario sobre la tolerancia al riesgo del inversor para su clasificación")

    st.write("""Por favor, repsonda a las siguientes preguntas para poder determinar su perfil de riesgo. """
             """De esta forma podremos recomendarle una cartera de inversión acorde a sus necesidades""")

    st.header("Información referente sobre su apetito de riesgo")

    # ----------------------------------------------------------
    # RISK APPETITE
    # ----------------------------------------------------------

    # We first create the dictionaries with the answers for each dropdown menu
    knowledge_options = {
        1: "1 - Muy poco conocimiento sobre productos financieros",
        2: "2 - Poco conocimiento sobre productos financieros",
        3: "3 - Conocimiento medio sobre productos financieros",
        4: "4 - Buen conocimiento sobre productos financieros",
        5: "5 - Alto conocimiento sobre productos financieros",
        6: "6 - Conocimiento experto sobre productos financieros"
    }

    risk_level_options = {
        1: "1 - Nivel de riesgo dispuesto a asumir: bajo",
        2: "2 - Nivel de riesgo dispuesto a asumir: medio-bajo",
        3: "3 - Nivel de riesgo dispuesto a asumir: medio",
        4: "4 - Nivel de riesgo dispuesto a asumir: medio-alto",
        5: "5 - Nivel de riesgo dispuesto a asumir: alto",
        6: "6 - Nivel de riesgo dispuesto a asumir: muy alto"
    }

    downside_reaction_options = {
        1: "1 - Ante una caída fuerte vendería toda la inversión",
        2: "2 - Ante una caída fuerte vendería una parte de la inversión",
        3: "3 - Ante una caída fuerte mantendría la inversión",
        4: "4 - Ante una caída fuerte compraría más para aprovechar la caída"
    }

    knowledge = st.selectbox("1) Conocimiento financiero y experiencia",
                             options=list(knowledge_options.keys()),
                             format_func= lambda x: knowledge_options[x],
                             index= 2
                             )

    st.write("Tu conocimiento seleccionado es: ", knowledge)

    risk_level = st.selectbox("2) Nivel de riesgo dispuesto a asumir",
                             options=list(risk_level_options.keys()),
                             format_func= lambda x: risk_level_options[x],
                             index= 2
                             )
    st.write("Tu nivel de riesgo dispues a asumir es: ", risk_level_options[risk_level])

    downside_reaction = st.selectbox("3) Reacción ante caídas fuertes del precio de los activos",
                             options=list(downside_reaction_options.keys()),
                             format_func= lambda x: downside_reaction_options[x],
                             index= 2
                             )

    st.write("Tu nivel de reacción ante caídas fuertes del precio de los activos es: ", downside_reaction_options[downside_reaction])

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

    income = st.selectbox(
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

    # We know submit and check
    if  st.button("Enviar formulario"):
        (sigma_min, sigma_max), RA, RC, RT = investor_target_volatility(
            knowledge=knowledge,
            risk_level=risk_level,
            downside_reaction=downside_reaction,
            liquidity_need=liquidity_need,
            annual_income= income,
            net_worth=net_worth,
            investment_horizon=investment_horizon,
            financial_goal_importance=financial_goal_importance
        )

        st.success("Formulario enviado correctamente")
        st.subheader("Resultados del perfil de riesgo")
        col1, col2, col3 = st.columns(3)
        col1.metric("Apetito de riesgo: ", RISK_PROFILE_DICTIONARY[RA])
        col2.metric("Capacidad de asumir riesgo: ", RISK_PROFILE_DICTIONARY[RC])
        col3.metric("Tolerancia al riesgo: ", RISK_PROFILE_DICTIONARY[RT])
        st.subheader("Rango de volatilidad recomendado de la cartera para el inversor")
        st.write(f"Rango recomendado de volatilidad anualizada: "
                 f"{sigma_min*100:.1f}% - {sigma_max*100:.1f}%")

        return answers
    return None