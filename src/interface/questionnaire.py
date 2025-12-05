import streamlit as st

def investor_questionnaire():
    st.set_page_config(page_title="Cuestionario sobre la tolerancia de riesgo del invesor", layout="centered")

    st.title("Cuestionario sobre la tolerancia al riesgo del inversor para su clasificación")

    st.write("""Por favor, repsonda a las siguientes preguntas para poder determinar su perfil de riesgo. """
             """De esta forma podremos recomendarle una cartera de inversión acorde a sus necesidades""")

    st.header("Información referente sobre su apetito de riesgo")

    # We first create the dictionaries with the answers for each dropdown menu
    knowledge_options = {
        1: "1 - Muy poco conocimiento sobre productos financieros",
        2: "2 - Poco conocimiento sobre productos financieros",
        3: "3 - Conocimiento medio sobre productos financieros",
        4: "4 - Buen conocimiento sobre productos financieros",
        5: "5 - Alto conocimiento sobre productos financieros",
        6: "6 - Conocimiento experto sobre productos financieros",
    }

    risk_level_options = {
        1: "1 - Nivel de riesgo dispuesto a asumir: bajo",
        2: "2 - Nivel de riesgo dispuesto a asumir: medio-bajo",
        3: "3 - Nivel de riesgo dispuesto a asumir: medio",
        4: "4 - Nivel de riesgo dispuesto a asumir: medio-alto",
        5: "5 - Nivel de riesgo dispuesto a asumir: alto",
        6: "6 - Nivel de riesgo dispuesto a asumir: muy alto",
    }

    downside_reaction_options = {
        1: "1 - Ante una caída fuerte vendería toda la inversión",
        2: "2 - Ante una caída fuerte vendería una parte de la inversión",
        3: "3 - Ante una caída fuerte mantendría la inversión",
        4: "4 - Ante una caída fuerte compraría más para aprovechar la caída",
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
    st.write("Tu nivel de riesgo dispues a asumir es: ", risk_level_options[risk_level],)

    downside_reaction = st.selectbox("3) Reacción ante caídas fuertes del precio de los activos",
                             options=list(downside_reaction_options.keys()),
                             format_func= lambda x: downside_reaction_options[x],
                             index= 2
                             )

    st.write("Tu nivel de reacción ante caídas fuertes del precio de los activos es: ", downside_reaction_options[downside_reaction],)