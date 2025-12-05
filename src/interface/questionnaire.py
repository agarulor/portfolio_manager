import streamlit as st

def investor_questionnaire():
    st.set_page_config(page_title="Cuestionario sobre la tolerancia de riesgo del invesor", layout="centered")

    st.title("Cuestionario sobre la tolerancia al riesgo del inversor para su clasificación")

    st.write("""Por favor, repsonda a las siguientes preguntas para poder determinar su perfil de riesgo. """
             """De esta forma podremos recomendarle una cartera de inversión acorde a sus necesidades""")

    st.header("Información referente sobre su apetito de riesgo")

    knowledge_options = {
        1: "1 - Muy poco conocimiento sobre productos financieros",
        2: "2 - Poco conocimiento sobre productos financieros",
        3: "3 - Conocimiento medio sobre productos financieros",
        4: "4 - Buen conocimiento sobre productos financieros",
        5: "5 - Alto conocimiento sobre productos financieros",
        6: "6 - Conocimiento experto sobre productos financieros",
    }

    knowledge = st.selectbox("Conocimiento financiero y experiencia",
                             options=list(knowledge_options.keys()),
                             format_func= lambda x: knowledge_options[x],
                             index= 2
                             )

    st.write("Tu conocimiento seleccionado es: ", knowledge)
