import sys
import os
import streamlit as st
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
ASSETS_PATH = "assets/"

def render():
    st.set_page_config(
        page_title="UOC - Robo Advisor",
        page_icon="📈",
        layout="wide"
    )

    logo_col_left, logo_col_center, logo_col_right = st.columns([2, 1, 2])

    with logo_col_center:
        st.image(
            str(ASSETS_PATH + "202-nova-marca-uoc.jpg"),
            width=280
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Roboto', sans-serif;
    }

    body { background-color: #F8FAFC; }

    .stButton > button {
        background-color: #73EDFF;
        color: #000078;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-size: 1.1rem;
        font-weight: 700;
        border: 2px solid #000078;
    }
    .stButton > button:hover { background-color: #73EDFF; }

    .card {
        background-color: white;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        height: 100%;
    }
    </style>
    """, unsafe_allow_html=True)

    # ---------- HERO ----------
    st.markdown("""
    <div style="text-align: center; margin-bottom: 2rem;">
        <h1 style="color:#000078;">TFG - Ciencia de datos aplicada a la gestión de carteras</h1>
        <h3 style="color:#000078; font-weight:400;">
            Proyecto para la creación de una cartera de inversión diversificada mediante un asesor automatizado
        </h3>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:flex; justify-content:center; margin-bottom:2.5rem;">
        <div style="
            background: linear-gradient(135deg, rgba(115,237,255,0.15), rgba(115,237,255,0.05));
            border: 2px solid #73EDFF;
            border-radius: 14px;
            padding: 0.9rem 1.6rem;
            box-shadow: 0 10px 30px rgba(115,237,255,0.15);
            text-align: center;
        ">
            <span style="
                color:#73EDFF;
                font-size:3.15rem;
                font-weight:800;
                letter-spacing:0.1em;
            ">
                ROBO-UOC ADVISOR
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style="text-align:center; color:#000078; font-size:1rem;">
        Aplicación para creación de carteras de inversión para inversores en función de su apetito y capacidad de
        asumir riesgo.
    </p>
    """, unsafe_allow_html=True)


    # ---------- CTA FINAL ----------
    st.markdown("## Antes de empezar, es necesario llevar a cabo un cuestionario personal")
    st.write("Responde unas preguntas, define tu perfil de riesgo y deja que la aplicación te ayude.")

    if st.button("Completar cuestionario personal", use_container_width=True):
        st.session_state["route"] = "onboarding"