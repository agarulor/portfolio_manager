import sys
import os
import streamlit as st
import base64
from pathlib import Path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
BASE_DIR = Path(__file__).resolve().parent
ASSETS_PATH = BASE_DIR / "assets"


def _img_to_base64(img_path: Path) -> str:
    data = img_path.read_bytes()
    return base64.b64encode(data).decode("utf-8")

def render():

    portada_path = ASSETS_PATH / "portada.png"
    portada_b64 = _img_to_base64(portada_path)
    logo_path = ASSETS_PATH / "202-nova-marca-uoc.jpg"
    logo_b64 = _img_to_base64(logo_path)

    st.markdown(f"""
    <div style="
    background-image: url('data:image/jpeg;base64,{portada_b64}');
    background-size: cover;
    background-position: center;
    padding: 8rem 1rem;
    border-radius: 18px;
    margin-bottom: -9rem
    ">
    
    <img src="data:image/jpeg;base64,{logo_b64}" style="
    position: absolute;
    top: 1.2rem;
    left: 50%;
    transform: translateX(-50%);
    width: 260px;
    opacity: 0.95;
    "/>
    <div style="text-align: center; margin-bottom: 2rem;">
    <h1 style="color:#000078;">TFG - Ciencia de datos aplicada a la gestión de carteras</h1>
    <h3 style="color:#000078; font-weight:400;">
    Proyecto para la creación de una cartera de inversión diversificada mediante un asesor automatizado
    </h3>
    </div>
    
    <div style="display:flex; justify-content:center; margin-bottom:1.5rem;">
    <div style="
    background: linear-gradient(135deg, rgba(115,237,255,0.15), rgba(115,237,255,0.05));
    border: 2px solid #FFFFFF;
    border-radius: 14px;
    padding: 0.9rem 1.6rem;
    box-shadow: 0 10px 30px rgba(115,237,255,0.15);
    text-align: center;
    ">
    <span style="
    display:block;
    color:#FFFFFF;
    font-size:3.15rem;
    font-weight:800;
    letter-spacing:0.1em;
    line-height:1.1;
    ">
    ROBO-UOC ADVISOR
    </span>
    </div>
    </div>    
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align: center; margin-top: -0.5rem; margin-bottom: -1.5rem";>
        <h3 style="color:#FFFFFF; font-weight:500;">
            Antes de empezar necesitamos hacerte un breve cuestionario
        </h3>
    </div>
    """, unsafe_allow_html=True)

    with st.container():
        st.markdown(
            """<div style="margin-top:-0.5rem; margin-bottom: -0.5rem"></div>""",
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns([2, 1, 2])
        with c2:
            if st.button("Completar cuestionario personal"):
                st.session_state["route"] = "questionnaire"
                st.rerun()


    st.markdown("""
    <hr style="
        border: none;
        height: 2px;
        background: linear-gradient(
            to right,
            transparent,
            #73EDFF,
            transparent
        );
        margin: 3rem 0 2rem 0;
    ">
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center; margin-top: 0.0rem; margin-bottom: 0.0rem;">
      <h2 style="color:#000078; margin-bottom:0.0rem;">¿Qué hace el Robo Advisor?</h2>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("""
        <div class="card" style="border-left: 6px solid #73EDFF; text-align:center">
            <h3 style="color:#000078; margin-top:0"> Diversificación</h3>
            <p style="color:#000078; margin-bottom:0">
                Construye una cartera diversificada ajustada al inversor.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="card" style="border-left: 6px solid #73EDFF; text-align:center">
            <h3 style="color:#000078; margin-top:0;"> Perfil de riesgo</h3>
            <p style="color:#000078; margin-bottom:0;">
                Ajusta la asignación de activos según el perfil de riesgo del inversor.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="card" style="border-left: 6px solid #73EDFF; text-align:center">
            <h3 style="color:#000078; margin-top:0;"> Seguimiento</h3>
            <p style="color:#000078; margin-bottom:0;">
                Permite hacer seguimiento de la cartera de invesión.
            </p>
        </div>
        """, unsafe_allow_html=True)


    st.markdown("""
    <hr style="
        border: none;
        height: 2px;
        background: linear-gradient(
            to right,
            transparent,
            #73EDFF,
            transparent
        );
        margin: 3rem 0 2rem 0;
    ">
    """, unsafe_allow_html=True)

    st.markdown("""
       <div style="text-align:center; margin-top: 0rem; margin-bottom: 0rem;">
         <h2 style="color:#000078; margin-bottom:0.25rem;">Cómo funciona</h2>
       </div>
       """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""
        <div class="card" style="border-left: 6px solid #73EDFF; text-align:center">
            <h3 style="color:#000078; margin-top:0;">Cuestionario</h3>
            <p style="color:#000078; margin-bottom:0;">
                Preguntas para determinar perfil y restricciones
            </p>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        st.markdown("""
        <div class="card" style="border-left: 6px solid #73EDFF; text-align:center;">
            <h3 style="color:#000078; margin-top:0;">Perfil</h3>
            <p style="color:#000078; margin-bottom:0;">
                Determina el perfil de riesgo del inversor a partir del cuestionario
            </p>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        st.markdown("""
        <div class="card" style="border-left: 6px solid #73EDFF; text-align:center;">
            <h3 style="color:#000078; margin-top:0;">Cartera</h3>
            <p style="color:#000078; margin-bottom:0;">
                Muestra la cartera recomendada
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <hr style="
        border: none;
        height: 2px;
        background: linear-gradient(
            to right,
            transparent,
            #73EDFF,
            transparent
        );
        margin: 3rem 0 2rem 0;
    ">
    """, unsafe_allow_html=True)

    #--------------- DISCLAIMER ---------------
    st.markdown("""
    <div style="
        display:flex;
        justify-content:center;
        margin-top:0.5rem;
    ">
        <div style="
            max-width:900px;
            background-color:rgba(115,237,255,0.05);
            border:1px solid rgba(115,237,255,0.35);
            border-radius:12px;
            padding:1.2rem 1.5rem;
            text-align:center;
        ">
            <p style="
                color:#000078;
                font-size:1rem;
                line-height:1.5;
                margin:0;
            ">
                <span style="display:block; font-weight:700; margin-bottom:0.3rem;">
                Disclaimer:
                </span>
                Esta aplicación es una herramienta desarrollada exclusivamente con fines académicos
                en el marco de un Trabajo de Fin de Grado (TFG).
                No constituye asesoramiento financiero, recomendación de inversión ni oferta de
                productos financieros.
                Cualquier decisión tomada a partir de esta herramienta es responsabilidad exclusiva del usuario.
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)