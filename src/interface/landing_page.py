import sys
import streamlit as st
import base64
from pathlib import Path
from interface.constants import ASSETS_PATH, PROJECT_ROOT

# checking and prepare folders
current_file_path = Path(__file__).resolve()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _img_to_base64(img_path: Path) -> str:
    """
    Computes  img to base64.

    Parameters
    ----------
    img_path : Path. img path.

    Returns
    -------
    str:  img to base64 output.
    """
    # Read the image file as bytes
    data = img_path.read_bytes()
    # Convert it to base64 so we can embed it directly in HTML
    return base64.b64encode(data).decode("utf-8")


def add_separation(margin_top: str = "1rem", margin_bottom: str = "1rem"):
    """
    Adds separation.

    Parameters
    ----------
    margin_top : str. margin top.
    margin_bottom : str. margin bottom.

    Returns
    -------
    Any: add separation output.
    """
    # Just a visual separator to make sections easier to read
    st.markdown(f"""
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
        margin-bottom: {margin_bottom};
        margin-top: {margin_top}
    ">
    """, unsafe_allow_html=True)


def render():
    """
    Renders .

    Parameters
    ----------


    Returns
    -------
    Any: render output.
    """
    # Load the cover image and logo and convert both to base64
    portada_path = ASSETS_PATH / "portada.png"
    portada_b64 = _img_to_base64(portada_path)
    logo_path = ASSETS_PATH / "202-nova-marca-uoc.jpg"
    logo_b64 = _img_to_base64(logo_path)

    # Main header section with background image, title and logo
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

    # Short intro text before starting the questionnaire
    st.markdown("""
    <div style="text-align: center; margin-top: -0.5rem; margin-bottom: -1.5rem";>
        <h3 style="color:#FFFFFF; font-weight:500;">
            Antes de empezar necesitamos hacerte un breve cuestionario
        </h3>
    </div>
    """, unsafe_allow_html=True)

    # Centered button that sends the user to the questionnaire
    with st.container():
        st.markdown(
            """<div style="margin-top:-0.5rem; margin-bottom: -0.5rem"></div>""",
            unsafe_allow_html=True
        )

        c1, c2, c3 = st.columns([2, 1, 2])
        with c2:
            if st.button("Completar cuestionario personal", width="stretch"):
                # Change route and force Streamlit to reload
                st.session_state["route"] = "questionnaire"
                st.rerun()

    add_separation()

    st.markdown("""
    <div style="text-align:center; margin-top: 0.0rem; margin-bottom: 0.0rem;">
      <h2 style="color:#000078; margin-bottom:0.0rem;">¿Qué hace el Robo Advisor?</h2>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        # Diversification card
        st.markdown("""
        <div class="card" style="border-left: 6px solid #73EDFF; text-align:center">
            <h3 style="color:#000078; margin-top:0"> Diversificación</h3>
            <p style="color:#000078; margin-bottom:0">
                Construye una cartera diversificada ajustada al inversor.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        # Risk profile card
        st.markdown("""
        <div class="card" style="border-left: 6px solid #73EDFF; text-align:center">
            <h3 style="color:#000078; margin-top:0;"> Perfil de riesgo</h3>
            <p style="color:#000078; margin-bottom:0;">
                Ajusta la asignación de activos según el perfil de riesgo del inversor.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        # Portfolio tracking card
        st.markdown("""
        <div class="card" style="border-left: 6px solid #73EDFF; text-align:center">
            <h3 style="color:#000078; margin-top:0;"> Seguimiento</h3>
            <p style="color:#000078; margin-bottom:0;">
                Permite hacer seguimiento de la cartera de invesión.
            </p>
        </div>
        """, unsafe_allow_html=True)

    add_separation()

    # Section explaining the flow of the app
    st.markdown("""
       <div style="text-align:center; margin-top: 0rem; margin-bottom: 0rem;">
         <h2 style="color:#000078; margin-bottom:0.25rem;">Cómo funciona</h2>
       </div>
       """, unsafe_allow_html=True)

    s1, s2, s3 = st.columns(3)
    with s1:
        # First step: questionnaire
        st.markdown("""
        <div class="card" style="border-left: 6px solid #73EDFF; text-align:center">
            <h3 style="color:#000078; margin-top:0;">Cuestionario</h3>
            <p style="color:#000078; margin-bottom:0;">
                Preguntas para determinar perfil y restricciones
            </p>
        </div>
        """, unsafe_allow_html=True)

    with s2:
        # Second step: calculate risk profile
        st.markdown("""
        <div class="card" style="border-left: 6px solid #73EDFF; text-align:center;">
            <h3 style="color:#000078; margin-top:0;">Perfil</h3>
            <p style="color:#000078; margin-bottom:0;">
                Determina el perfil de riesgo del inversor a partir del cuestionario
            </p>
        </div>
        """, unsafe_allow_html=True)

    with s3:
        # Final step: generate the portfolio
        st.markdown("""
        <div class="card" style="border-left: 6px solid #73EDFF; text-align:center;">
            <h3 style="color:#000078; margin-top:0;">Cartera</h3>
            <p style="color:#000078; margin-bottom:0;">
                Muestra la cartera recomendada
            </p>
        </div>
        """, unsafe_allow_html=True)

    add_separation()

    # DISCLAIMER
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