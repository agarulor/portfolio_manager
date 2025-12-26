import streamlit as st
from types import MappingProxyType
from pathlib import Path
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
BASE_DIR = Path(__file__).resolve().parent
ASSETS_PATH = BASE_DIR / "assets"
RISK_COLOR = MappingProxyType({1: "#2ecc71", 2: "#6bdc8b", 3: "#f1c40f", 4: "#f39c12", 5: "#e67e22", 6: "#e74c3c"})
RISK_PROFILE_DICTIONARY = MappingProxyType({
    1: "Perfil bajo de riesgo",
    2: "Perfil medio-bajo de riesgo",
    3: "Perfil medio de riesgo",
    4: "Perfil medio-alto de riesgo",
    5: "Perfil alto de riesgo",
    6: "Perfil agresivo de riesgo"
})


def apply_global_styles():
    """
    Computes apply global styles.

    Parameters
    ----------


    Returns
    -------
    Any: apply global styles output.
    """
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Roboto', sans-serif !important;
    }

    body {
        background-color: #F8FAFC;
    }

    .main > div {
        padding-top: 1rem;
    }

    .stButton > button {
        background-color: #73EDFF;
        color: #000078;
        border-radius: 8px;
        padding: 0.6rem 1.2rem;
        font-size: 1.1rem;
        font-weight: 700;
        border: 2px solid #000078;
    }

    .stButton > button:hover {
        background-color: #73EDFF;
    }

    .card {
        background-color: white;
        padding: 1.8rem;
        border-radius: 12px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        height: 100%;
    }

    }
    .progress-sticky {
        position: sticky;
        top: 0;
        z-index: 9999;        
        background: #F8FAFC;
        }

        .stApp {
        overflow: visible;
        }

    </style>
    """, unsafe_allow_html=True)


def render_sidebar_header():
    """
    Renders sidebar header.

    Parameters
    ----------


    Returns
    -------
    Any: render sidebar header output.
    """
    with st.sidebar:
        st.image(str(ASSETS_PATH / "202-nova-marca-uoc.jpg"), width="stretch")

        st.markdown("""
        <div style="margin-top:0.6rem; margin-bottom:0.8rem; text-align:center;">
            <div style="color:#000078; font-weight:700; font-size:1.05rem;">
                ROBO-UOC ADVISOR
            </div>
            <div style="color:#000078; opacity:0.85; font-size:0.9rem;">
                TFG - UOC
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")


def render_sidebar_profile_summary():
    """
    Renders sidebar profile summary.

    Parameters
    ----------


    Returns
    -------
    Any: render sidebar profile summary output.
    """
    if "risk_result" not in st.session_state:
        return

    res = st.session_state["risk_result"]
    RT = res["RT"]
    color = RISK_COLOR[RT]

    st.sidebar.markdown("---")

    st.sidebar.markdown(
        f"""
        <div style="text-align: center">
            <div style="font-size: 18px; font-weight: 1200; color: {color}; margin-top: 6px;">
                {RISK_PROFILE_DICTIONARY[RT]}
            </div>
            <div style="font-size: 28px; font-weight: 1200; color: {color}; margin-top: 6px;">
                {RT}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar():
    """
    Renders sidebar.

    Parameters
    ----------


    Returns
    -------
    Any: render sidebar output.
    """
    render_sidebar_header()
    """
    st.sidebar.title("Menú")

    page = st.sidebar.radio(
        "Opciones",
        options=["Perfil de riesgo", "Cartera de inversión"],
        index=0
    )

    return page"""


def render_portfolio():
    """
    Renders portfolio.

    Parameters
    ----------


    Returns
    -------
    Any: render portfolio output.
    """
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
        f"Volatilidad objetivo: **{sigma_min * 100:.1f}% – {sigma_max * 100:.1f}%**"
    )

    # Here you can later add your optimization and display the weights
    st.info("Aquí iría la construcción de la cartera (pesos por activo, gráficos, etc.).")


def header(text: str):
    """
    Computes header.

    Parameters
    ----------
    text : str. text.

    Returns
    -------
    Any: header output.
    """
    st.markdown(f"""
                <div style="display:flex; justify-content:center; margin-bottom:1.5rem;">
                <div style="
                background: linear-gradient(135deg, rgba(115,237,255,0.15), rgba(115,237,255,0.05));
                border: 2px solid #000078;
                border-radius: 12px;
                padding: 0.9rem 1.6rem;
                box-shadow: 0 10px 30px rgba(115,237,255,0.15);
                text-align: center;
                ">
                <span style="
                display:block;
                color:#000078;
                font-size:3.15rem;
                font-weight:800;
                letter-spacing:0.1em;
                line-height:1.1;
                ">
                {text}
                </span>
                </div>
                </div>    
                </div>""", unsafe_allow_html=True)


def subheader(text: str, font_size: str = "1.1rem", margin_bottom: str = "-1.0rem", font_weight: str = "600",
              color: str = "#000078"):
    """
    Computes subheader.

    Parameters
    ----------
    text : str. text.
    font_size : str. font size.
    margin_bottom : str. margin bottom.
    font_weight : str. font weight.
    color : str. color.

    Returns
    -------
    Any: subheader output.
    """
    st.markdown(f"""
            <div style="
            font-size: {font_size};
            font-weight: {font_weight};
            color: {color};
            line-height: 1.6;
            margin-bottom: {margin_bottom};
            text-align: center;
            max-width: 800px;
            margin-left: auto;
            margin-right: auto;
            ">
            {text}
            </div>
          """, unsafe_allow_html=True)