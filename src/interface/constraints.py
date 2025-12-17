import streamlit as st
from interface.main_interface import header, subheader

import plotly.graph_objects as go
FONT_SIZE = "0.9rem"
FONT_WEIGHT = "700"
FONT_COLOR = "#000078"

def render_slider(text:str, text_slider: str, key: str, min_value: float, max_value: float):
    st.markdown(
        f"""
         <div style="
         font-size: {FONT_SIZE};
         font-weight: {FONT_WEIGHT};
         color: {FONT_COLOR};
         margin-bottom: 1rem;
         margin-top: 2.0rem;
         text-align:center;
         ">
         {text} 
         </div>
         """,
        unsafe_allow_html=True
    )
    max_pct = st.slider(
        text_slider,
        min_value=10, max_value=100, value=25, step=1,
        label_visibility="collapsed",
        key= key
    )

    st.markdown(
        f"""
         <div style="
         font-size: 1.2rem;
         font-weight: {FONT_WEIGHT};
         color: {FONT_COLOR};
         margin-bottom: -1.0rem;
         margin-top: -1.0rem;
         text-align:center;
         "> 
        {max_pct}%
         </div>
         """,
        unsafe_allow_html=True
    )
    return max_pct


def render_number_input(text:str, min_value:float, init_value:float, step:float, key:str):
    st.markdown(f"""
    <div style="
        font-size: {FONT_SIZE};
        font-weight: {FONT_WEIGHT};
        color: {FONT_COLOR};
        text-align: center;
         margin-bottom: 1rem;
         margin-top: 1.0rem;
    ">
        {text}
    </div>
    """, unsafe_allow_html=True)

    amount = st.number_input(
        "Importe de la aportación (€)",
        min_value=min_value,
        value=init_value,
        step=step,
        format="%.2f",
        label_visibility="collapsed",
        key=key
    )

    st.markdown(
        f"""
         <div style="
         font-size: 1.2rem;
         font-weight: {FONT_WEIGHT};
         color: {FONT_COLOR};
         margin-bottom: 1rem;
         margin-top: 0.0rem;
         text-align:center;
         "> 
        {amount} EUR
         </div>
         """,
        unsafe_allow_html=True
    )
    return amount


def render_investor_constraints():


    subheader("Defina el grado de diversificación y el importe inicial para la propuesta de cartera",
              margin_bottom="1.0rem")

    with st.sidebar:
        max_sector_pct = render_slider(text="Selecciona el peso máximo por sector",
                                       text_slider="% máximo asignado a un sector",
                                       key="max_sector_pct",
                                       min_value=0.1,
                                       max_value=100)


        max_stock_pct = render_slider(text="Selecciona peso máximo por acción",
                                      text_slider="% máximo asignado a una acción",
                                      key="max_stock_pct",
                                      min_value=0.0,
                                      max_value=100)

        min_stock_pct = render_slider(text="Selecciona peso mínimo por acción",
                                      text_slider="% mínimo asignado a una acción",
                                      key="min_stock_pct",
                                      min_value = 0.0,
                                      max_value = 100)


        amount = render_number_input("Inversión inicial",
                                     min_value=0.0,
                                     init_value=10000.0,
                                     step=100.0,
                                     key="cash_contribution")



        generate_cartera = st.button("Generar cartera", use_container_width=True)

        # Save session state
    st.session_state["investor_constraints"] = {
        "max_sector_pct": max_sector_pct,
        "max_stock_pct": max_stock_pct,
        "min_stock_pct": min_stock_pct,
        "amount": amount
    }
    return

    """
    if submitted_cartera:
        # st.session_state["route"] = "portfolio"
        #st.rerun()

    elif not submitted_cartera:
        # Button not yet pushed
        return None
    return None
    """