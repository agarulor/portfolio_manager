import streamlit as st
from interface.main_interface import header, subheader
import datetime as dt

import plotly.graph_objects as go
FONT_SIZE = "0.9rem"
FONT_WEIGHT = "700"
FONT_COLOR = "#000078"
INITIAL_DATE = dt.date(2020, 10, 1)
END_DATE = dt.date(2025, 9, 30)

def render_slider(text:str,
                  text_slider: str,
                  key: str,
                  min_value: float,
                  max_value: float,
                  font_color: str = FONT_COLOR,
                  font_size: str = FONT_SIZE,
                  font_weight: str = FONT_WEIGHT):
    st.markdown(
        f"""
         <div style="
         font-size: {font_size}; 
         font-weight: {font_weight};
         color: {font_color};
         margin-bottom: 1rem;
         margin-top: 2.0rem;
         text-align:center;
         ">
         {text} 
         </div>
         """, unsafe_allow_html=True)
    max_pct = st.slider(
        text_slider,
        min_value=min_value, max_value=max_value, value=25.0, step=1.0,
        label_visibility="collapsed",
        key= key
    )

    st.markdown(
        f"""
         <div style="
         font-size: 1.2rem;
         font-weight: {font_weight};
         color: {font_color};
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


def render_number_input(text:str,
                        min_value:float,
                        init_value:float,
                        step:float,
                        key:str,
                        font_color: str = FONT_COLOR,
                        font_size: str = FONT_SIZE,
                        font_weight: str = FONT_WEIGHT):
    st.markdown(f"""
    <div style="
        font-size: {font_size};
        font-weight: {font_weight};
        color: {font_color};
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

    formatted_amount = (
        f"{amount:,.2f}"
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    st.markdown(
        f"""
         <div style="
         font-size: 1.2rem;
         font-weight: {font_weight};
         color: {font_color};
         margin-bottom: 1rem;
         margin-top: 0.0rem;
         text-align:center;
         "> 
        {formatted_amount} EUR
         </div>
         """,
        unsafe_allow_html=True
    )
    return amount


def render_date(text:str,
                reference_date: dt.date,
                key:str,
                font_color: str = FONT_COLOR,
                font_size: str = FONT_SIZE,
                font_weight: str = FONT_WEIGHT):
    st.markdown(f"""
    <div style="
        font-size: {font_size};
        font-weight: {font_weight};
        color: {font_color};
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 1.0rem;
    ">
        {text}
    </div>
    """, unsafe_allow_html=True)

    date = st.date_input(
        "Fecha de la aportación",
        value=reference_date,
        label_visibility="collapsed",
        key= key
    )
    st.markdown(
        f"""
         <div style="
         font-size: 1.2rem;
         font-weight: {font_weight};
         color: {font_color};
         margin-bottom: 1rem;
         margin-top: 0.0rem;
         text-align:center;
         "> 
        {date}
         </div>
         """,
        unsafe_allow_html=True
    )
    return date



def render_investor_constraints():

    subheader("Defina el grado de diversificación y el importe inicial para la propuesta de cartera",
              margin_bottom="1.0rem")

    c1, c2, c3 = st.columns(3)

    with c1:
        max_sector_pct = render_slider(text="Selecciona el peso máximo por sector",
                                       text_slider="% máximo asignado a un sector",
                                       key="max_sector_pct",
                                       min_value=0.0,
                                       max_value=100.0)


        with c2:
            max_stock_pct = render_slider(text="Selecciona peso máximo por acción",
                                          text_slider="% máximo asignado a una acción",
                                          key="max_stock_pct",
                                          min_value=0.0,
                                          max_value=100.0,
                                          font_color="#FF0000")

        with c3:
            min_stock_pct = render_slider(text="Selecciona peso mínimo por acción",
                                          text_slider="% mínimo asignado a una acción",
                                          key="min_stock_pct",
                                          min_value = 0.0,
                                          max_value = 100.0)





    s1, s2= st.columns(2)

    with s1:
        st1, st2 = st.columns(2)
        with st2:
            render_date("Fecha de inicio de datos", key="date" , reference_date=INITIAL_DATE)
    with s2:
        stt1, stt2 = st.columns(2)
        with stt1:
            render_date("Fecha de fin de datos", font_color="#FF0000", key="date2", reference_date=END_DATE)


    t1, t2, t3 = st.columns(3)
    with t2:
        amount = render_number_input("Inversión inicial",
                                     min_value=0.0,
                                     init_value=10000.0,
                                     step=100.0,
                                     key="cash_contribution")




    u1, u2, u3 = st.columns(3)
    with u2:
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

def render_initial_portfolio():

    # We first extract the status
    if "risk_result" not in st.session_state:
        st.warning("Primero completa el cuestionario de perfil de riesgo.")
        return

    if "investor_constraints" not in st.session_state:
        st.warning("Asegurate de seleccionar los porcentajes por sector y acciones y la inversión inicial")

    # We extract the risk values
    res = st.session_state["risk_result"]
    sigma_min = res["sigma_min"]
    sigma_max = res["sigma_max"]

    # We get the average
    sigma_investor = (sigma_min + sigma_max) / 2

    # We now extract the values for the portfolio allocations
    constraints = st.session_state["investor_constraints"]

    col1, col2 = st.columns(2)

    with col1:

        amount = constraints["amount"]
        max_sector_pct = constraints["max_sector_pct"] / 100
        max_stock_pct = constraints["max_stock_pct"] / 100
        min_stock_pct = constraints["min_stock_pct"] / 100

        st.markdown(f""" <div> <p> {max_stock_pct} + {max_sector_pct} = {min_stock_pct} + {amount}</p> </div>

""", unsafe_allow_html=True)