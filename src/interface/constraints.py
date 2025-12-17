import streamlit as st
from interface.main_interface import header, subheader
import plotly.graph_objects as go

def render_investor_constraints():
    header("PARÁMETROS DE LA CARTERA")

    subheader("Defina el grado de diversificación y el importe inicial para la propuesta de cartera",
              margin_bottom="1.0rem")

    col1, col2, col3, col4, col5 = st.columns(5)

    with col2:
        st.markdown("""
        <div style="
            width:  140%;
            margin-left: -6rem;
            border-right: 2px solid #73EDFF;
            background-color: #F5F7FA;
            padding: 6.5rem;
            border-radius: 12px;
            margin-bottom: -17rem;
        ">
        """, unsafe_allow_html=True)

        st.markdown(
            f"""
             <div style="
             font-size: 1.2rem;
             font-weight: 700;
             color: #000078;
             margin-bottom: -1.0rem;
             margin-top: 2.0rem;
             text-align:center;
             ">
             Selecciona el peso máximo que puede tener un sector 
             </div>
             """,
            unsafe_allow_html=True
        )
        max_sector_pct = st.slider(
            "% máximo asignado a un sector",
            min_value=10, max_value=100, value=25, step=1,
            label_visibility="collapsed",
            key="max_sector_pct",
        )

        st.markdown(
            f"""
             <div style="
             font-size: 2rem;
             font-weight: 700;
             color: #000078;
             margin-bottom: -1.0rem;
             margin-top: -1.0rem;
             text-align:center;
             "> 
            {max_sector_pct}%
             </div>
             """,
            unsafe_allow_html=True
        )


    with col4:
        with st.container(border=True):
            st.markdown("### 2) Acción")
            st.caption("Límite máximo del peso en una sola acción.")
            max_stock_pct = st.slider(
                "% máximo asignado a una acción",
                min_value=0, max_value=100, value=10, step=1,
                label_visibility="collapsed"
            )
            st.metric("Valor seleccionado", f"{max_stock_pct}%")
