import pandas as pd
import streamlit as st

#=======================================================
# ARCHIVO DE PRUEBA PARA TABLAS EN STREAMLIT
# SE COMPLETARÁ MÁS ADELANTE
#=======================================================
def show_table(df: pd.DataFrame, caption: str | None = None, use_container_width: bool = True):
    if caption:
        st.subheader(caption)
    st.dataframe(df, use_container_width=use_container_width)
