import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import seaborn as sns


def show_table(df: pd.DataFrame, caption: str | None = None, use_container_width: bool = True):
    if caption:
        st.subheader(caption)
    st.dataframe(df, use_container_width=use_container_width)
