import plotly.express as px
import pandas as pd
from typing import Optional
import streamlit as st
from interface.main_interface import subheader, header

def show_portfolio(
    df_weights: pd.DataFrame,
    title: str = "Composición de la cartera",
    label_name: str = "Activo",
    weight_col: Optional[str] = None,
    weights_in_percent: bool = True,
) -> None:
    """
    Muestra la composición (activos o sectores) usando Plotly Express en Streamlit.

    Acepta:
    - DataFrame con índice = etiqueta (ticker/sector) y 1 columna de pesos, o
    - DataFrame con múltiples columnas si indicas weight_col (o existe una columna típica).
    """

    if df_weights is None or df_weights.empty:
        st.warning("No hay datos para mostrar.")
        return

    # ----------------------------
    # 2) Prepare DF for Plotly
    # ----------------------------
    df_plot = df_weights[[weight_col]].copy()

    # Ensure number
    df_plot[weight_col] = pd.to_numeric(df_plot[weight_col], errors="coerce")
    df_plot = df_plot.dropna(subset=[weight_col])

    # Convert to % si if come from 1 to 0
    if not weights_in_percent:
        df_plot[weight_col] = df_plot[weight_col] * 100

    # reset index
    df_plot = df_plot.reset_index()
    df_plot.columns = [label_name, "Peso"]  # rename to Peso for the chart

    # We order it
    df_plot = df_plot.sort_values("Peso", ascending=True)

    # ----------------------------
    # 3) Plot
    # ----------------------------
    fig = px.bar(
        df_plot,
        x="Peso",
        y=label_name,
        orientation="h",
        text="Peso",
        title=title,
    )
    fig.update_traces(texttemplate="%{text:.2f}%",
                      textposition="outside",
                      marker=dict(
                          color=df_plot["Peso"],
                          colorscale="Cividis"
                      )
                      )
    axis_color = "#000078"
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            xanchor="center",
            font=dict(size=24, color=axis_color),
        ),
        xaxis=dict(
            title=dict(
                text="Peso (%)",
                font=dict(size=16, color=axis_color),
            ),
            tickfont=dict(size=13, color=axis_color),
        ),
        yaxis=dict(
            title=dict(
                text=label_name,
                font=dict(size=16, color=axis_color),
            ),
            tickfont=dict(size=13, color=axis_color),
            categoryorder="total ascending",
        ),
        hovermode="y",
    )

    st.plotly_chart(fig, use_container_width=True)