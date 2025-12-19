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
        colorscale: str = "PuBu"
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
    fig.update_traces(
        texttemplate="%{text:.2f}%",
        textposition="outside",
        textfont=dict(
            color="#000078",  # mismo azul que los ejes
            size=14
        ),
        marker=dict(
            color=df_plot["Peso"],
            colorscale=colorscale
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


def render_results_table(
        df: pd.DataFrame,
        title: str = "Resultados",
        percent_cols: Optional[list[str]] = None,
        float_cols: Optional[list[str]] = None,
        highlight: bool = True,
        hide_index: bool = True,
        height: int = 320,
        use_container_width: bool = True) -> None:

    """
    It renders a nice table with results (Sharpe ratio, returns, volatility, drawdown)
    :param df:
    :param title:
    :param percent_cols:
    :param float_cols:
    :param highlight:
    :param hide_index:
    :param height:
    :param use_container_width:
    :return:
    """

    if df is None or df.empty:
        st.warning("No hay datos para mostrar")
        return
    PRIMARY = "#000078"
    SECONDARY = "#1f3a5f"
    # We robustly create the columns or use typical columns
    percent_cols = percent_cols or ["Returns", "Volatility", "max_drawdown"]
    float_cols = float_cols or ["Sharpe Ratio"]

    # We create map with formats for columns
    fmt: Dict[str, str] = {}
    for c in percent_cols:
        if c in df.columns:
            fmt[c] = "{:.4f}%"
    for c in float_cols:
        if c in df.columns:
            fmt[c] = "{:.4f}"

    # we now set the styler

    styler = (
        df.style
        .format(fmt, na_rep="—")
        .set_properties(**{
            "text-align": "center",
            "font-size": "50px",
            "color": PRIMARY,
        })
        .set_table_styles([
            {
                "selector": "thead th",
                "props": [
                    ("text-align", "center"),
                    ("font-size", "18px"),
                    ("font-weight", "800"),
                    ("color", "white"),
                    ("background-color", SECONDARY),
                    ("padding", "10px"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("padding", "8px 10px"),
                ],
            },
        ])
    )

    if hide_index:
        styler = styler.hide(axis="index")


    if highlight:
        if "Returns" in df.columns:
            styler = styler.background_gradient(subset=["Returns"], cmap="Greens")
        if "Sharpe Ratio" in df.columns:
            styler = styler.background_gradient(subset=["Sharpe Ratio"], cmap="Greens")
        if "Volatility" in df.columns:
            styler = styler.background_gradient(subset=["Volatility"], cmap="Blues")
        if "max_drawdown" in df.columns:
            # Para drawdown normalmente queremos “menos malo” (más cercano a 0) como mejor
            styler = styler.background_gradient(subset=["max_drawdown"], cmap="Reds")

    if title:
        st.markdown(f"### **{title}**")

    st.table(styler)