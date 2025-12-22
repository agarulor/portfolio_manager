from portfolio_tools.markowitz import  compute_efficient_frontier, get_weights
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional, Literal
import streamlit as st
from portfolio_tools.risk_metrics import calculate_covariance, portfolio_returns, portfolio_volatility
from portfolio_management.investor_portfolios import get_cumulative_returns, get_total_results

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

    st.plotly_chart(fig, width="stretch")


def render_results_table(
        df: pd.DataFrame,
        percent_cols: Optional[list[str]] = None,
        float_cols: Optional[list[str]] = None,
        highlight: bool = True,
        hide_index: bool = False)-> None:

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
    percent_cols = percent_cols or ["Retorno anualizado", "Volatilidad", "Max Drawdown"]
    float_cols = float_cols or ["Ratio de Sharpe"]

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
            "font-size": "16px",
            "color": PRIMARY,
            "line-height": "1.5"
        })
        .set_table_styles([
            {
                "selector": "",
                "props": [("width", "80%"), ("border-collapse", "collapse")]
            },
            {
                "selector": "thead th",
                "props": [
                    ("text-align", "center"),
                    ("font-size", "16px"),
                    ("font-weight", "600"),
                    ("color", "white"),
                    ("background-color", SECONDARY),
                    ("padding", "8px"),
                ],
            },
            {
                "selector": "tbody td",
                "props": [
                    ("padding", "6px 8px"),
                ],
            },
        ])
    )

    if hide_index:
        styler = styler.hide(axis="index")


    if highlight:
        if "Retorno anualizado" in df.columns:
            styler = styler.background_gradient(subset=["Retorno anualizado"], cmap="Greens", low=0.6, high=0.0)
        if "Ratio de Sharpe" in df.columns:
            styler = styler.background_gradient(subset=["Ratio de Sharpe"], cmap="Greens", low=0.6, high=0.0)
        if "Volatilidad" in df.columns:
            styler = styler.background_gradient(subset=["Volatilidad"], cmap="Reds", low=0.6, high=0.0)
        if "Max Drawdown" in df.columns:
            styler = styler.background_gradient(subset=["Max Drawdown"], cmap="Reds", low=0.6, high=0.0)

    html_table = styler.to_html()
    st.markdown(html_table, unsafe_allow_html=True)

def show_markowitz_results(n_returns: int,
                           returns: pd.DataFrame,
                           df_results: pd.DataFrame,
    method: Literal["simple", "log"] = "simple",
    periods_per_year: int = 252):

    ef = compute_efficient_frontier(returns, n_returns, method, periods_per_year)

    fig = px.line(
        ef,
        x="Volatilidad",
        y="Retorno anualizado"
    )
    dfp = df_results.reset_index().rename(columns={"Tipo de portfolio": "Tipo de portfolio"})

    fig.add_trace(
                go.Scatter(
                    x=dfp["Volatilidad"] / 100,
                    y=dfp["Retorno anualizado"] / 100,
                    mode="markers+text",
                    text=dfp["Tipo de portfolio"],
                    textposition="top center",
                    name="Portfolios",
                    marker=dict(
                        color="red",
                        size=14,
                    )
                )
            )
    fig.update_layout(
        xaxis_title="Volatilidad (anualizada)",
        yaxis_title="Retorno anualizado",
        legend_title_text="",
        height=500
    )
    fig.update_xaxes(rangemode="tozero", showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, width="stretch")


def plot_portfolio_value(df_value: pd.DataFrame) -> None:

    # We order by date
    df_value = df_value.reset_index()
    df_value.columns = ["Fecha", "Valor"]

    fig = px.line(df_value,
                  x="Fecha",
                  y="Valor",
                  markers=False)

    fig.update_layout(template="plotly_white",
                      xaxis_title="Fecha",
                      yaxis_title="Valor (€)",
                      hovermode="x unified",
                      height=700,
                      )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, use_container_width=True)


def plot_portfolio_values(results: dict) -> None:

    # We first extract the keys
    names = list(results.keys())

    selected = st.multiselect("Carteras a comparar",
                              options=names,
                              default=names[:1] if names else names)
    if not selected:
        st.info("Por favor, seleccione al menos una cartera")
        return

    fig = go.Figure()

    for i, name in enumerate(selected):
        df = results[name]
        fig.add_trace(
            go.Scatter(
                x = df.index,
                y = df.values,
                mode = "lines",
                name = name,
                line=dict(width=4 if i == 0 else 2)

            )
        )

    fig.update_layout(template="plotly_white",
                      xaxis_title="Fecha",
                      yaxis_title="Valor (€)",
                      hovermode="x unified",
                      height=700)
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, width="stretch")
