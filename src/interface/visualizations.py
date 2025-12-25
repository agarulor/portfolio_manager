from portfolio_tools.markowitz import  compute_efficient_frontier, get_weights
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional, Literal
import streamlit as st
import numpy as np
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

def show_markowitz_results(n_returns: Optional[int] = None,
                           returns: Optional[pd.DataFrame] = None,
                           df_results: pd.DataFrame = None,
    method: Literal["simple", "log"] = "simple",
    periods_per_year: int = 252,
                           no_ef: bool = False):

    if not no_ef:
        ef = compute_efficient_frontier(returns, n_returns, method, periods_per_year)

        fig = px.line(
        ef,
        x="Volatilidad",
        y="Retorno anualizado"
        )
    else:
        fig = px.scatter()
        df_results.index.name = "Tipo de portfolio"

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

    st.plotly_chart(fig, width="stretch")



def plot_portfolio_values(results: dict | pd.DataFrame, key: str, portfolio_type: Literal["stock", "global"] = "global") -> None:
    if portfolio_type == "global":
        if not isinstance(results, dict):
            st.error("For portfolio_type = global, results must be a dict")
            return
        # We first extract the keys
        names = list(results.keys())

        selected = st.multiselect("Carteras a comparar",
                                  options=names,
                                  default=names[:1] if names else names,
                                  key=key)
        if not selected:
            st.info("Por favor, seleccione al menos una cartera")
            return


    elif portfolio_type == "stock":

        if not isinstance(results, pd.DataFrame):
            st.error("For portfolio_type = stock results must be a Pandas DataFrame.")
            return

        df_clean = results.copy()
        df_clean = df_clean.drop(columns=df_clean.columns[(df_clean == 0).all()], errors="ignore")
        df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isna().all()], errors="ignore")

        names = list(df_clean.columns)

        selected = st.multiselect("Evolución de activos de la cartera",
                                  options=names,
                                  default=names[:1] if names else names,
                                  key=key)
        if not selected:
            st.info("Por favor, seleccione al menos un activo")
            return
    else:
        print("Please, choose either stock or global")
        return

    fig = go.Figure()

    for i, name in enumerate(selected):
        series = results[name]
        # series could be Series or DataFrame
        if isinstance(series, pd.DataFrame):
            y = series.iloc[:, 0].to_numpy()
            x = series.index
        else:
            # Case of series
            x = series.index
            y = series.to_numpy()

        fig.add_trace(
            go.Scatter(
                x = x,
                y = y,
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


def plot_portfolio_values(results: dict | pd.DataFrame, key: str, portfolio_type: Literal["stock", "global"] = "global") -> None:
    if portfolio_type == "global":
        if not isinstance(results, dict):
            st.error("For portfolio_type = global, results must be a dict")
            return
        # We first extract the keys
        names = list(results.keys())

        selected = st.multiselect("Carteras a comparar",
                                  options=names,
                                  default=names[:1] if names else names,
                                  key=key)
        if not selected:
            st.info("Por favor, seleccione al menos una cartera")
            return


    elif portfolio_type == "stock":

        if not isinstance(results, pd.DataFrame):
            st.error("For portfolio_type = stock results must be a Pandas DataFrame.")
            return

        df_clean = results.copy()
        df_clean = df_clean.drop(columns=df_clean.columns[(df_clean == 0).all()], errors="ignore")
        df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isna().all()], errors="ignore")

        names = list(df_clean.columns)

        selected = st.multiselect("Evolución de activos de la cartera",
                                  options=names,
                                  default=names[:1] if names else names,
                                  key=key)
        if not selected:
            st.info("Por favor, seleccione al menos un activo")
            return
    else:
        print("Please, choose either stock or global")
        return

    fig = go.Figure()

    for i, name in enumerate(selected):
        series = results[name]
        # series could be Series or DataFrame
        if isinstance(series, pd.DataFrame):
            y = series.iloc[:, 0].to_numpy()
            x = series.index
        else:
            # Case of series
            x = series.index
            y = series.to_numpy()

        fig.add_trace(
            go.Scatter(
                x = x,
                y = y,
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


def plot_daily_returns_scatter(
    results: dict | pd.DataFrame,
    key: str,
    data_type: Literal["global", "stock"] = "global",
    y_in_percent: bool = True,
) -> None:
    """
    - global: results es dict[name -> pd.Series/pd.DataFrame] (cada uno es una serie de retornos diarios)
    - stock:  results es pd.DataFrame con columnas=activos, index=fechas (valores=retornos diarios)
    """

    if data_type == "global":
        if not isinstance(results, dict):
            st.error("Para data_type='global', results debe ser un dict.")
            return

        names = list(results.keys())
        selected = st.multiselect(
            "Series a comparar",
            options=names,
            default=names[:1] if names else [],
            key=f"{key}_sel",
        )
        if not selected:
            st.info("Selecciona al menos una serie.")
            return

    elif data_type == "stock":
        if not isinstance(results, pd.DataFrame):
            st.error("For data_type='stock', results must be a DataFrame.")
            return

        df_clean = results.copy()
        df_clean = df_clean.drop(columns=df_clean.columns[(df_clean == 0).all()], errors="ignore")
        df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isna().all()], errors="ignore")

        names = list(df_clean.columns)
        selected = st.multiselect(
            "Activos a comparar",
            options=names,
            default=names[:1] if names else [],
            key=f"{key}_sel",
        )
        if not selected:
            st.info("Selecciona al menos un activo.")
            return

        results = df_clean

    else:
        st.error("data_type must be 'global' or 'stock'.")
        return

    # --- Figura ---
    fig = go.Figure()

    for name in selected:
        series = results[name]

        if isinstance(series, pd.DataFrame):
            s = series.iloc[:, 0]
        else:
            s = series

        s = s.dropna()

        # si vienen en tanto por uno, opcionalmente convierte a %
        y = (s.to_numpy() * 100.0) if y_in_percent else s.to_numpy()
        x = s.index


        colors = np.where(y >= 0, "blue", "red")

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers",
                name=str(name),
                marker=dict(
                    color=colors,
                    size=6,
                    opacity=0.75,
                ),
            )
        )

    fig.add_hline(y=0, line_width=2, line_color="black")

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Fecha",
        yaxis_title="Retorno diario (%)" if y_in_percent else "Retorno diario",
        hovermode="x unified",
        height=600,
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, width="stretch")



def plot_portfolio_values_select(
    results: dict | pd.DataFrame,
    key: str,
    portfolio_type: Literal["stock", "global"] = "global",
    selected: list[str] | None = None,
    show_selector: bool = True,
        height: int = 500,
) -> None:

    # -----------------------
    # GLOBAL (dict)
    # -----------------------
    if portfolio_type == "global":
        if not isinstance(results, dict):
            st.error("For portfolio_type = global, results must be a dict")
            return

        names = list(results.keys())

        if selected is None and show_selector:
            selected = st.multiselect(
                "Carteras a comparar",
                options=names,
                default=names[:1] if names else [],
                key=key,
            )
        elif selected is None:
            selected = names[:1] if names else []

        # Validate existence
        valid = [s for s in selected if s in results]
        invalid = [s for s in selected if s not in results]

    # -----------------------
    # STOCK (DataFrame)
    # -----------------------
    elif portfolio_type == "stock":
        if not isinstance(results, pd.DataFrame):
            st.error("For portfolio_type = stock results must be a Pandas DataFrame.")
            return

        df_clean = results.copy()
        df_clean = df_clean.drop(columns=df_clean.columns[(df_clean == 0).all()], errors="ignore")
        df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isna().all()], errors="ignore")

        names = list(df_clean.columns)

        if selected is None and show_selector:
            selected = st.multiselect(
                "Evolución de activos de la cartera",
                options=names,
                default=names[:1] if names else [],
                key=key,
            )
        elif selected is None:
            selected = names[:1] if names else []

        valid = [s for s in selected if s in df_clean.columns]
        invalid = [s for s in selected if s not in df_clean.columns]

        results = df_clean  # usar DF limpio

    else:
        st.error("Please, choose either stock or global")
        return

    # -----------------------
    # VALIDATION & WARNINGS
    # -----------------------
    if invalid:
        st.warning(
            "Las siguientes selecciones no tienen datos disponibles y se han omitido:\n"
            + ", ".join(invalid)
        )

    if not valid:
        st.info("No hay series válidas para mostrar con la selección actual.")
        return

    # -----------------------
    # PLOT
    # -----------------------
    fig = go.Figure()

    for i, name in enumerate(valid):
        series = results[name]

        if isinstance(series, pd.DataFrame):
            x = series.index
            y = series.iloc[:, 0].to_numpy()
        else:
            x = series.index
            y = series.to_numpy()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line=dict(width=4 if i == 0 else 2),
            )
        )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Fecha",
        yaxis_title="Valor (€)",
        hovermode="x unified",
        height=height,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig)



def plot_daily_returns_scatter_base_only(
    results: dict | pd.DataFrame,
    base: str | None,
    data_type: Literal["global", "stock"] = "global",
    y_in_percent: bool = True,
    key: str | None = None,
        height: int = 500
):
    """
    Plots daily returns as a scatter chart for a single base series.

    Parameters
    ----------
    results : dict | pd.DataFrame
        Dictionary of return series (global) or DataFrame with assets as columns (stock).
    base : str | None
        Base portfolio or asset selected in the sidebar.
    data_type : Literal["global", "stock"]
        Type of data provided.
    y_in_percent : bool
        Whether to display returns in percentage.
    key : str | None
        Optional Streamlit key for the chart.
    """

    if not base:
        st.info("Selecciona una serie base en el lateral.")
        return

    # Extract base series
    if data_type == "global":
        if not isinstance(results, dict):
            st.error("Para data_type='global', results debe ser dict.")
            return
        series = results.get(base)

    else:  # stock
        if not isinstance(results, pd.DataFrame):
            st.error("Para data_type='stock', results debe ser DataFrame.")
            return
        series = results[base]

    if series is None:
        st.warning("No hay datos disponibles para la serie seleccionada.")
        return

    # Normalize to Series
    if isinstance(series, pd.DataFrame):
        s = series.iloc[:, 0]
    else:
        s = series

    s = s.dropna()
    y = (s.to_numpy() * 100.0) if y_in_percent else s.to_numpy()
    x = s.index
    colors = np.where(y >= 0, "blue", "red")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="markers",
            name=str(base),
            marker=dict(color=colors, size=6, opacity=0.75),
        )
    )

    fig.add_hline(y=0, line_width=2, line_color="black")

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Fecha",
        yaxis_title="Retorno diario (%)" if y_in_percent else "Retorno diario",
        hovermode="x unified",
        height=height,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, key=key)


def plot_daily_returns_distribution(
    results: dict | pd.DataFrame,
    base: str | None,
    key: str,
    data_type: Literal["global", "stock"] = "stock",
    y_in_percent: bool = True,
    nbins: int = 60,
        height: int = 500,
) -> None:
    """
    Plots the distribution of daily returns (histogram) and overlays:
    - Vertical line at 0
    - Vertical line at mean
    - Vertical lines at mean +/- 1 std

    Parameters
    ----------
    results : dict | pd.DataFrame
        Daily returns data. For "stock", columns are assets. For "global", dict values are series.
    base : str | None
        Base asset/portfolio selected in the sidebar.
    key : str
        Streamlit key for the chart.
    data_type : Literal["global", "stock"]
        Data type selector.
    y_in_percent : bool
        If True, converts returns to % (x100).
    nbins : int
        Number of histogram bins.

    Returns
    -------
    None
    """

    if not base:
        st.info("Selecciona un activo/cartera base en el lateral.")
        return

    # Extract series
    if data_type == "global":
        if not isinstance(results, dict):
            st.error("Para data_type='global', results debe ser un diccionario.")
            return
        series = results.get(base)
        if series is None:
            st.warning("No hay datos para la serie seleccionada.")
            return
        s = series.iloc[:, 0] if isinstance(series, pd.DataFrame) else series

    else:  # stock
        if not isinstance(results, pd.DataFrame):
            st.error("Para data_type='stock', results debe ser un DataFrame.")
            return
        if base not in results.columns:
            st.warning("El activo seleccionado no está disponible en los datos.")
            return
        s = results[base]

    s = s.dropna()
    x = (s.to_numpy() * 100.0) if y_in_percent else s.to_numpy()

    if x.size == 0:
        st.warning("No hay retornos suficientes para mostrar la distribución.")
        return

    # Stats
    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    left = mu - sigma
    right = mu + sigma

    # Figure
    fig = go.Figure()

    # Histogram
    fig.add_trace(
        go.Histogram(
            x=x,
            nbinsx=nbins,
            name="Distribución",
            opacity=0.75,
        )
    )

    # Vertical reference lines
    fig.add_vline(
        x=0,
        line_width=2,
        line_dash="dot",
        line_color="gray",
        annotation_text="0",
        annotation_position="top left",
    )

    mean_color = "orange"
    fig.add_vline(
        x=mu,
        line_width=3,
        line_dash="solid",
        line_color=mean_color,
        annotation_text="media",
        annotation_position="top right",
    )

    std_color = "purple"
    fig.add_vline(
        x=left,
        line_width=2,
        line_dash="dash",
        line_color=std_color,
        annotation_text="-1 std",
        annotation_position="top left",
    )
    fig.add_vline(
        x=right,
        line_width=2,
        line_dash="dash",
        line_color=std_color,
        annotation_text="+1 std",
        annotation_position="top left",
    )

    # Layout
    fig.update_layout(
        template="plotly_white",
        title="Distribución de rendimiento diario (%)" if y_in_percent else "Distribución de rendimiento diario",
        xaxis_title="Retorno diario (%)" if y_in_percent else "Retorno diario",
        yaxis_title="Frecuencia",
        height=height,
        bargap=0.05,
        showlegend=False,  # keep it clean; annotations already label the lines
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, key=key)