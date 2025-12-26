from portfolio_tools.markowitz import compute_efficient_frontier
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Optional, Literal, Dict
import streamlit as st
import numpy as np


def show_portfolio(
        df_weights: pd.DataFrame,
        title: str = "Composición de la cartera",
        label_name: str = "Activo",
        weight_col: Optional[str] = None,
        weights_in_percent: bool = True,
        colorscale: str = "PuBu"
) -> None:
    """
    Shows portfolio.

    Parameters
    ----------
    df_weights : pd.DataFrame. df weights.
    title : str. title.
    label_name : str. label name.
    weight_col : Optional[str]. weight col.
    weights_in_percent : bool. weights in percent.
    colorscale : str. colorscale.

    Returns
    -------
    None: None.
    """
    if df_weights is None or df_weights.empty:
        st.warning("No hay datos para mostrar.")
        return

    df_plot = df_weights[[weight_col]].copy()

    df_plot[weight_col] = pd.to_numeric(df_plot[weight_col], errors="coerce")
    df_plot = df_plot.dropna(subset=[weight_col])

    if not weights_in_percent:
        df_plot[weight_col] = df_plot[weight_col] * 100

    df_plot = df_plot.reset_index()
    df_plot.columns = [label_name, "Peso"]

    df_plot = df_plot.sort_values("Peso", ascending=True)

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
            color="#000078",
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
        hide_index: bool = False) -> None:
    """
    Renders results table.

    Parameters
    ----------
    df : pd.DataFrame. df.
    percent_cols : Optional[list[str]]. percent cols.
    float_cols : Optional[list[str]]. float cols.
    highlight : bool. highlight.
    hide_index : bool. hide index.

    Returns
    -------
    None: None.
    """
    if df is None or df.empty:
        st.warning("No hay datos para mostrar")
        return

    PRIMARY = "#000078"
    SECONDARY = "#1f3a5f"

    percent_cols = percent_cols or ["Retorno anualizado", "Volatilidad", "Max Drawdown"]
    float_cols = float_cols or ["Ratio de Sharpe"]

    fmt: Dict[str, str] = {}
    for c in percent_cols:
        if c in df.columns:
            fmt[c] = "{:.4f}%"
    for c in float_cols:
        if c in df.columns:
            fmt[c] = "{:.4f}"

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


def show_markowitz_results(
        n_returns: Optional[int] = None,
        returns: Optional[pd.DataFrame] = None,
        df_results: pd.DataFrame = None,
        method: Literal["simple", "log"] = "simple",
        periods_per_year: int = 252,
        no_ef: bool = False):
    """
    Shows markowitz results.

    Parameters
    ----------
    n_returns : Optional[int]. n returns.
    returns : Optional[pd.DataFrame]. Returns of the assets.
    df_results : pd.DataFrame. df results.
    method : Literal["simple", "log"]. method.
    periods_per_year : int. periods per year.
    no_ef : bool. no ef.

    Returns
    -------
    Any: show markowitz results output.
    """
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

    df_investor = dfp[dfp["Tipo de portfolio"].str.lower() == "investor"]
    if not df_investor.empty:
        fig.add_trace(
            go.Scatter(
                x=df_investor["Volatilidad"] / 100,
                y=df_investor["Retorno anualizado"] / 100,
                mode="markers+text",
                text=df_investor["Tipo de portfolio"],
                textposition="top center",
                name="Investor",
                marker=dict(
                    color="darkgreen",
                    size=20,
                    symbol="circle",
                ),
            )
        )

    df_others = dfp[dfp["Tipo de portfolio"].str.lower() != "investor"]
    if not df_others.empty:
        fig.add_trace(
            go.Scatter(
                x=df_others["Volatilidad"] / 100,
                y=df_others["Retorno anualizado"] / 100,
                mode="markers+text",
                text=df_others["Tipo de portfolio"],
                textposition="top center",
                name="Otros portfolios",
                marker=dict(
                    color="red",
                    size=14,
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Volatilidad (anualizada)",
        yaxis_title="Retorno anualizado",
        legend_title_text="",
        height=500,
    )

    fig.update_xaxes(
        rangemode="tozero",
        showgrid=True
    )

    fig.update_yaxes(
        showgrid=True
    )

    st.plotly_chart(fig, width="stretch")


def plot_portfolio_values(
        results: dict | pd.DataFrame,
        key: str,
        portfolio_type: Literal["stock", "global"] = "global") -> None:
    """
    Plots portfolio values.

    Parameters
    ----------
    results : dict | pd.DataFrame. results.
    key : str. key.
    portfolio_type : Literal["stock", "global"]. portfolio type.

    Returns
    -------
    None: None.
    """
    if portfolio_type == "global":
        if not isinstance(results, dict):
            st.error("Para portfolio_type=\"global\", results debe ser un diccionario.")
            return

        names = list(results.keys())

        selected = st.multiselect(
            "Carteras a comparar",
            options=names,
            default=names[:1] if names else names,
            key=key
        )
        if not selected:
            st.info("Por favor, seleccione al menos una cartera")
            return

    elif portfolio_type == "stock":
        if not isinstance(results, pd.DataFrame):
            st.error("Para portfolio_type=\"stock\", results debe ser un DataFrame de Pandas.")
            return

        df_clean = results.copy()
        df_clean = df_clean.drop(columns=df_clean.columns[(df_clean == 0).all()], errors="ignore")
        df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isna().all()], errors="ignore")

        names = list(df_clean.columns)

        selected = st.multiselect(
            "Evolución de activos de la cartera",
            options=names,
            default=names[:1] if names else names,
            key=key
        )
        if not selected:
            st.info("Por favor, seleccione al menos un activo")
            return

        results = df_clean
    else:
        return

    fig = go.Figure()

    for i, name in enumerate(selected):
        series = results[name]
        if isinstance(series, pd.DataFrame):
            y = series.iloc[:, 0].to_numpy()
            x = series.index
        else:
            x = series.index
            y = series.to_numpy()

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                name=name,
                line=dict(width=4 if i == 0 else 2)
            )
        )

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Fecha",
        yaxis_title="Valor",
        hovermode="x unified",
        height=700
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
        height: int = 500) -> None:
    """
    Plots portfolio values select.

    Parameters
    ----------
    results : dict | pd.DataFrame. results.
    key : str. key.
    portfolio_type : Literal["stock", "global"]. portfolio type.
    selected : list[str] | None. selected.
    show_selector : bool. show selector.
    height : int. height.

    Returns
    -------
    None: None.
    """
    if portfolio_type == "global":
        if not isinstance(results, dict):
            st.error("Para portfolio_type=\"global\", results debe ser un diccionario.")
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

        valid = [s for s in selected if s in results]
        invalid = [s for s in selected if s not in results]

    elif portfolio_type == "stock":
        if not isinstance(results, pd.DataFrame):
            st.error("Para portfolio_type=\"stock\", results debe ser un DataFrame de Pandas.")
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

        results = df_clean

    else:
        st.error("Por favor, elige \"stock\" o \"global\".")
        return

    if invalid:
        st.warning(
            "Las siguientes selecciones no tienen datos disponibles y se han omitido:\n"
            + ", ".join(invalid)
        )

    if not valid:
        st.info("No hay series válidas para mostrar con la selección actual.")
        return

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
        yaxis_title="Valor",
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
        height: int = 500):
    """
    Plots daily returns scatter base only.

    Parameters
    ----------
    results : dict | pd.DataFrame. results.
    base : str | None. base.
    data_type : Literal["global", "stock"]. data type.
    y_in_percent : bool. y in percent.
    key : str | None. key.
    height : int. height.

    Returns
    -------
    Any: plot daily returns scatter base only output.
    """
    if not base:
        st.info("Selecciona una serie base en el lateral.")
        return

    if data_type == "global":
        if not isinstance(results, dict):
            st.error("Para data_type='global', results debe ser dict.")
            return
        series = results.get(base)
    else:
        if not isinstance(results, pd.DataFrame):
            st.error("Para data_type='stock', results debe ser DataFrame.")
            return
        series = results[base]

    if series is None:
        st.warning("No hay datos disponibles para la serie seleccionada.")
        return

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
        height: int = 500) -> None:
    """
    Plots daily returns distribution.

    Parameters
    ----------
    results : dict | pd.DataFrame. results.
    base : str | None. base.
    key : str. key.
    data_type : Literal["global", "stock"]. data type.
    y_in_percent : bool. y in percent.
    nbins : int. nbins.
    height : int. height.

    Returns
    -------
    None: None.
    """
    if not base:
        st.info("Selecciona un activo/cartera base en el lateral.")
        return

    if data_type == "global":
        if not isinstance(results, dict):
            st.error("Para data_type='global', results debe ser un diccionario.")
            return
        series = results.get(base)
        if series is None:
            st.warning("No hay datos para la serie seleccionada.")
            return
        s = series.iloc[:, 0] if isinstance(series, pd.DataFrame) else series
    else:
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

    mu = float(np.mean(x))
    sigma = float(np.std(x, ddof=1)) if x.size > 1 else 0.0
    left = mu - sigma
    right = mu + sigma

    fig = go.Figure()

    fig.add_trace(
        go.Histogram(
            x=x,
            nbinsx=nbins,
            name="Distribución",
            opacity=0.75,
        )
    )

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

    fig.update_layout(
        template="plotly_white",
        xaxis_title="Retorno diario (%)" if y_in_percent else "Retorno diario",
        yaxis_title="Frecuencia",
        height=height,
        bargap=0.05,
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    st.plotly_chart(fig, key=key)
