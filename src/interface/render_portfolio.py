import sys
import os

import pandas as pd
from interface.constraints import render_investor_constraints
from portfolio_tools.risk_metrics import calculate_covariance  # si quieres usar covmat
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import streamlit as st
from typing import Literal, Optional
import plotly.express as px
from data_management.get_data import read_price_file, get_stock_prices

from data_management.save_data import save_preprocessed_data
from data_management.clean_data import clean_and_align_data
from portfolio_tools.return_metrics import calculate_daily_returns
from portfolio_tools.risk_metrics import calculate_covariance

from portfolio_tools.markowitz import plot_frontier
from portfolio_management.investor_portfolios import  get_investor_initial_portfolio, get_updated_results, get_cumulative_returns, get_sector_exposure_table
from data_management.dataset_preparation import split_data_markowtiz
from portfolio_management.portfolio_management import check_portfolio_weights, get_sector_weights_at_date, simulate_rebalance_6m_from_returns
from investor_information.investor_profile import investor_target_volatility
from types import MappingProxyType
from interface.tables import show_table

import os
import random
import numpy as np
RISK_PROFILE_DICTIONARY = MappingProxyType({
    1: "Perfil bajo de riesgo",
    2: "Perfil medio-bajo de riesgo",
    3: "Perfil medio de riesgo",
    4: "Perfil medio-alto de riesgo",
    5: "Perfil alto de riesgo",
    6: "Perfil agresivo de riesgo"
})


def show_portfolio(
    df_weights: pd.DataFrame,
    chart_type: Literal["pie", "bar"] = "bar",
    title: str = "Composición de la cartera",
    label_name: str = "Activo",
    weight_col: Optional[str] = None,          # <- NUEVO: columna de pesos a usar
    weights_in_percent: bool = True,           # <- si tus pesos vienen en 0-1 pon False
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

    df = df_weights.copy()

    # ----------------------------
    # 1) Elegir columna de pesos
    # ----------------------------
    if weight_col is None:
        # si tiene 1 columna, usamos esa
        if df.shape[1] == 1:
            weight_col = df.columns[0]
        else:
            # intentamos encontrar una columna típica
            candidates = ["Pesos", "Peso", "weight", "weights", "Weight", "Weights"]
            found = [c for c in candidates if c in df.columns]
            if not found:
                raise ValueError(
                    f"No se pudo inferir la columna de pesos. "
                    f"Columnas disponibles: {list(df.columns)}. "
                    f"Pasa el parámetro weight_col='...'."
                )
            weight_col = found[0]

    if weight_col not in df.columns:
        raise ValueError(f"La columna de pesos '{weight_col}' no existe en el DataFrame.")

    # ----------------------------
    # 2) Preparar DF para plotly
    # ----------------------------
    df_plot = df[[weight_col]].copy()

    # Asegurar numérico
    df_plot[weight_col] = pd.to_numeric(df_plot[weight_col], errors="coerce")
    df_plot = df_plot.dropna(subset=[weight_col])

    # Convertir a % si vienen en 0-1
    if not weights_in_percent:
        df_plot[weight_col] = df_plot[weight_col] * 100

    # reset index para tener la etiqueta como columna
    df_plot = df_plot.reset_index()
    df_plot.columns = [label_name, "Peso"]  # renombramos a "Peso" para el gráfico

    # Orden bonito
    df_plot = df_plot.sort_values("Peso", ascending=True)

    st.subheader(title)

    # ----------------------------
    # 3) Plot
    # ----------------------------
    if chart_type == "bar":
        fig = px.bar(
            df_plot,
            x="Peso",
            y=label_name,
            orientation="h",
            text="Peso",
            title=title,
        )
        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(
            xaxis_title="Peso (%)",
            yaxis_title=label_name,
            yaxis=dict(categoryorder="total ascending"),
            hovermode="y",
        )

    elif chart_type == "pie":
        fig = px.pie(
            df_plot,
            names=label_name,
            values="Peso",
            title=title,
        )
    else:
        raise ValueError("chart_type must be 'bar' or 'pie'")

    st.plotly_chart(fig, use_container_width=True)

def plot_portfolio_value(df_value: pd.DataFrame,
                         title: str = "Evolución de la cartera") -> None:

    # We order by date
    df_value = df_value.reset_index()
    df_value.columns = ["Fecha", "Valor"]

    fig = px.line(df_value,
                  x="Fecha",
                  y="Valor",
                  title = title,
                  markers=False)

    fig.update_layout(template="plotly_white",
                      xaxis_title="Fecha",
                      yaxis_title="Valor (€)",
                      hovermode= "x unified",
                      height=450,
    )
    st.plotly_chart(fig, use_container_width=True)


def render_portfolio():
    render_investor_constraints()
    dinero_invertido = 100
    if "risk_result" not in st.session_state:
        st.warning("Primero completa el cuestionario de perfil de riesgo.")
        return

    res = st.session_state["risk_result"]
    RT = res["RT"]
    sigma_min = res["sigma_min"]
    sigma_max = res["sigma_max"]

    st.header("Cartera de inversión recomendada")

    st.write(
        f"Perfil de riesgo final: **{RT} – {RISK_PROFILE_DICTIONARY[RT]}**"
    )
    st.write(
        f"Volatilidad objetivo: **{sigma_min*100:.1f}% – {sigma_max*100:.1f}%**"
    )

    sigma_recomended = (sigma_max + sigma_min) / 2
    # -----------------------------
    # AQUÍ IMPRIMES LA TABLA
    # -----------------------------
    st.subheader(f"Dinero Invertido **{dinero_invertido}**")
    st.subheader("Cartera de inversión recomendada")

    # covmat no lo usas realmente dentro de create_markowitz_table,
    # pero si quieres ser consistente puedes calcularlo:
    """
    price_data, sectors = get_stock_prices("data/input/ibex_eurostoxx.csv",
                                           "ticker_yahoo",
                                           "name",
                                           start_date="2020-10-01",
                                           )
    prices, report, summary = clean_and_align_data(price_data, beginning_data=True)
    print(sectors)
    f = calculate_daily_returns(prices, method="simple")

    train_set, test_set = split_data_markowtiz(returns=f, test_date_start="2024-10-01", test_date_end="2025-9-30")

    df_resultados, df_weights, weights = get_investor_initial_portfolio(train_set,
                                           min_w=0.025,
                                           max_w=0.15,
                                           rf_annual = 0.035,
                                            periods_per_year=256,
                                           custom_target_volatility=sigma_recomended,
                                                                        sectors_df=sectors,
                                                                        sector_max_weight=0.25,
                                                                        risk_free_ticker="RISK_FREE")

    sectores = get_sector_exposure_table(df_weights, sectors)

    #print(check_portfolio_weights(df))

    df_resultados_updated, money, stock_returns = get_updated_results(test_set, weights, initial_investment= 100, rf_annual=0.035, periods_per_year=254.5)
    print(df_resultados_updated)
    # Versión interactiva
    st.dataframe(
        df_resultados.style.format(
            {
                "Returns": "{:.4f}%",
                "Volatility": "{:.4f}%",
                "Sharpe Ratio": "{:.4f}",
                "max_drawdown": "{:.4f}%",
            }
        )
    )

    st.dataframe(
        sectores.style.format(
        )
    )

    # Versión interactiva
    st.dataframe(
        df_weights.style.format()
    )

    show_portfolio(
        df_weights=df_weights,
        chart_type="bar",
        title="Composición por activo",
        label_name="Activo",
        weight_col="Pesos",
        weights_in_percent=False  # porque tus pesos son 0-1
    )
    show_portfolio(
        df_weights=sectores.set_index("sector"),
        chart_type="bar",
        title="Composición por sector",
        label_name="Sector",
        weight_col="Pesos",
        weights_in_percent=True
    )

    # Versión interactiva
    st.dataframe(
        df_resultados_updated.style.format(
            {
                "Returns": "{:.8f}%",
                "Volatility": "{:.4}%",
                "Sharpe Ratio": "{:.4f}",
                "max_drawdown": "{:.4f}%",
            }
        )
    )

    plot_portfolio_value(money)

    st.write(money)
    print(check_portfolio_weights(stock_returns, "2025-01-31"))
    print(get_sector_weights_at_date(stock_returns, sectors,"2025-01-31"))

    total_value_rb, values_by_asset_rb, trades_log = simulate_rebalance_6m_from_returns(
        returns_test=test_set,
        target_weights=weights,  # tu np.ndarray objetivo
        initial_investment=100,
        months=6,
        rf_annual=0.035,
        periods_per_year=254.5,
        risk_free_ticker="RISK_FREE"
    )

    print(trades_log.head(20))
    """