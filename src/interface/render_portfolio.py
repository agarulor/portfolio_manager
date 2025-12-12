import sys
import os

import pandas as pd

from portfolio_tools.risk_metrics import calculate_covariance  # si quieres usar covmat
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import streamlit as st
from typing import Literal
import plotly.express as px
from data_management.get_data import read_price_file, get_stock_prices
from data_management.save_data import save_preprocessed_data
from data_management.clean_data import clean_and_align_data
from portfolio_tools.return_metrics import calculate_daily_returns
from portfolio_tools.risk_metrics import calculate_covariance

from portfolio_tools.markowitz import plot_frontier
from portfolio_management.investor_portfolios import  get_investor_initial_portfolio, get_updated_results, get_cumulative_returns
from data_management.dataset_preparation import split_data_markowtiz
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


def show_portfolio(df_weights: pd.DataFrame,
                   chart_type: Literal["pie", "bar"] = "bar",
                   title: "str" = "Composición de la cartera inicial") -> None:
    # We check that the DataFrame is correct
    if df_weights.shape[1] != 1:
        raise ValueError("DataFrame needs to have 1 weights column.")

    col_name = df_weights.columns[0]
    df_plot = df_weights.copy()
    # We reset the index to plot
    df_plot = df_plot.reset_index()
    df_plot.columns = ["Ticker", "Peso"]

    st.subheader(title)

    if chart_type == "bar":
        fig = px.bar(
            df_plot,
            x="Peso",
            y="Ticker",
            orientation="h",
            text="Peso",
            title=title,
        )
        fig.update_traces(texttemplate="%{text:.2f}", textposition="outside")
        fig.update_layout(
            xaxis_title="Peso (%)",
            yaxis_title="Activo",
            yaxis=dict(categoryorder="total ascending"),
        )

    elif chart_type == "pie":
        fig = px.pie(
            df_plot,
            names="Ticker",
            values="Peso",
            title=title,
        )

    else:
        raise ValueError("chart_type must be either 'bar' or 'pie'")

    st.plotly_chart(fig, use_container_width=True)


def render_portfolio():
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

    # -----------------------------
    # AQUÍ IMPRIMES LA TABLA
    # -----------------------------
    st.subheader(f"Dinero Invertido **{dinero_invertido}**")
    st.subheader("Cartera de inversión recomendada")

    # covmat no lo usas realmente dentro de create_markowitz_table,
    # pero si quieres ser consistente puedes calcularlo:
    e = read_price_file("data/processed/prices_20251207-210306.csv")
    f = calculate_daily_returns(e, method="simple")

    train_set, test_set = split_data_markowtiz(returns=f, test_date_start="2023-10-01", test_date_end="2025-09-30")

    df_resultados, df_weights, weights = get_investor_initial_portfolio(train_set,
                                           min_w=0.00,
                                           max_w=1,
                                           rf_annual = 0.035,
                                            periods_per_year=256,
                                           custom_target_volatility=0.15)

    df_resultados_updated, money = get_updated_results(test_set, weights, initial_investment= 100, rf_annual=0.035, periods_per_year=254.5)

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

    # Versión interactiva
    st.dataframe(
        df_weights.style.format()
    )

    show_portfolio(
        df_weights=df_weights,
        chart_type="bar",
        title="Distribución inicial de la cartera"
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

    #money = get_cumulative_returns(test_set, weights, dinero_invertido, rf_annual=0.035, periods_per_year=254   )

    st.write(money)