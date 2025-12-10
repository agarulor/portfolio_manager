import sys
import os

import pandas as pd

from portfolio_tools.risk_metrics import calculate_covariance  # si quieres usar covmat
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import streamlit as st
from data_management.get_data import read_price_file, get_stock_prices
from data_management.save_data import save_preprocessed_data
from data_management.clean_data import clean_and_align_data
from portfolio_tools.return_metrics import calculate_daily_returns
from portfolio_tools.risk_metrics import calculate_covariance

from portfolio_tools.markowitz import plot_frontier
from portfolio_management.markowitz_portfolios import get_initial_portfolio, show_initial_portfolio, \
    get_investor_initial_portfolio
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


def show_portfolio(returns: pd.DataFrame, weights: np.ndarray) -> None:
    a = 1

def render_portfolio():
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
    st.subheader("Cartera de inversión recomendada")

    # covmat no lo usas realmente dentro de create_markowitz_table,
    # pero si quieres ser consistente puedes calcularlo:
    e = read_price_file("data/processed/prices_20251207-210306.csv")
    f = calculate_daily_returns(e, method="simple")

    train_set, test_set = split_data_markowtiz(returns=f, test_date_start="2024-10-01", test_date_end="2025-09-30")

    covmat = calculate_covariance(train_set)


    df_resultados, df_weights = get_investor_initial_portfolio(train_set,
                                           min_w=0.0,
                                           max_w=0.19,
                                           rf_annual = 0.035,
                                           custom_target_volatility=0.25)

    # Versión interactiva
    st.dataframe(
        df_resultados.style.format(
            {
                "Returns": "{:.2f}%",
                "Volatility": "{:.2f}%",
                "Sharpe Ratio": "{:.2f}",
                "max_drawdown": "{:.2f}%",
            }
        )
    )

    # Versión interactiva
    st.dataframe(
        df_weights.style.format()
    )
