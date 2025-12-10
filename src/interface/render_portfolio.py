import sys
import os

from portfolio_management.markowitz_portfolios import create_markowitz_table  # ajusta el path a donde tengas la función
from portfolio_tools.risk_metrics import calculate_covariance  # si quieres usar covmat
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import streamlit as st
from data_management.get_data import read_price_file, get_stock_prices
from data_management.save_data import save_preprocessed_data
from data_management.clean_data import clean_and_align_data
from portfolio_tools.return_metrics import calculate_daily_returns
from portfolio_tools.risk_metrics import calculate_covariance

from portfolio_tools.markowitz import plot_frontier
from portfolio_management.markowitz_portfolios import create_markowitz_table, get_markowtiz_results
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
    st.subheader("Comparativa de carteras (Markowitz)")

    # covmat no lo usas realmente dentro de create_markowitz_table,
    # pero si quieres ser consistente puedes calcularlo:
    e = read_price_file("data/processed/prices_20251110-193638.csv")
    f = calculate_daily_returns(e, method="simple")

    train_set, test_set = split_data_markowtiz(returns=f, test_date_start="2024-10-01", test_date_end="2025-09-30")

    covmat = calculate_covariance(train_set)

    df_resultados = create_markowitz_table(train_returns=train_set,
                                           test_returns=train_set,
                                           min_w=0.025,
                                           max_w=0.15,
                                           rf_annual = 0.035,
                                           custom_target_volatility=0.15)

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

    # Si prefieres una tabla estática:
    # st.table(df_resultados)

    st.info("La tabla muestra rentabilidad, volatilidad, sharpe, max drawdown y pesos de cada tipo de cartera.")