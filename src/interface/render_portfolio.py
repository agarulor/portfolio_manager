import sys
import os
FILENAME_PATH = "data/input/ibex_eurostoxx.csv"
TICKER_COL = "ticker_yahoo"
COMPANIES_COL = "name"
START_DATE = "2005-01-01"
END_DATE = "2025-09-30"
import pandas as pd
from interface.render_initial_portfolio import render_investor_constraints, render_constraints_portfolio

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import streamlit as st

import plotly.express as px

from types import MappingProxyType

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
    render_constraints_portfolio()
"""


    df_resultados, df_weights, weights = get_investor_initial_portfolio(train_set,
                                           min_w=0.025,
                                           max_w=0.15,
                                           rf_annual = 0.035,
                                            periods_per_year=256,
                                           custom_target_volatility=sigma_recomended,
                                                                        sectors_df=sectors,
                                                                        sector_max_weight=max_sector_pct,
                                                                        risk_free_ticker="RISK_FREE")

    sectores = get_sector_exposure_table(df_weights, sectors)

    #print(check_portfolio_weights(df))

    df_resultados_updated, money, stock_returns = get_updated_results(test_set, weights, initial_investment = amount, rf_annual=0.035, periods_per_year=254.5)
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