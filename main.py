import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import numpy as np
import pandas as pd
import streamlit as st
from data_management.get_data import get_stock_prices, read_price_file
from data_management.clean_data import clean_and_align_data
from data_management.save_data import save_preprocessed_data
from portfolio_tools.return_metrics import calculate_daily_returns, annualize_returns, portfolio_returns, daily_portfolio_returns
from portfolio_tools.risk_metrics import calculate_covariance, portfolio_volatility, sharpe_ratio, neg_sharpe_ratio, \
    annualize_covariance, calculate_max_drawdown
from portfolio_tools.markowitz import minimize_volatility, plot_frontier, maximize_return, msr
from portfolio_management.markowitz_portfolios import get_markowtiz_results, create_markowitz_table
import matplotlib.pyplot as plt
from portfolio_tools.portfolio_allocation import split_data_markowtiz, split_data_ml
from outputs.tables import show_table


def main():
    st.title("Análisis de Carteras Markowitz")

    e = read_price_file("data/processed/prices_20251105-013233.csv")

    f = calculate_daily_returns(e, method="simple")
    train, test = split_data_markowtiz(f)
    covmat_train = calculate_covariance(train)

    pruba = create_markowitz_table(train, test, covmat_train)

    show_table(pruba, caption="Resultados Markowitz")

if __name__ == "__main__":
    main()



