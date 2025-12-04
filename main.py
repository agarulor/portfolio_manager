import sys
import os

import investor_information

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

from outputs.tables import show_table

import os
import random
import numpy as np

SEED = 42

# Para que el hashing de Python no cambie entre ejecuciones
os.environ["PYTHONHASHSEED"] = str(SEED)

# Semillas de Python, NumPy y TensorFlow
random.seed(SEED)
np.random.seed(SEED)


def main():
    """
    datos = get_stock_prices("data/input/eurostoxx50_csv.csv",
                             "ticker_yahoo",
                             "name",
                             start_date="2015-10-01",
                             )
    datos_2, report, summary = clean_and_align_data(datos, beginning_data=True)

    save_preprocessed_data(datos_2)

    print(datos_2.head())


    #st.title("Análisis de Carteras Markowitz")

    e = read_price_file("data/processed/prices_20251110-193424.csv")

    f = calculate_daily_returns(e, method="simple")



e = read_price_file("data/processed/prices_20251110-193638.csv")
f = calculate_daily_returns(e, method="simple")

train_set, test_set = split_data_markowtiz(returns=f, test_date_start="2025-06-01", test_date_end="2025-09-30")

g = calculate_covariance(train_set)

h = create_markowitz_table(train_set, train_set, g, rf = 0.00, min_w=0.0)
print(h.head())

plot_frontier(30, train_set, g,  method="simple")

#h = get_markowtiz_results()
"""
print(investor_target_volatility(6, 6, 3, 3, 3, 2, 3, 3))
if __name__ == "__main__":
    main()



