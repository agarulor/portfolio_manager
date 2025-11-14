import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import streamlit as st
from data_management.get_data import read_price_file
from portfolio_tools.return_metrics import calculate_daily_returns
from portfolio_tools.risk_metrics import calculate_covariance
from portfolio_tools.markowitz import plot_frontier
from portfolio_management.markowitz_portfolios import create_markowitz_table
from data_management.dataset_preparation import split_data_markowtiz, prepare_datasets_ml
from portfolio_management.ml_portfolio import run_lstm_model
from outputs.tables import show_table


def main():

    """
    datos = get_stock_prices("data/input/eurostoxx50_csv.csv",
                             "ticker_yahoo",
                             "name",
                             )
    datos_2, report, summary = clean_and_align_data(datos, beginning_data=True)

    save_preprocessed_data(datos_2)

    print(datos_2.head())

    """
    #st.title("Análisis de Carteras Markowitz")

    e = read_price_file("data/processed/prices_20251110-193638.csv")

    f = calculate_daily_returns(e, method="simple")
    #train, test = split_data_markowtiz(f)

    retorno = run_lstm_model(f, learning_rate=0.0003, batch_size=32, epochs=50)

    print(retorno["test_loss"])


    #covmat_train = calculate_covariance(train)

    #pruba = create_markowitz_table(train, test, covmat_train, rf = 0.00, min_w=0.0)

    #a = plot_frontier(30, train, covmat_train, rf= 0.0)

    #show_table(pruba, caption="Resultados Markowitz")


if __name__ == "__main__":
    main()



