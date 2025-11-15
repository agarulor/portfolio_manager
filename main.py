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
from portfolio_management.ml_portfolio import run_lstm_model, get_predictions_and_denormalize, plot_real_vs_predicted, grid_search_lstm, run_best_lstm_and_plot
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

    results_df, best_params = grid_search_lstm(
        returns=f,
        train_date_end="2022-09-30",
        val_date_end="2024-09-30",
        test_date_end="2025-09-30",
        window_size_list=[30, 60, 90],
        horizon_shift=1,
        lstm_units_list=[32, 64, 128],
        learning_rate_list=[1e-3, 5e-4],
        dropout_rate_list=[0.0, 0.1, 0.2],
        optimizer_name_list=["adam", "rmsprop"],
        epochs=75,
        batch_size_list=[16, 32, 64],
        verbose=0
    )

    print(results_df.sort_values("val_loss").head())
    print("Mejores hiperparámetros:", best_params)

    # 2) Reentrenar con los mejores y graficar
    exp_results = run_best_lstm_and_plot(
        returns=f,
        results_df=results_df,
        best_params=best_params,
        asset_idx=0,  # por ejemplo, primera acción
        asset_name="Asset 0",  # o el ticker
        n_points=200
    )


if __name__ == "__main__":
    main()



