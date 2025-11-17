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
from portfolio_management.XGBoost import run_xgb_experiment


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
    """
    e = read_price_file("data/processed/prices_20251110-193424.csv")

    f = calculate_daily_returns(e, method="simple")
    #train, test = split_data_markowtiz(f)

    result = run_lstm_model(f, window_size=60,
                            lstm_units=128,
                            learning_rate=0.0005,
                            dropout_rate=0.05,
                            batch_size=32,
                            epochs=100,
                            loss="mae",
                            optimizer_name="rmsprop")

    model = result["model"]
    scaler = result["scaler"]
    X_test = result["X_test"]
    y_test = result["y_test"]
    dates = result["y_test_index"]
    X_val = result["X_val"]
    y_val = result["y_val"]

    y_test_inv, y_pred_inv = get_predictions_and_denormalize(
        model=model,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler)

    y_val_inv, y_pred_val = get_predictions_and_denormalize(
        model=model,
        X_test=X_val,
        y_test=y_val,
        scaler=scaler)

    plot_real_vs_predicted(
        y_test_inv=y_test_inv,
        y_pred_inv=y_pred_inv,
        dates=dates,
        asset_idx=0,
        asset_name=f.columns[0],
        n_points=None
    )

    plot_real_vs_predicted(
        y_test_inv=y_val_inv,
        y_pred_inv=y_pred_val,

        asset_idx=0,
        asset_name=f.columns[0],
        n_points=None
    )
    #covmat_train = calculate_covariance(train)

    #pruba = create_markowitz_table(train, test, covmat_train, rf = 0.00, min_w=0.0)

    #a = plot_frontier(30, train, covmat_train, rf= 0.0)

    #show_table(pruba, caption="Resultados Markowitz")

e = read_price_file("data/processed/prices_20251110-193638.csv")
f = calculate_daily_returns(e, method="simple")
print(type(f))  # debería ser <class 'pandas.core.frame.DataFrame'>

xgb_results = run_xgb_experiment(
    returns=f,  # tu DataFrame de retornos
    train_date_end="2023-09-30",
    val_date_end="2024-09-30",
    test_date_end="2025-09-30",
    window_size=63,  # importante: >= max(lags)
    horizon_shift=1,
    lags=[1, 2, 5, 10, 21, 63],
    n_estimators=800,
    learning_rate=0.03,
    max_depth=4,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=3,
    reg_alpha=0.1,
)

# Escogemos un activo, por ejemplo el primero (columna 0)
asset_idx = 0
asset_name = f.columns[asset_idx]

plot_real_vs_predicted(
    y_test_inv=xgb_results["y_test_inv"],
    y_pred_inv=xgb_results["y_pred_inv"],
    dates=xgb_results["y_test_index"],
    asset_idx=asset_idx,
    asset_name=asset_name,
    n_points=200
)
"""
e = read_price_file("data/processed/prices_20251110-193638.csv")
f = calculate_daily_returns(e, method="simple")

results_df, best_params = grid_search_lstm(
    returns=f,
    train_date_end="2022-09-30",
    val_date_end="2024-09-30",
    test_date_end="2025-09-30",
    window_size_list=[20, 30, 60, 120],
    horizon_shift=1,
    lstm_units_list=[64, 128],
    learning_rate_list=[0.001, 0.0005],
    dropout_rate_list=[0, 0.1, 0.2],
    optimizer_name_list=["adam", "rmsprop"],
    epochs=50,
    batch_size_list=[16, 32, 64],
    ma_windows=[30]   # <-- aquí activas la media móvil 30 días
)

best_run = run_best_lstm_and_plot(
    returns=f,
    results_df=results_df,
    best_params=best_params,
    asset_idx=0,
    asset_name=f.columns[0],
    ma_windows=[30]
)
if __name__ == "__main__":
    main()



