import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import streamlit as st
from data_management.get_data import read_price_file, get_stock_prices
from data_management.save_data import save_preprocessed_data
from data_management.clean_data import clean_and_align_data
from portfolio_tools.return_metrics import calculate_daily_returns
from portfolio_tools.risk_metrics import calculate_covariance

from portfolio_tools.markowitz import plot_frontier
from portfolio_management.markowitz_portfolios import create_markowitz_table
from data_management.dataset_preparation import split_data_markowtiz, prepare_datasets_ml
from portfolio_management.ml_portfolio import run_lstm_model, get_predictions_and_denormalize, plot_real_vs_predicted, grid_search_lstm, run_best_lstm_and_plot
from outputs.tables import show_table
from portfolio_management.XGBoost import run_xgb_experiment
from portfolio_management.ml_portfolio4 import train_lstm_all_assets
from portfolio_management.visualization import  plot_asset, plot_validation, plot_equal_weight
from portfolio_management.ml_portfolio_old import  plot_equal_weight_portfolio_on_validation
import os
import random
import numpy as np
import tensorflow as tf

SEED = 42

# Para que el hashing de Python no cambie entre ejecuciones
os.environ["PYTHONHASHSEED"] = str(SEED)

# Semillas de Python, NumPy y TensorFlow
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

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



results_df, best_params = grid_search_lstm(
    returns=f,
    train_date_end="2023-09-30",
    val_date_end="2024-09-30",
    test_date_end="2025-09-30",
    window_size_list=[30],
    horizon_shift=1,
    lstm_units_list=[128],
    learning_rate_list=[0.001],
    dropout_rate_list=[0.35],
    optimizer_name_list=["adam"],
    epochs=125,
    batch_size_list=[32],
    ma_windows=[5, 21, 63],
    lookback= 0,
    loss="mse",
    verbose=0
)

best_run = run_best_lstm_and_plot(
    returns=f,
    results_df=results_df,
    best_params=best_params,
    asset_idx=0,
    asset_name=f.columns[0],
    ma_windows=[30]
)

"""
e = read_price_file("data/processed/prices_20251110-193638.csv")
f = calculate_daily_returns(e, method="simple")
e = e[["BBVA.MC"]]
results, real_df, pred_df = train_lstm_all_assets(
    prices_df=e,
    train_date_end="2024-09-30",
    val_date_end="2024-09-30",
    test_date_end="2025-09-30",
    window_size=60,
    lstm_units=256,
    learning_rate=0.0005,
    dropout_rate=0.0,
    optimizer_name="adam",
    loss="mse",
    epochs=200,
    batch_size=32,
    verbose=1,
    use_early_stopping=True,
    patience=15,
    min_delta=0.001,
    forecast=True
)
print(pred_df.head())
print(pred_df.tail())

print(real_df.head())
print(real_df.tail())

print(pred_df.iloc[:, 0].describe())
assets = e.columns
for asset in assets:

    plot_asset(
        real_df=real_df,
        pred_df=pred_df,
        asset=asset,
        n_points=200
)

plot_equal_weight(
    real_df=real_df,
    pred_df=pred_df,
    n_points=252    # por ejemplo, último año de validación
)


if __name__ == "__main__":
    main()



