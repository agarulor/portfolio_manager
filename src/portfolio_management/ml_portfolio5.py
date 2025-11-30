import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
from keras.src.optimizers import Adam, RMSprop, SGD, AdamW

from data_management.dataset_preparation import prepare_datasets_ml
from tensorflow.keras.layers import LSTM
from typing import Dict, Any, Tuple, List, Optional

import matplotlib.pyplot as plt


# ======================================================
# 0) FUNCIONES AUXILIARES: MEDIAS MÓVILES COMO FEATURES
# ======================================================

def add_moving_average_features(
    prices: pd.DataFrame,
    ma_windows: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Añade medias móviles de los retornos como nuevas columnas (features).

    Para cada columna de 'returns' y para cada ventana en ma_windows,
    se crea una columna nueva con el sufijo _maXX, por ejemplo:
        'ACS.MC' -> 'ACS.MC_ma30'

    Las primeras filas donde no se pueda calcular la MA (NaN) se eliminan.
    """
    if ma_windows is None:
        ma_windows = [30]  # por defecto, 30 días

    returns = prices.sort_index()
    df_list = [returns]

    for w in ma_windows:
        ma_df = returns.rolling(window=w).mean()
        ma_df = ma_df.add_suffix(f"_ma{w}")
        df_list.append(ma_df)

    df_with_ma = pd.concat(df_list, axis=1)
    df_with_ma = df_with_ma.dropna()

    return df_with_ma


# ======================================
# 1) DEFINICIÓN DEL MODELO LSTM
# ======================================

def create_lstm_model(window_size: int,
                      lstm_units: int = 64,
                      learning_rate: float = 0.001,
                      dropout_rate: float = 0.2,
                      optimizer_name: str = "adam",
                      loss="mse") -> tf.keras.models.Sequential:
    """
    Crea un modelo LSTM sencillo para predecir todos los activos a la vez.
    """
    inputs = tf.keras.Input(shape=(window_size, 1))
    x = LSTM(
        lstm_units,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        return_sequences=False
    )(inputs)

    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Elegimos optimizador
    if optimizer_name.lower() == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == "adamw":
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    elif optimizer_name.lower() == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError("Please, enter a valid optimizer name")

    model.compile(
        loss=loss,
        optimizer=optimizer,
    )

    return model


def train_lstm_model(model: tf.keras.models.Sequential,
                     X_train: np.ndarray,
                     y_train: np.ndarray,
                     X_val: np.ndarray,
                     y_val: np.ndarray,
                     epochs: int = 25,
                     batch_size: int = 64,
                     verbose: int = 1) -> tf.keras.callbacks.History:
    """
    Entrena el modelo LSTM.
    """
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    return history


# =====================================================
# 2) PIPELINE LSTM (CON MEDIAS MÓVILES COMO FEATURES)
# =====================================================

def run_lstm_model(prices: pd.DataFrame,
                   train_date_end: str = "2023-09-30",
                   val_date_end: str = "2024-09-30",
                   test_date_end: str = "2025-09-30",
                   lookback: int = 0,
                   window_size: int = 60,
                   horizon_shift: int = 1,
                   lstm_units: int = 64,
                   learning_rate: float = 0.001,
                   dropout_rate: float = 0.2,
                   optimizer_name: str = "adam",
                   loss="mse",
                   epochs: int = 25,
                   batch_size: int = 64,
                   verbose: int = 1,
                   ma_windows: Optional[List[int]] = None):
    """
    Pipeline LSTM completo:
    1) Añade medias móviles como features.
    2) Normaliza y crea rolling windows (prepare_datasets_ml).
    3) Entrena el modelo y evalúa en test.
    """

    # 1) Añadimos medias móviles a los retornos
    prices_with_ma = add_moving_average_features(prices, ma_windows=ma_windows)

    # 2) Preparamos datasets (usa los retornos + MAs)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, y_test_index = prepare_datasets_ml(
        prices_with_ma,
        train_date_end,
        val_date_end,
        test_date_end,
        lookback,
        window_size,
        horizon_shift
    )

    # 3) Creamos el modelo
    model = create_lstm_model(
        window_size,
        lstm_units,
        learning_rate,
        dropout_rate,
        optimizer_name,
        loss
    )

    # 4) Entrenamos
    history = train_lstm_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    # 5) Evaluamos en test
    test_loss = model.evaluate(X_test, y_test, verbose=verbose)

    return {
        "asset": asset,
        "model": model,
        "history": history,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "X_val": X_val,
        "y_val": y_val,
        "test_loss": test_loss,
        "y_test_index": y_test_index,
        "returns_with_ma": prices_with_ma
    }


# ==========================================
# 3) PREDICCIONES + DESNORMALIZACIÓN
# ==========================================

def get_predictions_and_denormalize(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler):
    """
    Obtiene predicciones del modelo y las des-normaliza
    usando el mismo scaler que se ajustó en train.
    """
    y_pred = model.predict(X_test, verbose=0)

    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    return y_test_inv, y_pred_inv


# ===================================================
# 4) GRÁFICO: REAL VS PREDICHO + PORTFOLIO EQUIPONDERADO
# ===================================================

def plot_real_vs_predicted(
    y_test_inv: np.ndarray,
    y_pred_inv: np.ndarray,
    dates=None,
    asset_idx: int = 0,
    asset_name: str | None = None,
    n_points: int | None = 200
):
    """
    1) Grafica retornos reales vs predichos para una acción concreta.
    2) Calcula un portfolio equiponderado (igual peso en todas las acciones),
       y grafica su retorno acumulado real vs predicho.
    3) Muestra rentabilidades anualizadas del portfolio equiponderado.
    """

    # 0) Preparar ejes de tiempo
    n_samples = y_test_inv.shape[0]

    if dates is None:
        x_axis = np.arange(n_samples)
    else:
        x_axis = np.array(dates)

    n = min(len(x_axis), n_samples)
    x_axis = x_axis[:n]
    y_test_inv = y_test_inv[:n, :]
    y_pred_inv = y_pred_inv[:n, :]

    # Recorte a últimos n_points
    if (n_points is not None) and (n > n_points):
        x_axis_slice = x_axis[-n_points:]
        y_test_slice = y_test_inv[-n_points:, :]
        y_pred_slice = y_pred_inv[-n_points:, :]
    else:
        x_axis_slice = x_axis
        y_test_slice = y_test_inv
        y_pred_slice = y_pred_inv

    # 1) Retornos diarios real vs predicho para una sola acción
    real_series = y_test_slice[:, asset_idx]
    pred_series = y_pred_slice[:, asset_idx]

    if asset_name is None:
        asset_name = f"Asset {asset_idx}"

    plt.figure(figsize=(12, 5))
    plt.plot(x_axis_slice, real_series, label="Real", linewidth=1)
    plt.plot(x_axis_slice, pred_series, label="Predicho", linewidth=1, linestyle="--")
    plt.title(f"Retornos diarios reales vs predichos - {asset_name}")
    plt.xlabel("Tiempo")
    plt.ylabel("Retorno diario")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 2) Portfolio equiponderado real vs predicho
    real_port_daily = y_test_slice.mean(axis=1)
    pred_port_daily = y_pred_slice.mean(axis=1)

    real_port_cum = (1.0 + real_port_daily).cumprod() - 1.0
    pred_port_cum = (1.0 + pred_port_daily).cumprod() - 1.0

    plt.figure(figsize=(12, 5))
    plt.plot(x_axis_slice, real_port_cum, label="Portfolio Real (EW)", linewidth=1)
    plt.plot(x_axis_slice, pred_port_cum, label="Portfolio Predicho (EW)", linewidth=1, linestyle="--")
    plt.axhline(0.0, color="black", linewidth=1, linestyle="--")
    plt.title("Retorno acumulado - Portfolio equiponderado (real vs predicho)")
    plt.xlabel("Tiempo")
    plt.ylabel("Retorno acumulado")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # 3) Rentabilidad anualizada
    n_days = len(real_port_daily)
    if n_days > 0:
        real_total = real_port_cum[-1]
        pred_total = pred_port_cum[-1]

        ann_factor = 252.0 / n_days
        real_ann = (1.0 + real_total) ** ann_factor - 1.0
        pred_ann = (1.0 + pred_total) ** ann_factor - 1.0

        print("=== Portfolio equiponderado (sobre ventana mostrada) ===")
        print(f"Retorno total REAL:     {real_total: .4%}")
        print(f"Retorno total PREDICHO: {pred_total: .4%}")
        print(f"Retorno anualizado REAL:     {real_ann: .4%}")
        print(f"Retorno anualizado PREDICHO: {pred_ann: .4%}")
    else:
        print("No hay suficientes datos para calcular rentabilidades del portfolio.")
