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
    returns: pd.DataFrame,
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

    returns = returns.sort_index()
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
                      n_features: int,
                      lstm_units: int = 64,
                      learning_rate: float = 0.001,
                      dropout_rate: float = 0.2,
                      optimizer_name: str = "adam",
                      loss="mse") -> tf.keras.models.Sequential:
    """
    Crea un modelo LSTM sencillo para predecir todos los activos a la vez.
    """
    inputs = tf.keras.Input(shape=(window_size, n_features))
    x = LSTM(
        lstm_units,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        return_sequences=False
    )(inputs)

    outputs = tf.keras.layers.Dense(n_features)(x)
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

def run_lstm_model(returns: pd.DataFrame,
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
    returns_with_ma = add_moving_average_features(returns, ma_windows=ma_windows)

    # 2) Preparamos datasets (usa los retornos + MAs)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, y_test_index = prepare_datasets_ml(
        returns_with_ma,
        train_date_end,
        val_date_end,
        test_date_end,
        lookback,
        window_size,
        horizon_shift
    )

    # 3) Creamos el modelo
    n_features = X_train.shape[2]
    model = create_lstm_model(
        window_size,
        n_features,
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
        "model": model,
        "history": history,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "X_val": X_val,
        "y_val": y_val,
        "test_loss": test_loss,
        "y_test_index": y_test_index,
        "returns_with_ma": returns_with_ma
    }


# ==========================================
# 3) PREDICCIONES + DESNORMALIZACIÓN
# ==========================================

def get_predictions_and_denormalize(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
):
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


# =========================================================
# 5) GRID SEARCH LSTM (CON LOG LIMPIO DE ESCENARIOS)
# =========================================================

def grid_search_lstm(
    returns: pd.DataFrame,
    train_date_end: str,
    val_date_end: str,
    test_date_end: str,
    window_size_list: List[int],
    horizon_shift: int,
    lstm_units_list: List[int],
    learning_rate_list: List[float],
    dropout_rate_list: List[float],
    optimizer_name_list: List[str],
    epochs: int,
    batch_size_list: List[int],
    lookback: int = 0,
    loss: str = "mse",
    verbose: int = 0,
    ma_windows: Optional[List[int]] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Grid search sobre varios hiperparámetros de la LSTM.
    - NO imprime pesos ni arrays.
    - Solo muestra:
        * Número de escenario (k / N)
        * Hiperparámetros del escenario
        * Cuántos escenarios quedan
    - Al final, imprime los mejores hiperparámetros.
    """

    # Añadimos medias móviles una única vez
    returns_with_ma = add_moving_average_features(returns, ma_windows=ma_windows)

    results = []

    # Todas las combinaciones de hiperparámetros
    combos = list(itertools.product(
        window_size_list,
        lstm_units_list,
        learning_rate_list,
        dropout_rate_list,
        optimizer_name_list,
        batch_size_list
    ))
    total_scenarios = len(combos)

    for scen_idx, (window_size,
                   lstm_units,
                   learning_rate,
                   dropout_rate,
                   optimizer_name,
                   batch_size) in enumerate(combos, start=1):

        remaining = total_scenarios - scen_idx

        print(
            f"\nEscenario {scen_idx}/{total_scenarios} "
            f"(quedan {remaining} por correr)"
        )
        print(
            f"  params -> window_size={window_size}, "
            f"lstm_units={lstm_units}, lr={learning_rate}, "
            f"dropout={dropout_rate}, opt={optimizer_name}, "
            f"batch_size={batch_size}"
        )

        # 1) Preparar datasets
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, y_test_index = prepare_datasets_ml(
            returns_with_ma,
            train_date_end=train_date_end,
            val_date_end=val_date_end,
            test_date_end=test_date_end,
            lookback=lookback,
            window_size=window_size,
            horizon_shift=horizon_shift
        )

        n_features = X_train.shape[2]

        # 2) Crear modelo
        model = create_lstm_model(
            window_size=window_size,
            n_features=n_features,
            lstm_units=lstm_units,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            optimizer_name=optimizer_name,
            loss=loss
        )

        # 3) Entrenar
        history = train_lstm_model(
            model,
            X_train, y_train,
            X_val, y_val,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )

        # 4) best val_loss
        val_loss_history = history.history.get("val_loss", None)
        if val_loss_history is not None:
            best_val_loss = float(np.min(val_loss_history))
        else:
            best_val_loss = float(history.history["loss"][-1])

        # 5) test_loss
        test_loss = float(model.evaluate(X_test, y_test, verbose=0))

        results.append({
            "train_date_end": train_date_end,
            "val_date_end": val_date_end,
            "test_date_end": test_date_end,
            "lookback": lookback,
            "window_size": window_size,
            "horizon_shift": horizon_shift,
            "lstm_units": lstm_units,
            "learning_rate": learning_rate,
            "dropout_rate": dropout_rate,
            "optimizer_name": optimizer_name,
            "epochs": epochs,
            "batch_size": batch_size,
            "val_loss": best_val_loss,
            "test_loss": test_loss,
        })

    results_df = pd.DataFrame(results)
    best_row = results_df.sort_values("val_loss", ascending=True).iloc[0]
    best_params: Dict[str, Any] = best_row.to_dict()

    # Mensaje final con los mejores hiperparámetros
    print("\n==============================")
    print(" Mejores hiperparámetros LSTM ")
    print("==============================")
    for k, v in best_params.items():
        print(f"{k}: {v}")

    return results_df, best_params


# =================================================================
# 6) REENTRENAR CON LOS MEJORES HP Y GRAFICAR (CON MAs INCLUIDAS)
# =================================================================

def run_best_lstm_and_plot(
    returns: pd.DataFrame,
    results_df: pd.DataFrame,
    best_params: Dict[str, Any],
    asset_idx: int = 0,
    asset_name: str | None = None,
    n_points: int | None = 200,
    loss: str = "mse",
    verbose: int = 1,
    ma_windows: Optional[List[int]] = None
):
    """
    Reentrena un modelo LSTM con los mejores hiperparámetros encontrados
    y muestra gráficos de real vs predicho (acción + portfolio EW).
    """

    window_size = int(best_params["window_size"])
    horizon_shift = int(best_params["horizon_shift"])
    train_date_end = best_params["train_date_end"]
    val_date_end = best_params["val_date_end"]
    test_date_end = best_params["test_date_end"]
    lookback = int(best_params["lookback"])

    lstm_units = int(best_params["lstm_units"])
    learning_rate = float(best_params["learning_rate"])
    dropout_rate = float(best_params["dropout_rate"])
    optimizer_name = str(best_params["optimizer_name"])
    epochs = int(best_params["epochs"])
    batch_size = int(best_params["batch_size"])

    # 0) Añadimos medias móviles
    returns_with_ma = add_moving_average_features(returns, ma_windows=ma_windows)

    # 1) Preparar datasets
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, y_test_index = prepare_datasets_ml(
        returns_with_ma,
        train_date_end=train_date_end,
        val_date_end=val_date_end,
        test_date_end=test_date_end,
        lookback=lookback,
        window_size=window_size,
        horizon_shift=horizon_shift
    )

    n_features = X_train.shape[2]

    # 2) Modelo
    model = create_lstm_model(
        window_size=window_size,
        n_features=n_features,
        lstm_units=lstm_units,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        optimizer_name=optimizer_name,
        loss=loss
    )

    # 3) Entrenar
    history = train_lstm_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    # 4) Evaluar en test
    test_loss = float(model.evaluate(X_test, y_test, verbose=verbose))
    print(f"\nTest loss con mejores hiperparámetros: {test_loss:.6f}")

    # 5) Predicciones desnormalizadas
    y_test_inv, y_pred_inv = get_predictions_and_denormalize(
        model=model,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler
    )

    # 6) Gráficos
    plot_real_vs_predicted(
        y_test_inv=y_test_inv,
        y_pred_inv=y_pred_inv,
        dates=y_test_index,
        asset_idx=asset_idx,
        asset_name=asset_name,
        n_points=n_points
    )

    return {
        "model": model,
        "history": history,
        "test_loss": test_loss,
        "y_test_inv": y_test_inv,
        "y_pred_inv": y_pred_inv,
        "y_test_index": y_test_index,
        "returns_with_ma": returns_with_ma
    }
