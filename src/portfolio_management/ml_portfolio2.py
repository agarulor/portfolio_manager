import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD


# ============================================================
# 1. PREPARAR DATOS PARA UNA ÚNICA ACCIÓN (UNIVARIANTE)
# ============================================================

def prepare_unistep_univariate_for_asset(
    prices_series: pd.Series,
    train_date_end: str,
    val_date_end: str,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           MinMaxScaler,
           np.ndarray]:
    """
    Prepara X_train, y_train, X_val, y_val para UNA sola acción (univariante),
    con horizonte 1, usando Rolling Window sobre TODO el histórico de esa serie,
    y asignando cada ejemplo a TRAIN/VAL según la fecha del target.

    prices_series: Serie de precios de UNA acción (index = fechas).
    train_date_end: último día de train (YYYY-MM-DD).
    val_date_end: último día de validación (YYYY-MM-DD).
    window_size: lookback (ej: 60, 120).
    """

    # Aseguramos orden temporal y float
    prices_series = prices_series.sort_index().astype(float)

    # Fechas de corte
    train_end = pd.to_datetime(train_date_end)
    val_end = pd.to_datetime(val_date_end)

    dates = prices_series.index.to_numpy()               # (n_days,)
    data = prices_series.to_numpy().reshape(-1, 1)       # (n_days, 1)
    n_days = data.shape[0]

    # Escalado SOLO con TRAIN de esta acción
    mask_train_scaler = dates <= train_end
    data_train_for_scaler = data[mask_train_scaler]

    if data_train_for_scaler.shape[0] == 0:
        raise ValueError("No hay datos anteriores o iguales a train_date_end para ajustar el scaler.")

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_train_for_scaler)

    data_scaled = scaler.transform(data)                 # (n_days, 1)

    # Rolling Window
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    val_dates_list = []

    for t in range(window_size, n_days):
        X_window = data_scaled[t - window_size:t, :]     # (window_size, 1)
        y_t = data_scaled[t, :]                          # (1,)
        date_t = dates[t]

        if date_t <= train_end:
            X_train_list.append(X_window)
            y_train_list.append(y_t)
        elif date_t <= val_end:
            X_val_list.append(X_window)
            y_val_list.append(y_t)
            val_dates_list.append(date_t)
        else:
            # Más allá de val_end -> futuro (no se usa aquí)
            pass

    def _stack_xy(X_list, y_list):
        if len(X_list) == 0:
            return (np.empty((0, window_size, 1)),
                    np.empty((0, 1)))
        return np.stack(X_list, axis=0), np.stack(y_list, axis=0)

    X_train, y_train = _stack_xy(X_train_list, y_train_list)
    X_val, y_val = _stack_xy(X_val_list, y_val_list)
    val_dates = np.array(val_dates_list)

    print(f"[{prices_series.name}] X_train: {X_train.shape}, X_val: {X_val.shape}")
    if len(val_dates) > 0:
        print(f"[{prices_series.name}] Validación: {val_dates[0]} -> {val_dates[-1]} "
              f"({len(val_dates)} días de target)")

    return X_train, y_train, X_val, y_val, scaler, val_dates


# ============================================================
# 2. MODELO LSTM UNIVARIANTE (UNA ACCIÓN)
# ============================================================

def create_univariate_lstm_model(
    window_size: int,
    lstm_units: int = 50,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.0,
    optimizer_name: str = "rmsprop",
    loss: str = "mse",
) -> Model:
    """
    LSTM univariante:
      input:  (window_size, 1)
      output: (1)  -> precio del día siguiente (escalado)
    """
    inputs = Input(shape=(window_size, 1))
    x = LSTM(
        lstm_units,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        return_sequences=False
    )(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)

    opt_name = optimizer_name.lower()
    if opt_name == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif opt_name == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    elif opt_name == "sgd":
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Optimizer no válido: {optimizer_name}")

    if loss == "huber":
        loss_fn = tf.keras.losses.Huber(delta=1.0)
    else:
        loss_fn = loss

    model.compile(
        loss=loss_fn,
        optimizer=optimizer,
    )
    return model


# ============================================================
# 3. ENTRENAR + PREDECIR PARA UNA ACCIÓN
# ============================================================

def train_and_validate_single_asset(
    prices_series: pd.Series,
    train_date_end: str,
    val_date_end: str,
    window_size: int = 60,
    lstm_units: int = 50,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.0,
    optimizer_name: str = "rmsprop",
    loss: str = "mse",
    epochs: int = 25,
    batch_size: int = 32,
    verbose: int = 1,
) -> Dict[str, Any]:
    """
    Entrena y valida una LSTM univariante sobre UNA acción concreta.
    Devuelve modelo, history, y precios reales/predichos en validación desescalados.
    """

    X_train, y_train, X_val, y_val, scaler, val_dates = prepare_unistep_univariate_for_asset(
        prices_series=prices_series,
        train_date_end=train_date_end,
        val_date_end=val_date_end,
        window_size=window_size
    )

    model = create_univariate_lstm_model(
        window_size=window_size,
        lstm_units=lstm_units,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        optimizer_name=optimizer_name,
        loss=loss
    )

    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,       # IMPORTANTE en series temporales
        verbose=verbose
    )

    # Predicciones en validación
    y_val_pred = model.predict(X_val)

    # Desescalar
    y_val_inv = scaler.inverse_transform(y_val)
    y_pred_inv = scaler.inverse_transform(y_val_pred)

    return {
        "asset": prices_series.name,
        "model": model,
        "history": history,
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "y_val_inv": y_val_inv,      # precios reales val
        "y_pred_inv": y_pred_inv,    # precios predichos val
        "val_dates": val_dates,
        "scaler": scaler
    }


# ============================================================
# 4. ENTRENAR TODAS LAS ACCIONES POR SEPARADO
# ============================================================

def train_lstm_unistep_all_assets_separately(
    prices_df: pd.DataFrame,
    train_date_end: str,
    val_date_end: str,
    window_size: int = 60,
    lstm_units: int = 50,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.0,
    optimizer_name: str = "rmsprop",
    loss: str = "mse",
    epochs: int = 25,
    batch_size: int = 32,
    verbose: int = 1,
) -> Dict[str, Dict[str, Any]]:
    """
    Recorre todas las columnas de prices_df (acciones) y:
      - prepara datos univariantes por acción,
      - entrena una LSTM por acción,
      - obtiene predicciones en validación.

    Devuelve un dict:
      results[asset] = {... info de esa acción ...}
    """
    results: Dict[str, Dict[str, Any]] = {}

    for col in prices_df.columns:
        print(f"\n==============================")
        print(f"Entrenando LSTM univariante para {col}")
        print(f"==============================\n")

        res_asset = train_and_validate_single_asset(
            prices_series=prices_df[col],
            train_date_end=train_date_end,
            val_date_end=val_date_end,
            window_size=window_size,
            lstm_units=lstm_units,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            optimizer_name=optimizer_name,
            loss=loss,
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose
        )

        results[col] = res_asset

    return results


# ============================================================
# 5. PLOT VALIDACIÓN PARA UNA ACCIÓN (USANDO RESULTADOS)
# ============================================================

def plot_validation_for_asset_from_results(
    results: Dict[str, Dict[str, Any]],
    asset: str,
    n_points: Optional[int] = 200
):
    """
    Dibuja precio real vs predicho en validación para una acción,
    usando el dict devuelto por train_lstm_unistep_all_assets_separately.
    """

    if asset not in results:
        raise ValueError(f"{asset} no está en results (keys: {list(results.keys())})")

    res = results[asset]
    y_val_inv = res["y_val_inv"].reshape(-1)
    y_pred_inv = res["y_pred_inv"].reshape(-1)
    val_dates = res["val_dates"]

    # Alineamos
    n = min(len(y_val_inv), len(y_pred_inv), len(val_dates))
    y_val_inv = y_val_inv[:n]
    y_pred_inv = y_pred_inv[:n]
    val_dates = val_dates[:n]

    if n_points is not None and n > n_points:
        y_val_inv = y_val_inv[-n_points:]
        y_pred_inv = y_pred_inv[-n_points:]
        val_dates = val_dates[-n_points:]

    if len(val_dates) == 0:
        print(f"No hay datos de validación para {asset}")
        return

    print(f"[{asset}] Validación desde {val_dates[0]} hasta {val_dates[-1]} (N={len(val_dates)})")

    plt.figure(figsize=(12, 5))
    plt.plot(val_dates, y_val_inv, label="Precio real (validación)", linewidth=1.5)
    plt.plot(val_dates, y_pred_inv, label="Precio predicho (validación)", linestyle="--", linewidth=1.5)
    plt.title(f"LSTM univariante - {asset}")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()

def build_validation_price_matrices_from_results(
    results: Dict[str, Dict[str, Any]]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    A partir del dict `results` (un modelo por activo), construye:
      - real_prices_val_df: precios reales de validación (fechas x activos)
      - pred_prices_val_df: precios predichos de validación (fechas x activos)
    """

    if not results:
        raise ValueError("El diccionario results está vacío.")

    # Tomamos un activo de referencia para las fechas
    first_asset = list(results.keys())[0]
    ref_dates = results[first_asset]["val_dates"]
    ref_dates = pd.to_datetime(ref_dates)
    real_df = pd.DataFrame(index=ref_dates)
    pred_df = pd.DataFrame(index=ref_dates)

    for asset, res in results.items():
        y_val_inv = res["y_val_inv"].reshape(-1)
        y_pred_inv = res["y_pred_inv"].reshape(-1)
        val_dates = pd.to_datetime(res["val_dates"])

        # Convertimos cada uno en Series (indexadas por fecha)
        s_real = pd.Series(y_val_inv, index=val_dates, name=asset)
        s_pred = pd.Series(y_pred_inv, index=val_dates, name=asset)

        # Reindexamos a las fechas de referencia (por si acaso difieren ligeramente)
        s_real = s_real.reindex(ref_dates)
        s_pred = s_pred.reindex(ref_dates)

        real_df[asset] = s_real
        pred_df[asset] = s_pred

    return real_df, pred_df

def plot_equal_weight_portfolio_from_results(
    results: Dict[str, Dict[str, Any]],
    n_points: Optional[int] = 200,
):
    """
    Calcula y grafica el portfolio equiponderado REAL vs PREDICHO
    a partir del diccionario `results` (un modelo por activo).
    """

    # 1) Construimos matrices de precios de validación (fechas x activos)
    real_prices_df, pred_prices_df = build_validation_price_matrices_from_results(results)

    # Eliminamos filas donde falten precios para algún activo (por seguridad)
    real_prices_df = real_prices_df.dropna(how="any")
    pred_prices_df = pred_prices_df.dropna(how="any")

    # Alineamos por seguridad
    real_prices_df, pred_prices_df = real_prices_df.align(pred_prices_df, join="inner", axis=0)

    if real_prices_df.shape[0] <= 1:
        print("No hay suficientes datos de validación para construir el portfolio.")
        return

    dates = real_prices_df.index

    # 2) Retornos diarios por activo: r_t = P_t / P_{t-1} - 1
    real_ret = real_prices_df.pct_change().iloc[1:]
    pred_ret = pred_prices_df.pct_change().iloc[1:]
    dates_ret = real_ret.index

    # 3) Portfolio equiponderado = media de retornos de todas las columnas
    real_port_ret = real_ret.mean(axis=1)
    pred_port_ret = pred_ret.mean(axis=1)

    # Recorte opcional de los últimos n_points
    if n_points is not None and len(real_port_ret) > n_points:
        real_port_ret = real_port_ret[-n_points:]
        pred_port_ret = pred_port_ret[-n_points:]
        dates_ret = dates_ret[-n_points:]

    # 4) Valor acumulado del portfolio (empezando en 1.0)
    real_port_val = (1.0 + real_port_ret).cumprod()
    pred_port_val = (1.0 + pred_port_ret).cumprod()

    # 5) Gráfico
    plt.figure(figsize=(12, 5))
    plt.plot(dates_ret, real_port_val, label="Portfolio real (EW)", linewidth=1.5)
    plt.plot(dates_ret, pred_port_val, label="Portfolio predicho (EW)", linestyle="--", linewidth=1.5)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.title("Portfolio equiponderado en validación (modelos univariantes por activo)")
    plt.xlabel("Fecha")
    plt.ylabel("Valor del portfolio (normalizado a 1.0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()

    # 6) Métricas resumen
    n_days = len(real_port_ret)
    if n_days > 0:
        real_total = real_port_val.iloc[-1] - 1.0
        pred_total = pred_port_val.iloc[-1] - 1.0

        ann_factor = 252.0 / n_days
        real_ann = (1.0 + real_total) ** ann_factor - 1.0
        pred_ann = (1.0 + pred_total) ** ann_factor - 1.0

        print("=== Portfolio equiponderado (VALIDACIÓN, modelos por activo) ===")
        print(f"N días: {n_days}")
        print(f"Retorno total REAL:     {real_total: .2%}")
        print(f"Retorno total PREDICHO: {pred_total: .2%}")
        print(f"Retorno anualizado REAL:     {real_ann: .2%}")
        print(f"Retorno anualizado PREDICHO: {pred_ann: .2%}")
    else:
        print("No hay suficientes datos de validación para calcular el portfolio.")