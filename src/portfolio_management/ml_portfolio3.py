import numpy as np
import pandas as pd
from typing import Optional, Tuple

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler, StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD


# ============================================================
# In this section we are going to create the tools required
# to forecast the future returns / share price of the
# different shares. The aim to is to find the best
# hyperparameters that will help us to do so with a
# LSTM Neuronal Network. From here, we will try to build
# The right combination of assets for a given investor
# ============================================================

# ============================================================
# 1. We prepare the data for a single share (univariate)
# ============================================================

def stack_xy(
        X_list: list,
        y_list: list,
        window_size: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Stack lists of X and y arrays.

    Parameters
    ----------
    X_list : list of np.ndarray.
    y_list : list of np.ndarray
    window_size : int

    Returns
    -------
    X : np.ndarray stacked
    y : np.ndarray stacked
    """
    if X_list:
        X = np.stack(X_list, axis=0)
        y = np.stack(y_list, axis=0)
    else:
        X = np.empty((0, window_size, 1))
        y = np.empty((0, 1))
    return X, y


def prepare_data_ml(
    prices_series: pd.Series,
    train_date_end: str,
    val_date_end: str,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           MinMaxScaler,
           np.ndarray]:
    """
    Prepare rolling-window training and validation datasets for a single asset,
    using horizon = 1 forecasting (i.e. T + 1 days).

    This function
    1. Splits the timeline into Train and Validation based on the target date.
    2. Fits a StandardScaler **only on the train portion** to avoid data leakage.
    3. Applies a rolling window of size `window_size` to create supervised
       learning samples
    4. Returns stacked arrays for TRAIN and VAL sets, the  scaler and
       the validation target dates.

    Parameters
    ----------
    prices_series : pd.Series
    train_date_end : str
    val_date_end : str
    window_size : int

    Returns
    -------
    X_train : np.ndarray
    y_train : np.ndarray
    X_val : np.ndarray
    y_val : np.ndarray
    scaler : StandardScaler
    val_dates : np.ndarray
    Notes
    """

    # We check that the order is correct
    prices_series = prices_series.sort_index().astype(float)

    # We include the cut-off dates
    train_end = pd.to_datetime(train_date_end)
    val_end = pd.to_datetime(val_date_end)

    # We get the dates and the number of days
    dates = prices_series.index.to_numpy()
    data = prices_series.to_numpy().reshape(-1, 1)
    n_days = data.shape[0]

    # We escalate the data, with only on the train part, to avoid leakages
    mask_train_scaler = dates <= train_end
    data_train_for_scaler = data[mask_train_scaler]

    if data_train_for_scaler.shape[0] == 0:
        raise ValueError("There is no data available for train_date_end to adjust the scaler.")

    # Now we can scalate
    scaler = StandardScaler()

    # We fit the data
    scaler.fit(data_train_for_scaler)

    # Now we transform the data
    data_scaled = scaler.transform(data)

    # Rolling Window
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    val_dates_list = []


    for t in range(window_size, n_days):
        X_window = data_scaled[t - window_size:t, :]
        y_t = data_scaled[t, :]
        date_t = dates[t]

        if date_t <= train_end:
            X_train_list.append(X_window)
            y_train_list.append(y_t)
        elif date_t <= val_end:
            X_val_list.append(X_window)
            y_val_list.append(y_t)
            val_dates_list.append(date_t)
        else:
            #If it is higher than the date, we pass (no used here)
            pass

    # We now make a stack of the data for train and val
    X_train, y_train = stack_xy(X_train_list, y_train_list, window_size)
    X_val, y_val = stack_xy(X_val_list, y_val_list, window_size)
    # We get val dates
    val_dates = np.array(val_dates_list)

    # We provide visual information
    print(f"[{prices_series.name}] X_train: {X_train.shape}, X_val: {X_val.shape}")
    if len(val_dates) > 0:
        print(f"[{prices_series.name}] Validation: {val_dates[0]} -> {val_dates[-1]} "
              f"({len(val_dates)} target days)")

    # Finally we return relevant information, including the scaler for de-scaling the data
    return X_train, y_train, X_val, y_val, scaler, val_dates


# ============================================================
# 2. We create the model for the asset
# ============================================================

def create_lstm_model(
    window_size: int,
    lstm_units: int = 50,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.0,
    optimizer_name: str = "rmsprop",
    loss: str = "mse",
) -> Model:
    """
    Create and compile a univariate LSTM model

    Parameters
    ----------
    window_size : int
    lstm_units : int, default 50
    learning_rate : float, default 0.001
    dropout_rate : float, default 0.0
    optimizer_name : {"adam", "rmsprop", "sgd"}, default "rmsprop"
    loss : {"mse", "huber", ...}, default "mse"

    Returns
    -------
    model : keras.Model
    """
    # We first create the inputs
    inputs = Input(shape=(window_size, 1))
    # We create the model
    x = LSTM(
        lstm_units,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        return_sequences=False
    )(inputs)
    outputs = Dense(1)(x)
    model = Model(inputs=inputs, outputs=outputs)

    # We define the optimizer
    opt_name = optimizer_name.lower()
    if opt_name == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif opt_name == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    elif opt_name == "sgd":
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Non-valid Optimizer: {optimizer_name}")

    # In case we opt for a huber loss
    if loss == "huber":
        loss_fn = tf.keras.losses.Huber(delta=1.0)
    else:
        loss_fn = loss

   # We compile the model
    model.compile(
        loss=loss_fn,
        optimizer=optimizer,
    )
    return model

# ============================================================
# 3. WE TRAIN THE MODEL FOR A SHARE
# ============================================================

def train_and_validate_asset(
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
) -> dict:
    """
    Train and validate a LSTM model

    Parameters
    ----------
    prices_series : pd.Series
    train_date_end : str
    val_date_end : str
    window_size : int, default 60
    lstm_units : int, default 50
    learning_rate : float, default 0.001
    dropout_rate : float, default 0.0
    optimizer_name : {"adam", "rmsprop", "sgd"}, default "rmsprop"
    loss : {"mse", "huber", ...}, default "mse"
    epochs : int, default 25
    batch_size : int, default 32
    verbose : int, default 1

    Returns
    -------
    results : dict
     containing:
        - **asset** : str
        - **model** : keras.Model
        - **history** : keras.callbacks.History
        - **X_train**, **y_train** : np.ndarray
        - **X_val**, **y_val** : np.ndarray
        - **y_val_inv** : np.ndarray
        - **y_pred_inv** : np.ndarray
        - **val_dates** : np.ndarray
        - **scaler** : StandardScaler
    """

    # We prepare data
    X_train, y_train, X_val, y_val, scaler, val_dates = prepare_data_ml(
        prices_series=prices_series,
        train_date_end=train_date_end,
        val_date_end=val_date_end,
        window_size=window_size
    )

    # We create the model
    model = create_lstm_model(
        window_size=window_size,
        lstm_units=lstm_units,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        optimizer_name=optimizer_name,
        loss=loss
    )

    # We do the fit
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        shuffle=False,
        verbose=verbose
    )

    # We predict the valuation
    y_val_pred = model.predict(X_val)

    # We de-scalate
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
        "y_val_inv": y_val_inv,
        "y_pred_inv": y_pred_inv,
        "val_dates": val_dates,
        "scaler": scaler
    }


# ============================================================
# 4. WE NOW PUT EVERYTHING TOGETHER BY
# TRAINING ALL THE SHARES UNDER ONE FUNCTION
# ============================================================

def train_lstm_all_assets(
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
) -> dict:
    """
    Train and validate LSTM model for each of the shares under one function

    Parameters
    ----------
    prices_df : pd.DataFrame
    train_date_end : str
    val_date_end : str
    window_size : int, default 60
    lstm_units : int, default 50
    learning_rate : float, default 0.001
    dropout_rate : float, default 0.0
    optimizer_name : {"adam", "rmsprop", "sgd"}, default "rmsprop"
    loss : {"mse", "huber", ...}, default "mse"
    epochs : int, default 25
    batch_size : int, default 32
    verbose : int, default 1

    Returns
    -------
    results : dict
        the key is the name of the share and the value is a dictionary containing:
            - **asset** : str
            - **model** : keras.Model
            - **history** : keras.callbacks.History
            - **X_train**, **y_train** : np.ndarray
            - **X_val**, **y_val** : np.ndarray
            - **y_val_inv** : np.ndarray
            - **y_pred_inv** : np.ndarray
            - **val_dates** : np.ndarray
            - **scaler** : StandardScaler
    """
    # We create the dict object
    results = {}

    for col in prices_df.columns:
        print(f"\n==============================")
        print(f"Training LSTM model for {col} share")
        print(f"==============================\n")

        res_asset = train_and_validate_asset(
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

        # We include both, key and value in the dictionary
        results[col] = res_asset

    return results


# ============================================================
# 5. PLOT VALIDATION FOR A SHARE (TO CHECK PERFORMANCE)
#   IN A VISUAL FASHION
# ============================================================

def plot_validation(
    results: dict,
    asset: str,
    n_points: Optional[int] = 200
):
    """
    Plot actual vs predicted validation prices for a given asset.

    Parameters
    ----------
    results : dict. Dictionary where each key is an asset ticker and each value contains:
    asset : str. The asset/ticker to plot.
    n_points : int, optional (default=200). Maximum number of most recent validation points to display.

    Raises
    ------
    ValueError
        If the requested asset is not found in the results dictionary.

    Returns
    -------
    None : It only displays the plot.

    """

    # Check if the asset wit want to plot is with the results
    if asset not in results:
        raise ValueError(f"{asset} no está en results (keys: {list(results.keys())})")

    # we extract the results from the asset
    res = results[asset]
    # We extract the values predicted and the real one (for the val)
    y_val_inv = res["y_val_inv"].reshape(-1)
    y_pred_inv = res["y_pred_inv"].reshape(-1)
    val_dates = res["val_dates"]

    # We align information
    n = min(len(y_val_inv), len(y_pred_inv), len(val_dates))
    y_val_inv = y_val_inv[:n]
    y_pred_inv = y_pred_inv[:n]
    val_dates = val_dates[:n]

   # We plot only the last n_points (to avoid showing too much info on the screen)
    if n_points is not None and n > n_points:
        y_val_inv = y_val_inv[-n_points:]
        y_pred_inv = y_pred_inv[-n_points:]
        val_dates = val_dates[-n_points:]

    # We check that there is validation data
    if len(val_dates) == 0:
        print(f"There is no validation data for {asset}")
        return

    print(f"[{asset}] validation from {val_dates[0]} until {val_dates[-1]} (N={len(val_dates)})")

    # We create and show the plot
    plt.figure(figsize=(12, 5))
    plt.plot(val_dates, y_val_inv, label="Actual share price (validation)", linewidth=1.5)
    plt.plot(val_dates, y_pred_inv, label="Forecast Price (validation)", linestyle="--", linewidth=1.5)
    plt.title(f"Forecasted vs. actual values - {asset}")
    plt.xlabel("Date")
    plt.ylabel("Share price (EUR)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()

def build_validation_price_matrices_from_results(
    results: dict
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
    ref_dates = pd.to_datetime(results[first_asset]["val_dates"])
    real_df = pd.DataFrame(index=ref_dates)
    pred_df = pd.DataFrame(index=ref_dates)

    for asset, res in results.items():
        y_val_inv = res["y_val_inv"].reshape(-1)
        y_pred_inv = res["y_pred_inv"].reshape(-1)
        val_dates = pd.to_datetime(res["val_dates"])

        s_real = pd.Series(y_val_inv, index=val_dates, name=asset)
        s_pred = pd.Series(y_pred_inv, index=val_dates, name=asset)

        # reindexamos a las fechas de referencia (por si difieren algún día)
        s_real = s_real.reindex(ref_dates)
        s_pred = s_pred.reindex(ref_dates)

        real_df[asset] = s_real
        pred_df[asset] = s_pred

    # Quitamos filas con NaNs en alguno de los activos
    real_df = real_df.dropna(how="any")
    pred_df = pred_df.dropna(how="any")

    # Alineamos por seguridad
    real_df, pred_df = real_df.align(pred_df, join="inner", axis=0)

    return real_df, pred_df

def plot_equal_weight_buy_and_hold_from_results(
    results: dict,
    n_points: Optional[int] = 200,
):
    """
    Portfolio equiponderado BUY & HOLD en validación, a partir de `results`
    (un modelo univariante por activo, entrenado con train_lstm_unistep_all_assets_separately).

    Supone que al inicio del periodo de validación se invierte el mismo capital
    en cada activo (1/N), y luego se dejan correr los precios sin rebalancear.
    """

    real_prices_df, pred_prices_df = build_validation_price_matrices_from_results(results)

    if real_prices_df.shape[0] <= 1:
        print("No hay suficientes datos de validación para construir el portfolio.")
        return

    print("Rango usado para el portfolio BUY & HOLD:")
    print("  Desde:", real_prices_df.index[0])
    print("  Hasta:", real_prices_df.index[-1])
    print("  Nº días:", real_prices_df.shape[0])

    # Fechas
    dates = real_prices_df.index.to_numpy()

    # Recorte final si se pide (últimos n_points días)
    if n_points is not None and len(dates) > n_points:
        dates = dates[-n_points:]
        real_prices_df = real_prices_df.iloc[-n_points:, :]
        pred_prices_df = pred_prices_df.iloc[-n_points:, :]

    # ============================
    # BUY & HOLD EQUIPONDERADO
    # ============================
    # Normalizamos precios por el valor inicial de cada activo
    # -> cada activo empieza en 1.0 el primer día de validación
    real_norm = real_prices_df / real_prices_df.iloc[0]
    pred_norm = pred_prices_df / pred_prices_df.iloc[0]

    # Portfolio equiponderado = media de los valores normalizados (mismo peso inicial)
    real_port_val = real_norm.mean(axis=1)   # Series indexada por fecha
    pred_port_val = pred_norm.mean(axis=1)

    # ============================
    # Gráfico
    # ============================
    plt.figure(figsize=(12, 5))
    plt.plot(real_port_val.index, real_port_val.values,
             label="Portfolio REAL (EW buy&hold)", linewidth=1.5)
    plt.plot(pred_port_val.index, pred_port_val.values,
             label="Portfolio PREDICHO (EW buy&hold)", linestyle="--", linewidth=1.5)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.title("Portfolio equiponderado BUY & HOLD (validación)\nModelos univariantes por activo")
    plt.xlabel("Fecha")
    plt.ylabel("Valor del portfolio (normalizado a 1.0 al inicio)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()

    # ============================
    # Métricas: retorno total y anualizado
    # ============================
    n_days = len(real_port_val) - 1  # días efectivos de inversión
    if n_days > 0:
        real_total = real_port_val.iloc[-1] - 1.0
        pred_total = pred_port_val.iloc[-1] - 1.0

        ann_factor = 252.0 / n_days
        real_ann = (1.0 + real_total) ** ann_factor - 1.0
        pred_ann = (1.0 + pred_total) ** ann_factor - 1.0

        print("=== Portfolio equiponderado BUY & HOLD (VALIDACIÓN, modelos por activo) ===")
        print(f"N días: {n_days}")
        print(f"Retorno total REAL:     {real_total: .2%}")
        print(f"Retorno total PREDICHO: {pred_total: .2%}")
        print(f"Retorno anualizado REAL:     {real_ann: .2%}")
        print(f"Retorno anualizado PREDICHO: {pred_ann: .2%}")
    else:
        print("No hay suficientes días para calcular rentabilidades.")