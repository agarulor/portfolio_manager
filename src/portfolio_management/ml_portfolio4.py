import pandas as pd
import numpy as np
from typing import Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from data_management.dataset_preparation import prepare_data_ml
from tensorflow.keras.callbacks import EarlyStopping

# ============================================================
# In this section we are going to create the tools required
# to forecast the future returns / share price of the
# different shares. The aim to is to find the best
# hyperparameters that will help us to do so with a
# LSTM Neuronal Network. From here, we will try to build
# The right combination of assets for a given investor
# ============================================================

# ============================================================
# 1. We create the model for the asset
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


def build_last_window_scaled(
    series: pd.Series,
    train_end_ts: pd.Timestamp,
    window_size: int,
    scaler
) -> Optional[np.ndarray]:
    """
    Construye la última ventana escalada (window_size x 1) a partir de la serie
    y el scaler ajustado sólo con TRAIN.

    Devuelve:
      - np.ndarray (window_size, 1) si hay suficientes días.
      - None si no hay suficientes datos.
    """
    series_train = series.loc[:train_end_ts]

    if series_train.shape[0] < window_size:
        print(f"⚠ {series.name}: menos de window_size={window_size} días hasta {train_end_ts.date()}.")
        return None

    last_window = series_train.tail(window_size).values.reshape(-1, 1)  # (window_size, 1)
    last_window_scaled = scaler.transform(last_window)                  # (window_size, 1)
    return last_window_scaled
# ============================================================
# 2. WE TRAIN THE MODEL FOR A SHARE
# ============================================================

# We first create a core model (for base model)
def train_model_core(
        X_train: np.ndarray,
        y_train: np.ndarray,
        window_size: int,
        lstm_units: int,
        learning_rate: float,
        dropout_rate: float,
        optimizer_name: str,
        loss: str,
        epochs: int,
        batch_size: int,
        verbose: int,
        use_early_stopping: bool = True,
        patience: int = 10,
        min_delta: float = 0.01
        ) -> Tuple[Model, tf.keras.callbacks.History]:

        # we create the model
        model = create_lstm_model(
            window_size=window_size,
            lstm_units=lstm_units,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            optimizer_name=optimizer_name,
            loss=loss
        )

        callbacks = []

        if use_early_stopping:
            monitor_metric =  "loss"
            es = EarlyStopping(
                monitor=monitor_metric,
                patience=patience,
                min_delta=min_delta,
                restore_best_weights=True,
            )
            callbacks.append(es)

        # we fit the model with the data
        history = model.fit(
            X_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=False,
            verbose=verbose
        )

        return model, history



# ============================================================
# 2. WE TRAIN THE MODEL AND PRODUCE THE FORECAST PER ASSET
# ============================================================

def forecast_iterative(
    prices_series: pd.Series,
    model: tf.keras.Model,
    scaler: StandardScaler | MinMaxScaler,
    last_window_scaled: np.ndarray,
    train_end_ts: pd.Timestamp,
    test_date_end: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs iterative one-step-ahead forecasting for a single asset using a trained
    univariate LSTM model.

    The model forecasts only day ahead (horizon = 1). It takes the last available scaled
    window from the training period, predicts the next value, appends the prediction to
    the window, shifts it, and repeats the process

    No real data beyond train_end_ts is used: the function strictly
    uses its own previous predictions to generate future forecasts.

    Parameters
    ----------
    prices_series : pd.Series
    model : keras.Model
    scaler : sklearn scaler (StandardScaler or MinMaxScaler)
    last_window_scaled : np.ndarray of shape (window_size, 1)
    train_end_ts : pd.Timestamp
    test_date_end : str

    Returns
    -------
    forecast_dates : np.ndarray
    forecast_prices : np.ndarray
    real_future_prices : np.ndarray
    """
    # We ensure chronological order and numeric values
    series_sorted = prices_series.sort_index().astype(float)

    # Build future index mask: (train_end_ts, test_end_ts]
    test_end_ts = pd.to_datetime(test_date_end)
    full_idx = series_sorted.index

    mask_future = (full_idx > train_end_ts) & (full_idx <= test_end_ts)
    forecast_dates = full_idx[mask_future]

    if len(forecast_dates) == 0:
        print(f"{prices_series.name} does not have any future days")
        return np.array([]), np.array([]), np.array([])

    real_future_prices = series_sorted.loc[forecast_dates].to_numpy()

    # Iterative forecasting
    window_scaled = np.array(last_window_scaled, copy=True).reshape(-1, 1)
    w_size = window_scaled.shape[0]

    preds_scaled = []
    print(f"[DEBUG] {prices_series.name} – número de días a forecast:",
          len(forecast_dates))
    print(f"[DEBUG] {prices_series.name} – primeras fechas forecast:",
          forecast_dates[:5])
    for i in range(len(forecast_dates)):
        X = window_scaled.reshape(1, w_size, 1)
        y_scaled = model.predict(X, verbose=0)[0, 0]
        if i == 0:
            print(f"[DEBUG] {prices_series.name} – primera predicción escalada:",
                  y_scaled)
        preds_scaled.append(y_scaled)


        # Window and append prediction
        window_scaled = np.vstack([window_scaled[1:], [[y_scaled]]])

    # We now transform back all predictions at once
    preds_scaled_arr = np.array(preds_scaled).reshape(-1, 1)
    forecast_prices = scaler.inverse_transform(preds_scaled_arr).reshape(-1)

    return np.array(forecast_dates), forecast_prices, real_future_prices

# Function to train and validate the data and hyperarmeters
def train_asset(
    prices_series: pd.Series,
    train_date_end: str,
    val_date_end: Optional[str] = None,
    test_date_end: Optional[str] = None,
    window_size: int = 60,
    lstm_units: int = 50,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.0,
    optimizer_name: str = "rmsprop",
    loss: str = "mse",
    epochs: int = 25,
    batch_size: int = 32,
    verbose: int = 1,
    use_early_stopping: bool = True,
    patience: int = 10,
    min_delta: float = 0.01,
    forecast: bool = False,
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
    forecast : bool, default False

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
    asset_name = prices_series.name

    if forecast:
        val_date_end = train_date_end
    # We prepare data
    X_train, y_train, X_val, y_val, scaler, val_dates = prepare_data_ml(
        prices_series=prices_series,
        train_date_end=train_date_end,
        val_date_end=val_date_end,
        window_size=window_size
    )

    # We create the model and fit it
    model, history = train_model_core(
        X_train=X_train,
        y_train=y_train,
        window_size=window_size,
        lstm_units=lstm_units,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        optimizer_name=optimizer_name,
        loss=loss,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
        use_early_stopping=use_early_stopping,
        patience=patience,
        min_delta=min_delta
    )

    if not forecast:
        if len(X_val) > 0:
            # We predict the valuation
            y_val_pred = model.predict(X_val)

            # We de-scalate
            y_val_inv = scaler.inverse_transform(y_val)
            y_pred_inv = scaler.inverse_transform(y_val_pred)
        else:
            y_val_inv = np.array([])
            y_pred_inv = np.array([])

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
    train_end_ts = pd.to_datetime(train_date_end)

    last_window_scaled = build_last_window_scaled(
        series=prices_series,
        train_end_ts=train_end_ts,
        window_size=window_size,
        scaler=scaler,
    )

    print(f"[DEBUG] {asset_name} – last_window (últimos 5 precios TRAIN):",
          prices_series.loc[:train_end_ts].tail(5).values)

    print(f"[DEBUG] {asset_name} – last_window_scaled (últimos 5):",
          last_window_scaled[-5:].reshape(-1))

    if last_window_scaled is None:
        # Not enough data for a full window: no forecast possible
        return {
            "asset": asset_name,
            "model": model,
            "history": history,
            "scaler": scaler,
            "train_date_end": train_end_ts,
            "last_window_scaled": None,
            "forecast_dates": np.array([]),
            "forecast_prices": np.array([]),
            "real_future_prices": np.array([]),
        }

    forecast_dates, forecast_prices, real_future_prices = forecast_iterative(
        prices_series=prices_series,
        model=model,
        scaler=scaler,
        last_window_scaled=last_window_scaled,
        train_end_ts=train_end_ts,
        test_date_end=test_date_end)

    print(f"\n[DEBUG] {asset_name} – y_train (primeros 5):",
          y_train[:5].reshape(-1))

    print(f"[DEBUG] {asset_name} – scaler.mean_:", getattr(scaler, "mean_", None))
    print(f"[DEBUG] {asset_name} – scaler.scale_:", getattr(scaler, "scale_", None))

    return {
        "asset": asset_name,
        "model": model,
        "history": history,
        "scaler": scaler,
        "train_date_end": train_end_ts,
        "last_window_scaled": last_window_scaled,
        "forecast_dates": forecast_dates,
        "forecast_prices": forecast_prices,
        "real_future_prices": real_future_prices,
    }

# ============================================================
# 3. WE NOW PUT EVERYTHING TOGETHER BY FOR VALIDATION
#  ============================================================

def build_price_df(
    results: dict,
    forecast: bool
) -> Tuple[Optional[pd.DataFrame], pd.DataFrame]:
    """
    Builds price dfs.

    Parameters
    ----------
    results : dict. Dictionary where each key is an asset name and each value is the output
        dictionary returned
    forecast : bool. If False → validation mode. If True  → forecast mode

    Returns
    -------
    real_df : pd.DataFrame. DataFrame of real validation prices
    pred_df : pd.DataFrame. DataFrame of predicted validation prices
    """
    if not results:
        raise ValueError("Empty results dictionary.")

    # For forecast
    if forecast:
        # Use the first asset to get the forecast date index
        first_asset = list(results.keys())[0]
        forecast_dates = pd.to_datetime(results[first_asset]["forecast_dates"])

        real_df = pd.DataFrame(index=forecast_dates)
        pred_df = pd.DataFrame(index=forecast_dates)

        for asset, res in results.items():
            # Predicted (forecast) prices
            preds = np.asarray(res["forecast_prices"]).reshape(-1)
            s_pred = pd.Series(preds, index=forecast_dates, name=asset)
            pred_df[asset] = s_pred

            # Real future prices (if present)
            if "real_future_prices" in res and res["real_future_prices"] is not None:
                real_vals = np.asarray(res["real_future_prices"]).reshape(-1)
                if len(real_vals) != len(forecast_dates):
                    raise ValueError(
                        f"Length mismatch for {asset}: "
                        f"real_future_prices={len(real_vals)}, "
                        f"forecast_dates={len(forecast_dates)}"
                    )
                s_real = pd.Series(real_vals, index=forecast_dates, name=asset)
            else:
                # If not present, fill with NaN (but en tu caso deberías tenerlos)
                s_real = pd.Series(index=forecast_dates, data=np.nan, name=asset)

            real_df[asset] = s_real

        return real_df, pred_df

    # for validation
    first_asset = list(results.keys())[0]
    ref_dates = pd.to_datetime(results[first_asset]["val_dates"])

    real_df = pd.DataFrame(index=ref_dates)
    pred_df = pd.DataFrame(index=ref_dates)

    for asset, res in results.items():
        y_val_inv = np.asarray(res["y_val_inv"]).reshape(-1)
        y_pred_inv = np.asarray(res["y_pred_inv"]).reshape(-1)
        val_dates = pd.to_datetime(res["val_dates"])

        s_real = pd.Series(y_val_inv, index=val_dates, name=asset).reindex(ref_dates)
        s_pred = pd.Series(y_pred_inv, index=val_dates, name=asset).reindex(ref_dates)

        real_df[asset] = s_real
        pred_df[asset] = s_pred

    # Clean and align
    real_df = real_df.dropna(how="any")
    pred_df = pred_df.dropna(how="any")
    real_df, pred_df = real_df.align(pred_df, join="inner", axis=0)

    return real_df, pred_df


def train_lstm_all_assets(
    prices_df: pd.DataFrame,
    train_date_end: str,
    val_date_end: Optional[str] = None,
    test_date_end: Optional[str] = None,
    window_size: int = 60,
    lstm_units: int = 50,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.0,
    optimizer_name: str = "rmsprop",
    loss: str = "mse",
    epochs: int = 25,
    batch_size: int = 32,
    verbose: int = 1,
    use_early_stopping: bool = True,
    patience: int = 10,
    min_delta: float = 0.01,
    forecast: bool = False,
) -> dict:
    """
    Train LSTM models for all assets and return both the per-asset results dict
    AND two DataFrames with real and predicted prices.

    Two modes:
    1) Validation mode (forecast=False)
    2) Forecast mode (forecast=True)

    Parameters
    ----------
    prices_df : pd.DataFrame
    train_date_end : str
    val_date_end : str, optional
    test_date_end : str, optional
    window_size, lstm_units, learning_rate, dropout_rate, optimizer_name,
    loss, epochs, batch_size, verbose, forecast : See train_asset for details.

    Returns
    -------
    results : dict
    real_df : pd.DataFrame
    pred_df : pd.DataFrame
    """
    # We create the dict object
    results = {}

    if not forecast and val_date_end is None:
        raise ValueError("val_date_end must be provided when forecast=False.")
    if forecast and test_date_end is None:
        raise ValueError("test_date_end must be provided when forecast=True.")

    for col in prices_df.columns:
        print("\n==============================")
        print(f"Training LSTM model for {col} (forecast={forecast})")
        print("==============================\n")

        res_asset = train_asset(
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
            verbose=verbose,
            forecast=forecast,
            test_date_end=test_date_end,
            use_early_stopping=use_early_stopping,
            patience=patience,
            min_delta=min_delta
        )


        if res_asset is not None:
            results[col] = res_asset

        # Construimos los DataFrames según el modo
    if not forecast:
        real_df, pred_df = build_price_df(results, forecast=False)
    else:
        real_df, pred_df = build_price_df(results, True)

    return results, real_df, pred_df


