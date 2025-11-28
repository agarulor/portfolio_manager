import pandas as pd
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD
from data_management.dataset_preparation import prepare_data_ml

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

# ============================================================
# 2. WE TRAIN THE MODEL FOR A SHARE
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
# 3. WE NOW PUT EVERYTHING TOGETHER BY
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
# 4. PLOT VALIDATION FOR A SHARE (TO CHECK PERFORMANCE)
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
    ValueError if the requested asset is not found in the results dictionary.

    Returns
    -------
    None : It only displays the plot.
    """
    # Check if the asset wit want to plot is with the results
    if asset not in results:
        raise ValueError(f"{asset} is not in results (keys: {list(results.keys())})")

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

def validation_price_matrices_from_results(
    results: dict
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Builds price matrices (real and predicted) from a results dictionary containing multiple assets.

    Parameters
    ----------
    results : dict. Dictionary where each key is an asset name and each value is the output
        dictionary returned

    Returns
    -------
    real_df : pd.DataFrame. DataFrame of real validation prices
    pred_df : pd.DataFrame. DataFrame of predicted validation prices
    """
    # We check if the dictionary is not empty
    if not results:
        raise ValueError("The dictionary is empty")

    # We take an asset to extract the dates
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

        # We re-index to the reference dates (just in case)
        s_real = s_real.reindex(ref_dates)
        s_pred = s_pred.reindex(ref_dates)

        real_df[asset] = s_real
        pred_df[asset] = s_pred

    # Remove rows with NAs
    real_df = real_df.dropna(how="any")
    pred_df = pred_df.dropna(how="any")

    # We align, just in case
    real_df, pred_df = real_df.align(pred_df, join="inner", axis=0)

    return real_df, pred_df


def plot_equal_weight_buy_and_hold_from_results(
    results: dict,
    n_points: Optional[int] = 200,
):
    """
    Plots an equally weighted  portfolio over the validation period with the
    real and predicted prices

    Parameters
    ----------
    results : dict. Dictionary where each key is an asset name and each value is the output
        dictionary returned
            - "y_pred_inv"  : inverse-scaled predicted validation prices

    n_points : int, optional (default=200). Maximum number of most recent validation points to display.

    Returns
    -------
    None : It only displays the plot.
    """

    # We get the real and predicted prices for each asset
    real_prices_df, pred_prices_df = validation_price_matrices_from_results(results)

    if real_prices_df.shape[0] <= 1:
        print("There is not enough validation data to build the portfolio")
        return

    print("Range for the theoretical portfolio:")
    print("  From:", real_prices_df.index[0])
    print("  Until:", real_prices_df.index[-1])
    print("  Number of days:", real_prices_df.shape[0])

    # dates
    dates = real_prices_df.index.to_numpy()

    # Final adjustement
    if n_points is not None and len(dates) > n_points:
        dates = dates[-n_points:]
        real_prices_df = real_prices_df.iloc[-n_points:, :]
        pred_prices_df = pred_prices_df.iloc[-n_points:, :]


    # Normalization of prices by the initial value of each asset
    # Each assets starts at 1 on the first day of validation
    real_norm = real_prices_df / real_prices_df.iloc[0]
    pred_norm = pred_prices_df / pred_prices_df.iloc[0]

    # Equally weighted portfolio, each asset has the same weight
    real_port_val = real_norm.mean(axis=1)
    pred_port_val = pred_norm.mean(axis=1)

    # moving to the plotting section
    plt.figure(figsize=(12, 5))
    plt.plot(real_port_val.index, real_port_val.values,
             label="Actual portfolio equally weighted", linewidth=1.5)
    plt.plot(pred_port_val.index, pred_port_val.values,
             label="Forecasted Portfolio equally weighted", linestyle="--", linewidth=1.5)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.title("Equally weighted porfolio actual / vs. forecast \n Validation data")
    plt.xlabel("Data")
    plt.ylabel("Portfolio value (normalized at 1.0 at the beginning)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()

    # Key metrics (for comparation purposes)
    n_days = len(real_port_val) - 1
    if n_days > 0:
        real_total = real_port_val.iloc[-1] - 1.0
        pred_total = pred_port_val.iloc[-1] - 1.0

        ann_factor = 252.0 / n_days
        real_ann = (1.0 + real_total) ** ann_factor - 1.0
        pred_ann = (1.0 + pred_total) ** ann_factor - 1.0

        print("=== Equally weighted portfolio (VALIDATION DATA - 1 model per asset) ===")
        print(f"Number of days: {n_days}")
        print(f"Total actual return:     {real_total: .2%}")
        print(f"Total forecasted return: {pred_total: .2%}")
        print(f"Annualized return - actual:     {real_ann: .2%}")
        print(f"Annualized return - forecasted: {pred_ann: .2%}")
    else:
        print("No hay suficientes días para calcular rentabilidades.")