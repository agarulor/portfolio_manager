import pandas as pd
import numpy as np
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
        verbose: int) -> Tuple[Model, tf.keras.callbacks.History]:

        # we create the model
        model = create_lstm_model(
            window_size=window_size,
            lstm_units=lstm_units,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            optimizer_name=optimizer_name,
            loss=loss
        )

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
# 2. WE TRAIN THE MODEL FOR A SHARE AND FOR VALIDATION
# ============================================================

# Function to train and validate the data and hyperarmeters
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
# 3. WE NOW PUT EVERYTHING TOGETHER BY FOR VALIDATION
#  ============================================================

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


