import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from data_management.dataset_preparation import prepare_datasets_ml
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD

def create_lstm_unistep_price_model(
    window_size: int,
    n_features: int,
    lstm_units: int = 64,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.0,
    optimizer_name: str = "adam",
    loss: str = "mse",
) -> Model:
    """
    Modelo LSTM uniperiodo sobre precios:
      Entrada: (window_size, n_features)  ->  60 precios anteriores (escalados)
      Salida:  (n_features)               ->  precio del día siguiente (escalado)
    """
    inputs = Input(shape=(window_size, n_features))
    x = LSTM(
        lstm_units,
        dropout=dropout_rate,
        recurrent_dropout=dropout_rate,
        return_sequences=False
    )(inputs)
    outputs = Dense(n_features)(x)
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

    model.compile(
        loss=loss,
        optimizer=optimizer,
    )
    return model


# ============================================================
# 3. ENTRENAMIENTO + VALIDACIÓN
# ============================================================

def train_lstm_with_validation_prices(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 25,
    batch_size: int = 64,
    verbose: int = 1
) -> tf.keras.callbacks.History:
    """
    Entrena la LSTM sobre precios con conjunto de entrenamiento
    y valida en el conjunto de validación.
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


def get_val_price_predictions_and_denormalize(
    model: Model,
    X_val: np.ndarray,
    y_val: np.ndarray,
    scaler: StandardScaler,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica model.predict(X_val) y desescala tanto y_val como y_pred
    usando el scaler entrenado sobre precios de TRAIN.

    Devuelve:
      y_val_inv  : precios reales (validación)
      y_pred_inv : precios predichos (validación)
    """
    y_pred = model.predict(X_val)
    y_val_inv  = scaler.inverse_transform(y_val)
    y_pred_inv = scaler.inverse_transform(y_pred)
    return y_val_inv, y_pred_inv


# ============================================================
# 4. GRÁFICO DE FINETUNING EN PRECIOS (VALIDACIÓN)
# ============================================================

def plot_finetuning_prices_on_validation_prices(
    prices_df: pd.DataFrame,
    y_val_inv: np.ndarray,
    y_pred_inv: np.ndarray,
    val_dates: np.ndarray,
    asset_col: Optional[str] = None,
    asset_idx: Optional[int] = None,
    n_points: Optional[int] = 200,
):
    """
    Grafica precios reales vs precios predichos sobre el CONJUNTO DE VALIDACIÓN
    para una acción concreta, usando precios desescalados.

    Parameters
    ----------
    prices_df : DataFrame
        DataFrame de precios originales (solo lo usamos para los nombres de columnas).
    y_val_inv : np.ndarray, shape (n_val_samples, n_features)
        Precios reales (desescalados) de validación.
    y_pred_inv : np.ndarray, shape (n_val_samples, n_features)
        Precios predichos (desescalados) de validación.
    val_dates : np.ndarray of datetime-like
        Fechas de cada predicción/real de validación.
    asset_col : str, opcional
        Nombre de la acción (columna de prices_df). Si se pasa, se ignora asset_idx.
    asset_idx : int, opcional
        Índice de la acción. Si no se pasa nada, se usa la primera columna.
    n_points : int, opcional
        Si no es None, se limita a los últimos n_points puntos de validación.
    """

    # Determinar acción
    if asset_col is not None:
        asset_idx = list(prices_df.columns).index(asset_col)
    elif asset_idx is None:
        asset_idx = 0
        asset_col = prices_df.columns[asset_idx]
    else:
        asset_col = prices_df.columns[asset_idx]

    # Extraemos precios reales y predichos de validación para esa acción
    real_prices_val = y_val_inv[:, asset_idx]
    pred_prices_val = y_pred_inv[:, asset_idx]

    # Limitamos a últimos n_points si se pide
    if n_points is not None and len(real_prices_val) > n_points:
        real_prices_val = real_prices_val[-n_points:]
        pred_prices_val = pred_prices_val[-n_points:]
        val_dates_slice = val_dates[-n_points:]
    else:
        val_dates_slice = val_dates

    # Gráfico
    plt.figure(figsize=(12, 5))
    plt.plot(val_dates_slice, real_prices_val, label="Precio real (validación)", linewidth=1.5)
    plt.plot(val_dates_slice, pred_prices_val, label="Precio predicho (validación)", linestyle="--", linewidth=1.5)
    plt.title(f"Fine-tuning LSTM (uniperiodo, PRECIOS) - {asset_col}")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


# ============================================================
# 5. PIPELINE COMPLETO: FINETUNING SOBRE PRECIOS + GRÁFICO
# ============================================================

def finetuning_unistep_on_prices_and_plot(
    prices_df: pd.DataFrame,
    train_date_end: str,
    val_date_end: str,
    window_size: int = 60,
    lstm_units: int = 128,
    learning_rate: float = 0.001,
    dropout_rate: float = 0.0,
    optimizer_name: str = "rmsprop",
    loss: str = "mse",
    epochs: int = 25,
    batch_size: int = 32,
    verbose: int = 1,
    asset_col: Optional[str] = None,
    asset_idx: Optional[int] = None,
    n_points: Optional[int] = 200,
) -> Dict[str, Any]:
    """
    Hace
    sobre PRECIOS (normalizados):

    1) Divide precios en TRAIN / VALIDACIÓN por fechas.
    2) Escala solo con TRAIN (StandardScaler).
    3) Construye ventanas:
       - X: 60 precios anteriores (escalados)
       - y: precio del día siguiente (escalado)
    4) Entrena LSTM uniperiodo en TRAIN y calibra con VALIDACIÓN.
    5) Obtiene predicciones en VALIDACIÓN, desescala a precios reales.
    6) Grafica precio real vs precio predicho en VALIDACIÓN.

    Devuelve un dict con modelo, history, datos de validación, etc.
    """

    # 1) Preparar train + validación (sobre precios)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, val_dates, y_test_index = prepare_datasets_ml(
        returns=prices_df,
        train_date_end=train_date_end,
        val_date_end=val_date_end,
        window_size=window_size
    )

    n_features = X_train.shape[2]

    # 2) Crear modelo
    model = create_lstm_unistep_price_model(
        window_size=window_size,
        n_features=n_features,
        lstm_units=lstm_units,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        optimizer_name=optimizer_name,
        loss=loss
    )

    # 3) Entrenar con validación
    history = train_lstm_with_validation_prices(
        model,
        X_train, y_train,
        X_val,   y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    # 4) Predicciones y desescalado en VALIDACIÓN
    y_val_inv, y_pred_inv = get_val_price_predictions_and_denormalize(
        model=model,
        X_val=X_val,
        y_val=y_val,
        scaler=scaler
    )

    # 5) Gráfico de precios en VALIDACIÓN
    plot_finetuning_prices_on_validation_prices(
        prices_df=prices_df,
        y_val_inv=y_val_inv,
        y_pred_inv=y_pred_inv,
        val_dates=val_dates,
        asset_col=asset_col,
        asset_idx=asset_idx,
        n_points=n_points
    )

    return {
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
