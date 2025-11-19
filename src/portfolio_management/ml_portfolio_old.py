import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.optimizers import Adam, RMSprop, SGD


# ============================================================
# 1. PREPARACIÓN DE DATOS SOBRE PRECIOS (TRAIN + VALIDACIÓN)
#    -> VENTANAS SOBRE TODA LA SERIE, SPLIT POR FECHA DEL TARGET
# ============================================================

def prepare_train_val_unistep_prices(
    prices_df: pd.DataFrame,
    train_date_end: str,
    val_date_end: str,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray,
           np.ndarray, np.ndarray,
           StandardScaler,
           np.ndarray]:
    """
    Prepara X_train, y_train, X_val, y_val para una LSTM UNIPERIODO
    (horizonte 1) usando Rolling Window **sobre PRECIOS**, de forma que:

      - Las ventanas se crean sobre
      - Cada ejemplo (ventana -> target) se asigna a TRAIN o VALIDACIÓN
        según la FECHA del target (día que se predice).

    Esto garantiza que:

      * La PRIMERA predicción de validación (primer día > train_date_end)
        usa los `window_size` días anteriores, aunque estén en TRAIN.

    Parameters
    ----------
    prices_df : DataFrame
        DataFrame de precios (index = fechas, columns = acciones).
    train_date_end : str
        Último día incluido en TRAIN (YYYY-MM-DD).
    val_date_end : str
        Último día incluido en VALIDACIÓN (YYYY-MM-DD).
    window_size : int
        Número de días de lookback (ej: 60, 120).

    Returns
    -------
    X_train, y_train, X_val, y_val, scaler, val_dates
    """

    # Aseguramos orden temporal
    prices_df = prices_df.sort_index()

    # Fechas de corte
    train_end = pd.to_datetime(train_date_end)
    val_end = pd.to_datetime(val_date_end)

    dates = prices_df.index.to_numpy()
    data = prices_df.to_numpy()   # (n_days, n_features)
    n_days, n_features = data.shape

    # --------------------
    # Escalado SOLO con TRAIN (sobre precios)
    # --------------------
    mask_train_scaler = dates <= train_end
    data_train_for_scaler = data[mask_train_scaler]

    scaler = StandardScaler()
    scaler.fit(data_train_for_scaler)

    data_scaled = scaler.transform(data)

    # ---------------------------------------
    # Rolling Window sobre TODA la serie
    # ---------------------------------------
    X_train_list, y_train_list = [], []
    X_val_list, y_val_list = [], []
    val_dates_list = []

    for t in range(window_size, n_days):
        # Ventana: precios escalados de los window_size días previos
        X_window = data_scaled[t - window_size:t, :]   # (window_size, n_features)
        # Target: precio escalado del día t
        y_t = data_scaled[t, :]                        # (n_features,)
        date_t = dates[t]                              # fecha del target

        if date_t <= train_end:
            X_train_list.append(X_window)
            y_train_list.append(y_t)
        elif date_t <= val_end:
            X_val_list.append(X_window)
            y_val_list.append(y_t)
            val_dates_list.append(date_t)
        else:
            # Más allá de val_end -> futuro o test (no se usa aquí)
            pass

    def _stack_xy(X_list, y_list):
        if len(X_list) == 0:
            return (np.empty((0, window_size, n_features)),
                    np.empty((0, n_features)))
        return np.stack(X_list, axis=0), np.stack(y_list, axis=0)

    X_train, y_train = _stack_xy(X_train_list, y_train_list)
    X_val, y_val = _stack_xy(X_val_list, y_val_list)
    val_dates = np.array(val_dates_list)

    print(f"[DATA-PRICES] X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"[DATA-PRICES] X_val:   {X_val.shape},   y_val:   {y_val.shape}")
    if len(val_dates) > 0:
        print(f"[DATA-PRICES] Validación: {val_dates[0]} -> {val_dates[-1]} "
              f"({len(val_dates)} días de target)")

    return X_train, y_train, X_val, y_val, scaler, val_dates


# ============================================================
# 2. MODELO LSTM UNIPERIODO SOBRE PRECIOS
# ============================================================

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
      Entrada: (window_size, n_features)  ->  window_size precios anteriores (escalados)
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

    # Permitir usar Huber por nombre
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
        shuffle=False,
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
    y_val_inv = scaler.inverse_transform(y_val)
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
    """

    # Determinar acción
    if asset_col is not None:
        asset_idx = list(prices_df.columns).index(asset_col)
    elif asset_idx is None:
        asset_idx = 0
        asset_col = prices_df.columns[asset_idx]
    else:
        asset_col = prices_df.columns[asset_idx]

    real_prices_val = np.asarray(y_val_inv)[:, asset_idx]
    pred_prices_val = np.asarray(y_pred_inv)[:, asset_idx]
    val_dates = np.array(val_dates)

    # Alineamos longitudes por seguridad
    n = min(len(real_prices_val), len(pred_prices_val), len(val_dates))
    real_prices_val = real_prices_val[:n]
    pred_prices_val = pred_prices_val[:n]
    val_dates = val_dates[:n]

    # Limitamos a últimos n_points si se pide
    if n_points is not None and n > n_points:
        real_prices_val = real_prices_val[-n_points:]
        pred_prices_val = pred_prices_val[-n_points:]
        val_dates_slice = val_dates[-n_points:]
    else:
        val_dates_slice = val_dates

    if len(val_dates_slice) == 0:
        print("⚠ No hay datos de validación para graficar.")
        return

    print("=== RANGO QUE SE ESTÁ GRAFICANDO (VALIDACIÓN) ===")
    print("Primera fecha en gráfico:", val_dates_slice[0])
    print("Última fecha en gráfico: ", val_dates_slice[-1])
    print("Número de puntos:", len(val_dates_slice))

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
    plt.gcf().autofmt_xdate()
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
    Fine-tuning uniperiodo trabajando DIRECTAMENTE sobre PRECIOS (normalizados):

    1) Divide precios en TRAIN / VALIDACIÓN por fechas (según la fecha del target).
    2) Escala solo con TRAIN (StandardScaler).
    3) Construye ventanas sobre :
       - X: window_size precios anteriores (escalados)
       - y: precio del día siguiente (escalado)
    4) Entrena LSTM uniperiodo en TRAIN y calibra con VALIDACIÓN.
    5) Obtiene predicciones en VALIDACIÓN, desescala a precios reales.
    6) Grafica precio real vs precio predicho en VALIDACIÓN para una acción.

    Devuelve un dict con modelo, history, datos de validación, etc.
    """

    # 1) Preparar train + validación (sobre precios) CON VENTANAS SOBRE
    X_train, y_train, X_val, y_val, scaler, val_dates = prepare_train_val_unistep_prices(
        prices_df=prices_df,
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
        X_val, y_val,
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


# ============================================================
# 6. PORTFOLIO EQUIPONDERADO EN VALIDACIÓN
# ============================================================

def plot_equal_weight_portfolio_on_validation(
    y_val_inv: np.ndarray,
    y_pred_inv: np.ndarray,
    val_dates: np.ndarray,
    n_points: Optional[int] = 200,
):
    """
    Compara la evolución de un portafolio equiponderado (igual peso en todas
    las acciones) usando PRECIOS de validación reales vs predichos.
    """

    # Aseguramos numpy arrays
    val_dates = np.array(val_dates)
    real_prices = np.asarray(y_val_inv)
    pred_prices = np.asarray(y_pred_inv)

    # Alineamos por seguridad
    n = min(len(val_dates), real_prices.shape[0], pred_prices.shape[0])
    val_dates = val_dates[-n:]
    real_prices = real_prices[-n:, :]
    pred_prices = pred_prices[-n:, :]

    # Retornos diarios por activo: r_t = P_t / P_{t-1} - 1
    real_ret = real_prices[1:, :] / real_prices[:-1, :] - 1.0
    pred_ret = pred_prices[1:, :] / pred_prices[:-1, :] - 1.0
    dates_ret = val_dates[1:]

    # Portafolio equiponderado = media de retornos de todas las columnas
    real_port_ret = real_ret.mean(axis=1)
    pred_port_ret = pred_ret.mean(axis=1)

    # Recorte final si se pide
    if n_points is not None and len(real_port_ret) > n_points:
        real_port_ret = real_port_ret[-n_points:]
        pred_port_ret = pred_port_ret[-n_points:]
        dates_ret = dates_ret[-n_points:]

    # Valor acumulado del portafolio (empezando en 1.0)
    real_port_val = (1.0 + real_port_ret).cumprod()
    pred_port_val = (1.0 + pred_port_ret).cumprod()

    # Gráfico
    plt.figure(figsize=(12, 5))
    plt.plot(dates_ret, real_port_val, label="Portfolio real (EW)", linewidth=1.5)
    plt.plot(dates_ret, pred_port_val, label="Portfolio predicho (EW)", linestyle="--", linewidth=1.5)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.title("Evolución del portfolio equiponderado (validación)\nReal vs Predicho")
    plt.xlabel("Fecha")
    plt.ylabel("Valor del portfolio (normalizado a 1.0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Métricas
    n_days = len(real_port_ret)
    if n_days > 0:
        real_total = real_port_val[-1] - 1.0
        pred_total = pred_port_val[-1] - 1.0

        ann_factor = 252.0 / n_days
        real_ann = (1.0 + real_total) ** ann_factor - 1.0
        pred_ann = (1.0 + pred_total) ** ann_factor - 1.0

        print("=== Portfolio equiponderado (VALIDACIÓN) ===")
        print(f"N días: {n_days}")
        print(f"Retorno total REAL:     {real_total: .2%}")
        print(f"Retorno total PREDICHO: {pred_total: .2%}")
        print(f"Retorno anualizado REAL:     {real_ann: .2%}")
        print(f"Retorno anualizado PREDICHO: {pred_ann: .2%}")
    else:
        print("No hay suficientes datos de validación para calcular el portfolio.")
