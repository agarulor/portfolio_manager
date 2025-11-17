import tensorflow as tf
import pandas as pd
import numpy as np
import itertools
from keras.src.optimizers import Adam, RMSprop, SGD, AdamW

from data_management.dataset_preparation import prepare_datasets_ml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from typing import Dict, Any, Tuple, List



def create_lstm_model(window_size: int,
                      n_features: int,
                      lstm_units: int = 64,
                      learning_rate: float = 0.001,
                      dropout_rate: float = 0.2,
                      optimizer_name: str = "adam",
                      loss = "mse") -> tf.keras.models.Sequential:
    # We define the model
    inputs = tf.keras.Input(shape=(window_size, n_features))
    x = LSTM(lstm_units,
             dropout=dropout_rate,
             recurrent_dropout=dropout_rate,
             return_sequences=False)(inputs)

    outputs = tf.keras.layers.Dense(n_features)(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # We create the model

    if optimizer_name.lower() == "adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == "adamw":
        optimizer = AdamW(learning_rate=learning_rate, weight_decay=1e-4)
    elif optimizer_name.lower() == "rmsprop":
        optimizer = RMSprop(learning_rate=learning_rate)
    elif optimizer_name.lower() == "sgd":
        optimizer = SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError("Please, enter a valid optimizer")

    model.compile(
        loss=loss,
        optimizer=optimizer,
    )


    return model

def train_lstm_model(model:tf.keras.models.Sequential,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     X_val: pd.DataFrame,
                     y_val: pd.Series,
                     epochs: int = 25,
                     batch_size: int = 64,
                     verbose: int = 1)\
        -> tf.keras.callbacks.History:
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose)

    return history

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
                   verbose: int = 1):
        """
        Normalizes data based on the train DataSet
        it creates rolling windows

        Parameters
        ----------
        returns : pd.DataFrame. Dataset with returns.
        train_date_end : str. End day for train data (YYYY-MM-DD).
        val_date_end : str. Ending day for validation data (YYYY-MM-DD).
        test_date_end : str. Ending day for test data (YYYY-MM-DD).
        lookback : int. Number of days to look back.
        window_size : int. Size of rolling window. 60 by default
        horizon_shift : int. Size of rolling window. 1 by default

        Returns
        ----------
        X_train, y_train, X_val, y_val, X_test, y_test : np.ndarray. Data ready for ml algorith
        scaler : StandardScaler. Scaled adjusted on train_df.
        """
        # we first split the data
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, y_test_index = prepare_datasets_ml(
            returns,
            train_date_end,
            val_date_end,
            test_date_end,
            lookback,
            window_size,
            horizon_shift)

       # we now create the model
        n_features = X_train.shape[2]
        model = create_lstm_model(
            window_size,
            n_features,
            lstm_units,
            learning_rate,
            dropout_rate,
            optimizer_name,
            loss)

        # We train the model

        history = train_lstm_model(model,
                                   X_train, y_train,
                                   X_val, y_val,
                                   epochs = epochs,
                                   batch_size = batch_size
                                   )

        test_loss = model.evaluate(X_test, y_test, verbose=verbose)

        return {
            "model": model,  # Modelo LSTM entrenado
            "history": history,  # Historial de entrenamiento (loss, val_loss)
            "scaler": scaler,  # StandardScaler ajustado al train
            "X_test": X_test,  # Últimas ventanas para predecir el futuro (test)
            "y_test": y_test,  # Target real del test (normalizado)
            "X_val": X_val,
            "y_val": y_val,
            "test_loss": test_loss,  # Pérdida RMSE/MSE/MAE en test,
            "y_test_index" :  y_test_index
        }


def get_predictions_and_denormalize(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    scaler,
):
    """
    Obtiene predicciones del modelo y las des-normaliza
    usando el mismo scaler que se ajustó en train.

    Parameters
    ----------
    model : tf.keras.Model
    X_test : np.ndarray, shape (n_samples, window_size, n_features)
    y_test : np.ndarray, shape (n_samples, n_features)
    scaler : StandardScaler

    Returns
    -------
    y_test_inv : np.ndarray
        y_test desescalado (retornos originales).
    y_pred_inv : np.ndarray
        predicciones desescaladas (retornos originales).
    """
    # Predicciones en el espacio normalizado
    y_pred = model.predict(X_test)

    # Desescalar ambos: y_test y y_pred
    y_test_inv = scaler.inverse_transform(y_test)
    y_pred_inv = scaler.inverse_transform(y_pred)

    return y_test_inv, y_pred_inv


import matplotlib.pyplot as plt

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

    Parameters
    ----------
    y_test_inv : np.ndarray, shape (n_samples, n_features)
        Retornos reales desescalados.
    y_pred_inv : np.ndarray, shape (n_samples, n_features)
        Retornos predichos desescalados.
    dates : array-like o pd.DatetimeIndex, opcional
        Fechas asociadas a cada fila de y_test/y_pred.
        Si es None, se usa un simple rango 0...n-1.
    asset_idx : int
        Índice de la columna (acción) a representar.
    asset_name : str, opcional
        Nombre para el gráfico (si no, usa f"Asset {asset_idx}".
    n_points : int, opcional
        Si no es None, limita los gráficos a los últimos n_points
        para que no queden demasiado cargados.
    """

    # -------------------------
    # 0) Preparar ejes de tiempo
    # -------------------------
    n_samples = y_test_inv.shape[0]

    if dates is None:
        x_axis = np.arange(n_samples)
    else:
        x_axis = np.array(dates)

    # Aseguramos misma longitud por seguridad
    n = min(len(x_axis), n_samples)
    x_axis = x_axis[:n]
    y_test_inv = y_test_inv[:n, :]
    y_pred_inv = y_pred_inv[:n, :]

    # Aplicar recorte a últimos n_points si se pide
    if (n_points is not None) and (n > n_points):
        x_axis_slice = x_axis[-n_points:]
        y_test_slice = y_test_inv[-n_points:, :]
        y_pred_slice = y_pred_inv[-n_points:, :]
    else:
        x_axis_slice = x_axis
        y_test_slice = y_test_inv
        y_pred_slice = y_pred_inv

    # ------------------------------------------------------------------
    # 1) Gráfico de retornos diarios real vs predicho para una sola acción
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------
    # 2) Portfolio equiponderado real vs predicho (todas las acciones)
    #    - retorno diario
    #    - retorno acumulado
    # ------------------------------------------------------------
    # Retorno diario equiponderado = media de columnas (igual peso)
    real_port_daily = y_test_slice.mean(axis=1)
    pred_port_daily = y_pred_slice.mean(axis=1)

    # Retorno acumulado
    real_port_cum = (1.0 + real_port_daily).cumprod() - 1.0
    pred_port_cum = (1.0 + pred_port_daily).cumprod() - 1.0

    # Gráfico de retorno acumulado equiponderado
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

    # ------------------------------------------------------------
    # 3) Rentabilidad anualizada del portfolio equiponderado
    # ------------------------------------------------------------
    # número de días en el periodo mostrado
    n_days = len(real_port_daily)
    if n_days > 0:
        real_total = real_port_cum[-1]
        pred_total = pred_port_cum[-1]

        # Anualización asumiendo 252 días de mercado
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
    verbose: int = 0
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Hace una búsqueda en rejilla (grid search) sobre varios hiperparámetros
    de la LSTM y devuelve un DataFrame con los resultados y el mejor set.
    """

    results = []

    # Todas las combinaciones de hiperparámetros
    for (window_size,
         lstm_units,
         learning_rate,
         dropout_rate,
         optimizer_name,
         batch_size) in itertools.product(
            window_size_list,
            lstm_units_list,
            learning_rate_list,
            dropout_rate_list,
            optimizer_name_list,
            batch_size_list
    ):
        print(
            f"Probando: window_size={window_size}, lstm_units={lstm_units}, "
            f"lr={learning_rate}, dropout={dropout_rate}, opt={optimizer_name}, "
            f"batch_size={batch_size}"
        )

        # 1) Preparar datasets con ESTA ventana y este horizonte
        X_train, y_train, X_val, y_val, X_test, y_test, scaler, y_test_index = prepare_datasets_ml(
            returns=returns,
            train_date_end=train_date_end,
            val_date_end=val_date_end,
            test_date_end=test_date_end,
            lookback=lookback,
            window_size=window_size,
            horizon_shift=horizon_shift
        )

        n_features = X_train.shape[2]

        # 2) Crear modelo con ESTA combinación
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

        # 4) Métricas: usamos la pérdida de validación mínima como criterio
        val_loss_history = history.history.get("val_loss", None)
        if val_loss_history is not None:
            best_val_loss = float(np.min(val_loss_history))
        else:
            # por si acaso no se guarda val_loss
            best_val_loss = float(history.history["loss"][-1])

        # 5) Evaluación en test (opcional pero útil)
        test_loss = float(model.evaluate(X_test, y_test, verbose=0))

        # 6) Guardamos resultados de esta combinación
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

    # Pasamos a DataFrame
    results_df = pd.DataFrame(results)

    # Elegimos el mejor por val_loss (menor es mejor)
    best_row = results_df.sort_values("val_loss", ascending=True).iloc[0]
    best_params: Dict[str, Any] = best_row.to_dict()

    return results_df, best_params

def run_best_lstm_and_plot(
    returns: pd.DataFrame,
    results_df: pd.DataFrame,
    best_params: Dict[str, Any],
    asset_idx: int = 0,
    asset_name: str | None = None,
    n_points: int | None = 200,
    loss: str = "mse",
    verbose: int = 1
):
    """
    Reentrena un modelo LSTM con los mejores hiperparámetros encontrados
    y muestra un gráfico de real vs predicho para un activo concreto.
    """

    # Extraemos parámetros (mismos nombres que en grid_search_lstm)
    window_size    = int(best_params["window_size"])
    horizon_shift  = int(best_params["horizon_shift"])
    train_date_end = best_params["train_date_end"]
    val_date_end   = best_params["val_date_end"]
    test_date_end  = best_params["test_date_end"]
    lookback       = int(best_params["lookback"])

    lstm_units     = int(best_params["lstm_units"])
    learning_rate  = float(best_params["learning_rate"])
    dropout_rate   = float(best_params["dropout_rate"])
    optimizer_name = str(best_params["optimizer_name"])
    epochs         = int(best_params["epochs"])
    batch_size     = int(best_params["batch_size"])

    # 1) Preparamos datasets (idéntico a grid search)
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, y_test_index = prepare_datasets_ml(
        returns=returns,
        train_date_end=train_date_end,
        val_date_end=val_date_end,
        test_date_end=test_date_end,
        lookback=lookback,
        window_size=window_size,
        horizon_shift=horizon_shift
    )

    n_features = X_train.shape[2]

    # 2) Creamos modelo con los mejores hiperparámetros
    model = create_lstm_model(
        window_size=window_size,
        n_features=n_features,
        lstm_units=lstm_units,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        optimizer_name=optimizer_name,
        loss=loss
    )

    # 3) Entrenamos
    history = train_lstm_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose
    )

    # 4) Evaluamos en test
    test_loss = float(model.evaluate(X_test, y_test, verbose=verbose))
    print(f"Test loss con mejores hiperparámetros: {test_loss:.6f}")

    # 5) Predicciones y des-normalización
    y_test_inv, y_pred_inv = get_predictions_and_denormalize(
        model=model,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler
    )

    # 6) Gráfico real vs predicho
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
        "y_test_index": y_test_index
    }