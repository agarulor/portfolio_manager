import tensorflow as tf
import pandas as pd
from keras.src.optimizers import Adam, RMSprop, SGD, AdamW

from data_management.dataset_preparation import prepare_datasets_ml
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def create_lstm_model(window_size: int,
                      n_features: int,
                      lstm_units: int = 64,
                      learning_rate: float = 0.001,
                      optimizer_name: str = "adam",
                      loss = "mse") -> tf.keras.models.Sequential:
    # We define the model
    inputs = tf.keras.Input(shape=(window_size, n_features))
    x = LSTM(lstm_units)(inputs)
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
                   train_date_end: str = "2020-09-30",
                   val_date_end: str = "2022-09-30",
                   test_date_end: str = "2024-09-30",
                   lookback: int = 0,
                   window_size: int = 60,
                   horizon_shift: int = 1,
                   lstm_units: int = 64,
                   learning_rate: float = 0.001,
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
        X_train, y_train, X_val, y_val, X_test, y_test, scaler = prepare_datasets_ml(
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
            "test_loss": test_loss  # Pérdida RMSE/MSE/MAE en test
        }