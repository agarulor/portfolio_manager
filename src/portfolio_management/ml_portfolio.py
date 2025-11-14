import tensorflow as tf
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


def create_lstm_model(window_size: int,
                      n_features: int,
                      lstm_units: int = 64,
                      learning_rate: float = 0.001,
                      optimizer: tf.keras.optimizers.Optimizer = "adam",
                      loss = "mse") -> tf.keras.models.Sequential:
    # We define the model
    model = Sequential()
    model.add(LSTM(lstm_units, input_shape=(window_size, n_features),))
    model.add(Dense(n_features))

    # We create the model
    optimizer = tf.keras.optimizers.get({"class_name": optimizer,
                                         "config":{"learning_rate":learning_rate}},
                                        loss=loss)

    return model

def train_lstm_model(model:tf.keras.models.Sequential,
                     X_train: pd.DataFrame,
                     y_train: pd.Series,
                     X_val: pd.DataFrame,
                     y_val: pd.Series,
                     epochs: int = 25,
                     batch_size: int = 64,
                     verbose: int = 1):
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose)

    return history

