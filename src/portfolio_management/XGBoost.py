import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List

from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error

from data_management.dataset_preparation import prepare_datasets_ml
import matplotlib.pyplot as plt

def flatten_windows(X: np.ndarray) -> np.ndarray:
    """
    Convierte X de shape (n_samples, window_size, n_features)
    a shape (n_samples, window_size * n_features) para modelos tipo árbol.

    Param
    ----------
    X : np.ndarray
        Tensor 3D (n_samples, window_size, n_features)

    Returns
    -------
    X_flat : np.ndarray
        Matriz 2D (n_samples, window_size * n_features)
    """
    n_samples = X.shape[0]
    return X.reshape(n_samples, -1)

def train_xgb_models_per_asset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_estimators: int = 300,
    learning_rate: float = 0.05,
    max_depth: int = 4,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    random_state: int = 42,
    verbosity: int = 0
) -> Tuple[List[XGBRegressor], float]:
    """
    Entrena un modelo XGBRegressor por activo (columna de y).

    Parameters
    ----------
    X_train, y_train : np.ndarray
        X_train: (n_samples, n_features_flat)
        y_train: (n_samples, n_assets)
    X_val, y_val : np.ndarray
        Con mismas dimensiones en la primera y última.
    (hiperparámetros de XGBoost)...

    Returns
    -------
    models : List[XGBRegressor]
        Lista de modelos, uno por activo.
    mean_val_mse : float
        MSE medio en validación (promedio sobre activos).
    """
    n_assets = y_train.shape[1]
    models: List[XGBRegressor] = []
    val_mses: List[float] = []

    for i in range(n_assets):
        y_train_i = y_train[:, i]
        y_val_i   = y_val[:, i]

        model = XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_lambda=reg_lambda,
            reg_alpha=reg_alpha,
            objective="reg:squarederror",
            random_state=random_state,
            verbosity=verbosity,
        )

        model.fit(
            X_train,
            y_train_i,
            eval_set=[(X_val, y_val_i)],
            verbose=False
        )

        y_val_pred_i = model.predict(X_val)
        mse_i = mean_squared_error(y_val_i, y_val_pred_i)
        val_mses.append(mse_i)

        models.append(model)

    mean_val_mse = float(np.mean(val_mses))
    return models, mean_val_mse

def predict_xgb_models(
    models: List[XGBRegressor],
    X: np.ndarray
) -> np.ndarray:
    """
    Aplica cada modelo XGBoost a X y devuelve matriz (n_samples, n_assets).

    Parameters
    ----------
    models : list de XGBRegressor
    X : np.ndarray, shape (n_samples, n_features_flat)

    Returns
    -------
    y_pred : np.ndarray, shape (n_samples, n_assets)
    """
    n_samples = X.shape[0]
    n_assets = len(models)
    y_pred = np.zeros((n_samples, n_assets), dtype=float)

    for i, model in enumerate(models):
        y_pred[:, i] = model.predict(X)

    return y_pred

def denormalize_targets(
    y: np.ndarray,
    scaler
) -> np.ndarray:
    """
    Des-normaliza y (n_samples, n_features) usando el mismo scaler
    que se usó para X / y durante el preprocesado.
    """
    return scaler.inverse_transform(y)

def run_xgb_experiment(
    returns: pd.DataFrame,
    train_date_end: str = "2023-09-30",
    val_date_end: str = "2024-09-30",
    test_date_end: str = "2025-09-30",
    lookback: int = 0,
    window_size: int = 60,
    horizon_shift: int = 1,
    # Hiperparámetros XGBoost
    n_estimators: int = 1000,
    learning_rate: float = 0.005,
    max_depth: int = 4,
    subsample: float = 0.8,
    colsample_bytree: float = 0.8,
    reg_lambda: float = 1.0,
    reg_alpha: float = 0.0,
    random_state: int = 42,
    verbosity: int = 0,
):
    """
    Pipeline completo con XGBoost:

    1) Prepara datasets (usa prepare_datasets_ml)
    2) Aplana ventanas para XGB
    3) Entrena un modelo por activo
    4) Evalúa en test
    5) Devuelve
    """

    # 1) Preparamos datasets igual que con LSTM
    X_train, y_train, X_val, y_val, X_test, y_test, scaler, y_test_index = prepare_datasets_ml(
        returns=returns,
        train_date_end=train_date_end,
        val_date_end=val_date_end,
        test_date_end=test_date_end,
        lookback=lookback,
        window_size=window_size,
        horizon_shift=horizon_shift
    )

    # 2) Aplanamos ventanas para XGBoost
    X_train_flat = flatten_windows(X_train)
    X_val_flat   = flatten_windows(X_val)
    X_test_flat  = flatten_windows(X_test)

    # 3) Entrenar modelos XGBoost (uno por activo)
    models, mean_val_mse = train_xgb_models_per_asset(
        X_train=X_train_flat,
        y_train=y_train,
        X_val=X_val_flat,
        y_val=y_val,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        reg_lambda=reg_lambda,
        reg_alpha=reg_alpha,
        random_state=random_state,
        verbosity=verbosity
    )

    # 4) Evaluación en test
    y_test_pred = predict_xgb_models(models, X_test_flat)
    test_mse = float(mean_squared_error(y_test, y_test_pred))

    # 5) Des-normalizar para interpretar en retornos
    y_test_inv  = denormalize_targets(y_test, scaler)
    y_pred_inv  = denormalize_targets(y_test_pred, scaler)

    results = {
        "models": models,
        "scaler": scaler,
        "X_test": X_test,
        "y_test": y_test,
        "y_test_index": y_test_index,
        "y_test_inv": y_test_inv,
        "y_pred_inv": y_pred_inv,
        "mean_val_mse": mean_val_mse,
        "test_mse": test_mse,
        "window_size": window_size,
        "horizon_shift": horizon_shift,
        "xgb_params": {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
            "reg_lambda": reg_lambda,
            "reg_alpha": reg_alpha,
            "random_state": random_state,
        },
        "train_date_end": train_date_end,
        "val_date_end": val_date_end,
        "test_date_end": test_date_end,
    }

    return results