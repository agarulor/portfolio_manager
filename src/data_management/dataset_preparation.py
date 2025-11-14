import numpy as np
import pandas as pd
from typing import Tuple

from sklearn.preprocessing import StandardScaler

def split_data_markowtiz(
        returns: pd.DataFrame,
        test_date_start: str = "2024-01-01",
        test_date_end: str = "2025-09-30") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data into training and test sets

    Parameters
    ----------
    returns : pd.DataFrame. Dataset with returns.
    test_date_start : str. Starting day for test data (YYYY-MM-DD).
    test_date_end : str. Ending day for test data (YYYY-MM-DD).

    Returns
    ----------
    train_set : pd.DataFrame. Training set with returns.
    test_set : pd.DataFrame. Test set with returns.
    """
    # we sort the returns in case they are not shorted
    sorted_returns = returns.sort_index()

    # We convert to datetime the index
    sorted_returns.index = pd.to_datetime(sorted_returns.index)

    # Now we check if the date exists
    time_start = pd.to_datetime(test_date_start)
    time_end = pd.to_datetime(test_date_end)

    # We adjust data if not in the data shared
    if time_start not in sorted_returns.index:
        pos = sorted_returns.index.searchsorted(time_start)
        if pos >= len(sorted_returns.index):
            raise ValueError("time_start is after the available data range.")
        time_start = sorted_returns.index[pos]
        print(f"time_start adjusted to: {time_start.date()}")


    if time_end not in sorted_returns.index:
        print(f"{time_end} not in data. Adjusting to previous available business day.")
        pos = sorted_returns.index.searchsorted(time_end) - 1
        if pos < 0:
            raise ValueError("time_end is before the available data range.")
        time_end = sorted_returns.index[pos]
        print(f"time_end adjusted to: {time_end.date()}")

    # We now divide data into training and test
    train_set = sorted_returns.loc[:time_start - pd.Timedelta(days=1)]
    test_set = sorted_returns.loc[time_start : time_end]

    print(test_set)

    return train_set, test_set

def split_data_ml(
        returns: pd.DataFrame,
        train_date_end: str = "2022-09-30",
        val_date_end: str = "2024-09-30",
        test_date_end: str = "2025-09-30",
        lookback: int = 0) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Splits data into training, validation and test sets

    Parameters
    ----------
    returns : pd.DataFrame. Dataset with returns.
    train_date_end : str. End day for train data (YYYY-MM-DD).
    val_date_end : str. Ending day for validation data (YYYY-MM-DD).
    test_date_end : str. Ending day for test data (YYYY-MM-DD).
    lookback : int. Number of days to look back.

    Returns
    ----------
    train_set : pd.DataFrame. Training set with returns.
    val_set : pd.DataFrame. Validation set with returns.
    test_set : pd.DataFrame. Test set with returns.
    """
    # we sort the returns in case they are not shorted
    sorted_returns = returns.sort_index()

    # We convert to datetime the index
    sorted_returns.index = pd.to_datetime(sorted_returns.index)

    # Now we check if the date exists
    time_train_end = pd.to_datetime(train_date_end)
    time_val_end = pd.to_datetime(val_date_end)
    time_test_end = pd.to_datetime(test_date_end)

    if time_train_end not in sorted_returns.index:
        pos = sorted_returns.index.searchsorted(time_train_end) - 1
        if pos < 0:
            raise ValueError(f"No available data on or before {time_train_end}.")
        time_train_end = sorted_returns.index[pos]

    if time_val_end not in sorted_returns.index:
        pos = sorted_returns.index.searchsorted(time_val_end) - 1
        if pos < 0:
            raise ValueError(f"No available data on or before {time_val_end}.")
        time_val_end = sorted_returns.index[pos]

    if time_test_end not in sorted_returns.index:
        pos = sorted_returns.index.searchsorted(time_test_end) - 1
        if pos < 0:
            raise ValueError(f"No available data on or before {time_test_end}.")
        time_test_end = sorted_returns.index[pos]

    time_val_start = sorted_returns.index[sorted_returns.index.get_loc(time_train_end) + 1]
    time_test_start = sorted_returns.index[sorted_returns.index.get_loc(time_val_end) + 1]

    # We split the main sets
    train_set = sorted_returns.loc[: train_date_end]
    val_set = sorted_returns.loc[ time_val_start: time_val_end]
    test_set = sorted_returns.loc[time_test_start: time_test_end]

    # We add the warming-up
    val_warm = sorted_returns.loc[time_val_start:].iloc[:lookback] if lookback > 0 else sorted_returns.iloc[0:0]
    test_warm = sorted_returns.loc[time_test_start:].iloc[:lookback] if lookback > 0 else sorted_returns.iloc[0:0]

    return train_set, val_set, test_set, val_warm, test_warm


def normalize_data(X_train: pd.DataFrame,
                   X_val: pd.DataFrame,
                   X_test: pd.DataFrame,
                   ) ->  Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, StandardScaler]:

    """
    Normalizes data based on the train DataSet

    Parameters
    ----------
    X_train : pd.DataFrame. Dataset with training data.
    X_val : pd.DataFrame. Dataset with valuation data.
    X_test : pd.DataFrame. Dataset with test data.

    Returns
    ----------
    train_scaled : pd.DataFrame. Training set train scaled data.
    val_scaled : pd.DataFrame. Validation set with val scaled data.
    test_scaled : pd.DataFrame. Test set with trest scaled data.
    scaler : StandardScaler. Scaled adjusted on train_df.
    """
    scaler = StandardScaler()
    scaler.fit(X_train.values)

    train_scaled = scaler.transform(X_train.values)
    val_scaled = scaler.transform(X_val.values)
    test_scaled = scaler.transform(X_test.values)

    train_scaled_df = pd.DataFrame(train_scaled, index=X_train.index, columns=X_train.columns)
    val_scaled_df   = pd.DataFrame(val_scaled,   index=X_val.index,   columns=X_val.columns)
    test_scaled_df  = pd.DataFrame(test_scaled,  index=X_test.index,  columns=X_test.columns)

    return train_scaled_df, val_scaled_df, test_scaled_df, scaler


def create_rolling_window(df_scaled: pd.DataFrame,
                          window_size: int = 60,
                          horizon_shift: int = 1)-> Tuple[np.ndarray, np.ndarray, pd.Index]:
    """
    Creates rolling window based on window_size from a scaled DataFrame.

    Parameters
    ----------
    df_scaled : pd.DataFrame. Dataset with training data.
    window_size : int. Size of rolling window. 60 by default
    horizon_shift : int. Size of rolling window. 1 by default

    Returns
    ----------
    X : np.ndarray. Rolling window data. To be used with the algorithm
        (n_samples, window_size, n_features).
    y : np.ndarray. return in T+1 period. (n_samples, n_features).
    y_index : pd.Index. Return index of y (dates), useful for testing
    """

    # first we extract the values from the datasets
    values = df_scaled.values

    # we then get the number of files (or timesteps)
    n_timesteps = values.shape[0]

    # We then extract the features
    n_features = values.shape[1]

    # We get the last index allowed by the rolling window
    last_start = n_timesteps - window_size - horizon_shift + 1

    # We then create the supporting lists for the process
    X_list = []
    y_list = []

    # And now we iterate
    for start in range(last_start):
        # Last day of current window
        end = start + window_size
        # We get the day to predict
        target_index = end + horizon_shift - 1

        # We get the time sequence and the features
        X_window = values[start:end, :]
        y_window = values[target_index, :]

        # We append the date
        X_list.append(X_window)
        y_list.append(y_window)

    # We create 3-d array with n_samples, window_size and n_features
    X = np.stack(X_list)
    y = np.stack(y_list)

    print(X)
    # index with dates for target (y)
    y_index = df_scaled.index[window_size + horizon_shift - 1: window_size + horizon_shift - 1 + X.shape[0]]
    return X, y, y_index

def prepare_datasets_ml(returns: pd.DataFrame,
                         train_date_end: str = "2022-09-30",
                         val_date_end: str = "2024-09-30",
                         test_date_end: str = "2025-09-30",
                         lookback: int = 0,
                         window_size: int = 60,
                         horizon_shift: int = 1) \
        -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:

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
    train, val, test, val_warm, test_warm = split_data_ml(returns,
                                                          train_date_end,
                                                          val_date_end,
                                                          test_date_end,
                                                          lookback)
    # normalize data
    train_norm, val_norm, test_norm, scaler = normalize_data(train, val, test)

    # We create rolling windows for each split

    X_train, y_train, y_train_index = create_rolling_window(train_norm, window_size, horizon_shift)
    X_val, y_val, y_val_index = create_rolling_window(val_norm, window_size, horizon_shift)
    X_test, y_test, y_test_index = create_rolling_window(test_norm, window_size, horizon_shift)

    return X_train, y_train, X_val, y_val, X_test, y_test, scaler
