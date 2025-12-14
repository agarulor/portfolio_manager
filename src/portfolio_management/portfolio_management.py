import pandas as pd
import numpy as np
from typing import Union
from datetime import datetime


def check_portfolio_weights(df_values: pd.DataFrame,
                            date: Union[str, datetime, pd.Timestamp]) -> pd.Series:
    """
    Helps to track the current weights of the assets of the portfolio
    It takes the weights of the assets of the portfolio at a given date

    :return:

    The weights of the assets of the portfolio
    """


    # We convert the date to datetime
    date = pd.to_datetime(date)

    # if index not date time, we try to convert it
    if not pd.api.types.is_datetime64_any_dtype(df_values.index):
        df_values = df_values.copy()
        df_values.index = pd.to_datetime(df_values.index)

    # - If the date exists it uses it
    # - if it doesn't exist it use the previous one
    if date in df_values.index:
        row = df_values.loc[date]

    else:
        df_sorted = df_values.sort_index()
        row = df_sorted.loc[df_sorted.index.asof(date)]

    total = row.sum()

    if total == 0 or pd.isna(total):
        return pd.Series(0.0, index = df_values.columns)

    weights = (row / total).fillna(0.0)

    # We only keep those with weights > 0
    weights = weights[weights > 0]

    return weights


def calculate_portfolio_daily_returns(weights: np.ndarray, returns: pd.DataFrame, rf_annual: float|None = None) -> pd.Series:
    """
    We calculate the daily returns of the portfolio. Helper function to keep track of volatility
    :param weights:
    :param returns:
    :return:
    """
    weights_risky = weights
    if len(weights) != returns.shape[1]:
        if rf_annual is not None:
            weights_risky = weights[:-1]
        else:
            raise ValueError("The number of weights is not equal to the number of assets")

    return (returns * weights_risky).sum(axis=1)






def check_portfolio_volatility():
    """
    Helps to track the current volatility of the porfolio and the assets of the portfolio
    It compares the last x days volatility to the historic volatility (i.e. 3 years or similar)
    :return:
    portfolio_volatility
    volatility of assets in the portfolio
    """


def check_portfolio_drawdowns():

    """
    This function helps to keep track of the portfolio's drawdowns
    there is a limit to the drawdowns based on the investor risk profile

    :return:
    global porfolio drawdowns
    asset by asset drawdowns
    """