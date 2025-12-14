import pandas as pd
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



    # We first sum the value of all the assets
    total = df_values.sum(axis=1)
    # We then divide by the values
    weights = df_values.div(total, axis=0).fillna(0.0)

    weights = weights[weights > 0]
    return weights






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