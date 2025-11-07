import pandas as pd
import numpy as np
from typing import Literal

def calculate_daily_returns(
        prices: pd.DataFrame,
        method: Literal["simple", "log"] = "simple",
        drop_first: bool = True) -> pd.DataFrame:
    """
    Compute daily returns from a price datasets with tickers from Yahoo

    Given a DataFrame of prices with dates as index and tickers as columns,
    it calculates the day-over-day returns it can use normal, i.e. simple
    return (i.e. (P_t / P_{t-1}) - 1) or logarithmic returns
    (i.e. ln(P_t) - ln(P_{t-1}))

    Parameters
    ----------
    prices : pandas.DataFrame
        Prices DataFrame. NaNs are allowed and will propagate to returns.
    method : {"simple", "log"}, default "simple"
        Return type:
        - "simple": (P_t / P_{t-1}) - 1
        - "log":    ln(P_t) - ln(P_{t-1})
    drop_first : bool, default True
        If True, drops the first row, as we would get a NaN.
        If False, we keep the first row and get a NaN

    Returns
    -------
    pandas.DataFrame
        Daily return DataFrame

    """
    # We first check which method is being used.
    # Simple is useful for this exercise. We calculate the % variation
    if method == "simple":
        returns = prices.pct_change()
    # We get the log and then % variation in price
    elif method == "log":
        returns = np.log(prices).diff()
    else:
        raise ValueError("Invalid method. It has to be 'simple' or 'log'")

    # If we choose to drop the first value
    # (usually because there is no previous value in the first period)
    if drop_first:
        returns = returns.iloc[1:]

    return returns


def annualize_returns(
    returns: pd.DataFrame,
    method: Literal["simple", "log"] = "simple",
    periods_per_year: int = 252
) -> pd.Series:
    """
    Calculates annualized returns from different periodic returns.

    The function calculates the annualized  mean return for each asset in a
     DataFrame of periodic returns. It can use simple or logarithmic approach

    Parameters
    ----------
    returns : pandas.DataFrame
        It is a DataFrame of returns

    method : {"simple", "log"}, default "simple"
        Return type:
        - "simple": (∏(1 + R_t))^(periods_per_year / T) - 1
        - "log": exp(periods_per_year × mean(r_t)) - 1
    periods_per_year : int, default 252
        Typical number of trading days or other periods
        Typical values:
        - 252 → on a daily basis
        - 52 → on a weekly basis
        - 12 → on a monthly basis

    Returns
    -------
    pandas.Series
        It returns the annualized returns for each company (decimal form).
        This means that a 0.12 annual return means a 12% annual return.

    """
    # If we don't choose the right method
    if method not in ("simple", "log"):
        raise ValueError("Method must be 'simple' or 'log'")

    # If the DataFrame is empty
    if returns.empty:
        raise ValueError("DataFrame is empty.")

    # We calculate, it depends if we choose simple or log
    if method == "simple":
        # For each column: compounded_growth = prod(1 + r)
        compounded_growth = (1 + returns).prod(skipna=True)
        # Effective T per column = count of non-NaN observations
        t = returns.notna().sum(axis=0).astype(float)
        annualized_returns = compounded_growth.pow(periods_per_year / t) - 1
    else:  # log returns
        mean_log = returns.mean(skipna=True)
        annualized_returns = np.exp(mean_log * periods_per_year) - 1

    return annualized_returns

def portfolio_returns(
        weights: np.ndarray,
        returns: pd.DataFrame,
        method: Literal["simple", "log"] = "simple",
        periods_per_year=252) -> float:
    """
    Computes the portfolio return from asset weights and returns of assets.

    Parameters
    ----------
    weights : np.ndarray. Portfolio of weights allocated to each asset in the portfolio.
    returns : pd.DataFrame. Returns of the assets
    method : {"simple", "log"}, default "simple"
        Return type:
        - "simple": (∏(1 + R_t))^(periods_per_year / T) - 1
        - "log": exp(periods_per_year × mean(r_t)) - 1
    periods_per_year : int, default 252
        Typical number of trading days or other periods
        Typical values:
        - 252 → on a daily basis
        - 52 → on a weekly basis
        - 12 → on a monthly basis

    Returns
    -------
    float: Portfolio return.
    """


    # We calculate the annualized returns
    annualized_returns = annualize_returns(returns,
                                           method=method,
                                           periods_per_year=periods_per_year)
    # We check length of weights and returns
    if len(weights) != len(annualized_returns):
        raise ValueError("Weights and returns must have same length.")
    # We calculate the portfolio returns
    return weights.T @ annualized_returns


def daily_portfolio_returns(
        weights: np.ndarray,
        returns: pd.DataFrame) -> pd.Series:
    """
    Computes the daily portfolio return from asset weights and returns of assets.

    Parameters
    ----------
    weights : np.ndarray. Portfolio of weights allocated to each asset in the portfolio.
    returns : pd.DataFrame. Returns of the assets

    Returns
    -------
    pd.Series: Portfolio daily return.
    """

    # We check length of weights and returns
    if len(weights) != returns.shape[1]:
        raise ValueError("Weights and returns must have same length.")
    # We calculate the portfolio returns
    return weights @ returns.T