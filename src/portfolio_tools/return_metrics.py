import pandas as pd
import numpy as np
from typing import Literal


def calculate_daily_returns(
        prices: pd.DataFrame,
        method: Literal["simple", "log"] = "simple",
        drop_first: bool = True) -> pd.DataFrame:
    """
    Calculates daily returns.

    Parameters
    ----------
    prices : pd.DataFrame. Prices of the assets.
    method : Literal["simple", "log"]. method.
    drop_first : bool. drop first.

    Returns
    -------
    pd.DataFrame: calculate daily returns output.
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
        periods_per_year: float = 252
) -> pd.Series:
    """
    Annualizes returns.

    Parameters
    ----------
    returns : pd.DataFrame. Returns of the assets.
    method : Literal["simple", "log"]. method.
    periods_per_year : float. periods per year.

    Returns
    -------
    pd.Series: annualize returns output.
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
        periods_per_year: float = 252) -> float:
    """
    Computes portfolio returns.

    Parameters
    ----------
    weights : np.ndarray. Portfolio weights allocated to each asset.
    returns : pd.DataFrame. Returns of the assets.
    method : Literal["simple", "log"]. method.
    periods_per_year : float. periods per year.

    Returns
    -------
    float: portfolio returns output.
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
