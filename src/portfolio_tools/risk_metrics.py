import pandas as pd
import numpy as np
from portfolio_tools.return_metrics import portfolio_returns


def calculate_standard_deviation(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates standard deviation.

    Parameters
    ----------
    returns : pd.DataFrame. Returns of the assets.

    Returns
    -------
    pd.DataFrame: calculate standard deviation output.
    """
    return returns.std(axis=0)


def annualize_standard_deviation(standard_deviations: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """
    Annualizes standard deviation.

    Parameters
    ----------
    standard_deviations : pd.DataFrame. standard deviations.
    periods_per_year : int. periods per year.

    Returns
    -------
    pd.DataFrame: annualize standard deviation output.
    """
    return standard_deviations * periods_per_year ** 0.5


def calculate_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates covariance.

    Parameters
    ----------
    returns : pd.DataFrame. Returns of the assets.

    Returns
    -------
    pd.DataFrame: calculate covariance output.
    """
    return returns.cov()


def annualize_covariance(covmat: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """
    Annualizes covariance.

    Parameters
    ----------
    covmat : pd.DataFrame. covmat.
    periods_per_year : int. periods per year.

    Returns
    -------
    pd.DataFrame: annualize covariance output.
    """
    return covmat * periods_per_year


def portfolio_volatility(
        weights: np.ndarray,
        covmat: pd.DataFrame,
        periods_per_year: int = 252) -> float:
    """
    Computes portfolio volatility.

    Parameters
    ----------
    weights : np.ndarray. Portfolio weights allocated to each asset.
    covmat : pd.DataFrame. covmat.
    periods_per_year : int. periods per year.

    Returns
    -------
    float: portfolio volatility output.
    """
    annualized_covmat = annualize_covariance(covmat, periods_per_year)
    return (weights.T @ annualized_covmat @ weights) ** 0.5


def neg_sharpe_ratio(
        weights: np.ndarray,
        returns: pd.DataFrame,
        covmat: np.ndarray,
        rf: float = 0,
        method="simple",
        periods_per_year: int = 252
) -> float:
    """
    Computes neg sharpe ratio.

    Parameters
    ----------
    weights : np.ndarray. Portfolio weights allocated to each asset.
    returns : pd.DataFrame. Returns of the assets.
    covmat : np.ndarray. covmat.
    rf : float. rf.
    method : Any. method.
    periods_per_year : int. periods per year.

    Returns
    -------
    float: neg sharpe ratio output.
    """
    returns_portfolio = portfolio_returns(weights, returns, method, periods_per_year)
    volatility_portfolio = portfolio_volatility(weights, covmat, periods_per_year)

    return -(returns_portfolio - rf) / volatility_portfolio


def calculate_max_drawdown(weights: np.ndarray, returns: pd.Series) -> float:
    """
    Calculates max drawdown.

    Parameters
    ----------
    weights : np.ndarray. Portfolio weights allocated to each asset.
    returns : pd.Series. Returns of the assets.

    Returns
    -------
    float: calculate max drawdown output.
    """
    returns_portfolio = returns.mul(weights, axis=1).sum(axis=1)

    # first we create the wealth index
    wealth_index = 1000 * (1 + returns_portfolio).cumprod()

    # Calculate previous max
    running_max = wealth_index.cummax()
    # calculate drawdowns
    drawdowns = (wealth_index / running_max) - 1

    # We return the max drawdown
    return abs(drawdowns.min())


def max_drawdown_from_value_series(value_series: pd.Series) -> float:
    # Asegurar orden
    """
    Computes max drawdown from value series.

    Parameters
    ----------
    value_series : pd.Series. value series.

    Returns
    -------
    float: max drawdown from value series output.
    """
    series = value_series.dropna().sort_index()

    # We compute the running maximum up to each point
    running_max = series.cummax()

    # Drawdown en cada punto
    drawdowns = series / running_max - 1

    # Maximum drawdown (lowest value)
    return (drawdowns.min()) * - 1