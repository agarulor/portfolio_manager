import pandas as pd
import numpy as np
from portfolio_tools.return_metrics import portfolio_returns

def calculate_variance(returns: pd.DataFrame) -> pd.DataFrame:

    """
    Calculates the variance of each asset's returns

    Parameters
    ----------
    returns : pd.DataFrame with assets returns

    Returns
    -------
    pd.Series : Variance per asset.
    """
    return returns.var(axis=0)

def calculate_standard_deviation(returns: pd.DataFrame) -> pd.DataFrame:

    """
    Calculates the standard deviation of each asset's returns

    Parameters
    ----------
    returns : pd.DataFrame with assets returns

    Returns
    -------
    pd.Series: Standard deviation per asset.
    """
    return returns.std(axis=0)


def annualize_variance(variances: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:

    """
    Annualize variances.

    Parameters
    ----------
    variances : pd.Series. Variance per asset.
    periods_per_year : int, default 252. Number of periods per year.

    Returns
    -------
    pd.Series: Annualized variance (Var × periods_per_year).
    """
    return variances * periods_per_year

def annualize_standard_deviation(standard_deviations: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:

    """
    Annualize standard deviations.

    Parameters
    ----------
    standard_deviations : pd.Series. Standard deviations per asset.
    periods_per_year : int, default 252. Number of periods per year.

    Returns
    -------
    pd.Series: Annualized Standard deviations (Var × periods_per_year).
    """
    return standard_deviations * periods_per_year**0.5

def calculate_covariance(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the covariance matrix of asset returns

    Parameters
    ----------
    returns : pd.DataFrame with returns from assets

    Returns
    -------
    pd.Series : covariance matrix
    """
    return returns.cov()


def annualize_covariance(covmat: pd.DataFrame, periods_per_year: int = 252) -> pd.DataFrame:
    """
    It annualizes a covariance matrix

     Parameters
    ----------
    covmat : pd.DataFrame. Covariance matrix of asset returns.
    periods_per_year : int, default 252. Number of periods per year.

    Returns
    -------
    pd.DataFrame: Annualized covariance matrix.

    """
    return covmat * periods_per_year



def portfolio_volatility(
        weights: np.ndarray,
        covmat: pd.DataFrame,
        periods_per_year: int = 252) -> float:
    """
    Computes the portfolio volatility from asset weights and a covariance matrix.

    Parameters
    ----------
    weights : np.ndarray. Portfolio of weights allocated to each asset in the portfolio.
    covmat : pd.DataFrame. Covariance matrix of asset returns.
    periods_per_year : int, default 252. Number of periods per year.

    Returns
    -------
    float: Portfolio volatility (standard deviation).
    """
    annualized_covmat = annualize_covariance(covmat, periods_per_year)
    return (weights.T @ annualized_covmat @ weights) ** 0.5


def sharpe_ratio(
        weights: np.ndarray,
        returns: pd.DataFrame,
        covmat: np.ndarray,
        rf: float = 0,
        method="simple",
        periods_per_year: int = 252
) -> float:
    """
    It calculates the Sharpe Ratio.
    Parameters
    ----------
    weights: np.ndarray. Portfolio of weights allocated to each asset in the portfolio.
    returns: pd.DataFrame. Expected return of the portfolio.
    covmat: np.ndarray. Covariance matrix of the portfolio.
    rf: float. Risk-free rate.
    method: str. "simple" or "log
    periods_per_year: int. Number of years over which to calculate volatility.

    Returns
    -------
    float: Sharpe Ratio.
    """
    returns_portfolio = portfolio_returns(weights, returns, method, periods_per_year)
    volatility_portfolio = portfolio_volatility(weights, covmat, periods_per_year)

    return (returns_portfolio - rf) / volatility_portfolio


def neg_sharpe_ratio(
        weights: np.ndarray,
        returns: pd.DataFrame,
        covmat: np.ndarray,
        rf: float = 0,
        method = "simple",
        periods_per_year: int = 252
        ) -> float:
    """
    It calculates the negative Sharpe Ratio.
    Parameters
    ----------
    weights: np.ndarray. Portfolio of weights allocated to each asset in the portfolio.
    returns: pd.DataFrame. Expected return of the portfolio.
    covmat: np.ndarray. Covariance matrix of the portfolio.
    rf: float. Risk-free rate.
    method: str. "simple" or "log
    periods_per_year: int. Number of years over which to calculate volatility.

    Returns
    -------
    float: Sharpe Ratio.
    """
    returns_portfolio = portfolio_returns(weights, returns, method, periods_per_year)
    volatility_portfolio = portfolio_volatility(weights, covmat, periods_per_year)

    return -(returns_portfolio - rf) / volatility_portfolio


def calculate_max_drawdown(weights: np.ndarray, returns: pd.Series)-> float:
    """
    Calculates the maximum drawdown from the temp returns.

    Parameters
    ----------
    weights: np.ndarray. Portfolio of weights allocated to each asset in the portfolio.
    returns: pd.DataFrame. Expected return of the portfolio.

    Returns
    -------
    float: maximum drawdown from the returns.
    """
    returns_portfolio = returns.mul(weights, axis=1).sum(axis=1)

    # first we create the wealth index
    wealth_index = 1000*(1+returns_portfolio).cumprod()

    # Calculate previous max
    running_max = wealth_index.cummax()
    # calculate drawdowns
    drawdowns = (wealth_index / running_max) - 1

    # We return the max drawdown
    return abs(drawdowns.min())