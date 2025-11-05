import pandas as pd
import numpy as np
from typing import Literal
from portfolio_tools.return_metrics import portfolio_returns, annualize_returns
from portfolio_tools.risk_metrics import portfolio_volatility
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def min_percentage_renormalize(w: np.ndarray, min_w: float = 0.00) -> np.ndarray:
    """
        Obtains the minimal percentage renormalized weight for the assets

        Parameters
        ----------
        w: np.ndarray. Expected return of the portfolio.
        min_w: float. Minimum weight of the portfolio.

        Returns
        -------
        np.ndarray: Optimal weight of the portfolio adjusted
        """

    w2 = w.copy()
    w2[w2 < min_w] = 0.0
    s = w2.sum()
    return w2 / s if s > 0 else w2



def minimize_volatility(target_return: float,
                        returns: pd.DataFrame,
                        covmat: np.ndarray,
                        method: Literal["simple", "log"] = "simple",
                        periods_per_year: int =252,
                        min_w: float = 0.01) -> np.ndarray:
    """
    Returns the optimal weight of the portfolio assets that minimize
    volatility for a given target return, returns and a covariance matrix.

    Parameters
    ----------
    target_return : float. Expected return of the portfolio.
    returns: pd.DataFrame. Expected return of the portfolio.
    covmat: np.ndarray. Covariance matrix of the portfolio.
    method: str. "simple" or "log
    periods_per_year: int. Number of years over which to calculate volatility.
    min_w: float. Minimum weight of the portfolio.

    Returns
    -------
    np.ndarray: Optimal weight of the portfolio.
    """

    # Number of assets in the spectrum of assets
    n = returns.shape[1]

    # Initial guess of weights (we start with evenly weights)
    init_guess = np.ones(n) / n

    # We ensure that there is no short-selling (i.e. no short positions)
    bounds = [(0, 1)] * n

    # We add the constraints to the model
    # Weights must sum 1 (fully invested)
    # Portfolio expected return is equal to the target return
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: portfolio_returns(w, returns, method, periods_per_year) - target_return}
    )
    result = minimize(lambda w: (portfolio_volatility(w, covmat, periods_per_year)),
                      init_guess,
                      method='SLSQP',
                      options={'disp': False},
                      constraints=constraints,
                      bounds=bounds)

    weights = min_percentage_renormalize(result.x, min_w)
    return weights


def maximize_return(target_volatility: float,
                    returns: pd.DataFrame,
                    covmat: np.ndarray,
                    method: Literal["simple", "log"] = "simple",
                    periods_per_year: int =252,
                    min_w: float = 0.01) -> np.ndarray:
    """
    Returns the optimal weight of the portfolio assets that maximize
    return for a given target volatility, returns and a covariance matrix.

    Parameters
    ----------
    target_volatility: float. Expected return of the portfolio.
    returns: pd.DataFrame. Expected return of the portfolio.
    covmat: np.ndarray. Covariance matrix of the portfolio.
    method: str. "simple" or "log
    periods_per_year: int. Number of years over which to calculate volatility.
    min_w: float. Minimum weight of the portfolio.

    Returns
    -------
    np.ndarray: Optimal weight of the portfolio.
    """

    # Number of assets in the spectrum of assets
    n = returns.shape[1]

    # Initial guess of weights (we start with evenly weights)
    init_guess = np.ones(n) / n

    # We ensure that there is no short-selling (i.e. no short positions)
    bounds = [(0, 1)] * n

    # We add the constraints to the model
    # Weights must sum 1 (fully invested)
    # Volatility constraint
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: portfolio_volatility(w, covmat, periods_per_year) - target_volatility}
    )
    result = minimize(lambda w: -portfolio_returns(w, returns, method, periods_per_year),
                      init_guess,
                      method='SLSQP',
                      options={'disp': False},
                      constraints=constraints,
                      bounds=bounds)

    weights = min_percentage_renormalize(result.x, min_w)
    return weights


def get_weights(n_returns: int,
                returns: pd.DataFrame,
                covmat: np.ndarray,
                method: Literal["simple", "log"] = "simple",
                periods_per_year: int =252) -> np.ndarray:

    """
    Returns the optimal weight of the portfolio assets that minimize
    volatility for a given target return, returns and a covariance matrix.

    Parameters
    ----------
    n_returns : int. Expected returns of the portfolio
    returns: pd.DataFrame. Expected return of the portfolio.
    covmat: np.ndarray. Covariance matrix of the portfolio.
    method: str. "simple" or "log
    periods_per_year: int. Number of years over which to calculate volatility.

    Returns
    -------
    np.ndarray: Optimal weight of the portfolio.
    """

    annualized_returns = annualize_returns(returns, method, periods_per_year)
    # We obtain a series of points based on the min and max returns
    target_returns = np.linspace(annualized_returns.min(), annualized_returns.max(), n_returns)
    # We now obtain the weights for each of the target_returns
    weights = [minimize_volatility(target_return,
                                    returns,
                                    covmat,
                                    method=method,
                                    periods_per_year=periods_per_year) for target_return in target_returns]


    print(weights)
    return weights


def plot_basic_frontier(n_returns: int,
                returns: pd.DataFrame,
                covmat: np.ndarray,
                method: Literal["simple", "log"] = "simple",
                periods_per_year: int = 252,
                        style: str = '.-',
                        legend: bool = False) -> plt.Figure:

    weights = get_weights(n_returns, returns, covmat, method, periods_per_year)
    retornos = [portfolio_returns(w, returns, method,periods_per_year) for w in weights]
    volatilities = [portfolio_volatility(w, covmat, periods_per_year) for w in weights]
    ef = pd.DataFrame({
        "Returns": retornos,
        "Volatility": volatilities
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    plt.show()
    return ax

