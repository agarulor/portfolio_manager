import pandas as pd
import numpy as np
from typing import Literal
from portfolio_tools.return_metrics import portfolio_returns
from portfolio_tools.risk_metrics import portfolio_volatility
from scipy.optimize import minimize


def minimize_volatility(target_return: float,
                        returns: pd.DataFrame,
                        covmat: np.ndarray,
                        method: Literal["simple", "log"] = "simple",
                        periods_per_year=252) -> np.ndarray:
    """
    Returns the optimal weight of the portfolio assets that minimize
    volatility for a given target return, returns and a covariance matrix.

    Parameters
    ----------
    target_return : float. Expected return of the portfolio.

    returns: pd.DataFrame. Expected return of the portfolio.

    covmat: np.ndarray. Covariance matrix of the portfolio.

    Returns
    -------
    np.ndarray: Optimal weight of the portfolio.
    :param periods_per_year:
    :param covmat:
    :param returns:
    :param target_return:
    :param method:
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
        {"type": "eq", "fun": lambda w: portfolio_returns(w,
                                                          returns,
                                                          method=method,
                                                          periods_per_year=periods_per_year) - target_return}
    )
    result = minimize(portfolio_volatility, init_guess,
                      args=(covmat,), method='SLSQP',
                      options={'disp': False},
                      constraints=constraints,
                      bounds=bounds)

    return result.x
