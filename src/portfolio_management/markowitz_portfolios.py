import pandas as pd
import numpy as np
from typing import Literal, Tuple
from portfolio_tools.markowitz import gmv, msr, ew, random_weights
from portfolio_tools.return_metrics import portfolio_returns
from portfolio_tools.risk_metrics import portfolio_volatility, calculate_max_drawdown



def get_markowtiz_results(returns: pd.DataFrame,
                     covmat: pd.DataFrame,
                     portfolio_type: Literal["msr", "gmv", "portfolio", "ew", "random"] = "msr",
                     rf: float = 0.0,
                     method: Literal["simple", "log"] = "simple",
                     periods_per_year: int = 252,
                     min_w: float = 0.00) -> Tuple[float, float, float]:
    """
    Returns the returns and volatility of a portfolio given weights of the portfolio

    Parameters
    ----------
    returns: pd.DataFrame. Expected return of the portfolio.
    covmat: np.ndarray. Covariance matrix of the portfolio.
    rf: float. Risk-free rate.
    portfolio_type: Literal["msr", "gmv", "portfolio", "ew", "random"] = "msr"
    method: str. "simple" or "log
    periods_per_year: int. Number of years over which to calculate volatility.
    min_w: float. Minimum weight of the portfolio.

    Returns
    -------
    np.ndarray: Weights of the portfolio.
    """

    if portfolio_type == "msr":
        weights = msr(returns, covmat, rf, method, periods_per_year)
    elif portfolio_type == "gmv":
        weights = gmv(covmat, min_w)
    elif portfolio_type == "portfolio":
        weights = 0
    elif portfolio_type == "ew":
        # We calculate the weights for an equally weighted portfolio
        weights = ew(returns)
    elif portfolio_type == "random":
        weights = random_weights(returns)

    else:
        raise ValueError(f"Unknown portfolio type: {portfolio_type}")

    # We get the returns
    pf_return = portfolio_returns(weights, returns, method, periods_per_year)
    # We get the volatility
    pf_volatility = portfolio_volatility(weights, covmat, periods_per_year)

    # We get the maximum drawdown
    max_drawdown = calculate_max_drawdown(weights, returns)
    print(f"Max Drawdown: {max_drawdown}")
    return pf_return, pf_volatility, max_drawdown