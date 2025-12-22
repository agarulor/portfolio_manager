import pandas as pd
import numpy as np
from typing import Literal, Tuple, Optional
from portfolio_tools.return_metrics import portfolio_returns, annualize_returns
from portfolio_tools.risk_metrics import (
    portfolio_volatility,
    neg_sharpe_ratio,
    calculate_max_drawdown,
    calculate_standard_deviation,
    annualize_standard_deviation)
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def min_max_percentage_renormalize(w: np.ndarray,
                                   min_w: float = 0.00,
                                   max_w: float = 1.00,
                                   tol: float = 1e-12) -> np.ndarray:
    """
        Obtains the minimal and maximum percentages renormalized weight for the assets

        Parameters
        ----------
        w: np.ndarray. Expected return of the portfolio.
        min_w: float. Minimum weight of the portfolio.
        max_w: float. Maximum weight of the portfolio.
        tol: float. Tolerance of the renormalization.

        Returns
        -------
        np.ndarray: Optimal weight of the portfolio adjusted
        """

    w2 = np.array(w, dtype=float, copy=True)

    # 1) Limpieza numérica
    w2[np.abs(w2) < tol] = 0.0

    mask_small = (w2 > 0) & (w2 < min_w)
    w2[mask_small] = 0.0

    if max_w < 1.0:
        w2 = np.minimum(w2, max_w)

    # 4) Renormalizar
    s = w2.sum()
    if s <= tol:
        raise ValueError("All waits are equal to 0. Relax percentages")

    w2 /= s
    return w2


def minimize_volatility(target_return: float,
                        returns: pd.DataFrame,
                        covmat: np.ndarray,
                        method: Literal["simple", "log"] = "simple",
                        periods_per_year: int =252,
                        min_w: float = 0.00,
                        max_w: float = 1.00) -> np.ndarray:
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
    max_w: float. Maximum weight of the portfolio.

    Returns
    -------
    np.ndarray: Optimal weight of the portfolio.
    """

    # Number of assets in the spectrum of assets
    n = returns.shape[1]

    # Initial guess of weights (we start with evenly weights)
    init_guess = np.ones(n) / n

    # We ensure that there is no short-selling (i.e. no short positions)
    bounds = [(0, max_w)] * n

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

    weights = min_max_percentage_renormalize(result.x, min_w, max_w)
    return weights


def maximize_return(target_volatility: float,
                    returns: pd.DataFrame,
                    covmat: np.ndarray,
                    method: Literal["simpl  e", "log"] = "simple",
                    periods_per_year: int = 252,
                    min_w: float = 0.00,
                    max_w: float = 1.00,
                    sectors_df: Optional[pd.DataFrame] = None,
                    ticker_col: str = "ticker",
                    sector_col: str = "sector",
                    sector_max_weight: Optional[float] = None,
                    risk_free_ticker: str = "RISK_FREE") -> np.ndarray:
    """
    Returns the optimal weight of the portfolio assets that maximize
    return for a given target volatility, returns and a covariance matrix.

    Parameters
    ----------
    target_volatility: float. target volatility of the portfolio.
    returns: pd.DataFrame. Past returns of the portfolio.
    covmat: np.ndarray. Covariance matrix of the portfolio.
    method: str. "simple" or "log"
    periods_per_year: int. Number of periods, n= 252 per yeras, 12 for months over which to calculate volatility.
    min_w: float. Minimum weight of the portfolio.
    max_w: float. Maximum weight of the portfolio.
    sectors_df: pd.DataFrame. Sectors dataframe.
    ticker_col: str. Column name of the ticker.
    sector_col: str. Column name of the sector column.
    sector_max_weight: float. Maximum weight of the sector column.

    Returns
    -------
    np.ndarray: Optimal weight of the portfolio.
    """

    # Number of assets in the spectrum of assets
    n = returns.shape[1]

    # Initial guess of weights (we start with evenly weights)
    init_guess = np.ones(n) / n

    # We ensure that there is no short-selling (i.e. no short positions)
    bounds = [(0, max_w)] * n

    # We add the constraints to the model
    # Weights must sum 1 (fully invested)
    # Volatility constraint
    constraints = [
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
        {"type": "eq", "fun": lambda w: portfolio_volatility(w, covmat, periods_per_year) - target_volatility}
    ]

    # We add sector restrictions
    if sectors_df is not None and sector_max_weight is not None:
        if ticker_col not in sectors_df.columns:
            raise ValueError(f"Column '{ticker_col}' not found in sectors_df")

        if sector_col not in sectors_df.columns:
            raise ValueError(f"Column '{sector_col}' not found in sectors_df")

        # We index by ticker
        df = sectors_df.set_index(ticker_col)

        tickers_to_check = [t for t in returns.columns if t != risk_free_ticker]

        missing = [t for t in tickers_to_check if t not in df.index]
        if missing:
            raise ValueError(f"There is / are tickers with no sector: {missing}")

        # We align with the order of returns.columns
        sector_series = df.reindex(returns.columns)[sector_col]

        # We creat a restriction for each sector
        for sect in sector_series.dropna().unique():
            idx = np.where(sector_series.values == sect)[0]
            if len(idx) == 0:
                continue

            # type "ineq": fun(w) >= 0
            #  sum_{i in industria} w_i <= industry_max_weight
            # => industry_max_weight - sum(w_i) >= 0
            constraints.append(
                {
                    "type": "ineq",
                    "fun": lambda w, idx=idx: sector_max_weight - np.sum(w[idx])
                }
            )

    result = minimize(lambda w: -portfolio_returns(w, returns, method, periods_per_year),
                      init_guess,
                      method='SLSQP',
                      options={'disp': False},
                      constraints=constraints,
                      bounds=bounds)

    weights = min_max_percentage_renormalize(result.x, min_w, max_w)
    return weights


def get_weights(n_returns: int,
                returns: pd.DataFrame,
                covmat: np.ndarray,
                method: Literal["simple", "log"] = "simple",
                periods_per_year: int =252,
                min_w: float = 0,
                max_w: float = 1.00) -> np.ndarray:

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
    min_w: float. Minimum weight of the portfolio.
    max_w: float. Maximum weight of the portfolio.

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
                                   periods_per_year=periods_per_year,
                                   min_w=min_w,
                                   max_w=max_w) for target_return in target_returns]

    return weights


def msr(returns,
        covmat: np.ndarray,
        rf: float = 0,
        method: Literal["simple", "log"] = "simple",
        periods_per_year: int = 252,
        min_w: float = 0.00,
        max_w: float = 1.00,
        ) -> np.ndarray:
    """
    Returns the weights of the portfolio assets that maximize the Sharpe Ratio
    given the risk-free rate, the covariance matrix of returns and the expected
    returns

    Parameters
    ----------
    returns: pd.DataFrame. Expected return of the portfolio.
    covmat: np.ndarray. Covariance matrix of the portfolio.
    rf: float. Risk-free rate.
    method: str. "simple" or "log
    periods_per_year: int. Number of years over which to calculate volatility.
    min_w: float. Minimum weight of the portfolio.
    max_w: float. Maximum weight of the portfolio.

    Returns
    -------
    np.ndarray: Weights of the portfolio.
    """
    # We get the number of assets
    n = returns.shape[1]

    # Initial guess of weights (we start with evenly weights)
    init_guess = np.ones(n) / n

    # We ensure that there is no short-selling (i.e. no short positions)
    bounds = [(0, max_w)] * n

    # constraints
    # Weights must sum 1 (fully invested)
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1},
    )

    # For the function, we need to create a function that gives us the
    # negative Sharpe Ratio
    weights = minimize(neg_sharpe_ratio,
                       init_guess,
                       args=(returns, covmat, rf, method, periods_per_year),
                       method='SLSQP',
                       options={'disp': False, 'maxiter': 1000},
                       constraints=constraints,
                       bounds=bounds
                       )

    weights = min_max_percentage_renormalize(weights.x, min_w, max_w)
    return weights


def gmv(covmat: pd.DataFrame,
        min_w: float = 0.0,
        max_w: float = 1.0) -> np.ndarray:
    """
    Returns the weights of the portfolio assets that helps to meet the GMV

    Parameters
    ----------
    covmat: np.ndarray. Covariance matrix of the portfolio.
    min_w: float. Minimum weight of the portfolio.
    max_w: float. Maximum weight of the portfolio.

    Returns
    -------
    np.ndarray: Weights of the portfolio.
    """

    # We get the number of assets
    n = covmat.shape[0]

    # Create initial guess
    init_guess = np.ones(n) / n

    # We ensure that there is no short-selling (i.e. no short positions)
    bounds = [(0, max_w)] * n

    # constraints
    # Weights must sum 1 (fully invested)
    constraints = (
        {"type": "eq", "fun": lambda w: np.sum(w) - 1.0},
    )

    # Minimize the function to get the MGV
    weights = minimize(lambda w: float(w @ covmat @ w),
                   init_guess,
                   method='COBYLA',
                   bounds=bounds,
                   constraints=constraints,
                   options={'maxiter': 10000})

    if not weights.success:
        raise ValueError(f"GMV optimization failed: {weights.message}")
    weights = min_max_percentage_renormalize(weights.x, min_w, max_w)
    return weights

def ew(returns: pd.DataFrame) -> np.ndarray:
    """
    Returns the weights of the portfolio assets for an equally weighted portfolio

    Parameters
    ----------
    returns: np.ndarray. Returns of the portfolio.

    Returns
    -------
    np.ndarray: Weights of the portfolio.
    """
    # We get the number of assets
    n = returns.shape[1]
    # We calculate the weights for an equally weighted portfolio
    return np.ones(n) / n


def random_weights(returns: pd.DataFrame) -> np.ndarray:
    """
    Returns the weights of the portfolio assets for a random weighted portfolio

    Parameters
    ----------
    returns: np.ndarray. Returns of the portfolio.

    Returns
    -------
    np.ndarray: Weights of the portfolio.
    """
    n = returns.shape[1]

    # We return the weights
    return np.random.dirichlet([0.15] * n)

def get_weights_from_min_volatility(n_volatilities: int,
                                    returns: pd.DataFrame,
                                    covmat: np.ndarray,
                                    method: Literal["simple", "log"] = "simple",
                                    periods_per_year: int =252,
                                    min_w: float = 0,
                                    max_w: float = 0) -> np.ndarray:
    """
    Returns the optimal weight of the portfolio assets that maximize returns
    for a given target volatility, returns and a covariance matrix.

    Parameters
    ----------
    n_volatilities : int. volatilities of the portfolio
    returns: pd.DataFrame. Expected return of the portfolio.
    covmat: np.ndarray. Covariance matrix of the portfolio.
    method: str. "simple" or "log
    periods_per_year: int. Number of years over which to calculate volatility.
    min_w: float. Minimum weight of the portfolio.
    max_w: float. Maximum weight of the portfolio.

    Returns
    -------
    np.ndarray: Optimal weight of the portfolio.
    """

    # we obtain the volatilities
    # We first get the min_volatility from gmv
   # gmv_return, gmv_volatility, gmv_drawdown = portfolio_output(returns, covmat, "gmv")

    # now we obtain the highest volatility
    stds = calculate_standard_deviation(returns)
    annualized_stds = annualize_standard_deviation(stds, periods_per_year)
    vol_max = annualized_stds.max()

    # We obtain a series of points based on the min and max volatility
    target_volatilities = np.linspace(0, vol_max, n_volatilities)
    # We now obtain the weights for each of the target_volatility
    weights = [maximize_return(target_volatility,
                               returns,
                               covmat,
                               method=method,
                               periods_per_year=periods_per_year,
                               min_w=min_w,
                               max_w= max_w) for target_volatility in target_volatilities]

    return weights


def portfolio_output(returns: pd.DataFrame,
                     covmat: pd.DataFrame,
                     portfolio_type: Literal["msr", "gmv", "ew", "random"] = "msr",
                     rf: float = 0.0,
                     method: Literal["simple", "log"] = "simple",
                     periods_per_year: int = 252,
                     min_w: float = 0.00,
                     max_w: float = 1.00) -> Tuple[float, float, float]:
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
    max_w: float. Maximum weight of the portfolio.

    Returns
    -------
    np.ndarray: Weights of the portfolio.
    """
    if portfolio_type == "msr":
        weights = msr(returns, covmat, rf, method, periods_per_year, min_w, max_w)
    elif portfolio_type == "gmv":
        weights = gmv(covmat, min_w, max_w)
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
    return pf_return, pf_volatility, max_drawdown


def get_cml(target_volatility: float,
            returns: pd.DataFrame,
            covmat: pd.DataFrame,
            rf: float = 0.0,
            method: Literal["simple", "log"] = "simple",
            periods_per_year: int = 252,
            min_w: float = 0.00,
            max_w: float = 1.00) -> pd.DataFrame:

    """
    It helps an investor to select a point of the CML given a
    volatility target

    Parameters
    ----------
    target_volatility: float. Target volatility of the portfolio.
    returns: pd.DataFrame. Expected return of the portfolio.
    covmat: np.ndarray. Covariance matrix of the portfolio.
    rf: float. Risk-free rate.
    method: str. "simple" or "log
    periods_per_year: int. Number of years over which to calculate volatility.
    min_w: float. Minimum weight of the portfolio.
    max_w: float. Maximum weight of the portfolio.

    Returns
    -------
    w_risky : np.ndarray. Weights over risk assets in the CML.
    w_rf : float. Weight of the risk-free asset
    port_return : float. Annualized return of the portfolio.
    port_vol : float Annualized volatility of the portfolio.
    sharpe : float
        Ratio de Sharpe del punto en la CML, usando rf.
    """
    # We first obtain the msr
    sharpe_w = msr(returns, covmat, rf, method, periods_per_year, min_w, max_w)

    # We get the returns
    pf_return = portfolio_returns(sharpe_w, returns, method, periods_per_year)
    # We get the volatility
    pf_volatility = portfolio_volatility(sharpe_w, covmat, periods_per_year)

    # We check if vol is lesser than 0
    if pf_volatility <= 0:
        raise ValueError("Volatility cannot be less than zero")

    # We escalate to meet target volatility
    a = target_volatility / pf_volatility
    weight_risky = a * sharpe_w
    weight_risk_free = 1.0 - a


    # We get the reamining information
    cml_pf_volatility = float(abs(a) * pf_volatility)
    cml_pf_return = float(rf + a * (pf_return - rf))

    return weight_risky, weight_risk_free, cml_pf_return, cml_pf_volatility
