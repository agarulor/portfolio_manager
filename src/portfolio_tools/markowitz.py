import pandas as pd
import numpy as np
import streamlit as st
from typing import Literal, Tuple, Optional
from portfolio_tools.return_metrics import portfolio_returns, annualize_returns
from portfolio_tools.risk_metrics import (
    portfolio_volatility,
    neg_sharpe_ratio,
    calculate_covariance)
from scipy.optimize import minimize
import matplotlib.pyplot as plt


def min_max_percentage_renormalize(w: np.ndarray,
                                   min_w: float = 0.00,
                                   max_w: float = 1.00,
                                   tol: float = 1e-12) -> np.ndarray:
    """
    Computes min max percentage renormalize.

    Parameters
    ----------
    w : np.ndarray. w.
    min_w : float. min w.
    max_w : float. max w.
    tol : float. tol.

    Returns
    -------
    np.ndarray: min max percentage renormalize output.
    """

    w2 = np.array(w, dtype=float, copy=True)

    # 1) Numeric cleanup
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
                        periods_per_year: int = 252,
                        min_w: float = 0.00,
                        max_w: float = 1.00) -> np.ndarray:
    """
    Computes minimize volatility.

    Parameters
    ----------
    target_return : float. target return.
    returns : pd.DataFrame. Returns of the assets.
    covmat : np.ndarray. covmat.
    method : Literal["simple", "log"]. method.
    periods_per_year : int. periods per year.
    min_w : float. min w.
    max_w : float. max w.

    Returns
    -------
    np.ndarray: minimize volatility output.
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
    Computes maximize return.

    Parameters
    ----------
    target_volatility : float. target volatility.
    returns : pd.DataFrame. Returns of the assets.
    covmat : np.ndarray. covmat.
    method : Literal["simpl  e", "log"]. method.
    periods_per_year : int. periods per year.
    min_w : float. min w.
    max_w : float. max w.
    sectors_df : Optional[pd.DataFrame]. sectors df.
    ticker_col : str. ticker col.
    sector_col : str. sector col.
    sector_max_weight : Optional[float]. sector max weight.
    risk_free_ticker : str. risk free ticker.

    Returns
    -------
    np.ndarray: maximize return output.
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
                periods_per_year: int = 252,
                min_w: float = 0,
                max_w: float = 1.00) -> np.ndarray:
    """
    Gets weights.

    Parameters
    ----------
    n_returns : int. n returns.
    returns : pd.DataFrame. Returns of the assets.
    covmat : np.ndarray. covmat.
    method : Literal["simple", "log"]. method.
    periods_per_year : int. periods per year.
    min_w : float. min w.
    max_w : float. max w.

    Returns
    -------
    np.ndarray: get weights output.
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
    Computes msr.

    Parameters
    ----------
    returns : Any. Returns of the assets.
    covmat : np.ndarray. covmat.
    rf : float. rf.
    method : Literal["simple", "log"]. method.
    periods_per_year : int. periods per year.
    min_w : float. min w.
    max_w : float. max w.

    Returns
    -------
    np.ndarray: msr output.
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
    Computes gmv.

    Parameters
    ----------
    covmat : pd.DataFrame. covmat.
    min_w : float. min w.
    max_w : float. max w.

    Returns
    -------
    np.ndarray: gmv output.
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
    Computes ew.

    Parameters
    ----------
    returns : pd.DataFrame. Returns of the assets.

    Returns
    -------
    np.ndarray: ew output.
    """
    # We get the number of assets
    n = returns.shape[1]
    # We calculate the weights for an equally weighted portfolio
    return np.ones(n) / n


def random_weights(returns: pd.DataFrame) -> np.ndarray:
    """
    Computes random weights.

    Parameters
    ----------
    returns : pd.DataFrame. Returns of the assets.

    Returns
    -------
    np.ndarray: random weights output.
    """
    n = returns.shape[1]

    # We return the weights
    return np.random.dirichlet([0.15] * n)


@st.cache_data(show_spinner=False)
def compute_efficient_frontier(
        returns: pd.DataFrame,
        n_returns: int,
        method: str,
        periods_per_year: int,
) -> pd.DataFrame:
    """
    Computes efficient frontier.

    Parameters
    ----------
    returns : pd.DataFrame. Returns of the assets.
    n_returns : int. n returns.
    method : str. method.
    periods_per_year : int. periods per year.

    Returns
    -------
    pd.DataFrame: compute efficient frontier output.
    """
    covmat = calculate_covariance(returns)
    weights = get_weights(n_returns, returns, covmat, method, periods_per_year)

    retornos = [portfolio_returns(w, returns, method, periods_per_year) for w in weights]
    volatilities = [portfolio_volatility(w, covmat, periods_per_year) for w in weights]

    return pd.DataFrame({
        "Retorno anualizado": retornos,
        "Volatilidad": volatilities
    })
