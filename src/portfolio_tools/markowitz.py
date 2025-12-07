import pandas as pd
import numpy as np
from typing import Literal, Tuple
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

    # 2) Cortar por abajo: 0 ó >= min_w
    mask_small = (w2 > 0) & (w2 < min_w)
    w2[mask_small] = 0.0

    # 3) Cortar por arriba
    if max_w < 1.0:
        w2 = np.minimum(w2, max_w)

    # 4) Renormalizar
    s = w2.sum()
    if s <= tol:
        raise ValueError(
            "Todos los pesos han quedado a cero tras aplicar min_w / max_w. "
            "Relaja las restricciones."
        )

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
                    max_w: float = 1.00) -> np.ndarray:
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
    # We extract values
    covmat_values = covmat.values

    # We get the number of assets
    n = covmat_values.shape[0]

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
    weights = minimize(lambda w: float(w @ covmat_values @ w),
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
                     portfolio_type: Literal["msr", "gmv", "portfolio", "ew", "random"] = "msr",
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


def plot_frontier(n_returns: int,
                  returns: pd.DataFrame,
                  covmat: np.ndarray,
                  rf: float = 0.0,
                  method: Literal["simple", "log"] = "simple",
                  periods_per_year: int = 252,
                  min_w: float = 0.00,
                  max_w: float = 1.00,
                  plot_msr = True,
                  plot_cml = True,
                  plot_gmv = True,
                  style: str = '.-',
                  legend: bool = False) -> plt.Figure:


    weights = get_weights(n_returns, returns, covmat, method, periods_per_year, min_w, max_w)
    weights_2 = get_weights_from_min_volatility(n_returns, returns, covmat, method, periods_per_year, min_w, max_w)
    retornos = [portfolio_returns(w, returns, method,periods_per_year) for w in weights]
    volatilities = [portfolio_volatility(w, covmat, periods_per_year) for w in weights]
    retornos_2 = [portfolio_returns(w, returns, method, periods_per_year) for w in weights_2]
    volatilities_2 = [portfolio_volatility(w, covmat, periods_per_year) for w in weights_2]
    pesos3= maximize_return(0.17, returns, covmat, max_w=max_w)
    rentabilidad = portfolio_returns(pesos3, returns, method, periods_per_year)
    volatilidad = portfolio_volatility(pesos3, covmat, periods_per_year)
    ef = pd.DataFrame({
        "Returns": retornos,
        "Volatility": volatilities
    })

    ef_2 = pd.DataFrame({
        "Returns": retornos_2,
        "Volatility": volatilities_2
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    ax.plot(ef_2["Volatility"], ef_2["Returns"], color = "red")
    ax.plot(volatilidad, rentabilidad, color = "orange", marker='o', markersize=14)
    if plot_msr:
        msr_return, msr_volatility, msr_drawdown = portfolio_output(returns,
                                                                    covmat,
                                                                    "msr",
                                                                    rf,
                                                                    method,
                                                                    periods_per_year,
                                                                    min_w,
                                                                    max_w)
        ax.plot(msr_volatility, msr_return, color='midnightblue', marker='o', markersize=12)

        if plot_cml:
            ax.set_xlim(left=0)
            cml_x = [0, msr_volatility]
            cml_y = [rf, msr_return]
            cml_weights, cml_rf_weight, cml_return, cml_volatility = get_cml(0.12,
                                                                             returns,
                                                                             covmat,
                                                                             rf, method,
                                                                             periods_per_year,
                                                                             min_w,
                                                                             max_w)
            ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)
            ax.plot(cml_volatility, cml_return, color='yellow', marker='o', markersize=16)

    if plot_gmv:
        gmv_return, gmv_volatility, gmv_drawdown = portfolio_output(returns,
                                                                    covmat,
                                                                    "gmv",
                                                                    min_w=min_w,
                                                                    max_w=max_w)
        ax.plot(gmv_volatility, gmv_return, color='red', marker='o', markersize=12)

    ew_return, ew_volatility, ew_drawdown = portfolio_output(returns, covmat, "ew")
    ax.plot(ew_volatility, ew_return, color='salmon', marker='o', markersize=12)

    random_return, random_volatility, random_drawdown = portfolio_output(returns, covmat, "random")
    ax.plot(random_volatility, random_return, color='black', marker='o', markersize=12)

    plt.show()
    return ax

