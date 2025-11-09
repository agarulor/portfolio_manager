import pandas as pd
import numpy as np
from typing import Literal, Tuple
from portfolio_tools.return_metrics import portfolio_returns, annualize_returns
from portfolio_tools.risk_metrics import portfolio_volatility, neg_sharpe_ratio, sharpe_ratio, calculate_max_drawdown
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
                        min_w: float = 0.00) -> np.ndarray:
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
                    min_w: float = 0.00) -> np.ndarray:
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
                periods_per_year: int =252,
                min_w: float = 0) -> np.ndarray:

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
                                   min_w=min_w) for target_return in target_returns]

    return weights


def msr(returns,
        covmat: np.ndarray,
        rf: float = 0,
        method: Literal["simple", "log"] = "simple",
        periods_per_year: int = 252,
        min_w: float = 0.00
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

    Returns
    -------
    np.ndarray: Weights of the portfolio.
    """
    # We get the number of assets
    n = returns.shape[1]

    # Initial guess of weights (we start with evenly weights)
    init_guess = np.ones(n) / n

    # We ensure that there is no short-selling (i.e. no short positions)
    bounds = [(0, 1)] * n

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
                       options={'disp': False},
                       constraints=constraints,
                       bounds=bounds
                       )

    # We adjust for min % of assets
    weights = min_percentage_renormalize(weights.x, min_w)

    return weights



def gmv(covmat: pd.DataFrame,
        min_w: float = 0.0) -> np.ndarray:
    """
    Returns the weights of the portfolio assets that helps to meet the GMV

    Parameters
    ----------
    covmat: np.ndarray. Covariance matrix of the portfolio.
    min_w: float. Minimum weight of the portfolio.

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
    bounds = [(0, 1.0)] * n

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
                   options={'maxiter': 1000})

    weights = min_percentage_renormalize(weights.x, min_w)
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



def portfolio_output(returns: pd.DataFrame,
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


def plot_frontier(n_returns: int,
                  returns: pd.DataFrame,
                  covmat: np.ndarray,
                  rf: float = 0.0,
                  method: Literal["simple", "log"] = "simple",
                  periods_per_year: int = 252,
                  min_w: float = 0.00,
                  plot_msr = True,
                  plot_cml = True,
                  plot_gmv = True,
                  style: str = '.-',
                  legend: bool = False) -> plt.Figure:


    weights = get_weights(n_returns, returns, covmat, method, periods_per_year, min_w)
    retornos = [portfolio_returns(w, returns, method,periods_per_year) for w in weights]
    volatilities = [portfolio_volatility(w, covmat, periods_per_year) for w in weights]
    ef = pd.DataFrame({
        "Returns": retornos,
        "Volatility": volatilities
    })
    ax = ef.plot.line(x="Volatility", y="Returns", style=style, legend=legend)
    if plot_msr:
        msr_return, msr_volatility, msr_drawdown = portfolio_output(returns, covmat, "msr", rf, method, periods_per_year, min_w)
        ax.plot(msr_volatility, msr_return, color='midnightblue', marker='o', markersize=12)

        if plot_cml:
            ax.set_xlim(left=0)
            cml_x = [0, msr_volatility]
            cml_y = [rf, msr_return]
            ax.plot(cml_x, cml_y, color='green', marker='o', linestyle='dashed', linewidth=2, markersize=10)

    if plot_gmv:
        gmv_return, gmv_volatility, gmv_drawdown = portfolio_output(returns, covmat, "gmv")
        ax.plot(gmv_volatility, gmv_return, color='red', marker='o', markersize=12)

    ew_return, ew_volatility, ew_drawdown = portfolio_output(returns, covmat, "ew")
    ax.plot(ew_volatility, ew_return, color='salmon', marker='o', markersize=12)

    random_return, random_volatility, random_drawdown = portfolio_output(returns, covmat, "random")
    ax.plot(random_volatility, random_return, color='black', marker='o', markersize=12)

    plt.show()
    return ax

