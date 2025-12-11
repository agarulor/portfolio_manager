import pandas as pd
import numpy as np
from typing import Literal, Tuple

from portfolio_tools.markowitz import  maximize_return
from portfolio_tools.return_metrics import portfolio_returns
from portfolio_tools.risk_metrics import portfolio_volatility, calculate_max_drawdown, calculate_covariance


# We create an aux function for the client case. In this case, we add the risk free asset to the potential portfolio

def add_risk_free_column(returns: pd.DataFrame,
                         rf_annual: float,
                         periods_per_year: float = 252) -> pd.DataFrame:
    # we adjust it to the number of periods (approximately)
    rf_per_period = (1 + rf_annual) ** (1 / periods_per_year) - 1

    rf_series = pd.Series(rf_per_period, index=returns.index, name="RISK_FREE")

    # We add this new column to the returns dataframe
    returns_ext = pd.concat([returns, rf_series], axis=1)

    return returns_ext

def add_risk_free_asset(
        returns: pd.DataFrame,
        covmat: pd.DataFrame,
        rf_annual: float,
        periods_per_year: float) -> Tuple[pd.DataFrame, np.ndarray]:

    returns_ext = add_risk_free_column(returns, rf_annual, periods_per_year)# we adjust it to the number of periods (approximately)
    # We increase the covmat matrix by adding the new asset, taking into account that it has a VAR = 0 and COV = 0
    # with any other risky asset
    n = covmat.shape[0]
    cov_ext = np.zeros((n + 1, n + 1))
    cov_ext[:n, :n] = covmat

    return returns_ext, cov_ext


def get_investor_weights(returns: pd.DataFrame,
                         covmat: pd.DataFrame,
                         method: Literal["simple", "log"] = "simple",
                         min_w: float = 0.00,
                         max_w: float = 1.00,
                         custom_target_volatility: float = 0.15) -> np.ndarray:

    weights = maximize_return(custom_target_volatility, returns, covmat, min_w = min_w, max_w=max_w)
    return weights


def get_results(returns: pd.DataFrame,
                covmat: pd.DataFrame,
                weights: pd.DataFrame,
                method: Literal["simple", "log"] = "simple",
                periods_per_year: float = 252,
                rf_annual: float|None = None) -> Tuple[float, float, float]:

    # We get the historical returns
    pf_return = portfolio_returns(weights, returns, method, periods_per_year)

    # We get the historical volatility
    pf_volatility = portfolio_volatility(weights, covmat, periods_per_year)

    # We calculate the sharpe ratio
    portfolio_sharpe_ratio = (pf_return - rf_annual) / pf_volatility

    # We get the maximum drawdown
    max_drawdown = calculate_max_drawdown(weights, returns)
    # We convert it into a dictionary and multiply by 100 to get %
    portfolio_information = {"Returns": float(round(pf_return * 100, 3)),
                             "Volatility": float(round(pf_volatility * 100, 3)),
                             "Sharpe Ratio": float(round(portfolio_sharpe_ratio, 3)),
                             "max_drawdown": float(round(max_drawdown * 100, 3)) }

    return portfolio_information

def get_investor_initial_portfolio(returns: pd.DataFrame,
                                   method: Literal["simple", "log"] = "simple",
                                   periods_per_year: float = 252,
                                   min_w: float = 0.00,
                                   max_w: float = 1.00,
                                   rf_annual: float | None = None,
                                   custom_target_volatility: float = 0.15) -> pd.DataFrame:
    """
    Returns the returns and volatility of a portfolio given weights of the portfolio

    Parameters
    ----------
    returns: pd.DataFrame. Expected return of the portfolio.
    method: str. "simple" or "log
    periods_per_year: int. Number of years over which to calculate volatility.
    min_w: float. Minimum weight of the portfolio.
    max_w: float. Maximum weight of the portfolio.
    custom_target_volatility: float. Custom target volatility.
    rf_annual: float. Risk free annual.
    Returns
    -------
    dictionary: With name of the type of model, returns, volatility, max drawdown and weights
    """

    covmat = calculate_covariance(returns)
    if rf_annual is not None:
        returns, covmat = add_risk_free_asset(returns, covmat, rf_annual, periods_per_year)
    weights = get_investor_weights(returns,
                                   covmat,
                                   method,
                                   min_w,
                                   max_w,
                                   custom_target_volatility)

    resultados = get_results(returns, covmat, weights, method, periods_per_year, rf_annual)
    # we now extract tickers

    tickers = returns.columns
    #  We create a dataframe with the results
    df_results = pd.DataFrame(resultados, index=[0])

    df_weights = pd.DataFrame({"Pesos": weights}, index=tickers)
    # put a name to the index
    df_weights.index.name = "Ticker"
    # We remove the values equal to 0
    df_weights = df_weights[df_weights["Pesos"] > 0]
    df_weights = df_weights.sort_values(by=["Pesos"], ascending=False)

    return df_results, df_weights, weights


def get_updated_results(returns: pd.DataFrame,
                        weights: pd.DataFrame,
                        method: Literal["simple", "log"] = "simple",
                        periods_per_year: float = 252,
                        rf_annual: float | None = None,):
    covmat = calculate_covariance(returns)
    if rf_annual is not None:
        returns, covmat = add_risk_free_asset(returns, covmat, rf_annual, periods_per_year)
    resultados = get_results(returns, covmat, weights, method, periods_per_year, rf_annual)
    df_results = pd.DataFrame(resultados, index=[0])
    return df_results


def get_cumulative_returns(returns: pd.DataFrame,
                           weights: np.ndarray,
                           initial_investment: float = 1,
                           rf_annual: float | None = None,
                           periods_per_year: float = 252) -> pd.DataFrame:

    money_invested = weights * initial_investment
    if rf_annual is not None:
        returns = add_risk_free_column(returns, rf_annual, periods_per_year)

    adjust_returns = (1+returns)

    adjusted_returns = adjust_returns.cumprod()

    adjusted_value = adjusted_returns * money_invested

    adjusted_total_value = adjusted_value.sum(axis=1)

    print(adjusted_total_value.shape)

    return adjusted_total_value




