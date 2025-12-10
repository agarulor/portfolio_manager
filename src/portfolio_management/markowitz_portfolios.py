import pandas as pd
import numpy as np
from typing import Literal, Tuple

from portfolio_tools.markowitz import gmv, msr, ew, random_weights, maximize_return
from portfolio_tools.return_metrics import portfolio_returns
from portfolio_tools.risk_metrics import portfolio_volatility, calculate_max_drawdown, calculate_covariance


# We create an aux function for the client case. In this case, we add the risk free asset to the potential portfolio
def add_risk_free_asset(
        returns: pd.DataFrame,
        covmat: pd.DataFrame,
        rf_annual: float,
        periods_per_year: int) -> Tuple[pd.DataFrame, np.ndarray]:

    # we adjust it to the number of periods (approximately)
    rf_per_period = (1 + rf_annual)**(1/periods_per_year) - 1

    rf_series = pd.Series(rf_per_period, index=returns.index, name="RISK_FREE")

    # We add this new column to the returns dataframe
    returns_ext = pd.concat([returns, rf_series], axis=1)

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
                periods_per_year: int = 252,
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
                                   periods_per_year: int = 252,
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
    weight_name: str. Name of the weight column.
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
    # We remove the values equal to 0
    df_weights = df_weights[df_weights["Pesos"] > 0]
    df_weights = df_weights.sort_values(by=["Pesos"], ascending=False)

    return df_results, df_weights






def get_initial_portfolio(returns: pd.DataFrame,
                          portfolio_type: Literal["msr", "gmv", "ew", "random", "investor"] = "msr",
                          method: Literal["simple", "log"] = "simple",
                          periods_per_year: int = 252,
                          min_w: float = 0.00,
                          max_w: float = 1.00,
                          rf_annual: float | None = None,
                          custom_target_volatility: float = 0.15,
                          weight_name: str = "weights") -> Tuple[float, float, float]:
    """
    Returns the returns and volatility of a portfolio given weights of the portfolio

    Parameters
    ----------
    returns: pd.DataFrame. Expected return of the portfolio.
    portfolio_type: Literal["msr", "gmv", "ew", "random", "custom"] = "msr"
    method: str. "simple" or "log
    periods_per_year: int. Number of years over which to calculate volatility.
    min_w: float. Minimum weight of the portfolio.
    max_w: float. Maximum weight of the portfolio.
    custom_target_volatility: float. Custom target volatility.
    weight_name: str. Name of the weight column.
    rf_annual: float. Risk free annual.
    Returns
    -------
    dictionary: With name of the type of model, returns, volatility, max drawdown and weights
    """

    covmat = calculate_covariance(returns)
    if portfolio_type == "msr":
        weights = msr(returns, covmat, rf_annual, method, periods_per_year)
    elif portfolio_type == "gmv":
        weights = gmv(covmat)
    elif portfolio_type == "ew":
        # We calculate the weights for an equally weighted portfolio
        weights = ew(returns)
    elif portfolio_type == "random":
        weights = random_weights(returns)
    elif portfolio_type == "investor":
        if rf_annual is not None:
            returns, covmat = add_risk_free_asset(returns, covmat, rf_annual, periods_per_year)
        weights = get_investor_weights(returns,
                                       covmat,
                                       method,
                                       min_w,
                                       max_w,
                                       custom_target_volatility)

    else:
        raise ValueError(f"Unknown portfolio type: {portfolio_type}")

    # We get the historical returns
    pf_return = portfolio_returns(weights, returns, method, periods_per_year)

    # We get the historical volatility
    pf_volatility = portfolio_volatility(weights, covmat, periods_per_year)

    # We calculate the sharpe ratio
    portfolio_sharpe_ratio = (pf_return - rf_annual) / pf_volatility

    # We get the maximum drawdown
    max_drawdown = calculate_max_drawdown(weights, returns)

    # We add a new element in case portfolio type is not custom and if we are using RF_Rate
    if portfolio_type != "investor" and rf_annual is not None:
        weights = np.append(weights, 0.0)

    # We convert it into a dictionary and multiply by 100 to get %
    portfolio_information = {"Model": portfolio_type,
                             "Returns": float(round(pf_return * 100, 3)),
                             "Volatility": float(round(pf_volatility * 100, 3)),
                             "Sharpe Ratio": float(round(portfolio_sharpe_ratio, 3)),
                             "max_drawdown": float(round(max_drawdown * 100, 3)),
                             weight_name: np.round(weights * 100, 3) }

    return portfolio_information


def show_initial_portfolio(returns: pd.DataFrame,
                           portfolio_types: Literal["msr", "gmv", "ew", "random", "investor"] = None,
                           method: Literal["simple", "log"] = "simple",
                           periods_per_year: int = 252,
                           min_w: float = 0.00,
                           max_w: float = 1.00,
                           rf_annual: float = None,
                           custom_target_volatility: float = 0.15,
                           weight_name: str = "weights",
                           ) -> pd.DataFrame:
    """
    Returns the returns and volatility of a portfolio given weights of the portfolio

    Parameters
    ----------
    returns: pd.DataFrame. Expected return of the portfolio.
    portfolio_types: List with they type of portfolio names.
    method: str. "simple" or "log
    periods_per_year: int. Number of years over which to calculate volatility.
    min_w: float. Minimum weight of the portfolio.
    max_w: float. Maximum weight of the portfolio.
    custom_target_volatility: float. Custom target volatility for potential investor
    weight_name: str. Name of the weight column.
    rf_annual: float. Risk free annual.

    Returns
    -------
    df_results : pd.DataFrame With name of the type of model, returns, volatility, max drawdown and weights
    """
    portfolio_results = []

    # extraemos el nombre de las columnas
    tickers = returns.columns
    if rf_annual is not None:
        tickers = tickers.append(pd.Index(["Risk Free"]))

    if portfolio_types is None:
        portfolio_types = ["msr", "gmv", "ew", "random", "investor"]

    for portfolio in portfolio_types:
        print(portfolio)
        resultados = get_initial_portfolio(
            returns,
            portfolio,
            method,
            periods_per_year,
            min_w,
            max_w,
            rf_annual,
            custom_target_volatility,
            weight_name)
        portfolio_results.append(resultados)


    df_results = pd.DataFrame(portfolio_results)

    # We extract the column weights
    new_columns = df_results[weight_name].apply(pd.Series)

    # we add the ticker names
    new_columns.columns = tickers

    # We join it to the dataframe replacing weights
    df_results = pd.concat([df_results.drop(weight_name, axis=1), new_columns], axis=1)

    return df_results
