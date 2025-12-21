import pandas as pd
import numpy as np
from typing import Literal, Tuple, Optional

from portfolio_tools.markowitz import  maximize_return, msr, gmv, ew, random_weights
from portfolio_tools.return_metrics import portfolio_returns
from portfolio_tools.risk_metrics import portfolio_volatility, calculate_max_drawdown, calculate_covariance, max_drawdown_from_value_series


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
                         custom_target_volatility: float = 0.15,
                         sectors_df: Optional[pd.DataFrame] = None,
                         ticker_col: str = "ticker",
                         sector_col: str = "sector",
                         sector_max_weight: Optional[float] = None,
                         risk_free_ticker: str = "RISK_FREE") -> np.ndarray:

    weights = maximize_return(custom_target_volatility,
                              returns,
                              covmat,
                              min_w = min_w,
                              max_w=max_w,
                              sectors_df=sectors_df,
                              ticker_col=ticker_col,
                              sector_col=sector_col,
                              sector_max_weight=sector_max_weight,
                              risk_free_ticker=risk_free_ticker)

    return weights


def get_results(returns: pd.DataFrame,
                covmat: pd.DataFrame,
                weights: pd.DataFrame,
                method: Literal["simple", "log"] = "simple",
                periods_per_year: float = 252,
                rf_annual: float|None = None) -> dict:

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
                                   portfolio_type: Literal["investor", "msr", "gmv", "ew", "random"] = "investor",
                                   periods_per_year: float = 252,
                                   min_w: float = 0.00,
                                   max_w: float = 1.00,
                                   rf_annual: float | None = None,
                                   custom_target_volatility: float = 0.15,
                                   sectors_df: Optional[pd.DataFrame] = None,
                                   ticker_col: str = "ticker",
                                   sector_col: str = "sector",
                                   sector_max_weight: Optional[float] = None,
                                   risk_free_ticker: str = "RISK_FREE"
                                   ) -> pd.DataFrame:
    """
    Returns the returns and volatility of a portfolio given weights of the portfolio

    Parameters
    ----------
    returns: pd.DataFrame. Expected return of the portfolio.
    method: str. "simple" or "log"
    portfolio_type: str. "investor" or "msr
    periods_per_year: int. Number of years over which to calculate volatility.
    min_w: float. Minimum weight of the portfolio.
    max_w: float. Maximum weight of the portfolio.
    custom_target_volatility: float. Custom target volatility.
    rf_annual: float. Risk free annual.
    return_type: str. "markowitz" or "actual
    Returns
    -------
    dictionary: With name of the type of model, returns, volatility, max drawdown and weights
    """

    covmat = calculate_covariance(returns)
    if rf_annual is not None:
        returns, covmat = add_risk_free_asset(returns, covmat, rf_annual, periods_per_year)
    if portfolio_type == "investor":
        weights = get_investor_weights(returns,
                                       covmat,
                                       method,
                                       min_w,
                                       max_w,
                                       custom_target_volatility,
                                       sectors_df=sectors_df,
                                       ticker_col=ticker_col,
                                       sector_col=sector_col,
                                       sector_max_weight=sector_max_weight,
                                       risk_free_ticker = risk_free_ticker
                                       )
    elif portfolio_type == "msr":
        weights = msr(returns, covmat, rf_annual, method, periods_per_year, min_w, max_w)
    elif portfolio_type == "gmv":
        weights = gmv(covmat, min_w, max_w)
    elif portfolio_type == "ew":
        # We calculate the weights for an equally weighted portfolio
        weights = ew(returns)
    elif portfolio_type == "random":
        weights = random_weights(returns)

    else:
        raise ValueError(f"Unknown portfolio type: {portfolio_type}")
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

    print(df_weights)
    print(df_results)
    return df_results, df_weights, weights


def get_cumulative_returns(returns: pd.DataFrame,
                           weights: np.ndarray,
                           initial_investment: float = 1,
                           rf_annual: float | None = None,
                           periods_per_year: float = 252
                           ) -> pd.DataFrame:

    money_invested = weights * initial_investment
    if rf_annual is not None:
        returns = add_risk_free_column(returns, rf_annual, periods_per_year)

    adjust_returns = (1+returns)

    adjusted_returns = adjust_returns.cumprod()

    adjusted_value = adjusted_returns * money_invested

    adjusted_total_value = adjusted_value.sum(axis=1)

    return adjusted_total_value, adjusted_value


def get_updated_results(returns: pd.DataFrame,
                        weights: pd.DataFrame,
                        initial_investment: float = 1,
                        method: Literal["simple", "log"] = "simple",
                        periods_per_year: float = 252,
                        rf_annual: float | None = None):

    covmat = calculate_covariance(returns)

    df_returns, df_stock_returns = get_cumulative_returns(returns, weights, initial_investment, rf_annual, periods_per_year)

    absolute_return = df_returns.iloc[-1] / initial_investment

    print(absolute_return)

    t = df_returns.shape[0]

    annualized_return = absolute_return**(periods_per_year / t) - 1

    print(f"Retorno {annualized_return}")

    if rf_annual is not None:
        returns, covmat = add_risk_free_asset(returns, covmat, rf_annual, periods_per_year)


    # We get the historical volatility
    pf_volatility = portfolio_volatility(weights, covmat, periods_per_year)

    # We calculate the sharpe ratio
    portfolio_sharpe_ratio = (annualized_return - rf_annual) / pf_volatility

    # We get the maximum drawdown
    max_drawdown = max_drawdown_from_value_series(df_returns)

    # We get the maximum drawdown
    resultados = {"Returno anualizado": float(round(annualized_return * 100, 3)),
                  "Volatility": float(round(pf_volatility * 100, 3)),
                  "Sharpe Ratio": float(round(portfolio_sharpe_ratio, 3)),
                  "max_drawdown": float(round(max_drawdown * 100, 3))}

    df_results = pd.DataFrame(resultados, index=[0])

    return df_results, df_returns, df_stock_returns


def get_sector_exposure_table(
    df_weights: pd.DataFrame,
    sectors: pd.DataFrame,
    weight_col: str = "Pesos",
    as_percent: bool = True,
) -> pd.DataFrame:
    """
    Devuelve una tabla con la exposición por sector a partir de:
    - df_weights: índice = ticker, columna = pesos
    - sectors: columnas = ticker, sector
    """

    # Pasamos el índice (Ticker) a columna
    df = df_weights.reset_index().rename(columns={"Ticker": "ticker"})

    # Unimos con sectores
    df = df.merge(
        sectors[["ticker", "sector"]],
        on="ticker",
        how="left"
    )

    # Excluimos activos sin sector (ej. RISK_FREE)
    df = df.dropna(subset=["sector"])

    # Agregamos por sector
    sector_table = (
        df.groupby("sector", as_index=False)[weight_col]
          .sum()
    )

    if as_percent:
        sector_table[weight_col] *= 100

    # Ordenamos de mayor a menor peso
    sector_table = sector_table.sort_values(
        by=weight_col, ascending=False
    ).reset_index(drop=True)

    return sector_table


def create_output_table_portfolios(returns: pd.DataFrame,
                                   method: Literal["simple", "log"] = "simple",
                                   list_portfolio_types=None,
                                   periods_per_year: float = 252,
                                   min_w: float = 0.00,
                                   max_w: float = 1.00,
                                   rf_annual: float | None = None,
                                   custom_target_volatility: float = 0.15,
                                   sectors_df: Optional[pd.DataFrame] = None,
                                   ticker_col: str = "ticker",
                                   sector_col: str = "sector",
                                   sector_max_weight: Optional[float] = None,
                                   risk_free_ticker: str = "RISK_FREE") -> pd.DataFrame:

    if list_portfolio_types is None:
        list_portfolio_types = ["investor", "msr", "gmv", "ew", "random"]

    results = []
    dict_weights = {}
    for portfolio_type in list_portfolio_types:
        df_results, df_weights, weights = get_investor_initial_portfolio(returns,
                                                                        method=method,
                                                                        portfolio_type=portfolio_type,
                                                                        min_w=min_w,
                                                                        max_w=max_w,
                                                                        rf_annual=rf_annual,
                                                                        periods_per_year=256,
                                                                        custom_target_volatility=custom_target_volatility,
                                                                        sectors_df=sectors_df,
                                                                        ticker_col=ticker_col,
                                                                        sector_col=sector_col,
                                                                        sector_max_weight=sector_max_weight,
                                                                        risk_free_ticker=risk_free_ticker)
        dict_weights[portfolio_type] = weights
        df_r = df_results.copy()
        df_r.index = [portfolio_type]
        df_r.index.name = "Tipo de portfolio"

        results.append(df_r)

    df_results_all = pd.concat(results)

    return df_results_all, dict_weights

