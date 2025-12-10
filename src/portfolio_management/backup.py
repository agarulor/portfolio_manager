def get_markowtiz_results(train_returns: pd.DataFrame,
                          test_returns: pd.DataFrame,
                          portfolio_type: Literal["msr", "gmv", "ew", "random", "custom"] = "msr",
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
    train_returns: pd.DataFrame. Expected return of the portfolio.
    test_returns: pd.DataFrame. Expected return of the portfolio.
    portfolio_type: Literal["msr", "gmv", "portfolio", "ew", "random"] = "msr"
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
    covmat_train = calculate_covariance(train_returns)
    if portfolio_type == "msr":
        weights = msr(train_returns, covmat_train, rf_annual, method, periods_per_year)
    elif portfolio_type == "gmv":
        weights = gmv(covmat_train)
    elif portfolio_type == "ew":
        # We calculate the weights for an equally weighted portfolio
        weights = ew(train_returns)
    elif portfolio_type == "random":
        weights = random_weights(train_returns)
    elif portfolio_type == "custom":
        weights = get_investor_weights(train_returns,
                                       covmat_train,
                                       method,
                                       periods_per_year,
                                       min_w,
                                       max_w,
                                       rf_annual,
                                       custom_target_volatility)

    else:
        raise ValueError(f"Unknown portfolio type: {portfolio_type}")

    # We get the returns
    pf_return = portfolio_returns(weights, test_returns, method, periods_per_year)

    # We get the volatility from the test returns
    new_covmat = calculate_covariance(test_returns)

    pf_volatility = portfolio_volatility(weights, new_covmat, periods_per_year)

    # We calculate the sharpe ratio
    portfolio_sharpe_ratio = (pf_return - rf_annual) / pf_volatility

    # We get the maximum drawdown
    max_drawdown = calculate_max_drawdown(weights, test_returns)

    # We add a new element in case portfolio type is not custom and if we are using RF_Rate
    if portfolio_type != "custom":
        weights = np.append(weights, 0.0)

    # We convert it into a dictionary and multiply by 100 to get %
    portfolio_information = {"Model": portfolio_type,
                             "Returns": float(round(pf_return * 100, 3)),
                             "Volatility": float(round(pf_volatility * 100, 3)),
                             "Sharpe Ratio": float(round(portfolio_sharpe_ratio, 3)),
                             "max_drawdown": float(round(max_drawdown * 100, 3)),
                             weight_name: np.round(weights * 100, 3) }

    return portfolio_information


def create_markowitz_table(train_returns: pd.DataFrame,
                           test_returns: pd.DataFrame,
                           portfolio_types=None,
                           method: Literal["simple", "log"] = "simple",
                           periods_per_year: int = 252,
                           min_w: float = 0.00,
                           max_w: float = 1.00,
                           rf_annual: float = None,
                           custom_target_volatility: float = 0.15,
                           weight_name: str = "weights"
                           ) -> pd.DataFrame:
    """
    Returns the returns and volatility of a portfolio given weights of the portfolio

    Parameters
    ----------
    train_returns: pd.DataFrame. Expected return of the portfolio.
    test_returns: pd.DataFrame. Expected return of the portfolio.
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
    tickers = train_returns.columns

    if portfolio_types is None:
        portfolio_types = ["msr", "gmv", "ew", "random", "custom"]

    for portfolio in portfolio_types:
        resultados = get_markowtiz_results(
            train_returns,
            test_returns,
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