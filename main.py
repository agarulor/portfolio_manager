import numpy as np
from data_management.get_data import get_stock_prices, read_price_file
from data_management.clean_data import clean_and_align_data
from data_management.save_data import save_preprocessed_data
from portfolio_tools.return_metrics import calculate_daily_returns, annualize_returns, portfolio_returns
from portfolio_tools.risk_metrics import calculate_covariance, portfolio_volatility, sharpe_ratio
from portfolio_tools.markowitz import minimize_volatility, plot_basic_frontier, maximize_return
if __name__ == "__main__":
    #a = get_stock_prices("data/input/eurostoxx50_csv.csv", "ticker_yahoo", "name",
    #                     start_date = "2024-11-03", end_date = "2025-11-03"
    #)
    #b, c, d = clean_and_align_data(a)
    #print(b.head())
    #save_preprocessed_data(b)
    e = read_price_file("data/processed/prices_20251105-014012.csv")
    f = calculate_daily_returns(e, method="simple")
    h = calculate_covariance(f)

    l = minimize_volatility(0.1, f , h)
    portfolio_ret = portfolio_returns(l, f)
    portfolio_vol = portfolio_volatility(l, h)
    print(portfolio_ret)
    print(portfolio_vol)
    print(l)

    a = plot_basic_frontier(20, f, h)
    print(a)





