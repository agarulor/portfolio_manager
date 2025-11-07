import numpy as np
import pandas as pd
from data_management.get_data import get_stock_prices, read_price_file
from data_management.clean_data import clean_and_align_data
from data_management.save_data import save_preprocessed_data
from portfolio_tools.return_metrics import calculate_daily_returns, annualize_returns, portfolio_returns, daily_portfolio_returns
from portfolio_tools.risk_metrics import calculate_covariance, portfolio_volatility, sharpe_ratio, neg_sharpe_ratio, \
    annualize_covariance, calculate_max_drawdown
from portfolio_tools.markowitz import minimize_volatility, plot_frontier, maximize_return, msr
import matplotlib.pyplot as plt
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

    annualized_volatility = annualize_covariance(h)


    l = minimize_volatility(0.5, f , h)
    portfolio_ret = portfolio_returns(l, f)
    portfolio_vol = portfolio_volatility(l, h)
    print(portfolio_ret)
    print(portfolio_vol)
    msr_w = msr(f, h)
    j = annualize_returns(f)
    k = daily_portfolio_returns(msr_w, f)
    plt.plot(k)
    print(f"Max drawdown {calculate_max_drawdown(k)}")
    plt.show()


    #a = plot_frontier(100, f, h)
    #print(a)
    #calculate_max_drawdown(f)







