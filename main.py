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
from portfolio_tools.portfolio_allocation import split_data_markowtiz, split_data_ml
if __name__ == "__main__":
    #a = get_stock_prices("data/input/eurostoxx50_csv.csv", "ticker_yahoo", "name",
    #                     start_date = "2024-11-03", end_date = "2025-11-03"
    #)
    #b, c, d = clean_and_align_data(a)
    #print(b.head())
    #save_preprocessed_data(b)
    e = read_price_file("data/processed/prices_20251105-013441.csv")

    f = calculate_daily_returns(e, method="simple")
    h = calculate_covariance(f)

    annualized_volatility = annualize_covariance(h)


    #l = minimize_volatility(0.5, f , h)
    #portfolio_ret = portfolio_returns(l, f)
    #portfolio_vol = portfolio_volatility(l, h)

    #print(f.head())
    train, test = split_data_markowtiz(f)
    covmat_train = calculate_covariance(train)
    #print(train.head())
    #print(train.tail())
    #print(test.head())
    #print(test.tail())

    a= plot_frontier(100, train, covmat_train)
    #print(a)

    train_2, val_2, test_2, warm_1, warm_2 = split_data_ml(f, "2024-12-30", "2025-06-30")








