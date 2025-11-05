import numpy as np
from data_management.get_data import get_stock_prices, read_price_file
from data_management.clean_data import clean_and_align_data
from data_management.save_data import save_preprocessed_data
from portfolio_tools.return_metrics import calculate_daily_returns, annualize_returns, portfolio_returns
from portfolio_tools.risk_metrics import calculate_covariance, portfolio_volatility, sharpe_ratio
if __name__ == "__main__":
    #a = get_stock_prices("data/input/eurostoxx50_csv.csv", "ticker_yahoo", "name",
    #                     start_date = "2024-11-03", end_date = "2025-11-03"
    #)
    #b, c, d = clean_and_align_data(a)
    #print(b.head())
    #save_preprocessed_data(b)
    e = read_price_file("data/processed/prices_20251105-013748.csv")
    f = calculate_daily_returns(e, method="log")
    h = calculate_covariance(f)
    n = 50

    # Genera pesos aleatorios positivos
    pesos = np.full(35, 0.02)

    # Normaliza para que sumen 1
    pesos = pesos / pesos.sum()
    i = portfolio_volatility(pesos, h)
    print(i)

    j = portfolio_returns(pesos, f, method="log")
    print(j)
    sharpe_ratio = sharpe_ratio(j, i)
    print(sharpe_ratio)






