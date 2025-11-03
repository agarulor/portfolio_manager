from data_management.get_data import get_stock_prices, read_price_file
from data_management.clean_data import clean_and_align_data
from data_management.save_data import save_preprocessed_data
from portfolio_tools.return_metrics import calculate_daily_returns, annualize_returns
if __name__ == "__main__":
    #a = get_stock_prices("data/input/eurostoxx50_csv.csv", "ticker_yahoo", "name")
    #b, c, d = clean_and_align_data(a)
    #print(b.head())
    #save_preprocessed_data(b)
    e = read_price_file("data/processed/prices_20251103-223944.csv")
    print(e.head())
    print(e.index.inferred_type == "datetime64")
    f = calculate_daily_returns(e, method="log")
    print(f.head(100))
    g = annualize_returns(f, method="log")
    print(g)



