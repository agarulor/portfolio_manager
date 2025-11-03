from data_management.get_data import get_stock_prices
from data_management.clean_data import clean_and_align_data
from data_management.save_data import save_preprocessed_data
if __name__ == "__main__":
    a = get_stock_prices("data/input/eurostoxx50_csv.csv", "ticker_yahoo", "name")
    b, c, d = clean_and_align_data(a)
    save_preprocessed_data(b)


