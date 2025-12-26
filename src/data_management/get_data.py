import pandas as pd
import os
from typing import Optional, Tuple
import yfinance as yf


def read_stock_ticker(file_path: str) -> Optional[pd.DataFrame]:
    """
    Reads stock ticker.

    Parameters
    ----------
    file_path : str. file path.

    Returns
    -------
    Optional[pd.DataFrame]: read stock ticker output.
    """
    # We first check if the file exists
    if not os.path.exists(file_path):
        print("The file path doesn't exists. Please provide a valid file path.")
        return None

    # We try to read csv file
    try:
        df = pd.read_csv(file_path)
    # It raises an EmptyDataError if empty
    except pd.errors.EmptyDataError:
        print("The file is empty. Please provide a valid file.")
        return None
    # It raises a ParserError if error while parsing csv file
    except pd.errors.ParserError:
        print("csv parsing error. Please, check file")
        return None
    # Any other error
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None

    # We validate file content
    if df.empty:
        print("The file has no valid rows")
        return None
    print(f"The file {file_path} has been successfully loaded")

    # we return the DF
    return df


def get_stock_prices(file_path: str,
                     ticker_col: str,
                     additional_tickers: Optional[list] = None,
                     adjusted: bool = False,
                     start_date: str = "2005-01-01",
                     end_date: str = "2025-09-30") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Gets stock prices.

    Parameters
    ----------
    file_path : str. file path.
    ticker_col : str. ticker col.
    additional_tickers : Optional[list]. additional tickers.
    adjusted : bool. adjusted.
    start_date : str. start date.
    end_date : str. end date.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]: get stock prices output.
    """

    # First we read the file with the tickers
    df_tickers = read_stock_ticker(file_path)
    # We check that the ticker col is the right col:
    if ticker_col not in df_tickers.columns:
        print(f"The ticker column {ticker_col} is invalid. Please provide a valid column.")
        return None

    # We build the ticker list

    tickers = list(df_tickers[ticker_col])
    if additional_tickers is not None:
        tickers = tickers + additional_tickers

    # In case there are no tickers in the column
    if not tickers:
        print("There are no tickers, please, provide a file with valid tickers")
        return None

    # We obtain now tickers from Yahoo Finance
    print(f"Downloading prices for {len(tickers)} tickers. Please wait...")

    # We download files from Yahoo Finance
    # We want the data adjusted by default to account for dividends, splits, etc.
    try:
        stock_data = yf.download(
            tickers=tickers,
            start=start_date,
            end=end_date,
            progress=False,
            auto_adjust=adjusted
        )
    # We throw an error if not downloaded correctly
    except Exception as e:
        print(f"Error downloading data from Yahoo Finance: {e}")
        return None

    # If there is no data from Yahoo Finance
    if stock_data is None or stock_data.empty:
        print("No data retrieved from Yahoo Finance")

    # We are just interested in the closing price
    if adjusted:
        close_price = "Adj Close"
    else:
        close_price = "Close"

    # We if the price field exists and the type of field:
    try:
        if isinstance(stock_data.columns, pd.MultiIndex):
            prices = stock_data[close_price]
        else:
            prices = stock_data[[close_price]]
    except KeyError:
        print(f"The price field {close_price} was not found. Please provide a valid field.")
        return None

    # Final confirmation
    print(F"Download completed successfully for {len(prices.columns)} stocks")
    print(f"{prices.shape[0]} rows in total")

    # We now get sector information
    print(f"Downloading sector/industry info for {len(tickers)} tickers. Please wait...")

    # Create an empty dictionary
    sector_data = {
        "ticker": [],
        "sector": [],
        "industry": [],
    }

    # We get tiker information
    for tkr in tickers:
        try:
            info = yf.Ticker(tkr).info
            sector = info.get("sector", None)
            industry = info.get("industry", None)
        except Exception as e:
            print(f"Error downloading data from Yahoo Finance: {e}")
            sector = None
            industry = None

        sector_data["ticker"].append(tkr)
        sector_data["sector"].append(sector)
        sector_data["industry"].append(industry)

    sectors_df = pd.DataFrame(sector_data)

    return prices, sectors_df


def exists_asset(ticker: str) -> bool:
    """
    Computes exists asset.

    Parameters
    ----------
    ticker : str. ticker.

    Returns
    -------
    bool: exists asset output.
    """
    try:
        t = yf.Ticker(ticker)
        data = t.history(period="1d")

        return not data.empty

    except:
        return False
