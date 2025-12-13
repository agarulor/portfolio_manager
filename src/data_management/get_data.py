import pandas as pd
import os
from typing import Optional
import yfinance as yf


def read_stock_ticker(file_path: str) -> Optional[pd.DataFrame]:
    """
    Read a csv file containing a list of stock tickers or asset names.
    This function loads a csv file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing the list of stock tickers.
        The file should include at least the column with the 'ticker' in Yahoo Finance format

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the data from the csv file.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    pd.errors.EmptyDataError
        If the file exists but is empty.
    pd.errors.ParserError
        If there is an error parsing the csv file.
    Exception
        For any other unexpected error.

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
                     companies_col: str,
                     adjusted: bool = False,
                     start_date: str = "2005-01-01",
                     end_date: str = "2025-09-30") -> Optional[pd.DataFrame]:
    """
    Download historical stock price data from Yahoo Finance.

    The function reads a csv file containing the stock tickers for Yahoo Finance and company names
    downloads the closing adjusted data, on a daily basis, for the tickers provided, with a starting and
    closing date. Returns a DataFrame with the adjusted closing prices (by default)

    Parameters
    ----------
    file_path : str
        Path to the csv file containing the list of stock tickers and company names.
    ticker_col : str
        Name of the column in the csv file with the Yahoo Finance tickers.
    companies_col : str
        Name of the column in the csv file with the company names.
    adjusted : bool, default=False
        If using or not adjusted prices if False, closing prices are obtained from
         column Adj Closing prices
    start_date : str, default="2005-01-01"
        Start date for download historical data
    end_date : str, default="2025-09-30"
        End date for download historical data

    Returns
    -------
    pandas.DataFrame or None
        A DataFrame containing the adjusted closing prices for each stock. Dates are in the rows
        and tickers in the columns. Returns None if the file, or filename is invalid or downloading fails.

    Raises
    ------
    Exception
        If an unexpected error occurs while downloading data from Yahoo Finance.
    KeyError
        If the expected price field ('Close' or 'Adj Close') is not present in
        the data returned by Yahoo Finance.
    """

    # First we read the file with the tickers
    df_tickers = read_stock_ticker(file_path)
    # We check that the ticker col is the right col:
    if ticker_col not in df_tickers.columns:
        print(f"The ticker column {ticker_col} is invalid. Please provide a valid column.")
        return None

    # We build the ticker list
    tickers = list(df_tickers[ticker_col])
    list(df_tickers[companies_col])

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


def read_price_file(
        file_path: str,
        index_col: str = "Date",
        parse_dates: bool = True) -> Optional[pd.DataFrame]:
    """
    Read a csv file containing a list of stock prices.
    This function loads a csv file into a pandas DataFrame.

    Parameters
    ----------
    file_path : str
        Path to the CSV file containing the list of stock prices.
    index_col : str, default="Date"
        Name of the column in the csv file with the Yahoo Finance tickers.
    parse_dates : bool, default=True
        If True, parse the date column into the datetime64 format.

    Returns
    -------
    pd.DataFrame
        A pandas DataFrame with the data from the csv file.

    Raises
    ------
    FileNotFoundError
        If the specified file path does not exist.
    pd.errors.EmptyDataError
        If the file exists but is empty.
    pd.errors.ParserError
        If there is an error parsing the csv file.
    Exception
        For any other unexpected error.

    """
    # We first check if the file exists
    if not os.path.exists(file_path):
        print("The file path doesn't exists. Please provide a valid file path.")
        return None

    # We try to read csv file
    try:
        df = pd.read_csv(file_path, index_col=index_col, parse_dates=parse_dates)
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