from portfolio_tools.markowitz import portfolio_output
import pandas as pd


def split_data_markowtiz(
        returns: pd.DataFrame,
        test_date_start: str = "2024-10-01",
        test_date_end: str = "2025-09-30") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits data into training and test sets

    Parameters
    ----------
    returns : pd.DataFrame. Dataset with returns.
    test_date_start : str. Starting day for test data (YYYY-MM-DD).
    test_date_end : str. Ending day for test data (YYYY-MM-DD).

    Returns
    ----------
    training_set : pd.DataFrame. Training set with returns.
    test_set : pd.DataFrame. Test set with returns.
    """
    # we sort the returns in case they are not shorted
    sorted_returns = returns.sort_index()

    # We convert to datetime the index
    sorted_returns.index = pd.to_datetime(sorted_returns.index)

    # Now we check if the date exists
    time_start = pd.to_datetime(test_date_start)
    time_end = pd.to_datetime(test_date_end)

    # We adjust data if not in the data shared
    if time_start not in sorted_returns.index:
        pos = sorted_returns.index.searchsorted(time_start)  # sin +1
        if pos >= len(sorted_returns.index):
            raise ValueError("time_start is after the available data range.")
        time_start = sorted_returns.index[pos]
        print(f"time_start adjusted to: {time_start.date()}")


    if time_end not in sorted_returns.index:
        print(f"{time_end} not in data. Adjusting to previous available business day.")
        pos = sorted_returns.index.searchsorted(time_end) - 1
        if pos < 0:
            raise ValueError("time_end is before the available data range.")
        time_end = sorted_returns.index[pos]
        print(f"time_end adjusted to: {time_end.date()}")

    # We now divide data into training and test
    training_set = sorted_returns.loc[:time_start - pd.Timedelta(days=1)]
    test_set = sorted_returns.loc[time_start : time_end]

    return training_set, test_set
