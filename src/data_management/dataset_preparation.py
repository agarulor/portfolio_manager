import pandas as pd


def split_data_markowtiz(
        returns: pd.DataFrame,
        test_date_start: str = "2024-01-01",
        test_date_end: str = "2025-09-30") -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Computes split data markowtiz.

    Parameters
    ----------
    returns : pd.DataFrame. Returns of the assets.
    test_date_start : str. test date start.
    test_date_end : str. test date end.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]: split data markowtiz output.
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
        pos = sorted_returns.index.searchsorted(time_start)
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
    train_set = sorted_returns.loc[:time_start - pd.Timedelta(days=1)]
    test_set = sorted_returns.loc[time_start: time_end]

    return train_set, test_set
