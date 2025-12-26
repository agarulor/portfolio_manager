import pandas as pd
from typing import Tuple


def clean_stock_data(prices: pd.DataFrame,
                     beginning_data=True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Cleans stock data.

    Parameters
    ----------
    prices : pd.DataFrame. Prices of the assets.
    beginning_data : Any. beginning data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]: clean stock data output.
    """

    # We create a copy of the dataset
    df = prices.copy()

    # First we drop all tickers, i.e. columns, that don´t have any price
    df = df.dropna(how="all")

    if beginning_data:
        first_row = df.iloc[0]
        columns_to_remove = first_row[first_row.isna()].index
        df = df.drop(columns=columns_to_remove)

    # We create a report which provides information about the cleaning process
    # It is a useful tool for reviewing and understanding issues
    report = df.notna().agg(
        ['first_valid_index', 'last_valid_index', 'sum']
    ).T
    report = report.rename(columns={'sum': 'total_valid'})
    report['total_rows'] = len(df)
    report['coverage'] = report['total_valid'] / report['total_rows']

    # We prepare an availability report to provide information about the dataset
    print(f"The data has been cleaned successfully {df.shape[0]} rows, with {df.shape[1]} columns kept")

    # We return the df and the report
    return df, report


# Next, given that we have assets from several countries, we need to align the dates
# We create a function that will help with that
def align_dates(prices: pd.DataFrame,
                max_ffill_days: int = 5,
                min_row_coverage: float = 0.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aligns dates.

    Parameters
    ----------
    prices : pd.DataFrame. Prices of the assets.
    max_ffill_days : int. max ffill days.
    min_row_coverage : float. min row coverage.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]: align dates output.
    """

    # We get the number of projects before alingment for reporting
    rows_before = prices.shape[0]
    cols_before = prices.shape[1]

    # We forward fill with a limit per column
    # i.e. if the number of days is more than max_fill_days
    # First, we count the number of cells to be filled for reporting purposes
    na_before = prices.isna().sum().sum()
    # We now fill it with the limit (https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.ffill.html)
    ffilled = prices.ffill(limit=max_ffill_days)

    # Now, we count the cells filled by ffill (for reporting)
    # We first get the number of na after
    na_after = ffilled.isna().sum().sum()
    # we Calculate the difference between before and after
    ffilled_cells = (na_before - na_after)

    # We drop the days with not enough coverage
    if min_row_coverage > 0:
        min_valid = int(cols_before * min_row_coverage)
        keep_data = ffilled.notna().sum(axis=1) >= min_valid
        aligned = ffilled.loc[keep_data]
        rows_dropped_sparse = int((~keep_data).sum())
    else:
        aligned = ffilled
        rows_dropped_sparse = 0

    # We get the number of rows after, for reporting purposes
    rows_after, cols_after = aligned.shape

    # we prepare the summary
    summary = pd.DataFrame({
        "rows_before": [rows_before],
        "rows_after": [rows_after],
        "cols_before": [cols_before],
        "cols_after": [cols_after],
        "rows_dropped": [rows_dropped_sparse],
        "ffilled_cells": [int(ffilled_cells)]
    })

    # We return the df with aligned calendar and the summary
    return aligned, summary


def clean_and_align_data(
        prices: pd.DataFrame,
        max_ffill_days: int = 5,
        min_row_coverage: float = 0.0,
        beginning_data=True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Cleans and align data.

    Parameters
    ----------
    prices : pd.DataFrame. Prices of the assets.
    max_ffill_days : int. max ffill days.
    min_row_coverage : float. min row coverage.
    beginning_data : Any. beginning data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: clean and align data output.
    """

    # We clean the df
    cleaned_df, cleaned_report = clean_stock_data(prices, beginning_data)
    # We align the calendars
    aligned_df, aligned_summary = align_dates(cleaned_df, max_ffill_days, min_row_coverage)

    # Return the aligned and cleaned df with the reports
    return aligned_df, cleaned_report, aligned_summary

