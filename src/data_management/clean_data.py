import pandas as pd
from typing import Tuple
def clean_stock_data(prices: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Clean a dataframe with stock prices. It removes empty columns (when no data available)
    It also generates a report on the availability of data, por decision-making purposes

    It removes all tickers (columns) with no valid prices at all
    It generates a summary report with information about the completeness of each
    series.

    Parameters
    ----------
    prices : pandas.DataFrame
        Is a DataFrame with stock prices and dates as index and tickers/company names as columns.

    Returns
    -------
    df : pandas.DataFrame
        The cleaned DataFrame
    report : pandas.DataFrame
        A summary DataFrame with key availability metrics:
            - first_valid_index : first date with a valid observation
            - last_valid_index  : last date with a valid observation
            - total_valid       : number of valid observations
            - total_rows        : total number of rows (dates)
            - coverage          : % valid observations over total_rows
    """

    # We create a copy of the dataset
    df = prices.copy()

    # First we drop all tickers, i.e. columns, that don´t have any price
    df = df.dropna(how="all")

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
                min_row_coverage: float = 0.5) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align trading dates across markets by forward-filling short gaps. It removes days with insufficient coverage.

    This function helps with the issue of different trading calendars, such as different holidays across countries.
    It does a limited forward-fill on missing price values to solve the issue with non-trading gaps.
    It also, optionally, drops rows (dates) where too there are many assets with no values on that day.

    Parameters
    ----------
    prices : pandas.DataFrame
        Price data frame with stock prices. Where NaN indicate non-trading days or unavailable data.
    max_ffill_days : int, default 5
        Maximum number of consecutive days to forward-fill per column.
        It avoids imputing long suspensions or inactive periods, which could be due to different issues.
    min_row_coverage : float, default 0.5
        Minimum % of non-missing columns required to keep a row.
        Example: 0.5 means at least 50% of tickers must have data on that date.

    Returns
    -------
    aligned : pandas.DataFrame
        Price DataFrame with dates aligned.
    summary : pandas.DataFrame
        A DataFrame summarizing the operation:
            - rows_before / rows_after
            - cols_before / cols_after
            - rows_dropped
            - ffilled_cells (number of values filled by forward-fill)
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
        min_row_coverage: float = 0.5)-> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    """
    End-to-end preprocessing pipeline for price data:
    clean missing columns and align trading calendars.

    This function first removes tickers that contain no data at all
    (`clean_stock_data`), then harmonizes calendars using limited
    forward-fill and row filtering (`align_dates`).

    Parameters
    ----------
    prices : pandas.DataFrame
        Price data frame with stock prices. Where NaN indicate non-trading days or unavailable data.
    max_ffill_days : int, default 5
        Maximum number of consecutive days to forward-fill per column.
        It avoids imputing long suspensions or inactive periods, which could be due to different issues.
    min_row_coverage : float, default 0.5
        Minimum % of non-missing columns required to keep a row.
        Example: 0.5 means at least 50% of tickers must have data on that date.

    Returns
    -------
    aligned_df : pandas.DataFrame
        Cleaned and aligned DataFrame.
    cleaned_report : pandas.DataFrame
        Output report from clean_stock_data
    aligned_summary : pandas.DataFrame
        Output summary from align_dates
    """

    # We clean the df
    cleaned_df, cleaned_report = clean_stock_data(prices)
    # We align the calendars
    aligned_df, aligned_summary = align_dates(cleaned_df, max_ffill_days, min_row_coverage)

    # Return the aligned and cleaned df with the reports
    return aligned_df, cleaned_report, aligned_summary

