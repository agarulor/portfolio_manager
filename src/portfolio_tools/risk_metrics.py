import pandas as pd
import numpy as np


def calculate_variance(returns: pd.DataFrame) -> pd.DataFrame:

    """
    Calculates the variance of each asset's returns

    Parameters
    ----------
    returns : pd.DataFrame with assets returns

    Returns
    -------
    pd.Series : Variance per asset.
    """
    return returns.var(axis=0)

def calculate_standard_deviation(returns: pd.DataFrame) -> pd.DataFrame:

    """
    Calculates the standard deviation of each asset's returns

    Parameters
    ----------
    returns : pd.DataFrame with assets returns

    Returns
    -------
    pd.Series: Standard deviation per asset.
    """
    return returns.std(axis=0)