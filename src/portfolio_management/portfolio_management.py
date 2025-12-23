import pandas as pd
from typing import Union
from datetime import datetime

def check_portfolio_weights(df_values: pd.DataFrame,
                            date: Union[str, datetime, pd.Timestamp]) -> pd.Series:
    """
    Helps to track the current weights of the assets of the portfolio
    It takes the weights of the assets of the portfolio at a given date

    :return:

    The weights of the assets of the portfolio
    """


    # We convert the date to datetime
    date = pd.to_datetime(date)

    # if index not date time, we try to convert it
    if not pd.api.types.is_datetime64_any_dtype(df_values.index):
        df_values = df_values.copy()
        df_values.index = pd.to_datetime(df_values.index)

    # - If the date exists it uses it
    # - if it doesn't exist it use the previous one
    if date in df_values.index:
        row = df_values.loc[date]

    else:
        df_sorted = df_values.sort_index()
        row = df_sorted.loc[df_sorted.index.asof(date)]

    total = row.sum()

    if total == 0 or pd.isna(total):
        weights = pd.DataFrame({"Pesos": 0.0}, index=df_values.columns)
        print(type(weights))
    else:
        weights = (row / total).fillna(0.0).to_frame(name="Pesos")

    weights = weights[weights["Pesos"] > 0]
    weights.index.name = "Ticker"

    return weights


def get_sector_weights_at_date(
    df_values: pd.DataFrame,
    sectors: pd.DataFrame,
    date: Union[str, datetime, pd.Timestamp],
    as_percent: bool = True,
    ticker_col: str = "ticker",
    sector_col: str = "sector",
) -> pd.Series:
    """
    Devuelve pesos por sector en una fecha, a partir de df_values (valores monetarios).
    - ajusta la fecha a la inmediatamente anterior disponible si no existe
    - ignora tickers sin sector (ej. RISK_FREE)
    """
    w = check_portfolio_weights(df_values, date)  # Series index=ticker

    # map ticker -> sector
    m = sectors.set_index(ticker_col)[sector_col]

    df = w.copy()
    df["sector"] = df.index.map(m)
    df = df.dropna(subset=["sector"])
    sec = df.groupby("sector")["Pesos"].sum().sort_values(ascending=False)
    sec = sec.to_frame(name="Pesos")

    if as_percent:
        sec = sec * 100

    return sec