import pandas as pd
import numpy as np
from typing import Union, Tuple
from datetime import datetime
from portfolio_management.investor_portfolios import add_risk_free_column
from portfolio_tools.risk_metrics import annualize_standard_deviation


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
        return pd.Series(0.0, index = df_values.columns)

    weights = (row / total).fillna(0.0)

    # We only keep those with weights > 0
    weights = weights[weights > 0]

    return weights


def get_sector_weights_at_date(
    df_values: pd.DataFrame,
    sectors: pd.DataFrame,
    date: Union[str, datetime, pd.Timestamp],
    as_percent: bool = False,
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

    df = w.to_frame("weight")
    df["sector"] = df.index.map(m)
    df = df.dropna(subset=["sector"])  # ignora RISK_FREE u otros sin sector

    sec = df.groupby("sector")["weight"].sum().sort_values(ascending=False)

    if as_percent:
        sec = sec * 100

    return sec


def rebalance_dates(index: pd.DatetimeIndex, months: int = 6) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index).sort_values()
    sched = pd.date_range(start=idx.min(), end=idx.max(), freq=pd.DateOffset(months=months))
    chosen = []
    for d in sched:
        eligible = idx[idx <= d]
        if len(eligible):
            chosen.append(eligible.max())
    return pd.DatetimeIndex(sorted(set(chosen)))


def simulate_rebalance_6m_from_returns(
    returns_test: pd.DataFrame,
    target_weights: np.ndarray,
    initial_investment: float = 100.0,
    months: int = 6,
    rf_annual: float | None = None,
    periods_per_year: float = 252,
    risk_free_ticker: str = "RISK_FREE",
) -> Tuple[pd.Series, pd.DataFrame, pd.DataFrame]:
    """
    Simula cartera con rebalanceo cada `months` meses.

    returns_test: DF retornos diarios (fechas x tickers) SIN RF normalmente
    target_weights: np.ndarray (orden = returns_train.columns si incluiste RF al optimizar)
    Devuelve:
      - total_value: Serie valor total
      - values_by_asset: DF valor por activo
      - trades_log: DF compras/ventas (en €) en cada rebalanceo
    """
    returns_test = returns_test.sort_index().copy()

    # Si target_weights incluye RF, necesitamos que returns_test tenga RF como columna
    if rf_annual is not None and risk_free_ticker not in returns_test.columns:
        returns_test = add_risk_free_column(returns_test, rf_annual, periods_per_year)

    # Comprobar dimensiones
    if len(target_weights) != returns_test.shape[1]:
        raise ValueError(
            f"Mismatch: len(target_weights)={len(target_weights)} vs returns_test.columns={returns_test.shape[1]}"
        )

    tickers = list(returns_test.columns)

    # normalizar pesos objetivo por si acaso
    w0 = np.array(target_weights, dtype=float)
    w0 = w0 / w0.sum()

    # valores por activo
    values = pd.DataFrame(index=returns_test.index, columns=tickers, dtype=float)
    values.iloc[0] = initial_investment * w0

    rb = set(rebalance_dates(returns_test.index, months=months))
    trades = []

    for i in range(1, len(returns_test.index)):
        today = returns_test.index[i]
        yday = returns_test.index[i - 1]

        # evolucion buy&hold: V_t = V_{t-1} * (1 + r_t)
        values.loc[today] = values.loc[yday].values * (1.0 + returns_test.loc[today].values)

        # rebalance semestral: volver a pesos objetivo
        if today in rb:
            v_now = values.loc[today]
            total = float(v_now.sum())

            target_values = total * w0
            trade_values = target_values - v_now.values

            for tkr, cur, tgt, trd in zip(tickers, v_now.values, target_values, trade_values):
                if abs(trd) > 1e-10:
                    trades.append({
                        "date": today,
                        "ticker": tkr,
                        "current_value": float(cur),
                        "target_value": float(tgt),
                        "trade_value": float(trd),  # + comprar, - vender
                    })

            # aplicar rebalance
            values.loc[today] = target_values

    total_value = values.sum(axis=1)
    trades_log = pd.DataFrame(trades)
    if not trades_log.empty:
        trades_log = trades_log.sort_values(["date", "trade_value"])

    return total_value, values, trades_log