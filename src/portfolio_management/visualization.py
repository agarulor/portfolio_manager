import pandas as pd
import matplotlib.pyplot as plt
from typing import Optional, Tuple



# ============================================================
# 4. PLOT VALIDATION FOR A SHARE (TO CHECK PERFORMANCE)
#   IN A VISUAL FASHION
# ============================================================

def plot_validation(
    results: dict,
    asset: str,
    n_points: Optional[int] = 200
):
    """
    Plot actual vs predicted validation prices for a given asset.

    Parameters
    ----------
    results : dict. Dictionary where each key is an asset ticker and each value contains:
    asset : str. The asset/ticker to plot.
    n_points : int, optional (default=200). Maximum number of most recent validation points to display.

    Raises
    ------
    ValueError if the requested asset is not found in the results dictionary.

    Returns
    -------
    None : It only displays the plot.
    """
    # Check if the asset wit want to plot is with the results
    if asset not in results:
        raise ValueError(f"{asset} is not in results (keys: {list(results.keys())})")

    # we extract the results from the asset
    res = results[asset]
    # We extract the values predicted and the real one (for the val)
    y_val_inv = res["y_val_inv"].reshape(-1)
    y_pred_inv = res["y_pred_inv"].reshape(-1)
    val_dates = res["val_dates"]

    # We align information
    n = min(len(y_val_inv), len(y_pred_inv), len(val_dates))
    y_val_inv = y_val_inv[:n]
    y_pred_inv = y_pred_inv[:n]
    val_dates = val_dates[:n]

   # We plot only the last n_points (to avoid showing too much info on the screen)
    if n_points is not None and n > n_points:
        y_val_inv = y_val_inv[-n_points:]
        y_pred_inv = y_pred_inv[-n_points:]
        val_dates = val_dates[-n_points:]

    # We check that there is validation data
    if len(val_dates) == 0:
        print(f"There is no validation data for {asset}")
        return

    print(f"[{asset}] validation from {val_dates[0]} until {val_dates[-1]} (N={len(val_dates)})")

    # We create and show the plot
    plt.figure(figsize=(12, 5))
    plt.plot(val_dates, y_val_inv, label="Actual share price (validation)", linewidth=1.5)
    plt.plot(val_dates, y_pred_inv, label="Forecast Price (validation)", linestyle="--", linewidth=1.5)
    plt.title(f"Forecasted vs. actual values - {asset}")
    plt.xlabel("Date")
    plt.ylabel("Share price (EUR)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()




def plot_equal_weight_buy_and_hold_from_results(
    results: dict,
    n_points: Optional[int] = 200,
):
    """
    Plots an equally weighted  portfolio over the validation period with the
    real and predicted prices

    Parameters
    ----------
    results : dict. Dictionary where each key is an asset name and each value is the output
        dictionary returned
            - "y_pred_inv"  : inverse-scaled predicted validation prices

    n_points : int, optional (default=200). Maximum number of most recent validation points to display.

    Returns
    -------
    None : It only displays the plot.
    """

    # We get the real and predicted prices for each asset
    real_prices_df, pred_prices_df = validation_price_matrices_from_results(results)

    if real_prices_df.shape[0] <= 1:
        print("There is not enough validation data to build the portfolio")
        return

    print("Range for the theoretical portfolio:")
    print("  From:", real_prices_df.index[0])
    print("  Until:", real_prices_df.index[-1])
    print("  Number of days:", real_prices_df.shape[0])

    # dates
    dates = real_prices_df.index.to_numpy()

    # Final adjustement
    if n_points is not None and len(dates) > n_points:
        dates = dates[-n_points:]
        real_prices_df = real_prices_df.iloc[-n_points:, :]
        pred_prices_df = pred_prices_df.iloc[-n_points:, :]


    # Normalization of prices by the initial value of each asset
    # Each assets starts at 1 on the first day of validation
    real_norm = real_prices_df / real_prices_df.iloc[0]
    pred_norm = pred_prices_df / pred_prices_df.iloc[0]

    # Equally weighted portfolio, each asset has the same weight
    real_port_val = real_norm.mean(axis=1)
    pred_port_val = pred_norm.mean(axis=1)

    # moving to the plotting section
    plt.figure(figsize=(12, 5))
    plt.plot(real_port_val.index, real_port_val.values,
             label="Actual portfolio equally weighted", linewidth=1.5)
    plt.plot(pred_port_val.index, pred_port_val.values,
             label="Forecasted Portfolio equally weighted", linestyle="--", linewidth=1.5)
    plt.axhline(1.0, color="black", linestyle="--", linewidth=1)
    plt.title("Equally weighted porfolio actual / vs. forecast \n Validation data")
    plt.xlabel("Data")
    plt.ylabel("Portfolio value (normalized at 1.0 at the beginning)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()

    # Key metrics (for comparation purposes)
    n_days = len(real_port_val) - 1
    if n_days > 0:
        real_total = real_port_val.iloc[-1] - 1.0
        pred_total = pred_port_val.iloc[-1] - 1.0

        ann_factor = 252.0 / n_days
        real_ann = (1.0 + real_total) ** ann_factor - 1.0
        pred_ann = (1.0 + pred_total) ** ann_factor - 1.0

        print("=== Equally weighted portfolio (VALIDATION DATA - 1 model per asset) ===")
        print(f"Number of days: {n_days}")
        print(f"Total actual return:     {real_total: .2%}")
        print(f"Total forecasted return: {pred_total: .2%}")
        print(f"Annualized return - actual:     {real_ann: .2%}")
        print(f"Annualized return - forecasted: {pred_ann: .2%}")
    else:
        print("No hay suficientes días para calcular rentabilidades.")


def plot_asset(
    real_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    asset: str,
    n_points: Optional[int] = 200,
    title_prefix: str = "Validation",
) -> None:
    """
    Plot actual vs predicted prices for a single asset using
    already built DataFrames (real_df, pred_df).

    Parameters
    ----------
    real_df : pd.DataFrame
        Real prices. Index = dates, columns = assets.
    pred_df : pd.DataFrame
        Predicted prices. Same shape/axes as real_df (aligned).
    asset : str
        Column name (ticker) to plot.
    n_points : int, optional
        Maximum number of most recent points to display.
    title_prefix : str, optional
        Text to prefix the plot title, e.g. "Validation" or "Forecast".
    """
    if asset not in real_df.columns or asset not in pred_df.columns:
        raise ValueError(f"{asset} not found in DataFrames columns.")

    y_real = real_df[asset].to_numpy()
    y_pred = pred_df[asset].to_numpy()
    dates = real_df.index.to_numpy()

    n = min(len(y_real), len(y_pred), len(dates))
    y_real, y_pred, dates = y_real[:n], y_pred[:n], dates[:n]

    if n_points is not None and n > n_points:
        y_real = y_real[-n_points:]
        y_pred = y_pred[-n_points:]
        dates = dates[-n_points:]

    if len(dates) == 0:
        print(f"No data to plot for {asset}")
        return

    print(f"[{asset}] {title_prefix.lower()} from {dates[0]} to {dates[-1]} (N={len(dates)})")

    plt.figure(figsize=(12, 5))
    plt.plot(dates, y_real, label="Actual price", linewidth=1.5)
    plt.plot(dates, y_pred, label="Predicted price", linestyle="--", linewidth=1.5)
    plt.title(f"{title_prefix} – Actual vs Predicted – {asset}")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()


def plot_equal_weight(
    real_df: pd.DataFrame,
    pred_df: pd.DataFrame,
    n_points: Optional[int] = 200,
    title_prefix: str = "Validation",
) -> None:
    """
    Plot an equally weighted buy-and-hold portfolio using
    real and predicted price DataFrames.

    Parameters
    ----------
    real_df : pd.DataFrame
        Real prices. Index = dates, columns = assets.
    pred_df : pd.DataFrame
        Predicted prices. Same index/columns as real_df.
    n_points : int, optional
        Maximum number of most recent points to display.
    title_prefix : str, optional
        Text to include in the plot title, e.g. "Validation" or "Forecast".
    """
    if real_df.shape[0] <= 1:
        print("Not enough data to build the portfolio.")
        return

    # recorte de fechas si hace falta
    dates = real_df.index.to_numpy()
    if n_points is not None and len(dates) > n_points:
        real_df = real_df.iloc[-n_points:, :]
        pred_df = pred_df.iloc[-n_points:, :]

    print(f"Portfolio {title_prefix} period:")
    print("  From:", real_df.index[0])
    print("  To:  ", real_df.index[-1])
    print("  Days:", real_df.shape[0])

    # Normalización (buy & hold con pesos fijos al inicio)
    real_norm = real_df / real_df.iloc[0]
    pred_norm = pred_df / pred_df.iloc[0]

    real_port_val = real_norm.mean(axis=1)
    pred_port_val = pred_norm.mean(axis=1)

    plt.figure(figsize=(12, 5))
    plt.plot(real_port_val.index, real_port_val.values,
             label="Actual EW portfolio", linewidth=1.5)
    plt.plot(pred_port_val.index, pred_port_val.values,
             label="Predicted EW portfolio", linestyle="--", linewidth=1.5)
    plt.axhline(1.0, linestyle="--", linewidth=1)

    plt.title(f"Equally Weighted Portfolio – {title_prefix}")
    plt.xlabel("Date")
    plt.ylabel("Portfolio value (normalized to 1.0)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.gcf().autofmt_xdate()
    plt.show()

    # Métricas básicas
    n_days = len(real_port_val) - 1
    if n_days > 0:
        real_total = real_port_val.iloc[-1] - 1.0
        pred_total = pred_port_val.iloc[-1] - 1.0
        ann_factor = 252.0 / n_days
        real_ann = (1.0 + real_total) ** ann_factor - 1.0
        pred_ann = (1.0 + pred_total) ** ann_factor - 1.0

        print(f"=== EW Portfolio ({title_prefix}) ===")
        print(f"Days:                  {n_days}")
        print(f"Total actual return:   {real_total: .2%}")
        print(f"Total predicted return:{pred_total: .2%}")
        print(f"Annualized actual:     {real_ann: .2%}")
        print(f"Annualized predicted:  {pred_ann: .2%}")