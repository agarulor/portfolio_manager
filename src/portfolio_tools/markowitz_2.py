import numpy as np
import pandas as pd
from typing import Literal, Tuple, List

import cvxpy as cp  # usamos cvxpy para el MIP

from portfolio_tools.return_metrics import portfolio_returns, annualize_returns
from portfolio_tools.risk_metrics import (
    portfolio_volatility,
    calculate_max_drawdown,
    calculate_standard_deviation,
    annualize_standard_deviation,
)

# ---------------------------------------------------------------------
# Configuración del solver MIP
# ---------------------------------------------------------------------

# Cambia esto si instalas otro solver (por ejemplo "ECOS_BB" o "GUROBI")
MIP_SOLVER = "SCIP"


def _solve_cvxpy_problem(prob: cp.Problem) -> None:
    """Resuelve un problema cvxpy con el solver MIP configurado."""
    prob.solve(solver=MIP_SOLVER)

    if prob.status not in ["optimal", "optimal_inaccurate"]:
        raise ValueError(f"Optimización fallida. Status: {prob.status}")


# ---------------------------------------------------------------------
# 1) GMV con lógica 0 ó [min_w, max_w]
# ---------------------------------------------------------------------

def gmv_mip(
    covmat: pd.DataFrame | np.ndarray,
    min_w: float = 0.0,
    max_w: float = 1.0
) -> np.ndarray:
    """
    GMV con restricciones:
    - fully invested
    - long-only
    - cada activo tiene peso 0 o en [min_w, max_w]
    """
    cov = np.asarray(covmat)  # sirve tanto para DataFrame como para ndarray
    n = cov.shape[0]

    # Variables
    w = cp.Variable(n)                # pesos
    z = cp.Variable(n, boolean=True)  # binarios: 1 si el activo se usa

    # Objetivo: minimizar varianza (sin anualizar; no afecta a los pesos)
    objective = cp.Minimize(cp.quad_form(w, cov))

    constraints = [
        cp.sum(w) == 1.0,   # fully invested
        w >= 0,             # no cortos
        w >= min_w * z,
        w <= max_w * z,
    ]

    prob = cp.Problem(objective, constraints)
    _solve_cvxpy_problem(prob)

    return np.array(w.value).flatten()


# ---------------------------------------------------------------------
# 2) Minimizar volatilidad dado retorno objetivo (con MIP)
# ---------------------------------------------------------------------

def minimize_volatility_mip(
    target_return: float,
    returns: pd.DataFrame,
    covmat: np.ndarray | pd.DataFrame,
    method: Literal["simple", "log"] = "simple",
    periods_per_year: int = 252,
    min_w: float = 0.0,
    max_w: float = 1.0,
) -> np.ndarray:
    """
    Minimiza la volatilidad (varianza) sujeta a:
    - retorno esperado >= target_return
    - fully invested
    - long-only
    - 0 ó [min_w, max_w] por activo
    """
    cov = np.asarray(covmat)
    n = cov.shape[0]

    # retornos esperados anualizados por activo
    mu = annualize_returns(returns, method, periods_per_year).values  # shape (n,)

    w = cp.Variable(n)
    z = cp.Variable(n, boolean=True)

    # Objetivo: minimizar varianza anualizada
    objective = cp.Minimize(cp.quad_form(w, cov * periods_per_year))

    constraints = [
        cp.sum(w) == 1.0,
        w >= 0,
        mu @ w >= target_return,  # usamos >= en vez de == por robustez numérica
        w >= min_w * z,
        w <= max_w * z,
    ]

    prob = cp.Problem(objective, constraints)
    _solve_cvxpy_problem(prob)

    return np.array(w.value).flatten()


# ---------------------------------------------------------------------
# 3) Maximizar rentabilidad dado objetivo de volatilidad (con MIP)
# ---------------------------------------------------------------------

def maximize_return_mip(
    target_volatility: float,
    returns: pd.DataFrame,
    covmat: np.ndarray | pd.DataFrame,
    method: Literal["simple", "log"] = "simple",
    periods_per_year: int = 252,
    min_w: float = 0.0,
    max_w: float = 1.0,
) -> np.ndarray:
    """
    Maximiza la rentabilidad esperada sujeta a:
    - volatilidad anualizada <= target_volatility
    - fully invested
    - long-only
    - 0 ó [min_w, max_w] por activo
    """
    cov = np.asarray(covmat)
    n = cov.shape[0]

    mu = annualize_returns(returns, method, periods_per_year).values  # shape (n,)

    w = cp.Variable(n)
    z = cp.Variable(n, boolean=True)

    objective = cp.Maximize(mu @ w)

    # var_anual = w^T (Σ * periods_per_year) w
    # restricción: var_anual <= target_volatility^2
    cov_annual = cov * periods_per_year
    max_var = target_volatility**2

    constraints = [
        cp.sum(w) == 1.0,
        w >= 0,
        cp.quad_form(w, cov_annual) <= max_var,
        w >= min_w * z,
        w <= max_w * z,
    ]

    prob = cp.Problem(objective, constraints)
    _solve_cvxpy_problem(prob)

    return np.array(w.value).flatten()


# ---------------------------------------------------------------------
# 4) get_weights: barrer retornos objetivo usando minimize_volatility_mip
# ---------------------------------------------------------------------

def get_weights_mip(
    n_returns: int,
    returns: pd.DataFrame,
    covmat: np.ndarray | pd.DataFrame,
    method: Literal["simple", "log"] = "simple",
    periods_per_year: int = 252,
    min_w: float = 0.0,
    max_w: float = 1.0,
) -> List[np.ndarray]:
    """
    Devuelve una lista de vectores de pesos que minimizan la volatilidad
    para una rejilla de retornos objetivo, usando el modelo MIP.
    """
    annualized_returns = annualize_returns(returns, method, periods_per_year)
    ret_min = float(annualized_returns.min())
    ret_max = float(annualized_returns.max())

    target_returns = np.linspace(ret_min, ret_max, n_returns)

    weights_list = [
        minimize_volatility_mip(
            target_return=tr,
            returns=returns,
            covmat=covmat,
            method=method,
            periods_per_year=periods_per_year,
            min_w=min_w,
            max_w=max_w,
        )
        for tr in target_returns
    ]

    return weights_list


# ---------------------------------------------------------------------
# 5) get_weights_from_min_volatility: barrer volatilidades objetivo usando maximize_return_mip
# ---------------------------------------------------------------------

def get_weights_from_min_volatility_mip(
    n_volatilities: int,
    returns: pd.DataFrame,
    covmat: np.ndarray | pd.DataFrame,
    method: Literal["simple", "log"] = "simple",
    periods_per_year: int = 252,
    min_w: float = 0.0,
    max_w: float = 1.0,
) -> List[np.ndarray]:
    """
    Devuelve una lista de vectores de pesos que maximizan la rentabilidad esperada
    para una rejilla de volatilidades objetivo entre:
    - la volatilidad de la GMV (MIP)
    - la volatilidad del activo más volátil
    """
    cov = np.asarray(covmat)

    # 1) Volatilidad mínima: GMV
    w_gmv = gmv_mip(covmat, min_w=min_w, max_w=max_w)
    gmv_volatility = portfolio_volatility(w_gmv, cov, periods_per_year)

    # 2) Volatilidad máxima: activo más volátil
    stds = calculate_standard_deviation(returns)
    annualized_stds = annualize_standard_deviation(stds, periods_per_year)
    vol_max = float(annualized_stds.max())

    target_vols = np.linspace(gmv_volatility, vol_max, n_volatilities)

    weights_list = [
        maximize_return_mip(
            target_volatility=tv,
            returns=returns,
            covmat=cov,
            method=method,
            periods_per_year=periods_per_year,
            min_w=min_w,
            max_w=max_w,
        )
        for tv in target_vols
    ]

    return weights_list