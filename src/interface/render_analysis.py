import streamlit as st
from typing import Optional
import pandas as pd
from interface.main_interface import subheader, header
from interface.render_initial_portfolio import reset_portfolio_results
from interface.render_portfolio_results import show_portfolio_returns
from portfolio_tools.return_metrics import calculate_daily_returns

from portfolio_tools.investor_portfolios import get_cumulative_returns
from interface.visualizations import (
    plot_portfolio_values_select,
    plot_daily_returns_scatter_base_only,
    plot_daily_returns_distribution
)
from interface.constants import  RISK_PROFILE_DICTIONARY


def render_sidebar_analysis_selection(options: list[str], prefix: str = "analysis_stock") -> None:
    """
    Renders sidebar analysis selection.

    Parameters
    ----------
    options : list[str]. options.
    prefix : str. prefix.

    Returns
    -------
    None: None.
    """
    # Small separator so the sidebar doesn't look like a wall of text
    st.sidebar.markdown("---")
    st.sidebar.header("Selección de activos")

    # Keys to store selections in session_state (so they survive reruns)
    base_key = f"{prefix}_base"
    compare_key = f"{prefix}_compare"

    # If we don't have anything to pick from, just bail out nicely
    if not options:
        st.sidebar.info("No hay activos disponibles.")
        st.session_state[base_key] = None
        st.session_state[compare_key] = []
        return

    # If base isn't set yet (or it's not valid anymore), default to the first option
    if base_key not in st.session_state or st.session_state[base_key] not in options:
        st.session_state[base_key] = options[0]

    # Base asset selector (the one everything is compared against)
    base = st.sidebar.selectbox(
        "Base",
        options=options,
        index=options.index(st.session_state[base_key]),
        key=base_key,
    )

    # "Compare" list is everything except the base
    compare_options = [x for x in options if x != base]

    # If compare list doesn't exist yet, start it empty
    # If it does exist, clean it up so it only keeps valid values
    if compare_key not in st.session_state:
        st.session_state[compare_key] = []
    else:
        st.session_state[compare_key] = [x for x in st.session_state[compare_key] if x in compare_options]

    # Multi-select for comparables
    # (no default when using key, Streamlit gets grumpy otherwise)
    st.sidebar.multiselect(
        "Comparar con",
        options=compare_options,
        key=compare_key,
    )


def get_analysis_selection(prefix: str = "analysis_stock") -> tuple[Optional[str], list[str]]:
    """
    Gets analysis selection.

    Parameters
    ----------
    prefix : str. prefix.

    Returns
    -------
    tuple[Optional[str], list[str]]: get analysis selection output.
    """
    # Pull whatever the user picked from session_state
    base = st.session_state.get(f"{prefix}_base")
    compare = st.session_state.get(f"{prefix}_compare", [])

    # Just in case: remove base from compare if it sneaks in
    compare = [x for x in compare if x != base]
    return base, compare


def siderbar_show_charts():
    """
    Computes siderbar show charts.

    Parameters
    ----------


    Returns
    -------
    Any: siderbar show charts output.
    """
    # Section header + separator so the sidebar feels structured
    st.sidebar.markdown("---")
    st.sidebar.header("Selecciona visualizaciones")

    # Default values (so checkboxes are checked the first time)
    # Historic charts
    st.session_state.setdefault("show_resultado_historico", True)
    st.session_state.setdefault("show_evolucion_historica_precio", True)
    st.session_state.setdefault("retorno_diario_historico", True)
    st.session_state.setdefault("distribucion_historico", True)

    # Recent charts
    st.session_state.setdefault("show_resultado_recientes", True)
    st.session_state.setdefault("show_evolucion_reciente_precio", True)
    st.session_state.setdefault("retorno_diario_reciente", True)
    st.session_state.setdefault("distribucion_reciente", True)

    # Checkboxes for historic charts
    st.sidebar.checkbox("Resultados históricos", key="show_resultado_historico")
    st.sidebar.checkbox("Evolución histórica precio", key="show_evolucion_historica_precio")
    st.sidebar.checkbox("Retorno diario histórico", key="retorno_diario_historico")
    st.sidebar.checkbox("Distribución histórica", key="distribucion_historico")

    # Checkboxes for recent charts
    st.sidebar.checkbox("Resultados recientes", key="show_resultado_recientes")
    st.sidebar.checkbox("Evolución reciente precio", key="show_evolucion_reciente_precio")
    st.sidebar.checkbox("Retorno diario reciente", key="retorno_diario_reciente")
    st.sidebar.checkbox("Distribución reciente", key="distribucion_reciente")


def render_sidebar_display(options: list[str]) -> None:
    """
    Renders sidebar display.

    Parameters
    ----------
    options : list[str]. options.

    Returns
    -------
    None: None.
    """
    # Main sidebar navigation area
    st.sidebar.header("Navegación")

    # Back to questionnaire (and also wipe current portfolio results)
    if st.sidebar.button("Volver a cuestionario", use_container_width=True):
        reset_portfolio_results()
        st.session_state["route"] = "questionnaire"
        st.rerun()

    # Back to the initial portfolio screen
    if st.sidebar.button("Volver a cartera inicial", width="stretch"):
        st.session_state["route"] = "portfolio"
        st.rerun()

    # Go to portfolio performance results page
    if st.sidebar.button("Ver evolución cartera", width="stretch", type="primary"):
        st.session_state["route"] = "results"
        st.rerun()

    # Jump into the analysis page
    if st.sidebar.button("Ir a análisis de datos", width="stretch"):
        st.session_state["route"] = "analysis"
        st.rerun()

    # Asset selector for analysis charts (base + comparables)
    render_sidebar_analysis_selection(options, prefix="analysis_stock")

    # Chart selection checkboxes
    siderbar_show_charts()

    # Investor profile box on the sidebar (nice and visible)
    st.sidebar.markdown("---")
    st.sidebar.header("Perfil del inversor")

    risk = st.session_state.get("risk_result")
    if not risk:
        # If no profile yet, show a friendly message + shortcut button
        st.sidebar.info("Completa el cuestionario para ver tu perfil.")
        if st.sidebar.button("Ir al cuestionario", use_container_width=True):
            st.session_state["route"] = "questionnaire"
            st.rerun()
        return

    # Pull out profile + volatility range
    perfil = risk.get("RA", "—")
    sigma_min = risk.get("sigma_min", None)
    sigma_max = risk.get("sigma_max", None)

    # Try to map the profile number to a human label
    try:
        perfil_text = RISK_PROFILE_DICTIONARY.get(perfil, "Perfil no disponible")
    except Exception:
        perfil_text = "Perfil no disponible"

    # Show the profile in a little styled box
    st.sidebar.markdown(
        f"""
        <div style="
            border: 1px solid rgba(0,0,0,0.10);
            border-radius: 12px;
            padding: 10px 12px;
            background: #D6FAFF;
            opacity: 0.9
        ">
            <div style="font-size: 1.2rem; color: #000078; font-weight:900; text-align: center">Perfil</div>
            <div style="font-size: 2.5rem; color: #000078; font-weight: 900; text-align: center">{perfil}</div>
            <div style="font-size: 1.1rem; color: #000078; font-weight: 900; text-align: center">{perfil_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Add volatility range under the box if we have it
    if sigma_min is not None and sigma_max is not None:
        st.sidebar.caption(f"Volatilidad recomendada: {sigma_min:.2f}–{sigma_max:.2f}")


def _get_analysis_options_from_initial_data() -> list[str]:
    """
    Computes  get analysis options from initial data.

    Parameters
    ----------

    Returns
    -------
    list[str]:  get analysis options from initial data output.
    """
    # Grab the initial dataset from session_state
    initial_data = st.session_state.get("initial_data")
    if initial_data is None:
        return []

    # Returns for stocks + portfolio values that we created earlier
    historic_returns = initial_data.get("train_set")
    historic_portfolio_values = st.session_state.get("dict_pf_returns")

    # If data isn't there (or it's empty), nothing to show
    if historic_returns is None or not isinstance(historic_returns, pd.DataFrame) or historic_returns.empty:
        return []

    if historic_portfolio_values is None or not isinstance(historic_portfolio_values, dict):
        return []

    df_clean = historic_returns.copy()

    # Clean up useless columns:
    # - columns that are all zeros
    # - columns that are completely NaN
    df_clean = df_clean.drop(columns=df_clean.columns[(df_clean == 0).all()], errors="ignore")
    df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isna().all()], errors="ignore")

    # Portfolio series keys (like "Investor", "GMV", etc.)
    historic_keys = list(historic_portfolio_values.keys())

    # Combine: stocks + portfolios into a single selector list
    return list(df_clean.columns) + historic_keys


def render_historic_performance() -> None:
    """
    Renders historic perisk-freeormance.

    Parameters
    ----------

    Returns
    -------
    None: None.
    """
    # If user hasn't generated a portfolio yet, we can't show analysis
    initial_data = st.session_state.get("initial_data")
    if initial_data is None:
        st.warning("Primero genera la cartera para poder ver el análisis.")
        return

    # Grab portfolio values and turn them into returns we can plot
    initial_amount = st.session_state["investor_constraints_draft"]["amount"]
    recent_portfolio_values = st.session_state.get("dict_pf_returns_forecast")
    historic_portfolio_values = st.session_state.get("dict_pf_returns")

    # Normalize recent values to 1 unit (so it's comparable)
    df_recent_portfolio_values = pd.DataFrame(recent_portfolio_values) / initial_amount
    df_recent_portfolio_returns = calculate_daily_returns(df_recent_portfolio_values)

    df_historic_portfolio_values = pd.DataFrame(historic_portfolio_values)
    df_historic_portfolio_returns = calculate_daily_returns(df_historic_portfolio_values)

    # Raw stock returns + prices
    historic_returns = initial_data.get("train_set")
    recent_returns = initial_data.get("test_set")

    historic_prices = initial_data.get("train_price")
    recent_prices = initial_data.get("test_price")

    # Human-friendly name for plotting titles
    base_name = st.session_state.get("analysis_stock_base")

    # If something is missing, stop before everything explodes
    if historic_returns is None or recent_returns is None:
        st.warning("No hay datos de retornos para mostrar.")
        return

    # Compute cumulative returns and also add portfolio curves so everything is in one place
    cum_returns_historic = get_cumulative_returns(historic_returns)
    cum_returns_historic = cum_returns_historic.join(df_historic_portfolio_values)

    cum_returns_recent = get_cumulative_returns(recent_returns)
    cum_returns_recent = cum_returns_recent.join(df_recent_portfolio_values)

    # Add portfolio daily returns to the returns DataFrames
    historic_returns = historic_returns.join(df_historic_portfolio_returns)
    recent_returns = recent_returns.join(df_recent_portfolio_returns)

    # What user picked in the sidebar (base + comparables)
    base, compare = get_analysis_selection(prefix="analysis_stock")
    selected = [base] + compare if base else []

    # If user didn't pick a base yet, tell them what to do
    if not base:
        st.info("Selecciona un activo base en el lateral para ver las visualizaciones.")
        return

    # First row: cumulative returns + price evolution (historic)
    c1, c2 = st.columns(2)
    if st.session_state.get("show_resultado_historico", True):
        with c1:
            subheader(f"Resultados históricos de {base_name} (y comparable)", font_size="1.8rem")
            plot_portfolio_values_select(
                cum_returns_historic,
                key="historic_cum",
                portfolio_type="stock",
                selected=selected,
                show_selector=False,
            )

    if st.session_state.get("show_evolucion_historica_precio", True):
        with c2:
            subheader(f"Evolución histórica del precio de {base_name} (y comparable)", font_size="1.8rem")
            plot_portfolio_values_select(
                historic_prices,
                key="historic_price",
                portfolio_type="stock",
                selected=selected,
                show_selector=False,
            )

    # Second row: daily return scatter + distribution (historic)
    d1, d2 = st.columns(2)
    if st.session_state.get("retorno_diario_historico", True):
        with d1:
            subheader(f"Retorno diario histórico de {base_name} en %", font_size="1.8rem")
            plot_daily_returns_scatter_base_only(
                historic_returns,
                key="daily_scatter_historic",
                data_type="stock",
                base=base,
                y_in_percent=True
            )

    if st.session_state.get("distribucion_historico", True):
        with d2:
            subheader(f"Distribución del rendimiento diario histórico de {base_name}", font_size="1.8rem")
            plot_daily_returns_distribution(
                historic_returns,
                base=base,
                data_type="stock",
                y_in_percent=True,
                key="dist_daily_returns_historic",
            )

    # Third row: cumulative returns + price evolution (recent)
    e1, e2 = st.columns(2)
    if st.session_state.get("show_resultado_recientes", True):
        with e1:
            subheader(f"Resultados recientes de {base_name} (y comparable)", font_size="1.8rem")
            plot_portfolio_values_select(
                cum_returns_recent,
                key="recent_cum",
                portfolio_type="stock",
                selected=selected,
                show_selector=False,
            )

    if st.session_state.get("show_evolucion_reciente_precio", True):
        with e2:
            subheader(f"Evolución reciente del precio de {base_name} (y comparable)", font_size="1.8rem")
            plot_portfolio_values_select(
                recent_prices,
                key="recent_price",
                portfolio_type="stock",
                selected=selected,
                show_selector=False,
            )

    # Fourth row: daily return scatter + distribution (recent)
    f1, f2 = st.columns(2)
    if st.session_state.get("retorno_diario_reciente", True):
        with f1:
            subheader(f"Retorno diario reciente de {base_name} en %", font_size="1.8rem")
            plot_daily_returns_scatter_base_only(
                recent_returns,
                key="daily_scatter_recent",
                data_type="stock",
                base=base,
                y_in_percent=True
            )

    if st.session_state.get("distribucion_reciente", True):
        with f2:
            subheader(f"Distribución del rendimiento diario reciente de {base_name}", font_size="1.8rem")
            plot_daily_returns_distribution(
                recent_returns,
                base=base,
                data_type="stock",
                y_in_percent=True,
                key="dist_daily_returns_recent",
            )


def render_analysis() -> None:
    """
    Renders analysis.

    Parameters
    ----------

    Returns
    -------
    None: None.
    """
    # Main page title
    header("ANÁLISIS DE CARTERA Y OTROS ACTIVOS")
    st.write("")
    st.write("")

    # Show portfolio performance summary first
    show_portfolio_returns()

    # Build the list of things the user can pick in the sidebar
    options = _get_analysis_options_from_initial_data()

    # Sidebar navigation + chart toggles + profile box
    render_sidebar_display(options)

    # Main charts and analysis
    render_historic_performance()