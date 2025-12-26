import streamlit as st
from types import MappingProxyType
from typing import Optional, Literal
import pandas as pd
from interface.main_interface import subheader, header
from interface.render_portfolio_results import show_portfolio_returns
from portfolio_tools.return_metrics import calculate_daily_returns


from portfolio_management.investor_portfolios import get_cumulative_returns
from interface.visualizations import (
    plot_portfolio_values_select,
    plot_daily_returns_scatter_base_only,
    plot_daily_returns_distribution
)

PERIODS_PER_YEAR = 255

RISK_PROFILE_DICTIONARY = MappingProxyType({
    1: "Perfil bajo de riesgo",
    2: "Perfil medio-bajo de riesgo",
    3: "Perfil medio de riesgo",
    4: "Perfil medio-alto de riesgo",
    5: "Perfil alto de riesgo",
    6: "Perfil agresivo de riesgo"
})


def render_sidebar_analysis_selection(options: list[str], prefix: str = "analysis_stock") -> None:
    """
    Renders a global selector in the sidebar to control the whole analysis page.

    Parameters
    ----------
    options: list[str]. Available assets/series names to select from.
    prefix: str. Prefix used to namespace session_state keys for this page.

    Returns
    -------
    None
    """
    st.sidebar.markdown("---")
    st.sidebar.header("Selección de activos")

    base_key = f"{prefix}_base"
    compare_key = f"{prefix}_compare"

    if not options:
        st.sidebar.info("No hay activos disponibles.")
        st.session_state[base_key] = None
        st.session_state[compare_key] = []
        return

    # Initialize base if missing or invalid
    if base_key not in st.session_state or st.session_state[base_key] not in options:
        st.session_state[base_key] = options[0]

    base = st.sidebar.selectbox(
        "Base",
        options=options,
        index=options.index(st.session_state[base_key]),
        key=base_key,
    )

    compare_options = [x for x in options if x != base]

    # Initialize compare if missing (and keep only valid values)
    if compare_key not in st.session_state:
        st.session_state[compare_key] = []
    else:
        st.session_state[compare_key] = [x for x in st.session_state[compare_key] if x in compare_options]

    # IMPORTANT: no default=... when using key
    st.sidebar.multiselect(
        "Comparar con",
        options=compare_options,
        key=compare_key,
    )


def get_analysis_selection(prefix: str = "analysis_stock") -> tuple[Optional[str], list[str]]:
    """
    Reads the current base and comparison selection for the analysis page.

    Parameters
    ----------
    prefix: str. Namespace prefix used for session_state keys.

    Returns
    -------
    base: Optional[str]. The base series name.
    compare: list[str]. A list of comparison series names (base removed if present).
    """
    base = st.session_state.get(f"{prefix}_base")
    compare = st.session_state.get(f"{prefix}_compare", [])
    compare = [x for x in compare if x != base]
    return base, compare


def siderbar_show_charts():
    st.sidebar.markdown("---")
    st.sidebar.header("Selecciona visualizaciones")

    # Selection charts historic
    st.session_state.setdefault("show_resultado_historico", True)
    st.session_state.setdefault("show_evolucion_historica_precio", True)
    st.session_state.setdefault("retorno_diario_historico", True)
    st.session_state.setdefault("distribucion_historico", True)

    # Selection charts recent
    st.session_state.setdefault("show_resultado_recientes", True)
    st.session_state.setdefault("show_evolucion_reciente_precio", True)
    st.session_state.setdefault("retorno_diario_reciente", True)
    st.session_state.setdefault("distribucion_reciente", True)

    # Show charts historic
    st.sidebar.checkbox("Resultados históricos", key="show_resultado_historico")
    st.sidebar.checkbox("Evolución histórica precio", key="show_evolucion_historica_precio")
    st.sidebar.checkbox("Retorno diario histórico", key="retorno_diario_historico")
    st.sidebar.checkbox("Distribución histórica", key="distribucion_historico")

    # Selection charts recent
    st.sidebar.checkbox("Resultados recientes", key="show_resultado_recientes")
    st.sidebar.checkbox("Evolución reciente precio", key="show_evolucion_reciente_precio")
    st.sidebar.checkbox("Retorno diario reciente", key="retorno_diario_reciente")
    st.sidebar.checkbox("Distribución reciente", key="distribucion_reciente")



def render_sidebar_display(options: list[str]) -> None:
    """
    Renders the analysis sidebar: navigation, global selectors, and investor profile.

    Parameters
    ----------
    options: list[str]. Available asset/series names for the analysis selector.

    Returns
    -------
    None
    """
    st.sidebar.header("Navegación")

    if st.sidebar.button("Volver a cuestionario", use_container_width=True):
        # External messages in Spanish
        reset_portfolio_results()
        st.session_state["route"] = "questionnaire"
        st.rerun()

    if st.sidebar.button("Volver a cartera inicial", width="stretch"):
        st.session_state["route"] = "portfolio"
        st.rerun()

    if st.session_state.get("step2_enabled", False):
        if st.sidebar.button("Ver evolución cartera", use_container_width=True, type="primary"):
            st.session_state["route"] = "results"
            st.rerun()
    else:
        st.sidebar.button("Ver evolución cartera", use_container_width=True, disabled=True)

    if st.sidebar.button("Ir a análisis de datos", use_container_width=True):
        st.session_state["route"] = "analysis"
        st.rerun()

    # Global selection for all analysis charts
    render_sidebar_analysis_selection(options, prefix="analysis_stock")

    # Checkbox
    siderbar_show_charts()
    # Investor profile (rendered nicely)
    st.sidebar.markdown("---")
    st.sidebar.header("Perfil del inversor")

    risk = st.session_state.get("risk_result")
    if not risk:
        st.sidebar.info("Completa el cuestionario para ver tu perfil.")
        if st.sidebar.button("Ir al cuestionario", use_container_width=True):
            st.session_state["route"] = "questionnaire"
            st.rerun()
        return

    perfil = risk.get("RA", "—")
    sigma_min = risk.get("sigma_min", None)
    sigma_max = risk.get("sigma_max", None)

    # External content in Spanish
    try:
        perfil_text = RISK_PROFILE_DICTIONARY.get(perfil, "Perfil no disponible")
    except Exception:
        perfil_text = "Perfil no disponible"

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

    if sigma_min is not None and sigma_max is not None:
        st.sidebar.caption(f"Volatilidad recomendada: {sigma_min:.2f}–{sigma_max:.2f}")


def _get_analysis_options_from_initial_data() -> list[str]:
    """
    Builds the list of available assets for the analysis page from session_state['initial_data'].

    Returns
    -------
    options: list[str]. Cleaned list of column names (all-zero/all-NaN removed).
    """
    initial_data = st.session_state.get("initial_data")
    if initial_data is None:
        return []

    historic_returns = initial_data.get("train_set")
    historic_portfolio_values = st.session_state.get("dict_pf_returns")
    if historic_returns is None or not isinstance(historic_returns, pd.DataFrame) or historic_returns.empty:
        return []

    if historic_portfolio_values is None or not isinstance(historic_portfolio_values, dict):
        return []

    df_clean = historic_returns.copy()

    # Drop columns that are all zeros or all NaNs
    df_clean = df_clean.drop(columns=df_clean.columns[(df_clean == 0).all()], errors="ignore")
    df_clean = df_clean.drop(columns=df_clean.columns[df_clean.isna().all()], errors="ignore")

    historic_keys = list(historic_portfolio_values.keys())

    return list(df_clean.columns) + historic_keys


def render_historic_performance() -> None:
    """
    Renders the analysis charts using the global sidebar selection:
    - Base + comparisons for cumulative returns and prices
    - Base-only for daily returns scatter (color-coded)

    Returns
    -------
    None
    """
    initial_data = st.session_state.get("initial_data")
    if initial_data is None:
        st.warning("Primero genera la cartera para poder ver el análisis.")
        return

    # We get the data
    initial_amount = st.session_state["investor_constraints_draft"]["amount"]
    recent_portfolio_values = st.session_state.get("dict_pf_returns_forecast")
    historic_portfolio_values = st.session_state.get("dict_pf_returns")


    df_recent_portfolio_values = pd.DataFrame(recent_portfolio_values) / initial_amount
    df_recent_portfolio_returns = calculate_daily_returns(df_recent_portfolio_values)

    df_historic_portfolio_values = pd.DataFrame(historic_portfolio_values)
    df_historic_portfolio_returns = calculate_daily_returns(df_historic_portfolio_values)

    historic_returns = initial_data.get("train_set")
    recent_returns = initial_data.get("test_set")


    historic_prices = initial_data.get("train_price")
    recent_prices = initial_data.get("test_price")
    base_name = st.session_state.get("analysis_stock_base")

    if historic_returns is None or recent_returns is None:
        st.warning("No hay datos de retornos para mostrar.")
        return

    # Compute cumulative returns
    cum_returns_historic = get_cumulative_returns(historic_returns)
    cum_returns_historic = cum_returns_historic.join(df_historic_portfolio_values)
    cum_returns_recent = get_cumulative_returns(recent_returns)
    cum_returns_recent = cum_returns_recent.join(df_recent_portfolio_values)


    historic_returns = historic_returns.join(df_historic_portfolio_returns)
    recent_returns = recent_returns.join(df_recent_portfolio_returns)

    # Global selection from sidebar
    base, compare = get_analysis_selection(prefix="analysis_stock")
    selected = [base] + compare if base else []

    if not base:
        st.info("Selecciona un activo base en el lateral para ver las visualizaciones.")
        return

    # Plots: cumulative returns (base + compare)
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

    d1, d2 = st.columns(2)
    if st.session_state.get("retorno_diario_historico", True):
        with d1:
            #  Plot: daily returns scatter
            subheader(f"Retorno diario histórico de {base_name} en %", font_size="1.8remrem")
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
    f1, f2 = st.columns(2)

    if st.session_state.get("retorno_diario_reciente", True):
        with f1:
            #  Plot: daily recent resutls
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
    Main entrypoint for the Analysis page.

    It renders:
    - A sidebar with navigation + investor profile + global selector (base vs compare)
    - A set of charts driven by that global selection

    External UI messages are shown in Spanish; internal comments/messages remain in English.
    """
    header("ANÁLISIS DE CARTERA Y OTROS ACTIVOS")
    st.write("")
    st.write("")

    show_portfolio_returns()
    # Build available options for sidebar selector
    options = _get_analysis_options_from_initial_data()

    # Render sidebar (needs options)
    render_sidebar_display(options)

    # Render page content
    render_historic_performance()