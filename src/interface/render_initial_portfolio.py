import streamlit as st
from typing import Optional
import pandas as pd
import datetime as dt
from interface.main_interface import subheader, header
from data_management.get_data import get_stock_prices, exists_asset
from data_management.dataset_preparation import split_data_markowtiz
from data_management.clean_data import clean_and_align_data
from portfolio_tools.return_metrics import calculate_daily_returns
from interface.landing_page import add_separation
from portfolio_tools.portfolio_management import get_sector_weights_at_date, check_portfolio_weights
from interface.visualizations import show_portfolio, render_results_table, show_markowitz_results, plot_portfolio_values
from portfolio_tools.investor_portfolios import get_sector_exposure_table, create_output_table_portfolios, \
    render_historical_portfolios_results

from interface.constants import (RISK_PROFILE_DICTIONARY,
                                 PERIODS_PER_YEAR,
                                 FILENAME_PATH,
                                 TICKER_COL,
                                 START_DATE,
                                 ENDING_DATE,
                                 FONT_SIZE,
                                 FONT_COLOR,
                                 FONT_WEIGHT,
                                 INITIAL_DATE,
                                 END_DATE,
                                 INITIAL_DATE_PORTFOLIO,
                                 END_DATE_PORTFOLIO)







def render_slider(text: str,
                  text_slider: str,
                  key: str,
                  min_value: float,
                  max_value: float,
                  value=25.0,
                  font_color: str = FONT_COLOR,
                  font_size: str = FONT_SIZE,
                  font_weight: str = FONT_WEIGHT):
    """
    Renders slider.

    Parameters
    ----------
    text : str. text.
    text_slider : str. text slider.
    key : str. key.
    min_value : float. min value.
    max_value : float. max value.
    value : Any. value.
    font_color : str. font color.
    font_size : str. font size.
    font_weight : str. font weight.

    Returns
    -------
    Any: render slider output.
    """
    st.markdown(
        f"""
         <div style="
         font-size: {font_size}; 
         font-weight: {font_weight};
         color: {font_color};
         margin-bottom: 1rem;
         margin-top: 2.0rem;
         text-align:center;
         ">
         {text} 
         </div>
         """, unsafe_allow_html=True)
    max_pct = st.slider(
        text_slider,
        min_value=min_value, max_value=max_value, value=value, step=0.5,
        label_visibility="collapsed",
        key=key
    )

    st.markdown(
        f"""
         <div style="
         font-size: 1.2rem;
         font-weight: {font_weight};
         color: {font_color};
         margin-bottom: -1.0rem;
         margin-top: -1.0rem;
         text-align:center;
         "> 
        {max_pct}%
         </div>
         """,
        unsafe_allow_html=True
    )
    return max_pct


def render_number_input(text: str,
                        min_value: float,
                        init_value: float,
                        step: float,
                        key: str,
                        font_color: str = FONT_COLOR,
                        font_size: str = FONT_SIZE,
                        font_weight: str = FONT_WEIGHT,
                        unit: str = "EUR",
                        number_format: str = ".2f",):
    """
    Renders number input.

    Parameters
    ----------
    text : str. text.
    min_value : float. min value.
    init_value : float. init value.
    step : float. step.
    key : str. key.
    font_color : str. font color.
    font_size : str. font size.
    font_weight : str. font weight.
    unit : str. unit.
    number_format : str. format.

    Returns
    -------
    Any: render number input output.
    """
    st.markdown(f"""
    <div style="
        font-size: {font_size};
        font-weight: {font_weight};
        color: {font_color};
        text-align: center;
         margin-bottom: 1rem;
         margin-top: 1.0rem;
    ">
        {text}
    </div>
    """, unsafe_allow_html=True)

    amount = st.number_input(
        text,
        min_value=min_value,
        value=init_value,
        step=step,
        format="%"+number_format,
        label_visibility="collapsed",
        key=key
    )

    formatted_amount = (
        format(amount, number_format)
        .replace(",", "X")
        .replace(".", ",")
        .replace("X", ".")
    )

    st.markdown(
        f"""
         <div style="
         font-size: 1.2rem;
         font-weight: {font_weight};
         color: {font_color};
         margin-bottom: 1rem;
         margin-top: 0.0rem;
         text-align:center;
         "> 
        {formatted_amount} {unit}
         </div>
         """,
        unsafe_allow_html=True
    )
    return amount


def render_date(text: str,
                reference_date: dt.date,
                key: str,
                font_color: str = FONT_COLOR,
                font_size: str = FONT_SIZE,
                font_weight: str = FONT_WEIGHT):
    """
    Renders date.

    Parameters
    ----------
    text : str. text.
    reference_date : dt.date. reference date.
    key : str. key.
    font_color : str. font color.
    font_size : str. font size.
    font_weight : str. font weight.

    Returns
    -------
    Any: render date output.
    """
    st.markdown(f"""
    <div style="
        font-size: {font_size};
        font-weight: {font_weight};
        color: {font_color};
        text-align: center;
        margin-bottom: 1rem;
        margin-top: 1.0rem;
    ">
        {text}
    </div>
    """, unsafe_allow_html=True)

    date = st.date_input(
        "Fecha de la aportación",
        value=reference_date,
        label_visibility="collapsed",
        key=key
    )
    st.markdown(
        f"""
         <div style="
         font-size: 1.2rem;
         font-weight: {font_weight};
         color: {font_color};
         margin-bottom: 1rem;
         margin-top: 0.0rem;
         text-align:center;
         "> 
        {date}
         </div>
         """,
        unsafe_allow_html=True
    )
    return date


def add_assets():
    """
    Adds assets.

    Parameters
    ----------


    Returns
    -------
    Any: add assets output.
    """
    existing_assets = pd.read_csv(FILENAME_PATH)
    column_with_tickers = list(existing_assets[TICKER_COL])
    # Initialize session state
    st.session_state.setdefault("custom_tickers", [])
    with st.container(border=True):
        b1, b2 = st.columns(2)
        with b1:
            subheader("Añadir activo de manera manual a lista inicial", font_size="1.8rem", color="#006400")

            ticker = st.text_input("Ticker de Yahoo",
                                   placeholder="ej: SAN.MC, BBVA.MC, AAPL",
                                   key="ticker")

            d1, d2, d3 = st.columns(3)
            with d2:
                add = st.button("Añadir activo", type="primary")

            if add:
                t = ticker.strip().upper()
                if not t:
                    st.warning("Introduce un ticker válido.")

                if t in column_with_tickers:
                    st.warning(f"El ticker {t} ya está en la lista inicial")
                elif t in st.session_state["custom_tickers"]:
                    st.warning(f"El ticker {t} ya está añadido.")
                elif not exists_asset(t):
                    st.warning(f"El ticker {t} no existe en Yahoo Finance.")

                else:
                    st.session_state["custom_tickers"].append(t)
                    st.success(f"Ticker {t} añadido.")
                    st.session_state["ticker_input"] = ""
            if st.session_state["custom_tickers"]:
                e1, e2 = st.columns(2)
                with e1:
                    st.write("Nuevos activos añadidos hasta el momento")
                    st.write(", ".join(st.session_state["custom_tickers"]))
                with b2:

                    subheader("Eliminar activos añadidos", font_size="1.8rem", color="#FF0000")

                    to_remove = st.multiselect(
                        "Eliminar activos añadidos",
                        options=st.session_state["custom_tickers"],
                        key="remove_custom_tickers",
                    )

                    g1, g2, g3 = st.columns(3)
                    with g2:
                        if st.button("Eliminar seleccionadas"):
                            st.session_state["custom_tickers"] = [
                                t for t in st.session_state["custom_tickers"] if t not in to_remove
                            ]
                            st.rerun()


def render_investor_constraints():
    """
    Renders investor constraints.

    Parameters
    ----------


    Returns
    -------
    Any: render investor constraints output.
    """
    subheader("Defina el grado de diversificación y el importe inicial para la propuesta de cartera",
              margin_bottom="1.0rem")

    add_separation()
    with st.container(border=False):
        subheader("Pesos de la cartera", font_size="1.8rem")
        c1, c2, c3 = st.columns(3, gap="large")

        with c1:
            max_sector_pct = render_slider(text="Selecciona el peso máximo por sector",
                                           text_slider="% máximo asignado a un sector",
                                           key="max_sector_pct",
                                           min_value=0.0,
                                           max_value=100.0)

        with c2:
            max_stock_pct = render_slider(text="Selecciona peso máximo por acción",
                                          text_slider="% máximo asignado a una acción",
                                          key="max_stock_pct",
                                          min_value=0.0,
                                          max_value=100.0,
                                          value=12.5,
                                          font_color="#FF0000")

        with c3:
            min_stock_pct = render_slider(text="Selecciona peso mínimo por acción",
                                          text_slider="% mínimo asignado a una acción",
                                          key="min_stock_pct",
                                          min_value=0.0,
                                          max_value=100.0,
                                          value=2.5)

    with st.container(border=True):
        subheader("Selección de fechas para extracción de datos", font_size="1.8rem")
        subheader("(Se recomienda no ir más allá de 5 años atrás en el tiempo)")
        s1, s2 = st.columns(2)

        with s1:
            st1, st2 = st.columns(2)
            with st2:
                data_start_date = render_date("Fecha de inicio de extracción de datos", key="date",
                                              reference_date=INITIAL_DATE)
        with s2:
            stt1, stt2 = st.columns(2)
            with stt1:
                data_end_date = render_date("Fecha de fin de extracción de datos", font_color="#FF0000", key="date2",
                                            reference_date=END_DATE)

    with st.container(border=True):
        subheader("Selección de fechas para análisis de cartera", font_size="1.8rem", color="#006400")
        w1, w2 = st.columns(2)

        with w1:
            wt1, wt2 = st.columns(2)
            with wt2:
                date_portfolio_start = render_date("Fecha de inicio de generación de cartera", key="date_portfolio",
                                                   reference_date=INITIAL_DATE_PORTFOLIO)
        with w2:
            wtt1, wtt2 = st.columns(2)
            with wtt1:
                date_portfolio_end = render_date("Fecha de fin de generación de cartera", font_color="#006400",
                                                 key="date_portfolio2", reference_date=END_DATE_PORTFOLIO)

    with st.container(border=False):
        subheader("Inversión inicial, tipo de interés libre de riesgo y volatilidad escogida", font_size="1.8rem")
        t1, t2, t3 = st.columns(3)
        with t1:
            amount = render_number_input("Inversión inicial",
                                         min_value=0.0,
                                         init_value=10000.0,
                                         step=100.0,
                                         key="cash_contribution")
        with t2:
            risk_free_rate = render_number_input("Tipo de interés libre de riesgo",
                                                 min_value=0.0,
                                                 init_value=3.2,
                                                 step=0.01,
                                                 font_color="#006400",
                                                 key="risk_free_rate",
                                                 unit="%")

        # Investor profile
        profile = st.session_state["risk_result"]
        volatility = (profile["sigma_min"] + profile["sigma_max"]) / 2
        with t3:
            volatility_selected = render_number_input("Volatilidad escogida",
                                                      min_value=0.0,
                                                      init_value=volatility,
                                                      step=0.0005,
                                                      font_color="#FF0000",
                                                      key="selected_volatility",
                                                      unit="",
                                                      number_format=".4f")

    add_assets()

    # Save session state
    st.session_state["investor_constraints_draft"] = {
        "max_sector_pct": max_sector_pct,
        "max_stock_pct": max_stock_pct,
        "min_stock_pct": min_stock_pct,
        "data_start_date": data_start_date,
        "data_end_date": data_end_date,
        "date_portfolio_start": date_portfolio_start,
        "date_portfolio_end": date_portfolio_end,
        "amount": amount,
        "risk_free_rate": risk_free_rate / 100,
        "volatility": volatility_selected
    }


def get_clean_initial_data(filename_path: str = FILENAME_PATH,
                           ticker_col: str = TICKER_COL,
                           additional_tickers: Optional[list] = None,
                           adjusted: bool = False,
                           start_date: str = START_DATE,
                           end_date: str = ENDING_DATE,
                           initial_date_portfolio: str = INITIAL_DATE_PORTFOLIO,
                           end_date_portfolio: str = END_DATE_PORTFOLIO):
    """
    Gets clean initial data.

    Parameters
    ----------
    filename_path : str. filename path.
    ticker_col : str. ticker col.
    additional_tickers : Optional[list]. additional tickers.
    adjusted : bool. adjusted.
    start_date : str. start date.
    end_date : str. end date.
    initial_date_portfolio : str. initial date portfolio.
    end_date_portfolio : str. end date portfolio.

    Returns
    -------
    Any: get clean initial data output.
    """
    price_data, sectors = get_stock_prices(file_path=filename_path,
                                           ticker_col=ticker_col,
                                           additional_tickers=additional_tickers,
                                           adjusted=adjusted,
                                           start_date=start_date,
                                           end_date=end_date,
                                           )

    prices, report, summary = clean_and_align_data(price_data, beginning_data=True)
    daily_returns = calculate_daily_returns(prices, method="simple")
    train_set, test_set = split_data_markowtiz(returns=daily_returns, test_date_start=initial_date_portfolio,
                                               test_date_end=end_date_portfolio)
    train_price, test_price = split_data_markowtiz(returns=prices, test_date_start=initial_date_portfolio,
                                                   test_date_end=end_date_portfolio)

    # We return relevant data
    return {
        "prices": prices,
        "daily_returns": daily_returns,
        "train_set": train_set,
        "test_set": test_set,
        "train_price": train_price,
        "test_price": test_price,
        "sectors": sectors
    }


def get_initial_data():
    """
    Gets initial data.

    Parameters
    ----------


    Returns
    -------
    Any: get initial data output.
    """
    constraints = st.session_state["investor_constraints_draft"]
    data_start_date = constraints["data_start_date"]
    data_end_date = constraints["data_end_date"]
    data_portfolio_start = constraints["date_portfolio_start"]
    data_portfolio_end = constraints["date_portfolio_end"]
    additional_tickers = st.session_state["custom_tickers"]
    st.session_state["initial_data"] = get_clean_initial_data(
        additional_tickers=additional_tickers,
        start_date=data_start_date.isoformat(),
        end_date=data_end_date.isoformat(),
        initial_date_portfolio=data_portfolio_start,
        end_date_portfolio=data_portfolio_end
    )
    st.session_state.data_ready = True


def get_initial_portfolio():
    # We get the relevant information constraints
    """
    Gets initial portfolio.

    Parameters
    ----------


    Returns
    -------
    Any: get initial portfolio output.
    """
    constraints = st.session_state["investor_constraints_draft"]
    max_stock_pct = constraints["max_stock_pct"] / 100
    min_stock_pct = constraints["min_stock_pct"] / 100
    max_sector_pct = constraints["max_sector_pct"] / 100
    risk_free_rate = constraints["risk_free_rate"]
    volatility = constraints["volatility"]


    # portfolio data
    resultados = st.session_state["initial_data"]
    train_set = resultados["train_set"]
    sectors = resultados["sectors"]
    st.session_state["initial_results"] = create_output_table_portfolios(train_set,
                                                                         min_w=min_stock_pct,
                                                                         max_w=max_stock_pct,
                                                                         rf_annual=risk_free_rate,
                                                                         periods_per_year=PERIODS_PER_YEAR,
                                                                         custom_target_volatility=volatility,
                                                                         sectors_df=sectors,
                                                                         sector_max_weight=max_sector_pct,
                                                                         risk_free_ticker="RISK_FREE")


def create_historical_portfolio_visualizations():
    """
    Creates historical portfolio visualizations.

    Parameters
    ----------


    Returns
    -------
    Any: create historical portfolio visualizations output.
    """
    if not st.session_state.get("data_ready", False):
        return
    df_weights = st.session_state["initial_results"][1]["investor"]
    sectors = st.session_state["initial_data"]["sectors"]
    sectores = get_sector_exposure_table(df_weights, sectors)
    df_results = st.session_state["initial_results"][0]
    df_returns = st.session_state["initial_data"]["train_set"]

    with st.container(border=False):
        col1, col2 = st.columns(2)
        if st.session_state.get("show_alloc_assets", True):
            with col1:
                show_portfolio(
                    df_weights=df_weights,
                    title="Composición por activo",
                    label_name="Activo",
                    weight_col="Pesos",
                    weights_in_percent=False
                )
        if st.session_state.get("show_alloc_sectors", True):
            with col2:
                show_portfolio(
                    df_weights=sectores.set_index("sector"),
                    title="Composición por sector",
                    label_name="Sector",
                    weight_col="Pesos",
                    weights_in_percent=True
                )

    # We now render the main table of results and comparable portfolios
    with st.container(border=False):
        c1, c2 = st.columns(2)
        if st.session_state.get("show_results_table", True):
            with c1:
                subheader("Resultados de la cartera", font_size="1.8rem")
                render_results_table(df_results)

        # We now render the efficient frontier and the comparable portfolios
        if st.session_state.get("show_frontier", True):
            with c2:
                subheader("Frontera eficiente y portfolios", font_size="1.8rem")
                show_markowitz_results(n_returns=100, returns=df_returns, df_results=df_results,
                                       periods_per_year=PERIODS_PER_YEAR)


def create_historical_results_visualizations():
    """
    Creates historical results visualizations.

    Parameters
    ----------


    Returns
    -------
    Any: create historical results visualizations output.
    """
    if st.session_state.get("show_historical", True):
        with st.container(border=False):
            subheader("Resultados históricos de la cartera antes de inversión", font_size="1.8rem")
            dict_pf_returns = st.session_state.get("dict_pf_returns")
            if dict_pf_returns is None:
                st.info("Pulsa **Generar cartera** para calcular los resultados históricos.")
                return

            plot_portfolio_values(dict_pf_returns, key="historic_portfolio")
    if st.session_state.get("show_historical_stocks", True):

        with st.container(border=False):
            subheader("Resultados históricos de los activos de la cartera", font_size="1.8rem")
            dict_stock_results = st.session_state.get("dict_stock_results")
            if dict_stock_results is None:
                st.info("Pulsa **Generar cartera** para calcular los resultados históricos de los activos.")
                return

            investor_results = dict_stock_results["investor"]
            plot_portfolio_values(investor_results, key="investor_portfolio", portfolio_type="stock")


def reset_portfolio_results():
    """
    Computes reset portfolio results.

    Parameters
    ----------


    Returns
    -------
    Any: reset portfolio results output.
    """
    keys_to_reset = [
        "initial_data",
        "initial_results",
        "dict_pf_returns",
        "dict_stock_results",
        "data_ready",
    ]
    for k in keys_to_reset:
        st.session_state[k] = None

    st.session_state["data_ready"] = False


def render_sidebar_display_options():
    """
    Renders sidebar display options.

    Parameters
    ----------


    Returns
    -------
    Any: render sidebar display options output.
    """
    # Robust init
    st.session_state.setdefault("data_ready", False)
    st.session_state.setdefault("viz_ready", False)
    st.session_state.setdefault("step2_enabled", False)
    st.session_state.setdefault("initial_data", None)
    st.session_state.setdefault("initial_results", None)

    st.sidebar.header("Navegación")

    if st.sidebar.button("Volver a cuestionario", use_container_width=True):
        reset_portfolio_results()
        st.session_state["route"] = "questionnaire"
        st.rerun()

    if st.sidebar.button("Volver a cartera inicial", width="stretch"):
        st.session_state["route"] = "portfolio"
        st.rerun()

    # Buttons are  enabled once visualizations were rendered at least once
    nav_enabled = bool(st.session_state.get("viz_ready", False))

    if st.sidebar.button(
            "Ver evolución cartera",
            width="stretch",
            type="primary",
            disabled=not nav_enabled,
    ):
        st.session_state["route"] = "results"
        st.rerun()

    analysis_enabled = (
            nav_enabled
            and st.session_state.get("initial_data") is not None
            and st.session_state.get("initial_results") is not None
    )

    if st.sidebar.button(
            "Ir a análisis de datos",
            width="stretch",
            disabled=not analysis_enabled,
    ):
        st.session_state["route"] = "analysis"
        st.rerun()

    if not nav_enabled:
        st.sidebar.caption(
            "Genera la cartera y espera a que se carguen las visualizaciones para desbloquear navegación.")
    elif not analysis_enabled:
        st.sidebar.caption("El análisis se desbloquea cuando existen datos y resultados de cartera.")

    st.sidebar.markdown("---")
    st.sidebar.header("Selecciona visualizaciones")

    st.session_state.setdefault("show_alloc_assets", True)
    st.session_state.setdefault("show_alloc_sectors", True)
    st.session_state.setdefault("show_results_table", True)
    st.session_state.setdefault("show_frontier", True)
    st.session_state.setdefault("show_historical", True)
    st.session_state.setdefault("show_historical_stocks", True)

    st.sidebar.checkbox("Composición por activo", key="show_alloc_assets")
    st.sidebar.checkbox("Composición por sector", key="show_alloc_sectors")
    st.sidebar.checkbox("Tabla de resultados", key="show_results_table")
    st.sidebar.checkbox("Frontera eficiente", key="show_frontier")
    st.sidebar.checkbox("Histórico (valor cartera)", key="show_historical")
    st.sidebar.checkbox("Histórico (valor activos)", key="show_historical_stocks")

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

    st.sidebar.markdown(
        f"""
        <div style="
            border: 1px solid rgba(0,0,0,0.10);
            border-radius: 12px;
            padding: 10px 12px;
            background: #D6FAFF;
            opacity: 0.8
        ">
            <div style="font-size: 1.2rem; color: #000078; font-weight:900; text-align: center">Perfil</div>
            <div style="font-size: 2.5rem; color: #000078; font-weight: 900; text-align: center">{perfil}</div>
            <div style="font-size: 1.2rem; color: #000078; font-weight: 900; text-align: center">{RISK_PROFILE_DICTIONARY.get(perfil, "Perfil no disponible")}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if sigma_min is not None and sigma_max is not None:
        st.sidebar.caption(f"Volatilidad recomendada: {sigma_min:.2f}–{sigma_max:.2f}")


def forecast_portfolio():
    """
    Computes forecast portfolio.

    Parameters
    ----------


    Returns
    -------
    Any: forecast portfolio output.
    """
    df_returns = st.session_state["initial_data"]["test_set"]
    weights = st.session_state["initial_results"][2]
    rf_annual = st.session_state["investor_constraints_draft"]["risk_free_rate"]
    amount = st.session_state["investor_constraints_draft"]["amount"]

    dict_pf_returns_forecast, dict_stock_results_forecast, dict_pf_results_forecasts = render_historical_portfolios_results(
        df_returns,
        amount,
        weights,
        periods_per_year=PERIODS_PER_YEAR,
        rf_annual=rf_annual
    )
    st.session_state["dict_pf_returns_forecast"] = dict_pf_returns_forecast
    st.session_state["dict_stock_results_forecast"] = dict_stock_results_forecast
    st.session_state["dict_pf_results_forecasts"] = dict_pf_results_forecasts

    st.session_state["data_ready"] = True
    st.session_state["step2_enabled"] = True
    resultados_forecast = st.session_state["dict_pf_returns_forecast"]
    end_date = resultados_forecast["investor"].index[-1]
    resultados = st.session_state["initial_data"]
    sectors = resultados["sectors"]
    st.session_state["forecast_sector_weights"] = get_sector_weights_at_date(dict_stock_results_forecast["investor"],
                                                                             sectors, end_date)
    st.session_state["forecast_asset_weights"] = check_portfolio_weights(dict_stock_results_forecast["investor"],
                                                                         end_date)


def render_constraints_portfolio():
    """
    Renders constraints portfolio.

    Parameters
    ----------


    Returns
    -------
    Any: render constraints portfolio output.
    """
    header("AJUSTES DE LA CARTERA INICIAL")

    # Session state initialization
    st.session_state.setdefault("data_ready", False)

    st.session_state.setdefault("viz_ready", False)
    st.session_state.setdefault("data_bundle", None)
    st.session_state.setdefault("investor_constraints_applied", None)
    st.session_state.setdefault("investor_constraints_draft", None)
    st.session_state.setdefault("risk_result", None)
    st.session_state.setdefault("dict_pf_returns", None)
    st.session_state.setdefault("dict_stock_results", None)
    st.session_state.setdefault("dict_pf_returns_forecast", None)
    st.session_state.setdefault("dict_stock_results_forecast", None)
    st.session_state.setdefault("dict_pf_results_forecasts", None)
    st.session_state.setdefault("initial_data", None)
    st.session_state.setdefault("initial_results", None)
    st.session_state.setdefault("step2_enabled", False)
    st.session_state.setdefault("custom_tickers", [])

    # Sidebar must be rendered every run (logo + nav + options)
    render_sidebar_display_options()

    # Inputs
    with st.container(border=False):
        render_investor_constraints()

    # Main action button
    c1, c2, c3 = st.columns(3)
    with c2:
        clicked = st.button("Generar cartera", width="stretch", type="primary")

    # Compute block (only when clicked)
    if clicked:
        # Disable navigation until charts are drawn
        st.session_state["viz_ready"] = False

        with st.spinner("Procesando datos..."):
            # Compute + store
            get_initial_data()
            get_initial_portfolio()

            df_returns_train = st.session_state["initial_data"]["train_set"]
            weights = st.session_state["initial_results"][2]
            rf_annual = st.session_state["investor_constraints_draft"]["risk_free_rate"]

            dict_pf_returns, dict_stock_results, _ = render_historical_portfolios_results(
                df_returns_train,
                1,
                weights,
                periods_per_year=PERIODS_PER_YEAR,
                rf_annual=rf_annual,
            )
            st.session_state["dict_pf_returns"] = dict_pf_returns
            st.session_state["dict_stock_results"] = dict_stock_results

            # Forecast/test calculations (writes dict_pf_returns_forecast, etc.)
            forecast_portfolio()

            st.session_state["data_ready"] = True
            st.session_state["step2_enabled"] = True

        # Re-run to render charts in a clean branch
        st.rerun()

    # If nothing computed yet
    if not st.session_state.get("data_ready", False):
        st.info("Configura parámetros y pulsa **Generar cartera**.")
        return

    # Render results + visualizations
    st.write("")
    header("RESULTADOS")

    with st.spinner("Generando visualizaciones..."):
        create_historical_portfolio_visualizations()
        create_historical_results_visualizations()

    # At this point, all visualizations on this page have been rendered once
    if not st.session_state.get("viz_ready", False):
        st.session_state["viz_ready"] = True
        # Re-run so sidebar buttons become enabled immediately
        st.rerun()