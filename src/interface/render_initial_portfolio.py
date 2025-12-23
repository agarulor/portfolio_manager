import streamlit as st
from interface.main_interface import subheader, header
from data_management.get_data import get_stock_prices, read_price_file
from data_management.dataset_preparation import split_data_markowtiz
from data_management.clean_data import clean_and_align_data
from portfolio_tools.return_metrics import calculate_daily_returns
from interface.landing_page import add_separation
from interface.visualizations import show_portfolio, render_results_table, show_markowitz_results, plot_portfolio_values
from portfolio_management.investor_portfolios import get_sector_exposure_table, create_output_table_portfolios, render_historical_portfolios_results
import datetime as dt
PERIODS_PER_YEAR = 255

import plotly.graph_objects as go
FILENAME_PATH = "data/input/ibex_eurostoxx.csv"
TICKER_COL = "ticker_yahoo"
COMPANIES_COL = "name"
START_DATE = "2005-01-01"
ENDING_DATE = "2025-09-30"
FONT_SIZE = "0.9rem"
FONT_WEIGHT = "700"
FONT_COLOR = "#000078"
INITIAL_DATE = dt.date(2020, 10, 1)
END_DATE = dt.date(2025, 9, 30)




def get_clean_initial_data(filename_path: str = FILENAME_PATH,
                           ticker_col: str = TICKER_COL,
                           adjusted: bool = False,
                           companies_col: str = COMPANIES_COL,
                           start_date: str = START_DATE,
                           end_date: str = ENDING_DATE):

    """
    price_data, sectors = get_stock_prices(file_path=filename_path,
                                           ticker_col=ticker_col,
                                           adjusted=adjusted,
                                           companies_col= companies_col,
                                           start_date=start_date,
                                           end_date=end_date,
                                           )
    """
    price_data = read_price_file("data/processed/prices_20251218-234715.csv")
    sectors = read_price_file("data/processed/sectores_20251218-234715.csv")
    prices, report, summary = clean_and_align_data(price_data, beginning_data=True)
    #save_preprocessed_data(prices, file_prefix="prices")
    #save_preprocessed_data(sectors, file_prefix="sectores")



    daily_returns = calculate_daily_returns(prices, method="simple")

    train_set, test_set = split_data_markowtiz(returns=daily_returns, test_date_start="2024-10-01", test_date_end="2025-9-30")

    # We return relevant data
    return {
        "prices": prices,
        "daily_returns": daily_returns,
        "train_set": train_set,
        "test_set": test_set,
        "sectors": sectors
    }


def render_slider(text:str,
                  text_slider: str,
                  key: str,
                  min_value: float,
                  max_value: float,
                  value = 25.0,
                  font_color: str = FONT_COLOR,
                  font_size: str = FONT_SIZE,
                  font_weight: str = FONT_WEIGHT):
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
        key= key
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


def render_number_input(text:str,
                        min_value:float,
                        init_value:float,
                        step:float,
                        key:str,
                        font_color: str = FONT_COLOR,
                        font_size: str = FONT_SIZE,
                        font_weight: str = FONT_WEIGHT,
                        unit: str = "EUR"):
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
        format="%.2f",
        label_visibility="collapsed",
        key=key
    )

    formatted_amount = (
        f"{amount:,.2f}"
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


def render_date(text:str,
                reference_date: dt.date,
                key:str,
                font_color: str = FONT_COLOR,
                font_size: str = FONT_SIZE,
                font_weight: str = FONT_WEIGHT):
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
        key= key
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


def render_investor_constraints():

    subheader("Defina el grado de diversificación y el importe inicial para la propuesta de cartera",
              margin_bottom="1.0rem")

    add_separation()
    with st.container(border=True):
        subheader("Pesos de la cartera", font_size="2.0rem")
        c1, c2, c3 = st.columns(3, gap = "large")

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
                                          value=15.0,
                                          font_color="#FF0000")

        with c3:
            min_stock_pct = render_slider(text="Selecciona peso mínimo por acción",
                                          text_slider="% mínimo asignado a una acción",
                                          key="min_stock_pct",
                                          min_value = 0.0,
                                          max_value = 100.0,
                                          value=2.5)

    with st.container(border=True):
        subheader("Selección de fechas", font_size="2.0rem")
        s1, s2= st.columns(2)

        with s1:
            st1, st2 = st.columns(2)
            with st2:
                data_start_date = render_date("Fecha de inicio de datos", key="date" , reference_date=INITIAL_DATE)
        with s2:
            stt1, stt2 = st.columns(2)
            with stt1:
                data_end_date = render_date("Fecha de fin de datos", font_color="#FF0000", key="date2", reference_date=END_DATE)

    with st.container(border=True):
        subheader("Inversión inicial y tipo de interés libre de riesgo", font_size="2.0rem")
        t1, t2 = st.columns(2)
        with t1:
            tt1, tt2 = st.columns(2)
            with tt1, tt2:
                amount = render_number_input("Inversión inicial",
                                             min_value=0.0,
                                             init_value=10000.0,
                                             step=100.0,
                                             key="cash_contribution")

        with t2:
            tt3, tt4 = st.columns(2)
            with tt3:
                risk_free_rate = render_number_input("Tipo de interés libre de riesgo",
                                                     min_value=0.0,
                                                     init_value=3.0,
                                                     step=0.01,
                                                     font_color="#FF0000",
                                                     key="risk_free_rate",
                                                     unit="%")

    # Save session state
    st.session_state["investor_constraints_draft"] = {
        "max_sector_pct": max_sector_pct,
        "max_stock_pct": max_stock_pct,
        "min_stock_pct": min_stock_pct,
        "data_start_date": data_start_date,
        "data_end_date": data_end_date,
        "amount": amount,
        "risk_free_rate": risk_free_rate/100,
    }

def get_initial_data():

    constraints = st.session_state["investor_constraints_draft"]
    data_start_date = constraints["data_start_date"]
    data_end_date = constraints["data_end_date"]
    st.session_state["initial_data"] = get_clean_initial_data(
        start_date=data_start_date.isoformat(),
        end_date=data_end_date.isoformat(),
    )
    st.session_state.data_ready = True

def get_initial_portfolio():
    # We get the relevant information constraints
    constraints = st.session_state["investor_constraints_draft"]
    max_stock_pct = constraints["max_stock_pct"] / 100
    min_stock_pct = constraints["min_stock_pct"] / 100
    max_sector_pct = constraints["max_sector_pct"] / 100
    risk_free_rate = constraints["risk_free_rate"]
    investment_amount = constraints["amount"]

    # Investor profile
    profile = st.session_state["risk_result"]
    volatility = (profile["sigma_min"] + profile["sigma_max"]) / 2

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

    if not st.session_state.get("data_ready", False):
        return
    df_weights = st.session_state["initial_results"][1]["investor"]
    sectors = st.session_state["initial_data"]["sectors"]
    sectores = get_sector_exposure_table(df_weights, sectors)
    df_results = st.session_state["initial_results"][0]
    df_returns = st.session_state["initial_data"]["train_set"]

    with st.container(border=True):
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
    with st.container(border=True):
        c1, c2 = st.columns(2)
        if st.session_state.get("show_results_table", True):
            with c1:
                subheader("Resultados de la cartera", font_size="2.0rem")
                render_results_table(df_results)

    # We now render the efficient frontier and the comparable portfolios
        if st.session_state.get("show_frontier", True):
            with c2:
                subheader("Frontera eficiente y portfolios", font_size="2.0rem")
                show_markowitz_results(n_returns=100, returns= df_returns, df_results=df_results, periods_per_year=PERIODS_PER_YEAR)


def create_historical_results_visualizations():
    if st.session_state.get("show_historical", True):
        with st.container(border=True):
            subheader("Resultados históricos de la cartera antes de inversión", font_size="2.0rem")
            dict_pf_returns = st.session_state.get("dict_pf_returns")
            if dict_pf_returns is None:
                st.info("Pulsa **Generar cartera** para calcular los resultados históricos.")
                return

            plot_portfolio_values(dict_pf_returns, key="historic_portfolio")
    if st.session_state.get("show_historical_stocks", True):

        with st.container(border=True):
            subheader("Resultados históricos de las acciones de la cartera", font_size="2.0rem")
            dict_stock_results = st.session_state.get("dict_stock_results")
            if dict_stock_results is None:
                st.info("Pulsa **Generar cartera** para calcular los resultados históricos de las acciones.")
                return

            investor_results = dict_stock_results["investor"]
            plot_portfolio_values(investor_results, key="investor_portfolio", portfolio_type="stock")

def reset_portfolio_results():
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

    st.sidebar.header("Navegación")
    if st.sidebar.button("Volver a cuestionario", use_container_width=True):
        reset_portfolio_results()
        st.session_state["route"] = "questionnaire"
        st.rerun()

    if st.session_state["step2_enabled"]:
        if st.sidebar.button("Ver evolución cartera", width="stretch"):
            st.session_state["route"] = "results"
            st.rerun()

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
    st.sidebar.checkbox("Histórico (valor acciones)", key="show_historical_stocks")




def render_constraints_portfolio():
    header("AJUSTES DE LA CARTERA INICIAL")
    st.session_state.setdefault("data_ready", False)
    st.session_state.setdefault("data_bundle", None)
    st.session_state.setdefault("investor_constraints_applied", None)
    st.session_state.setdefault("investor_constraints_draft", None)
    st.session_state.setdefault("risk_result", None)
    st.session_state.setdefault("dict_pf_returns", None)
    st.session_state.setdefault("dict_stock_results", None)
    st.session_state.setdefault("initial_results", None)
    st.session_state.setdefault("step2_enabled", False)

    render_sidebar_display_options()

    with st.container(border=True):
        render_investor_constraints()
        if "data_ready" not in st.session_state:
            st.session_state.data_ready = False
        if "data_bundle" not in st.session_state:
            st.session_state.data_bundle = None

    c1, c2, c3 = st.columns(3)
    with c2:
        clicked = st.button("Generar cartera", width="stretch", type="primary")

    if clicked:
        draft = st.session_state["investor_constraints_draft"]

        with (st.spinner("Procesando datos...")):
            # We store the draft values at that point in applied and we avoid re-runs
            get_initial_data()
            get_initial_portfolio()

            df_returns = st.session_state["initial_data"]["train_set"]
            weights = st.session_state["initial_results"][2]
            rf_annual = st.session_state["investor_constraints_draft"]["risk_free_rate"]

            dict_pf_returns, dict_stock_results, dict_pf_results = render_historical_portfolios_results(
                df_returns,
                1,
                weights,
                periods_per_year=PERIODS_PER_YEAR,
                rf_annual=rf_annual
            )
            st.session_state["dict_pf_returns"] = dict_pf_returns
            st.session_state["dict_stock_results"] = dict_stock_results

        st.session_state["data_ready"] = True
        st.session_state["step2_enabled"] = True


    if st.session_state.get("data_ready"):
        header("RESULTADOS")
        create_historical_portfolio_visualizations()
        create_historical_results_visualizations()

    else:
        st.info("Configura parámetros y pulsa **Generar cartera**.")

