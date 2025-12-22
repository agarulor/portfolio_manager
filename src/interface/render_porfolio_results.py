import streamlit as st
from interface.main_interface import subheader, header
from portfolio_management.investor_portfolios import get_sector_exposure_table, create_output_table_portfolios, render_historical_portfolios_results
from interface.render_initial_portfolio import  reset_portfolio_results
from interface.visualizations import show_portfolio, render_results_table, show_markowitz_results, plot_portfolio_values
PERIODS_PER_YEAR = 255

def render_sidebar_display_results():

    st.sidebar.header("Navegación")
    if st.sidebar.button("Volver a cuestionario", width="stretch"):
        reset_portfolio_results()
        st.session_state["route"] = "questionnaire"
        st.rerun()


    if st.sidebar.button("Volver a cartera inicial", width="stretch"):
        st.session_state["route"] = "portfolio"
        st.rerun()


def create_historical_portfolio_visualizations():

    if not st.session_state.get("data_ready", False):
        return


    # We now render the main table of results and comparable portfolios
    with st.container(border=True):
        if st.session_state.get("show_results_table", True):
            subheader("Resultados de la cartera", font_size="2.0rem")
            render_results_table(df_results)


def create_results_visualizations():
    #if not st.session_state.get("show_historical", True):
     #   return
    with st.container(border=True):
        subheader("Resultados de la cartera de inversión", font_size="2.0rem")
        dict_pf_results = st.session_state.get("dict_pf_results_forecast")
        if dict_pf_results is None:
            st.info("Pulsa **Generar cartera** para calcular los resultados históricos.")
            return

        plot_portfolio_values(dict_pf_results)


def render_results():
    render_sidebar_display_results()

    df_returns = st.session_state["initial_data"]["test_set"]
    weights = st.session_state["initial_results"][2]
    rf_annual = st.session_state["investor_constraints_draft"]["risk_free_rate"]
    amount = st.session_state["investor_constraints_draft"]["amount"]

    st.session_state["forecast_results"] = create_output_table_portfolios(test_set,
                                                                         min_w=min_stock_pct,
                                                                         max_w=max_stock_pct,
                                                                         rf_annual=risk_free_rate,
                                                                         periods_per_year=PERIODS_PER_YEAR,
                                                                         custom_target_volatility=volatility,
                                                                         sectors_df=sectors,
                                                                         sector_max_weight=max_sector_pct,
                                                                         risk_free_ticker="RISK_FREE")


    dict_pf_results, dict_stock_results = render_historical_portfolios_results(
        df_returns,
        amount,
        weights,
        periods_per_year=PERIODS_PER_YEAR,
        rf_annual=rf_annual
    )
    st.session_state["dict_pf_results_forecast"] = dict_pf_results
    st.session_state["dict_stock_results_forecast"] = dict_stock_results

    st.session_state["data_ready"] = True
    st.session_state["step2_enabled"] = True


    if st.session_state.get("data_ready"):
        header("EVOLUCIÓN CARTERA")
        create_results_visualizations()