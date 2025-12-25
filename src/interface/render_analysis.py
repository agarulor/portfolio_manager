import streamlit as st
from types import MappingProxyType
PERIODS_PER_YEAR = 255
from interface.visualizations import show_portfolio, render_results_table, plot_portfolio_values, plot_daily_returns_scatter
from portfolio_management.investor_portfolios import get_cumulative_returns

RISK_PROFILE_DICTIONARY = MappingProxyType({
    1: "Perfil bajo de riesgo",
    2: "Perfil medio-bajo de riesgo",
    3: "Perfil medio de riesgo",
    4: "Perfil medio-alto de riesgo",
    5: "Perfil alto de riesgo",
    6: "Perfil agresivo de riesgo"
})

def render_sidebar_display():
    st.sidebar.header("Navegación")

    if st.sidebar.button("Volver a cuestionario", use_container_width=True):
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
            <div style="font-size: 1.2rem; color: #000078; font-weight: 900; text-align: center">{RISK_PROFILE_DICTIONARY[perfil]}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if sigma_min is not None and sigma_max is not None:
        st.sidebar.caption(f"Volatilidad recomendada: {sigma_min:.2f}–{sigma_max:.2f}")


def render_historic_perfomance():
    #if st.session_state.get("show_stock_results", True):
    historic_returns = st.session_state["initial_data"]["train_set"]
    historic_prices = st.session_state["initial_data"]["train_price"]
    recent_prices = st.session_state["initial_data"]["test_price"]
    recent_returns = st.session_state["initial_data"]["test_set"]
    cum_returns_historic = get_cumulative_returns(historic_returns)
    cum_returns_recent = get_cumulative_returns(recent_returns)

    c1, c2 = st.columns(2)
    with c1:
        plot_portfolio_values(cum_returns_historic , "historic_value", "stock")
    with c2:
        plot_portfolio_values(cum_returns_recent, "recent_value", "stock")

    plot_daily_returns_scatter(historic_returns, key="returns_color", data_type="stock")
    s1, s2 = st.columns(2)
    with s1:
        plot_portfolio_values(historic_prices, "historic_price", "stock")
    with s2:
        plot_portfolio_values(recent_prices, "recent_price", "stock")


def render_analysis():
    render_sidebar_display()
    render_historic_perfomance()