import streamlit as st
from interface.main_interface import subheader, header
from interface.render_initial_portfolio import reset_portfolio_results
from interface.visualizations import show_portfolio, render_results_table, plot_portfolio_values, show_markowitz_results
from types import MappingProxyType

PERIODS_PER_YEAR = 255

RISK_PROFILE_DICTIONARY = MappingProxyType({
    1: "Perfil bajo de riesgo",
    2: "Perfil medio-bajo de riesgo",
    3: "Perfil medio de riesgo",
    4: "Perfil medio-alto de riesgo",
    5: "Perfil alto de riesgo",
    6: "Perfil agresivo de riesgo"
})


def render_sidebar_display_results():
    """
    Renders sidebar display results.

    Parameters
    ----------


    Returns
    -------
    Any: render sidebar display results output.
    """
    st.sidebar.header("Navegación")

    if st.sidebar.button("Volver a cuestionario", width="stretch"):
        reset_portfolio_results()
        st.session_state["route"] = "questionnaire"
        st.rerun()

    if st.sidebar.button("Volver a cartera inicial", width="stretch"):
        st.session_state["route"] = "portfolio"
        st.rerun()


    if st.sidebar.button("Ver evolución cartera", width="stretch", type="primary"):
        st.session_state["route"] = "results"
        st.rerun()

    if st.sidebar.button("Ir a análisis de datos", width="stretch"):
        st.session_state["route"] = "analysis"
        st.rerun()

    st.sidebar.header("Selecciona visualizaciones")

    st.session_state.setdefault("show_alloc_assets_forecast", True)
    st.session_state.setdefault("show_alloc_sectors_forecast", True)
    st.session_state.setdefault("show_results_table_forecast", True)
    st.session_state.setdefault("show_riesgo_rentabilidad", True)
    st.session_state.setdefault("show_portfolio_results", True)
    st.session_state.setdefault("show_stock_results", True)

    st.sidebar.checkbox("Composición por activo", key="show_alloc_assets_forecast")
    st.sidebar.checkbox("Composición por sector", key="show_alloc_sectors_forecast")
    st.sidebar.checkbox("Tabla de resultados", key="show_results_table_forecast")
    st.sidebar.checkbox("Riesgo / Rentabilidad", key="show_riesgo_rentabilidad")
    st.sidebar.checkbox("Histórico (valor cartera)", key="show_portfolio_results")
    st.sidebar.checkbox("Histórico (valor acciones)", key="show_stock_results")

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


def show_portfolio_returns():
    """
    Shows portfolio returns.

    Parameters
    ----------


    Returns
    -------
    Any: show portfolio returns output.
    """
    initial_amount = st.session_state["investor_constraints_draft"]["amount"]
    resultados = st.session_state["dict_pf_returns_forecast"]
    final_amount = resultados["investor"].iloc[-1]
    profit_abs = final_amount - initial_amount
    profit_pct = (final_amount / initial_amount - 1.0) * 100 if initial_amount else 0.0
    end_date = resultados["investor"].index[-1]
    start_date = resultados["investor"].index[1]

    with st.container(border=False):
        subheader("Rendimiento de la cartera", font_size="1.8rem", margin_bottom="3.0rem")
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            st.metric(
                "Importe inicial",
                f"{initial_amount:,.2f} €".replace(",", "X").replace(".", ",").replace("X", ".")
            )
        with c2:
            st.metric(
                "Importe final",
                f"{final_amount:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."),
                delta=f"{profit_abs:,.2f} €".replace(",", "X").replace(".", ",").replace("X", "."),
            )

        with c3:
            st.metric(
                "Fecha de inversión inicial",
                start_date.strftime("%d/%m/%Y") if hasattr(start_date, "strftime") else str(start_date)
            )


        with c4:
            st.metric(
                "Fecha de cálculo de resultados",
                end_date.strftime("%d/%m/%Y") if hasattr(end_date, "strftime") else str(end_date)
            )

        with c5:
            st.metric(
                "Ganancia total",
                f"{profit_pct:.2f}%"
            )
        st.write("")
        st.write("")


def create_portfolio_visualizations():
    """
    Creates portfolio visualizations.

    Parameters
    ----------


    Returns
    -------
    Any: create portfolio visualizations output.
    """
    if not st.session_state.get("data_ready", False):
        return

    df_results = st.session_state["dict_pf_results_forecasts"]
    if df_results.index.nlevels > 1:
        df_results.index = df_results.index.droplevel(1)
    df_weights = st.session_state.get("forecast_asset_weights")
    df_sectors = st.session_state.get("forecast_sector_weights")

    with st.container(border=False):
        col1, col2 = st.columns(2)
        if st.session_state.get("show_alloc_assets_forecast", True):
            with col1:
                show_portfolio(
                    df_weights=df_weights,
                    title="Composición por activo",
                    label_name="Activo",
                    weight_col="Pesos",
                    weights_in_percent=False
                )

        if st.session_state.get("show_alloc_sectors_forecast", True):
            with col2:
                show_portfolio(
                    df_weights=df_sectors,
                    title="Composición por sector",
                    label_name="Sector",
                    weight_col="Pesos",
                    weights_in_percent=True)
        st.write("")

    with st.container(border=False):
        u1, u2 = st.columns(2)
        # We now render the main table of results and comparable portfolios

        with u1:
            if st.session_state.get("show_results_table_forecast", True):
                subheader("Resultados de la cartera", font_size="1.8rem", margin_bottom="3.0rem")
                render_results_table(df_results)
                st.write("")

        # We now render the efficient frontier and the comparable portfolios

        with u2:
            if st.session_state.get("show_riesgo_rentabilidad", True):
                subheader("Riesgo / rentabilidad", font_size="1.8rem")
                show_markowitz_results(df_results=df_results,
                                       periods_per_year=PERIODS_PER_YEAR, no_ef=True)


def create_results_visualizations():
    """
    Creates results visualizations.

    Parameters
    ----------


    Returns
    -------
    Any: create results visualizations output.
    """
    if st.session_state.get("show_portfolio_results", True):

        with st.container(border=False):
            subheader("Resultados de la cartera de inversión", font_size="1.8rem", margin_bottom="3.0rem")
            dict_pf_returns_forecast = st.session_state.get("dict_pf_returns_forecast")
            if dict_pf_returns_forecast is None:
                st.info("Pulsa **Generar cartera** para calcular los resultados históricos.")
                return

            plot_portfolio_values(dict_pf_returns_forecast, key="forecast_portfolio")

    if st.session_state.get("show_stock_results", True):

        with st.container(border=False):
            subheader("Resultados de las acciones de la cartera", font_size="1.8rem", margin_bottom="3.0rem")
            dict_stock_results_forecast = st.session_state.get("dict_stock_results_forecast")
            if dict_stock_results_forecast is None:
                st.info("Pulsa **Generar cartera** para calcular los resultados de las acciones.")
                return

            investor_results = dict_stock_results_forecast["investor"]
            plot_portfolio_values(investor_results, key="investor_portfolio_forecast", portfolio_type="stock")


def render_results():
    """
    Renders results.

    Parameters
    ----------


    Returns
    -------
    Any: render results output.
    """
    render_sidebar_display_results()

    if st.session_state.get("data_ready"):
        header("EVOLUCIÓN CARTERA")
        st.write("")
        show_portfolio_returns()
        create_portfolio_visualizations()
        create_results_visualizations()