import sys
import os
FILENAME_PATH = "data/input/ibex_eurostoxx.csv"
TICKER_COL = "ticker_yahoo"
COMPANIES_COL = "name"
START_DATE = "2005-01-01"
END_DATE = "2025-09-30"
import pandas as pd
from interface.render_initial_portfolio import render_investor_constraints, render_constraints_portfolio

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import streamlit as st

import plotly.express as px

from types import MappingProxyType

import os
import random
import numpy as np
RISK_PROFILE_DICTIONARY = MappingProxyType({
    1: "Perfil bajo de riesgo",
    2: "Perfil medio-bajo de riesgo",
    3: "Perfil medio de riesgo",
    4: "Perfil medio-alto de riesgo",
    5: "Perfil alto de riesgo",
    6: "Perfil agresivo de riesgo"
})


def render_portfolio():


    render_constraints_portfolio()
    if not st.session_state.get("data_ready", False):
        return

