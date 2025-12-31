import datetime as dt
from types import MappingProxyType
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parent
ASSETS_PATH = BASE_DIR / "assets"
PROJECT_ROOT = BASE_DIR.parent.parent
FILENAME_PATH = "data/input/ibex_eurostoxx.csv"

# Investor related topics
RISK_PROFILE_DICTIONARY = MappingProxyType({
    1: "Perfil bajo de riesgo",
    2: "Perfil medio-bajo de riesgo",
    3: "Perfil medio de riesgo",
    4: "Perfil medio-alto de riesgo",
    5: "Perfil alto de riesgo",
    6: "Perfil agresivo de riesgo"
})

QUESTION_KEYS = [
    "knowledge",
    "risk_level",
    "downside_reaction",
    "liquidity_need",
    "annual_income",
    "net_worth",
    "investment_horizon",
    "financial_goal",
]

 # Colors and fonts
RISK_COLOR = MappingProxyType({1: "#2ecc71", 2: "#6bdc8b", 3: "#f1c40f", 4: "#f39c12", 5: "#e67e22", 6: "#e74c3c"})
FONT_SIZE = "0.9rem"
FONT_WEIGHT = "700"
FONT_COLOR = "#000078"

# Portfolio basic info
TICKER_COL = "ticker_yahoo"
START_DATE = "2005-01-01"
ENDING_DATE = "2025-09-30"
INITIAL_DATE = dt.date(2020, 10, 1)
END_DATE = dt.date(2025, 9, 30)
INITIAL_DATE_PORTFOLIO = dt.date(2024, 10, 1)
END_DATE_PORTFOLIO = dt.date(2025, 9, 30)
PERIODS_PER_YEAR = 255