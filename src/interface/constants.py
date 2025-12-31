from types import MappingProxyType
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

RISK_COLOR = MappingProxyType({1: "#2ecc71", 2: "#6bdc8b", 3: "#f1c40f", 4: "#f39c12", 5: "#e67e22", 6: "#e74c3c"})