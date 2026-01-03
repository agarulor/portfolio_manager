# To avoid changes to a dictionary
from types import MappingProxyType

KNOWLEDGE_WEIGHT = 0.3
RISK_LEVEL_WEIGHT = 0.4
DOWNSIDE_REACTION_WEIGHT = 0.3
VOLATILITY_MAPPING = MappingProxyType({
    1: (0.00, 0.065),
    2: (0.065, 0.11),
    3: (0.11, 0.155),
    4: (0.155, 0.19),
    5: (0.19, 0.23),
    6: (0.23, 0.33)
})


def risk_appetite(knowledge: int, risk_level: int, downside_reaction: int) -> int:
    """
    Computes risk appetite.

    Parameters
    ----------
    knowledge : int. knowledge.
    risk_level : int. risk level.
    downside_reaction : int. downside reaction.

    Returns
    -------
    int: risk appetite output.
    """
    # We first need to reescalate the downside_reaction
    if downside_reaction == 1:
        downside_reaction_r = 1
    elif downside_reaction == 2:
        downside_reaction_r = 3
    elif downside_reaction == 3:
        downside_reaction_r = 5
    elif downside_reaction == 4:
        downside_reaction_r = 6
    else:
        raise ValueError("Value must be between 1 and 4")

    # Weighted Average
    risk_appetite_score = round(
        KNOWLEDGE_WEIGHT * knowledge + RISK_LEVEL_WEIGHT * risk_level + DOWNSIDE_REACTION_WEIGHT * downside_reaction_r)

    # We ensure that we are within the values
    risk_appetite_score = max(1, min(6, risk_appetite_score))

    return risk_appetite_score


def risk_capacity(
        liquidity_need: int,
        annual_income: int,
        net_worth: int,
        investment_horizon: int,
        financial_goal_importance: int) -> float:
    """
    Computes risk capacity.

    Parameters
    ----------
    liquidity_need : int. liquidity need.
    annual_income : int. annual income.
    net_worth : int. net worth.
    investment_horizon : int. investment horizon.
    financial_goal_importance : int. financial goal importance.

    Returns
    -------
    float: risk capacity output.
    """
    # We first need to reescalate from 1 to 3 to 1 to 5
    if financial_goal_importance == 1:
        goal_score_rescaled = 1
    elif financial_goal_importance == 2:
        goal_score_rescaled = 3
    elif financial_goal_importance == 3:
        goal_score_rescaled = 5
    else:
        raise ValueError("financial_goal_importance must a value between 1 and 3")

    # We calculate the average score on a 1–5 scale
    average_score = (
                            liquidity_need +
                            annual_income +
                            net_worth +
                            investment_horizon +
                            goal_score_rescaled
                    ) / 5.0

    # As we need to provide the number in a scale from 1 to 6, we need to map
    # the average score from 1 to 5 to 1 to 6
    risk_capacity_score = round(1 + (average_score - 1) * 5 / 4)

    # We ensure that we are within the values
    risk_capacity_score = max(1, min(6, risk_capacity_score))

    return risk_capacity_score


def risk_tolerance(appetite: int, capacity: int) -> float:
    """
    Computes risk tolerance.

    Parameters
    ----------
    appetite : int. appetite.
    capacity : int. capacity.

    Returns
    -------
    float: risk tolerance output.
    """
    return min(appetite, capacity)


def investor_target_volatility(knowledge: int,
                               risk_level: int,
                               downside_reaction: int,
                               liquidity_need: int,
                               annual_income: int,
                               net_worth: int,
                               investment_horizon: int,
                               financial_goal_importance: int
                               ):
    """
    Computes investor target volatility.

    Parameters
    ----------
    knowledge : int. knowledge.
    risk_level : int. risk level.
    downside_reaction : int. downside reaction.
    liquidity_need : int. liquidity need.
    annual_income : int. annual income.
    net_worth : int. net worth.
    investment_horizon : int. investment horizon.
    financial_goal_importance : int. financial goal importance.

    Returns
    -------
    Any: investor target volatility output.
    """
    # calculate risk appetite
    RA = risk_appetite(knowledge, risk_level, downside_reaction)
    # calculate risk capacity
    RC = risk_capacity(liquidity_need, annual_income, net_worth, investment_horizon, financial_goal_importance)
    # calculate total risk_tolerance
    RT = risk_tolerance(RA, RC)

    # return volatility level
    return VOLATILITY_MAPPING[RT], RA, RC, RT

