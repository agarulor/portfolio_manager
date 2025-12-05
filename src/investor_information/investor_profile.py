from typing import Tuple
from types import MappingProxyType # to avoid changes to a dictionary
KNOWLEDGE_WEIGHT = 0.3
RISK_LEVEL_WEIGHT = 0.4
DOWNSIDE_REACTION_WEIGHT = 0.3
VOLATILITY_MAPPING = MappingProxyType( {
    1: (0.00, 0.10),
    2: (0.10, 0.15),
    3: (0.15, 0.20),
    4: (0.20, 0.30),
    5: (0.30, 0.45),
    6: (0.45, 0.70)
})



def risk_appetite(knowledge:int, risk_level: int, downside_reaction: int) -> int:
    """
    This function calculates the risk appetite of an investor
    The value will be between 1 and 6 depending on the risk appetite,
    being 1 little risk appetite or 6 very high risk appetite

    Parameters
    ----------
    knowledge : int (1-6)
        Integer value representing the knowledge of the investor
        Weight: 0.3
    risk_level : int (1-6)
        Integer value representing the risk level of the investor
        Weight: 0.4
    downside_reaction : int (1-4)
        Integer value representing the downside reaction of the investor when
        market falls. It will be normalized to values 1 to 6
        (1: sell all, 2: sell part of the asset/s, 3: keep and 4: buy even more).
        Weight: 0.3

    Returns
    -------
    risk_appetite_score : int
        value will be between 1 and 6 depending on the risk appetite. Each factor will have
        a different weight
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
        KNOWLEDGE_WEIGHT * knowledge + RISK_LEVEL_WEIGHT * risk_level + DOWNSIDE_REACTION_WEIGHT * downside_reaction)

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
    Calculates investor's actual capacity to assume risk (risk capacity),
    The value will be between 1 and 6 depending on the risk capacity,
    being 1 little risk appetite or 6 very high risk capacity. Values need to be
    reescalated to a 1-6 scale

    Parameters
    ----------
    liquidity_need : int (1–5)
        Level of liquidity needed
        1 = Immediate
        2 = High liquidity needed
        3 = Medium liquidity needed
        4 = Low liquidity needed
        5 = Very low liquidity needed
        Higher values imply greater ability to bear investment risk.

    annual_income : int (1–5)
        Annual income level.
        1 = Very low income
        2 = Low income
        3 = Medium income
        4 = High income
        5 = Very high income
        Higher values imply greater risk-bearing ability.

    net_worth : int (1–5)
        Investor's accumulated savings and financial wealth.
        1 = Very low net worth
        2 = Low net worth
        3 = Medium net worth
        4 = High net worth
        5 = Very high net worth
        Higher values imply greater financial resilience and risk capacity.

    investment_horizon : int (1–5)
        Investment time horizon.
        1 = Very short term
        2 = Short term
        3 = Medium term
        4 = Long term
        5 = Very long term
        Longer horizons implies more capacity to undertake risk

    financial_goal_importance : int (1–3)
        Criticality of the main financial objective.
        1 = Critical objective (failure cannot be afforded, e.g., home purchase)
        2 = Moderately important objective
        3 = Flexible or long-term wealth accumulation objective
        Less critical objectives means greater risk capacity.

    Returns
    -------
    risk_capacity_score : int (1–6)
        Discrete score representing the investor's real capacity to assume investment risk:
        1 = Very low risk capacity
        2 = Low risk capacity
        3 = Medium risk capacity
        4 = Medium-high risk capacity
        5 = High risk capacity
        6 = Very high risk capacity
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


def risk_tolerance(appetite: int, capacity: int)-> float:
    """"
    This function calculates the risk tolerance of an investor
    Following MiFid, the final value will be the min of
    risk appetite and risk tolerance.

    Parameters
    ----------
    appetite : int
        Value of risk appetite
    capacity : int
        value of risk capacity

    Returns
    -------
    risk_tolerance : float
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
                               ) :
    """
    Computes the investor's target volatility range from risk appetite and risk capacity

    Parameters
    ----------
    knowledge : int (1–6)
        Financial knowledge and experience level.

    risk_level : int (1–6)
        Declared willingness to assume investment risk.

    downside_reaction : int (1–4)
        Behavioral reaction to significant market downturns.

    liquidity_need : int (1–5)
        Required level of short-term liquidity.

    annual_income : int (1–5)
        Investor's annual income level.

    net_worth : int (1–5)
        Investor's accumulated financial wealth.

    investment_horizon : int (1–5)
        Planned investment time horizon.

    financial_goal_importance : int (1–3)
        Criticality of the main financial objective.

    Returns
    -------
    target_volatility_range : Tuple[float, float]
        A tuple (min_volatility, max_volatility) representing the recommended target annualized volatility interval
        for the investor’s portfolio.
    """
    # calculate risk appetite
    RA =risk_appetite(knowledge, risk_level, downside_reaction)
    # calculate risk capacity
    RC = risk_capacity(liquidity_need, annual_income, net_worth, investment_horizon, financial_goal_importance)
    # calculate total risk_tolerance
    RT = risk_tolerance(RA, RC)

    # return volatility level
    return VOLATILITY_MAPPING[RT], RA, RC, RT



