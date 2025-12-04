from typing import Tuple
KNOWLEDGE_WEIGHT = 0.3
RISK_LEVEL_WEIGHT = 0.4
DOWNSIDE_REACTION_WEIGHT = 0.3

def risk_appetite(knowledge:int, risk_level: int, downside_reaction: int) -> int:
    """
    This function calculates the risk appetite of an investor
    The value will be between 1 and 6 depending on the risk appetite,
    being 1 little risk appetite or 6 very high risk appetite

    Parameters
    ----------
    knowledge : int
        Integer value representing the knowledge of the investor
        (value must be between 1 and 6). Weight: 0.3
    risk_level : int
        Integer value representing the risk level of the investor
        (value must be between 1 and 6). Weight: 0.4
    downside_reaction : int
        Integer value representing the downside reaction of the investor when
        market falls
        (value must be between 1 and 4) and it will be normalized to values 1 to 6
        (1: sell all, 2: sell part of the asset/s, 3: keep and 4: buy even more).
        Weight: 0.3

    Returns
    -------
    risk_appetite : int
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
    RA = KNOWLEDGE_WEIGHT * knowledge + RISK_LEVEL_WEIGHT * risk_level + DOWNSIDE_REACTION_WEIGHT * downside_reaction

    # We ensure that we are within the values
    RA = max(1, min(6, RA))

    return RA



def risk_capacity():

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
    return min(risk_appetite, risk_capacity)




