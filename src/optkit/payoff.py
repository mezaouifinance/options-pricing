from __future__ import annotations
from .types import normalize_option_type

def payoff_vanilla(ST: float, K: float, option_type: str) -> float:
    ot = normalize_option_type(option_type)
    if ot == "call":
        return max(0.0, ST - K)
    return max(0.0, K - ST)
