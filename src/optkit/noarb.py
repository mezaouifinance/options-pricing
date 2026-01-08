from __future__ import annotations
import math
from .types import normalize_option_type
from .utils import discount_factor

def bounds_vanilla(S: float, K: float, r: float, T: float, option_type: str) -> tuple[float, float]:
    """
    Classic no-arbitrage bounds for European options (no dividends).
    Call: max(0, S - K e^{-rT}) <= C <= S
    Put : max(0, K e^{-rT} - S) <= P <= K e^{-rT}
    """
    ot = normalize_option_type(option_type)
    df = discount_factor(r, T)
    if ot == "call":
        lower = max(0.0, S - K * df)
        upper = S
    else:
        lower = max(0.0, K * df - S)
        upper = K * df
    return lower, upper

def put_call_parity_residual(C: float, P: float, S: float, K: float, r: float, T: float) -> float:
    # C - P - (S - K e^{-rT}) should be ~ 0
    return (C - P) - (S - K * math.exp(-r * T))

