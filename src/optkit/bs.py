from __future__ import annotations
import math
from .types import normalize_option_type
from .utils import norm_cdf, norm_pdf, assert_positive, discount_factor
from .payoff import payoff_vanilla

def _d1_d2(S: float, K: float, r: float, sigma: float, T: float) -> tuple[float, float]:
    assert_positive("S", S)
    assert_positive("K", K)
    assert_positive("sigma", sigma)
    assert_positive("T", T)
    vol_sqrt = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma * sigma) * T) / vol_sqrt
    d2 = d1 - vol_sqrt
    return d1, d2

def bs_price(S: float, K: float, r: float, sigma: float, T: float, option_type: str = "call") -> float:
    ot = normalize_option_type(option_type)
    assert_positive("S", S)
    assert_positive("K", K)
    assert_positive("T", T, allow_zero=True)
    assert_positive("sigma", sigma, allow_zero=True)

    # Handle expiry or zero vol in a controlled way:
    if T == 0 or sigma == 0:
        # Under risk-neutral, ST is deterministic if sigma=0 (S*e^{rT}) in BS model.
        # Discounted payoff = e^{-rT} * payoff(S e^{rT}, K)
        ST = S * math.exp(r * T)
        return math.exp(-r * T) * payoff_vanilla(ST, K, ot)

    d1, d2 = _d1_d2(S, K, r, sigma, T)
    df = discount_factor(r, T)

    if ot == "call":
        return S * norm_cdf(d1) - K * df * norm_cdf(d2)
    else:
        return K * df * norm_cdf(-d2) - S * norm_cdf(-d1)

def bs_delta(S: float, K: float, r: float, sigma: float, T: float, option_type: str = "call") -> float:
    ot = normalize_option_type(option_type)
    if T == 0 or sigma == 0:
        # Delta is not well-defined at the kink; give a stable convention:
        # if intrinsic in-the-money -> 1 for call, -1 for put; else 0
        intrinsic = payoff_vanilla(S, K, ot)  # using spot at expiry convention
        if ot == "call":
            return 1.0 if intrinsic > 0 else 0.0
        return -1.0 if intrinsic > 0 else 0.0

    d1, _ = _d1_d2(S, K, r, sigma, T)
    if ot == "call":
        return norm_cdf(d1)
    return norm_cdf(d1) - 1.0

def bs_gamma(S: float, K: float, r: float, sigma: float, T: float, option_type: str = "call") -> float:
    # Same gamma for call/put in BS
    if T == 0 or sigma == 0:
        return 0.0
    d1, _ = _d1_d2(S, K, r, sigma, T)
    return norm_pdf(d1) / (S * sigma * math.sqrt(T))

def bs_vega(S: float, K: float, r: float, sigma: float, T: float, option_type: str = "call") -> float:
    # Same vega for call/put in BS
    if T == 0 or sigma == 0:
        return 0.0
    d1, _ = _d1_d2(S, K, r, sigma, T)
    return S * norm_pdf(d1) * math.sqrt(T)
