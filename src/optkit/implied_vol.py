from __future__ import annotations
import math
from .types import normalize_option_type
from .bs import bs_price, bs_vega
from .noarb import bounds_vanilla

def implied_vol(
    target_price: float,
    S: float,
    K: float,
    r: float,
    T: float,
    option_type: str = "call",
    *,
    initial: float = 0.2,
    tol: float = 1e-10,
    max_iter: int = 100,
    vol_lower: float = 1e-8,
    vol_upper: float = 5.0,
) -> float:
    """
    Implied volatility for European vanilla option under Black-Scholes (no dividends).
    Strategy:
    - validate target_price is within no-arb bounds (loose check)
    - Newton iterations with vega
    - if Newton fails, fallback to bisection on [vol_lower, vol_upper]
    """
    ot = normalize_option_type(option_type)
    if target_price < 0:
        raise ValueError("target_price must be >= 0")
    if T <= 0:
        raise ValueError("T must be > 0 for implied vol")

    lower, upper = bounds_vanilla(S, K, r, T, ot)
    # Allow tiny epsilon for numerical/market microstructure
    eps = 1e-8
    if target_price < lower - eps or target_price > upper + eps:
        raise ValueError(f"target_price outside no-arbitrage bounds: [{lower}, {upper}]")

    # Newton
    sigma = max(vol_lower, min(vol_upper, float(initial)))
    for _ in range(max_iter):
        price = bs_price(S, K, r, sigma, T, ot)
        diff = price - target_price
        if abs(diff) < tol:
            return sigma

        v = bs_vega(S, K, r, sigma, T, ot)
        if v <= 1e-12:
            break  # fallback

        sigma_new = sigma - diff / v
        if not (vol_lower < sigma_new < vol_upper):
            break
        sigma = sigma_new

    # Bisection (robust)
    lo, hi = vol_lower, vol_upper
    f_lo = bs_price(S, K, r, lo, T, ot) - target_price
    f_hi = bs_price(S, K, r, hi, T, ot) - target_price

    # Expand upper bound if needed (rare but possible)
    expand_count = 0
    while f_lo * f_hi > 0 and expand_count < 10:
        hi *= 2.0
        if hi > 20.0:
            break
        f_hi = bs_price(S, K, r, hi, T, ot) - target_price
        expand_count += 1

    if f_lo * f_hi > 0:
        raise RuntimeError("Failed to bracket implied vol; check inputs/price.")

    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = bs_price(S, K, r, mid, T, ot) - target_price
        if abs(f_mid) < tol or (hi - lo) < 1e-12:
            return mid
        if f_lo * f_mid <= 0:
            hi, f_hi = mid, f_mid
        else:
            lo, f_lo = mid, f_mid

    return 0.5 * (lo + hi)

