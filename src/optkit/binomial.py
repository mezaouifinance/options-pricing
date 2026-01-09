from __future__ import annotations
import math
import numpy as np
from .types import normalize_option_type
from .payoff import payoff_vanilla
from .utils import assert_positive

def crr_price(
    S: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    N: int,
    option_type: str = "call",
    american: bool = False,
) -> float:
    """
    CRR binomial tree pricing (European by default; American if american=True).
    No dividends.

    Parameters
    - N: number of steps (>=1)
    """
    ot = normalize_option_type(option_type)
    assert_positive("S", S)
    assert_positive("K", K)
    assert_positive("sigma", sigma, allow_zero=True)
    assert_positive("T", T, allow_zero=True)
    if N < 1:
        raise ValueError("N must be >= 1")

    if T == 0:
        return payoff_vanilla(S, K, ot)

    dt = T / N
    # If sigma == 0, underlying deterministic: S_t = S exp(rT).
    if sigma == 0:
        ST = S * math.exp(r * T)
        return math.exp(-r * T) * payoff_vanilla(ST, K, ot)

    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp(r * dt) - d) / (u - d)
    # Numerical sanity
    if not (0.0 <= p <= 1.0):
        # Can happen for extreme dt/sigma; users should increase N.
        raise ValueError(f"Risk-neutral probability out of bounds: p={p}. Consider increasing N.")

    # Terminal stock prices: S * u^j d^(N-j), j=0..N
    j = np.arange(N + 1)
    ST = S * (u ** j) * (d ** (N - j))
    V = np.maximum(0.0, ST - K) if ot == "call" else np.maximum(0.0, K - ST)

    if not american:
        # Backward induction (European)
        for _ in range(N):
            V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        return float(V[0])

    # American: allow early exercise
    # We need stock prices at each node while stepping backward.
    # At time step n (from N down to 0), there are n+1 nodes.
    for n in range(N, 0, -1):
        V = disc * (p * V[1:] + (1.0 - p) * V[:-1])
        j = np.arange(n)
        S_n = S * (u ** j) * (d ** (n - 1 - j))
        exercise = (S_n - K) if ot == "call" else (K - S_n)
        V = np.maximum(V, np.maximum(0.0, exercise))
    return float(V[0])

