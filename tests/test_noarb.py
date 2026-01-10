import math
from optkit.noarb import bounds_vanilla, put_call_parity_residual

def test_bounds_basic_call_put():
    S, K, r, T = 100, 110, 0.03, 1.0
    cL, cU = bounds_vanilla(S, K, r, T, "call")
    pL, pU = bounds_vanilla(S, K, r, T, "put")
    assert 0 <= cL <= cU <= S
    assert 0 <= pL <= pU <= K * math.exp(-r * T)

def test_put_call_parity_residual_zero_for_constructed_prices():
    S, K, r, T = 100, 100, 0.05, 2.0
    # Construct prices satisfying parity
    df = math.exp(-r * T)
    C = 12.0
    P = C - (S - K * df)
    res = put_call_parity_residual(C, P, S, K, r, T)
    assert abs(res) < 1e-12
