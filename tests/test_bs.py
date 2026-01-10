import math
from optkit.bs import bs_price, bs_delta, bs_gamma, bs_vega
from optkit.noarb import put_call_parity_residual, bounds_vanilla

def test_bs_put_call_parity():
    S, K, r, T, sigma = 100, 100, 0.03, 1.0, 0.2
    C = bs_price(S, K, r, sigma, T, "call")
    P = bs_price(S, K, r, sigma, T, "put")
    res = put_call_parity_residual(C, P, S, K, r, T)
    assert abs(res) < 1e-10

def test_bs_noarb_bounds():
    S, K, r, T, sigma = 100, 120, 0.01, 0.5, 0.25
    for ot in ("call", "put"):
        price = bs_price(S, K, r, sigma, T, ot)
        lo, hi = bounds_vanilla(S, K, r, T, ot)
        assert lo - 1e-12 <= price <= hi + 1e-12

def test_bs_greeks_signs_sane():
    S, K, r, T, sigma = 100, 100, 0.02, 1.0, 0.3
    dc = bs_delta(S, K, r, sigma, T, "call")
    dp = bs_delta(S, K, r, sigma, T, "put")
    g = bs_gamma(S, K, r, sigma, T, "call")
    v = bs_vega(S, K, r, sigma, T, "call")

    assert 0.0 <= dc <= 1.0
    assert -1.0 <= dp <= 0.0
    assert g >= 0.0
    assert v >= 0.0

def test_bs_T0_matches_intrinsic_discounted():
    S, K, r, T, sigma = 105, 100, 0.05, 0.0, 0.2
    C = bs_price(S, K, r, sigma, T, "call")
    assert abs(C - max(0.0, S - K)) < 1e-12
