import math
from optkit.bs import bs_price
from optkit.binomial import crr_price

def test_crr_converges_to_bs_call():
    S, K, r, T, sigma = 100, 100, 0.03, 1.0, 0.2
    bs = bs_price(S, K, r, sigma, T, "call")
    # Convergence should improve with N
    p50 = crr_price(S, K, r, sigma, T, 50, "call", american=False)
    p200 = crr_price(S, K, r, sigma, T, 200, "call", american=False)
    assert abs(p200 - bs) <= abs(p50 - bs) + 1e-6

def test_american_put_ge_european_put():
    S, K, r, T, sigma = 100, 110, 0.03, 1.0, 0.25
    euro = crr_price(S, K, r, sigma, T, 200, "put", american=False)
    amer = crr_price(S, K, r, sigma, T, 200, "put", american=True)
    assert amer + 1e-12 >= euro
