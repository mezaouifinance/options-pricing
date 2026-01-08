from optkit.bs import bs_price
from optkit.implied_vol import implied_vol

def test_implied_vol_roundtrip_call():
    S, K, r, T, sigma = 100, 100, 0.01, 1.0, 0.33
    price = bs_price(S, K, r, sigma, T, "call")
    iv = implied_vol(price, S, K, r, T, "call")
    assert abs(iv - sigma) < 1e-7

def test_implied_vol_roundtrip_put():
    S, K, r, T, sigma = 100, 120, 0.02, 0.5, 0.4
    price = bs_price(S, K, r, sigma, T, "put")
    iv = implied_vol(price, S, K, r, T, "put")
    assert abs(iv - sigma) < 1e-7
