import math
import numpy as np

from optkit.bs import bs_price
from optkit.binomial import crr_price

def run():
    S, K, r, T, sigma = 100.0, 100.0, 0.03, 1.0, 0.2
    bs = bs_price(S, K, r, sigma, T, "call")

    Ns = [10, 25, 50, 100, 200, 400, 800, 1200]
    rows = []
    for N in Ns:
        crr = crr_price(S, K, r, sigma, T, N, "call", american=False)
        err = abs(crr - bs)
        rows.append((N, crr, err))

    print("CRR convergence to Black–Scholes (European Call)")
    print(f"Params: S={S}, K={K}, r={r}, T={T}, sigma={sigma}")
    print(f"BS price: {bs:.10f}\n")
    print(f"{'N':>6}  {'CRR':>14}  {'|CRR-BS|':>14}")
    for N, crr, err in rows:
        print(f"{N:6d}  {crr:14.10f}  {err:14.10f}")

    # Quick sanity: error should typically decrease as N increases (not strictly monotone always)
    # But if it explodes, something is off.
    worst = max(e for _, _, e in rows)
    if not math.isfinite(worst):
        raise RuntimeError("Non-finite error encountered.")

if __name__ == "__main__":
    run()
