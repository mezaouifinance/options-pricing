![CI](https://github.com/mezaouifinance/options-pricing/actions/workflows/ci.yml/badge.svg)

# Options Pricing Toolkit

Python library for pricing vanilla options and computing Greeks under two models:

| Model | European | American | Greeks |
|-------|----------|----------|--------|
| Black–Scholes (closed-form) | ✓ | — | Δ, Γ, ν, θ, ρ |
| CRR Binomial tree | ✓ | ✓ | — |

Includes an implied volatility solver and no-arbitrage validation.

---

## Installation

```bash
git clone https://github.com/mezaouifinance/options-pricing.git
cd options-pricing
pip install -e .
```

Requires Python ≥ 3.10 and NumPy ≥ 1.24.

---

## Quick start

```python
from optkit.bs import bs_price, bs_delta, bs_vega
from optkit.binomial import crr_price
from optkit.implied_vol import implied_vol

S, K, r, sigma, T = 100, 100, 0.03, 0.20, 1.0

# Black-Scholes
call  = bs_price(S, K, r, sigma, T, "call")     # 9.5768...
delta = bs_delta(S, K, r, sigma, T, "call")     # 0.5596...
vega  = bs_vega(S, K, r, sigma, T)              # 37.52...

# CRR binomial (European & American)
eu_put = crr_price(S, K, r, sigma, T, N=200, option_type="put")
am_put = crr_price(S, K, r, sigma, T, N=200, option_type="put", american=True)
assert am_put >= eu_put  # early-exercise premium

# Implied volatility (Newton + bisection fallback)
iv = implied_vol(call, S, K, r, T, "call")      # ~0.20
```

---

## Run tests

```bash
pytest
```

---

## Run convergence study

```bash
python scripts/crr_convergence.py
```

Prints CRR error vs. N for a European call, illustrating O(1/N) convergence to Black-Scholes.

---

## Mathematical validation

### Put-call parity (no dividends)

```
C - P = S - K * exp(-rT)
```

Covered by `tests/test_bs.py`.

### No-arbitrage bounds (European, no dividends)

```
Call: max(0, S - K*exp(-rT)) <= C <= S
Put:  max(0, K*exp(-rT) - S) <= P <= K*exp(-rT)
```

Covered by `tests/test_noarb.py`.

### CRR convergence

CRR prices converge to the Black-Scholes closed-form as N increases.
Reproduce with `python scripts/crr_convergence.py`.

![CRR convergence](figures/crr_convergence.png)

Error decays approximately as O(1/N) on the log-log scale, consistent with the theoretical rate.

---

## Project structure

```
options-pricing/
├── src/optkit/
│   ├── bs.py           # Black-Scholes price + Greeks
│   ├── binomial.py     # CRR binomial tree
│   ├── implied_vol.py  # Newton + bisection IV solver
│   ├── noarb.py        # No-arbitrage bounds + put-call parity
│   ├── payoff.py       # Vanilla payoff
│   ├── types.py        # OptionType enum
│   └── utils.py        # norm_cdf, norm_pdf, discount_factor
├── tests/
│   ├── test_bs.py
│   ├── test_binomial.py
│   ├── test_implied_vol.py
│   └── test_noarb.py
├── scripts/
│   └── crr_convergence.py
├── pyproject.toml
└── .github/workflows/ci.yml
```
