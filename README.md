![CI](https://github.com/mezaouifinance/options-pricing/actions/workflows/ci.yml/badge.svg)

# Options Pricing Toolkit
Python toolkit for pricing vanilla options using:
- Black–Scholes (European options)
- Cox–Ross–Rubinstein binomial tree (European & American)

Includes Greeks, implied volatility, tests, and reproducible notebooks.

## Features
- Black–Scholes call/put pricing + Greeks
- CRR binomial pricing (EU/US) + convergence study
- Implied volatility solver (robust root-finding)
- Unit tests (put-call parity, convergence, IV consistency)

## Validation (why this is correct)

### Put–call parity (no dividends)
Black–Scholes prices satisfy:
C - P = S - K e^{-rT}
Covered by unit tests.

### No-arbitrage bounds (European, no dividends)
- Call: max(0, S - K e^{-rT}) <= C <= S
- Put : max(0, K e^{-rT} - S) <= P <= K e^{-rT}
Covered by unit tests.

### Binomial CRR convergence
CRR prices converge to the Black–Scholes closed-form price as the number of steps increases.
Reproduce via:
```bash
python notebooks/01_crr_convergence.py
