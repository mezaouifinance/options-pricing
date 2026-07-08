from __future__ import annotations
from dataclasses import dataclass, field
import pandas as pd
from .bs import bs_price, bs_delta, bs_gamma, bs_vega, bs_theta, bs_rho


@dataclass
class Position:
    strike: float
    maturity: float        # years to expiry at start of day
    option_type: str       # "call" or "put"
    quantity: float = 1.0  # positive = long
    label: str = ""


@dataclass
class MarketSnapshot:
    spot: float
    vol: float
    rate: float


def greeks_pnl(
    position: Position,
    prev: MarketSnapshot,
    today: MarketSnapshot,
    dt: float = 1 / 252,
) -> dict:
    S, K = prev.spot, position.strike
    r, sigma, T = prev.rate, prev.vol, position.maturity
    ot, qty = position.option_type, position.quantity

    dS = today.spot - prev.spot
    dv = today.vol - prev.vol
    dr = today.rate - prev.rate

    delta_pnl = qty * bs_delta(S, K, r, sigma, T, ot) * dS
    gamma_pnl = qty * 0.5 * bs_gamma(S, K, r, sigma, T) * dS ** 2
    vega_pnl  = qty * bs_vega(S, K, r, sigma, T) * dv
    theta_pnl = qty * bs_theta(S, K, r, sigma, T, ot) * dt
    rho_pnl   = qty * bs_rho(S, K, r, sigma, T, ot) * dr

    price_prev  = bs_price(S, K, r, sigma, T, ot)
    price_today = bs_price(today.spot, K, today.rate, today.vol, max(T - dt, 0.0), ot)
    actual_pnl  = qty * (price_today - price_prev)

    explained    = delta_pnl + gamma_pnl + vega_pnl + theta_pnl + rho_pnl
    unexplained  = actual_pnl - explained

    return {
        "label":       position.label or f"K={K} {ot}",
        "actual":      actual_pnl,
        "delta":       delta_pnl,
        "gamma":       gamma_pnl,
        "vega":        vega_pnl,
        "theta":       theta_pnl,
        "rho":         rho_pnl,
        "unexplained": unexplained,
    }


def portfolio_pnl_explain(
    positions: list[Position],
    prev: MarketSnapshot,
    today: MarketSnapshot,
    dt: float = 1 / 252,
) -> pd.DataFrame:
    rows = [greeks_pnl(p, prev, today, dt) for p in positions]
    df = pd.DataFrame(rows).set_index("label")
    df.loc["TOTAL"] = df.sum()
    return df
