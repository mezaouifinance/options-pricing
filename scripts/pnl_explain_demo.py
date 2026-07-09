"""
P&L Explain demo — long ATM call on SPY, COVID crash period (Jan–Jun 2020).

Decomposes daily option P&L into Greek contributions using optkit.pnl.
Saves figure to figures/pnl_explain_spy_2020.png.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from optkit.pnl import Position, MarketSnapshot, greeks_pnl

TICKER     = "SPY"
START      = "2020-01-02"
END        = "2020-06-30"
VOL_WINDOW = 21
RATE       = 0.02
T0         = 0.5   # 6-month ATM call


def load_data(ticker, start, end, vol_window):
    prices = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    prices = prices["Close"].squeeze().dropna()
    log_ret = np.log(prices / prices.shift(1))
    rolling_vol = log_ret.rolling(vol_window).std() * np.sqrt(252)
    # align: drop first vol_window days where rolling vol is NaN
    valid = rolling_vol.dropna().index
    return prices.loc[valid], rolling_vol.loc[valid]


def run_explain(prices, rolling_vol, rate, K, T0):
    rows = []
    for i in range(1, len(prices)):
        T_prev = T0 - (i - 1) / 252
        if T_prev <= 1 / 252:
            break
        prev  = MarketSnapshot(float(prices.iloc[i - 1]), float(rolling_vol.iloc[i - 1]), rate)
        today = MarketSnapshot(float(prices.iloc[i]),     float(rolling_vol.iloc[i]),     rate)
        pos   = Position(strike=K, maturity=T_prev, option_type="call", quantity=1.0)
        row   = greeks_pnl(pos, prev, today)
        row["date"] = prices.index[i]
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def plot(prices, df, K, output="figures/pnl_explain_spy_2020.png"):
    components = ["delta", "gamma", "vega", "theta", "rho"]
    colors     = ["#2196F3", "#4CAF50", "#FF9800", "#9C27B0", "#F44336"]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    fig.suptitle(f"P&L Explain — long ATM call on {TICKER}  (K={K:.0f}, T=6m, Jan–Jun 2020)",
                 fontsize=13, fontweight="bold")

    # --- Panel 1: SPY price ---
    ax = axes[0]
    ax.plot(prices.loc[df.index], color="#1565C0", lw=1.5)
    ax.axhline(K, color="grey", ls="--", lw=0.8, label=f"strike K={K:.0f}")
    ax.set_ylabel("SPY price ($)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 2: daily P&L waterfall ---
    ax = axes[1]
    cum = df[components].cumsum()
    for col, color in zip(components, colors):
        ax.plot(cum[col], label=col.capitalize(), lw=1.2, color=color)
    ax.plot(df["actual"].cumsum(), color="black", lw=1.8, ls="--", label="Actual (cumul.)")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("Cumulative P&L ($)")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # --- Panel 3: unexplained residual ---
    ax = axes[2]
    ax.bar(df.index, df["unexplained"], color="coral", width=1.0, alpha=0.7, label="Unexplained")
    ax.axhline(0, color="grey", lw=0.6)
    ax.set_ylabel("Daily unexplained ($)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Figure saved → {output}")


def summary(df):
    components = ["delta", "gamma", "vega", "theta", "rho", "unexplained", "actual"]
    total = df[components].sum().rename("Cumulative P&L ($)")
    pct   = (total / abs(total["actual"]) * 100).rename("% of |actual|")
    out   = pd.concat([total, pct], axis=1).drop("actual", errors="ignore")
    print("\n=== Cumulative P&L attribution ===")
    print(out.to_string(float_format="{:.2f}".format))
    print(f"\nActual total P&L : ${total['actual']:.2f}")
    explained = total[["delta", "gamma", "vega", "theta", "rho"]].sum()
    print(f"Explained        : ${explained:.2f}  ({explained / total['actual'] * 100:.1f}%)")
    print(f"Unexplained      : ${total['unexplained']:.2f}")


if __name__ == "__main__":
    print(f"Loading {TICKER} data…")
    prices, rolling_vol = load_data(TICKER, START, END, VOL_WINDOW)

    K = round(float(prices.iloc[0]))
    print(f"Strike (ATM at inception): {K}")

    df = run_explain(prices, rolling_vol, RATE, K, T0)
    summary(df)
    plot(prices, df, K)
