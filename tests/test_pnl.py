from optkit.pnl import Position, MarketSnapshot, greeks_pnl, portfolio_pnl_explain

PREV  = MarketSnapshot(spot=100.0, vol=0.20, rate=0.03)
TODAY = MarketSnapshot(spot=101.0, vol=0.20, rate=0.03)  # spot +1, nothing else moves
POS   = Position(strike=100.0, maturity=1.0, option_type="call", quantity=1.0)


def test_greeks_pnl_keys():
    result = greeks_pnl(POS, PREV, TODAY)
    assert {"actual", "delta", "gamma", "vega", "theta", "rho", "unexplained"}.issubset(result)


def test_delta_dominates_small_move():
    result = greeks_pnl(POS, PREV, TODAY)
    # delta pnl should carry most of the actual pnl for a small spot move
    assert abs(result["delta"]) > abs(result["gamma"])
    assert abs(result["delta"]) > abs(result["vega"])


def test_unexplained_small():
    # for a small move, higher-order terms keep unexplained tight
    result = greeks_pnl(POS, PREV, TODAY)
    assert abs(result["unexplained"]) < abs(result["actual"]) * 0.05


def test_long_call_gains_on_up_move():
    result = greeks_pnl(POS, PREV, TODAY)
    assert result["actual"] > 0


def test_short_call_loses_on_up_move():
    short = Position(strike=100.0, maturity=1.0, option_type="call", quantity=-1.0)
    result = greeks_pnl(short, PREV, TODAY)
    assert result["actual"] < 0


def test_vega_pnl_on_vol_move():
    today_vol_up = MarketSnapshot(spot=100.0, vol=0.21, rate=0.03)
    result = greeks_pnl(POS, PREV, today_vol_up)
    assert result["vega"] > 0  # long call benefits from vol increase


def test_portfolio_pnl_totals():
    call = Position(strike=100.0, maturity=1.0, option_type="call", quantity=1.0, label="call")
    put  = Position(strike=100.0, maturity=1.0, option_type="put",  quantity=1.0, label="put")
    df = portfolio_pnl_explain([call, put], PREV, TODAY)
    assert "TOTAL" in df.index
    assert abs(df.loc["TOTAL", "actual"] - (df.loc["call", "actual"] + df.loc["put", "actual"])) < 1e-10
