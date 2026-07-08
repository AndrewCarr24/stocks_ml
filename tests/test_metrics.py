import numpy as np
import pandas as pd
import pytest

from stocks_ml.backtest.metrics import (
    cagr, deflated_sharpe, max_drawdown, nav_to_returns, regime_flags,
    regime_summaries, summarize,
)


def _nav(daily_ret, n=504, start="2020-01-01"):
    idx = pd.bdate_range(start, periods=n)
    return pd.Series(100.0 * np.cumprod(np.full(n, 1 + daily_ret)), index=idx)


def test_cagr_known_growth():
    nav = _nav(0.001)
    days = (nav.index[-1] - nav.index[0]).days
    expected = (nav.iloc[-1] / nav.iloc[0]) ** (365.25 / days) - 1
    assert cagr(nav) == pytest.approx(expected)


def test_max_drawdown_constructed():
    nav = pd.Series([100, 120, 90, 110, 80.0],
                    index=pd.bdate_range("2020-01-01", periods=5))
    assert max_drawdown(nav) == pytest.approx(1 - 80 / 120)


def test_deflated_sharpe_decreases_with_trials():
    rng = np.random.default_rng(5)
    rets = pd.Series(rng.normal(0.001, 0.01, 750),
                     index=pd.bdate_range("2020-01-01", periods=750))
    d1 = deflated_sharpe(rets, n_trials=1)
    d20 = deflated_sharpe(rets, n_trials=20)
    assert 0 <= d20 < d1 <= 1


def test_regime_flags():
    idx = pd.bdate_range("2020-01-01", periods=300)
    up_then_down = np.concatenate([np.full(250, 1.005), np.full(50, 0.97)])
    spy = pd.Series(100 * np.cumprod(up_then_down), index=idx)
    vix = pd.Series(np.concatenate([np.full(250, 15.0), np.full(50, 40.0)]), index=idx)
    flags = regime_flags(spy, vix)
    assert bool(flags["bull"].iloc[249])
    assert not bool(flags["bull"].iloc[-1])
    assert bool(flags["high_vol"].iloc[-1])


def test_summaries_have_expected_keys():
    nav = _nav(0.0005)
    s = summarize(nav, n_trials=12)
    for key in ("terminal_100", "cagr", "sharpe", "sortino", "max_drawdown",
                "worst_week", "longest_underwater_days", "deflated_sharpe"):
        assert key in s
    idx = nav.index
    flags = pd.DataFrame({"bull": True, "high_vol": False}, index=idx)
    rs = regime_summaries(nav, flags)
    assert set(rs) == {"bull", "bear", "high_vol", "low_vol"}
    assert rs["bear"]["n_days"] == 0
