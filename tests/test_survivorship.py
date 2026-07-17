import numpy as np
import pandas as pd
import pytest

from stocks_ml.backtest.survivorship import (
    classify_reason,
    compute_haircuts,
    measure_post_removal,
    parse_baseline_report,
    run_torture,
)

# ---------------------------------------------------------------------------
# classify_reason
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("reason,expected", [
    (None, "unknown"),
    (float("nan"), "unknown"),
    ("Some unrelated corporate action", "unknown"),
    ("Market capitalization change.", "decline"),
    ("Company filed for Chapter 11 bankruptcy", "decline"),
    ("Placed into FDIC receivership", "decline"),
    ("Delisted from NYSE", "decline"),
    ("Moved to S&P MidCap 400", "decline"),
    ("Acquired by Example Corp", "acquisition"),
    ("Merged with Example Corp", "acquisition"),
    ("Taken private by Example Capital", "acquisition"),
    ("Purchased by Example Corp", "acquisition"),
    ("Spun off from Example Corp", "restructuring"),
    ("Spin-off completed", "restructuring"),
    ("Acquired after bankruptcy filing", "acquisition"),  # order matters: acquisition wins ties
])
def test_classify_reason(reason, expected):
    assert classify_reason(reason) == expected


# ---------------------------------------------------------------------------
# measure_post_removal
# ---------------------------------------------------------------------------

def _close_frame(ticker: str, dates: pd.DatetimeIndex, close: pd.Series) -> pd.DataFrame:
    return pd.DataFrame({"date": dates, "ticker": ticker, "close": close.values})


def test_measure_post_removal_halves_over_horizon():
    dates = pd.bdate_range("2020-01-02", periods=400)
    close = pd.Series(100.0, index=dates)
    removal_date = dates[50]
    after_idx = dates[dates > removal_date]
    horizon = after_idx[:126]
    close.loc[horizon] = np.linspace(100.0, 50.0, len(horizon))
    close.loc[after_idx[126:]] = 50.0

    prices = _close_frame("ZZZ", dates, close)
    removals = pd.DataFrame({"ticker": ["ZZZ"], "date": [removal_date],
                             "reason": ["Market capitalization change."]})

    out = measure_post_removal(prices, removals, horizon_days=126)
    assert len(out) == 1
    row = out.iloc[0]
    assert row["post_ret"] == pytest.approx(-0.5, abs=1e-6)
    assert row["truncated"] == False  # noqa: E712
    assert row["reason_class"] == "decline"
    assert row["ticker"] == "ZZZ"
    assert row["date"] == removal_date


def test_measure_post_removal_anchor_uses_last_close_on_or_before():
    dates = pd.bdate_range("2020-01-02", periods=200)
    close = pd.Series(100.0, index=dates)
    close.iloc[30] = 77.0
    # a timestamp strictly between dates[30] and dates[31] -- never itself a
    # trading day, so the anchor must fall back to the last close ON/BEFORE it
    removal_date = dates[30] + pd.Timedelta(hours=12)

    prices = _close_frame("XXX", dates, close)
    removals = pd.DataFrame({"ticker": ["XXX"], "date": [removal_date], "reason": [None]})

    out = measure_post_removal(prices, removals, horizon_days=5)
    row = out.iloc[0]
    expected_outcome = close.iloc[35]  # dates[31..35] are the 5 trading days after
    assert row["post_ret"] == pytest.approx(expected_outcome / 77.0 - 1)


def test_measure_post_removal_flags_truncated_when_series_ends_soon():
    all_dates = pd.bdate_range("2020-01-02", periods=60)
    removal_date = all_dates[40]
    dates = all_dates[: 40 + 10 + 1]  # series ends 10 trading days after removal
    close = pd.Series(np.linspace(100.0, 40.0, len(dates)), index=dates)

    prices = _close_frame("YYY", dates, close)
    removals = pd.DataFrame({"ticker": ["YYY"], "date": [removal_date], "reason": [None]})

    out = measure_post_removal(prices, removals, horizon_days=126)
    row = out.iloc[0]
    assert row["truncated"] == True  # noqa: E712
    assert row["reason_class"] == "unknown"
    # outcome falls back to the final available close since the series ends
    # before the full horizon; anchor is the close at the removal date itself
    anchor = close.loc[removal_date]
    assert row["post_ret"] == pytest.approx(close.iloc[-1] / anchor - 1)


def test_measure_post_removal_skips_tickers_with_no_price_data():
    dates = pd.bdate_range("2020-01-02", periods=100)
    close = pd.Series(100.0, index=dates)
    prices = _close_frame("KNOWN", dates, close)
    removals = pd.DataFrame({"ticker": ["UNKNOWN"], "date": [dates[50]], "reason": [None]})
    out = measure_post_removal(prices, removals)
    assert out.empty


# ---------------------------------------------------------------------------
# compute_haircuts
# ---------------------------------------------------------------------------

def test_compute_haircuts_maps_class_medians_and_q25_to_haircuts():
    measured = pd.DataFrame([
        {"ticker": "A1", "date": pd.Timestamp("2010-01-01"), "reason_class": "decline",
         "post_ret": -0.6, "truncated": False},
        {"ticker": "A2", "date": pd.Timestamp("2011-01-01"), "reason_class": "decline",
         "post_ret": -0.4, "truncated": False},
        {"ticker": "A3", "date": pd.Timestamp("2012-01-01"), "reason_class": "decline",
         "post_ret": -0.2, "truncated": False},
        {"ticker": "A4", "date": pd.Timestamp("2013-01-01"), "reason_class": "decline",
         "post_ret": -0.9, "truncated": True},
        {"ticker": "B1", "date": pd.Timestamp("2010-06-01"), "reason_class": "acquisition",
         "post_ret": 0.3, "truncated": False},
        {"ticker": "B2", "date": pd.Timestamp("2011-06-01"), "reason_class": "acquisition",
         "post_ret": -0.1, "truncated": False},
    ])

    out = compute_haircuts(measured)
    assert set(out.keys()) == {"class_haircuts", "per_event"}

    # decline: non-truncated post_rets = [-0.6, -0.4, -0.2] -> median -0.4 -> haircut 0.4
    assert out["class_haircuts"]["decline"] == pytest.approx(0.4)
    # acquisition: non-truncated post_rets = [0.3, -0.1] -> median 0.1 (non-negative) -> haircut 0.0
    assert out["class_haircuts"]["acquisition"] == pytest.approx(0.0)

    per_event = out["per_event"].set_index("ticker")
    assert list(out["per_event"].columns) == ["ticker", "date", "haircut"]
    # non-truncated events take the class haircut
    assert per_event.loc["A1", "haircut"] == pytest.approx(0.4)
    assert per_event.loc["A2", "haircut"] == pytest.approx(0.4)
    assert per_event.loc["B2", "haircut"] == pytest.approx(0.0)
    # truncated event takes the class's (more punitive) 25th-percentile haircut:
    # q25 of [-0.6, -0.4, -0.2] = -0.5 -> haircut 0.5
    assert per_event.loc["A4", "haircut"] == pytest.approx(0.5)


def test_compute_haircuts_nonneg_median_class_gets_zero():
    measured = pd.DataFrame([
        {"ticker": "C1", "date": pd.Timestamp("2010-01-01"), "reason_class": "restructuring",
         "post_ret": 0.05, "truncated": False},
        {"ticker": "C2", "date": pd.Timestamp("2011-01-01"), "reason_class": "restructuring",
         "post_ret": 0.10, "truncated": False},
    ])
    out = compute_haircuts(measured)
    assert out["class_haircuts"]["restructuring"] == 0.0


# ---------------------------------------------------------------------------
# parse_baseline_report
# ---------------------------------------------------------------------------

SAMPLE_BASELINE_MD = """\
# Backtest report

Champion model: **xgb** · strategies × candidates tried: **12**

## Headline ($100 invested at start)

| strategy | $100 → | CAGR | Sharpe | Deflated Sharpe | Sortino | max DD | worst week | underwater (d) | costs $ | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| equal_topk | $12,170 | 25.1% | 0.76 | 0.97 | 1.28 | 66.7% | -33.7% | 805 | 2,068.53 | 278 |
| vol_scaled | $328 | 5.7% | 0.53 | 0.78 | 0.81 | 33.3% | -13.9% | 2338 | 84.25 | 278 |
| kelly | $949 | 11.1% | 0.60 | 0.86 | 0.93 | 50.5% | -20.0% | 806 | 179.45 | 278 |
| spy_hold | $959 | 11.1% | 0.65 | – | 1.02 | 55.2% | -19.8% | 1773 | – | – |
| cash | $145 | 1.7% | 14.68 | – | 28,391.73 | 0.0% | -0.0% | 16 | – | – |

## Stress windows (total return)

| window | equal_topk | vol_scaled | kelly | spy_hold | cash |
|---|---|---|---|---|---|
| GFC | -8.4% | -5.3% | -30.5% | -36.6% | 0.2% |
| Q4 2018 | -20.5% | -10.7% | -14.4% | -13.8% | 0.6% |

## Honesty notes

- some note
"""


def test_parse_baseline_report_extracts_headline_and_gfc_verbatim(tmp_path):
    path = tmp_path / "backtest.md"
    path.write_text(SAMPLE_BASELINE_MD)
    out = parse_baseline_report(path)
    assert out["headline"]["equal_topk"]["$100 →"] == "$12,170"
    assert out["headline"]["equal_topk"]["CAGR"] == "25.1%"
    assert out["headline"]["equal_topk"]["max DD"] == "66.7%"
    assert out["headline"]["kelly"]["$100 →"] == "$949"
    assert out["gfc"]["equal_topk"] == "-8.4%"
    assert out["gfc"]["vol_scaled"] == "-5.3%"
    assert out["gfc"]["kelly"] == "-30.5%"


# ---------------------------------------------------------------------------
# run_torture wiring (synthetic data, no network, cheap candidates)
# ---------------------------------------------------------------------------

def test_run_torture_writes_report(synthetic_store, tiny_cfg, tmp_path):
    from stocks_ml.features.panel import build_panel
    from stocks_ml.models.candidates import MomentumRank, ZeroForecast
    from stocks_ml.models.champion import run_training

    build_panel(synthetic_store, tiny_cfg)
    prices = synthetic_store.read("prices")
    tickers = [t for t in sorted(prices["ticker"].unique()) if t != "SPY"]
    dates = sorted(prices["date"].unique())
    removals = pd.DataFrame({
        "ticker": [tickers[0], tickers[1]],
        "date": [pd.Timestamp(dates[300]), pd.Timestamp(dates[400])],
        "reason": ["Acquired by BigCo", "Market capitalization change."],
    })
    synthetic_store.write("removals", removals)

    models_dir = tmp_path / "models"
    run_training(synthetic_store, tiny_cfg,
                candidates={"zero": ZeroForecast(), "momentum": MomentumRank()},
                out_dir=models_dir)

    baseline_path = tmp_path / "baseline_backtest.md"
    baseline_path.write_text(SAMPLE_BASELINE_MD)

    out = run_torture(synthetic_store, tiny_cfg, models_dir=models_dir,
                      out_dir=tmp_path / "torture_reports", baseline_report=baseline_path)

    assert out.exists()
    text = out.read_text()
    assert "equal_topk" in text and "vol_scaled" in text and "kelly" in text
    assert "haircut" in text.lower()
    assert "GFC" in text
    assert "decline" in text or "acquisition" in text
