"""live/r5.py: the champion's weekly rules and paper ledger."""
import json

import numpy as np
import pandas as pd
import pytest

from stocks_ml.live import r5

D = pd.Timestamp
SMAP = {"A1": "Tech", "A2": "Tech", "A3": "Tech", "B1": "Fin", "B2": "Fin", "B3": "Fin",
        "C1": "Ind", "C2": "Ind", "D1": "Util", "E1": "Ener", "F1": "Cons", "G1": "Heal",
        "H1": "Mat", "I1": "Real", "J1": "Comm"}
RANKED = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "D1", "E1", "F1", "G1", "H1", "I1", "J1"]


# ---- schedule ----
def test_sleeve_schedule_cycles_from_the_anchor():
    assert r5.week_index(r5.ANCHOR) == 0 and r5.due_sleeve(r5.ANCHOR) == 0
    assert r5.due_sleeve(r5.ANCHOR + pd.Timedelta(weeks=1)) == 1
    assert r5.due_sleeve(r5.ANCHOR + pd.Timedelta(weeks=4)) == 0
    assert r5.due_sleeve(r5.ANCHOR + pd.Timedelta(weeks=4, days=-1)) == 3   # a holiday Thursday
    assert r5.friday_of("2026-09-01") == D("2026-09-04")
    assert r5.friday_of("2026-09-04") == D("2026-09-04")
    assert r5.last_friday("2026-09-05") == D("2026-09-04")
    assert r5.last_friday("2026-09-04") == D("2026-09-04")


def test_rotation_fills_empty_sleeves_then_one_per_week():
    t0 = D("2026-08-28")                              # sleeve 2 due
    sleeves, rotated = r5.rotate_sleeves({}, t0, RANKED, SMAP)
    assert rotated == [0, 1, 2, 3]
    picked = sleeves["0"]["names"]
    assert picked == ["A1", "A2", "B1", "B2", "C1", "C2"]   # cap 2 per sector, top-6 by rank
    assert all(s["names"] == picked and s["since"] == "2026-08-28" for s in sleeves.values())

    t1 = t0 + pd.Timedelta(weeks=1)                   # sleeve 3 due
    ranked2 = list(reversed(RANKED))
    sleeves2, rotated2 = r5.rotate_sleeves(sleeves, t1, ranked2, SMAP)
    assert rotated2 == [3]
    assert sleeves2["3"]["names"] == ["J1", "I1", "H1", "G1", "F1", "E1"]
    assert sleeves2["3"]["since"] == "2026-09-04"
    for k in ("0", "1", "2"):
        assert sleeves2[k] == sleeves[k]


def test_rotation_catches_up_a_stale_sleeve():
    t = D("2026-08-28")
    sleeves = {str(k): {"names": ["A1"], "since": "2026-08-21"} for k in range(4)}
    sleeves["1"]["since"] = "2026-07-17"              # six weeks old: the job skipped its week
    out, rotated = r5.rotate_sleeves(sleeves, t, RANKED, SMAP)
    assert rotated == [1, 2]                          # stale + due
    assert out["0"]["since"] == "2026-08-21" and out["3"]["since"] == "2026-08-21"


def test_rotation_is_idempotent_on_the_same_day():
    t = D("2026-08-28")
    s1, _ = r5.rotate_sleeves({}, t, RANKED, SMAP)
    s2, rotated = r5.rotate_sleeves(s1, t, RANKED, SMAP)
    assert s2 == s1 and rotated == [2]


# ---- ballast + weights ----
def test_ballast_per_window_matches_simulate_rule():
    idx = pd.date_range("2024-01-05", periods=60, freq="W-FRI")
    rising = pd.Series(np.arange(60, dtype=float) + 100, index=idx)
    assert r5.ballast_state(rising, idx[-1]) == {"30": "SPY", "40": "SPY", "52": "SPY"}
    falling = pd.Series(200 - np.arange(60, dtype=float), index=idx)
    assert r5.ballast_state(falling, idx[-1]) == {"30": "IEF", "40": "IEF", "52": "IEF"}
    # too little history for the 52-week third: not "below", so SPY
    assert r5.ballast_state(falling.iloc[-45:], idx[-1]) == {"30": "IEF", "40": "IEF", "52": "SPY"}
    # closes after t's Friday are not used
    assert r5.ballast_state(falling, idx[-1] - pd.Timedelta(weeks=20))["30"] == "IEF"
    mixed = falling.copy()
    mixed.iloc[-1] = 300.0                            # last close above every mean
    assert r5.ballast_state(mixed, idx[-1]) == {"30": "SPY", "40": "SPY", "52": "SPY"}


def test_target_weights_sum_to_one_with_double_counting():
    sleeves = {"0": {"names": ["A1", "A2", "B1", "B2", "C1", "C2"]},
               "1": {"names": ["A1", "A3", "B1", "B3", "C1", "D1"]},
               "2": {"names": ["E1", "F1", "G1", "H1", "I1", "J1"]},
               "3": {"names": ["E1", "F1", "G1", "H1", "I1", "J1"]}}
    w = r5.target_weights(sleeves, {"30": "IEF", "40": "SPY", "52": "SPY"})
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["A1"] == pytest.approx(2 * 0.7 / 24) and w["A2"] == pytest.approx(0.7 / 24)
    assert w["SPY"] == pytest.approx(0.2) and w["IEF"] == pytest.approx(0.1)
    assert list(w)[:2] == ["SPY", "IEF"]             # sorted by weight
    assert r5.sleeve_counts(sleeves)["A1"] == 2


# ---- ledger ----
def market(days=("2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28",
                 "2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04")):
    idx = pd.to_datetime(list(days))
    close = pd.DataFrame({"AAA": 10.0, "BBB": 20.0, "SPY": 500.0, "IEF": 90.0}, index=idx)
    close["AAA"] = np.linspace(10, 19, len(idx))     # 10, 11, ..., 19
    op = close.copy() - 0.5
    return close, op


def test_fill_waits_for_the_first_open_after_the_decision():
    cw, ow = market()
    led = r5.R5Ledger.new(100.0, "2026-08-28")
    led.pending = {"decision_date": "2026-08-28", "weights": {"AAA": 0.7, "SPY": 0.3}}
    assert led.fill_pending(cw, ow, "2026-08-28") == []           # same day: not yet
    assert led.pending is not None
    fills = led.fill_pending(cw, ow, "2026-09-04")
    assert [f[0] for f in fills] == ["2026-08-31", "2026-08-31"]   # Monday's open
    aaa = [f for f in fills if f[1] == "AAA"][0]
    assert aaa[3] == pytest.approx(ow.loc["2026-08-31", "AAA"])
    fee = 5.0 / 1e4
    # buys are sized net of fees from a 100 NAV: 70 and 30 dollars scaled by 1/(1+fee)
    spent = sum(f[2] * f[3] * (1 + fee) for f in fills)
    assert spent == pytest.approx(100.0)
    assert led.cash == pytest.approx(0.0, abs=1e-9) and led.cash >= -1e-9
    assert led.pending is None
    assert led.bench["units"] == pytest.approx(100.0 / (ow.loc["2026-08-31", "SPY"] * (1 + fee)))
    nav, bench = led.mark(cw, "2026-09-04")
    assert nav == pytest.approx(led.positions["AAA"] * 19.0 + led.positions["SPY"] * 500.0)
    assert led.refs["AAA"] == ["2026-09-04", 19.0]
    assert led.nav_history == [["2026-09-04", nav, bench]]
    led.mark(cw, "2026-09-04")                                       # rerun: one row
    assert len(led.nav_history) == 1


def test_fill_sells_first_and_never_overdraws():
    cw, ow = market()
    led = r5.R5Ledger.new(100.0, "2026-08-21")
    led.cash, led.positions = 0.0, {"AAA": 5.0, "BBB": 2.5}          # 50 + 50 at 08-24 opens
    led.bench = {"cash": 0.0, "units": 0.2, "ref": None}
    led.pending = {"decision_date": "2026-08-21", "weights": {"BBB": 0.5, "SPY": 0.5}}
    fills = led.fill_pending(cw, ow, "2026-08-28")
    # AAA exits in full, BBB is trimmed (NAV at the open is 96.25 -> 48.125 target),
    # then SPY is bought with what is left; sells precede buys
    assert [f[1] for f in fills] == ["AAA", "BBB", "SPY"]
    assert [f[2] < 0 for f in fills] == [True, True, False]
    assert "AAA" not in led.positions and 2.4 < led.positions["BBB"] < 2.5
    assert led.positions["BBB"] * 19.5 == pytest.approx(48.125)
    assert led.cash == pytest.approx(0.0, abs=1e-9) and led.cash >= -1e-9
    assert led.trades == fills


def test_fill_skips_dust_rebalances_but_not_exits():
    cw, ow = market()
    led = r5.R5Ledger.new(100.0, "2026-08-21")
    led.cash, led.positions = 0.0, {"AAA": 5.0, "BBB": 2.5, "SPY": 0.001}
    led.bench = {"cash": 0.0, "units": 0.2, "ref": None}
    nav = 5.0 * 9.5 + 2.5 * 19.5 + 0.001 * 499.5
    led.pending = {"decision_date": "2026-08-21",
                   "weights": {"AAA": 5.0 * 9.5 / nav + 0.001, "BBB": 2.5 * 19.5 / nav - 0.001}}
    fills = led.fill_pending(cw, ow, "2026-08-28")
    assert [f[1] for f in fills] == ["SPY"]                          # the exit runs, dust does not
    assert "SPY" not in led.positions and led.positions["AAA"] == 5.0


def test_rebase_preserves_value_after_a_readjustment():
    cw, ow = market()
    led = r5.R5Ledger.new(100.0, "2026-08-28")
    led.positions = {"AAA": 10.0}
    led.bench = {"cash": 0.0, "units": 0.1, "ref": None}
    led.mark(cw, "2026-08-28")
    value = led.positions["AAA"] * cw.loc["2026-08-28", "AAA"]
    cw2 = cw.copy()
    cw2["AAA"] *= 0.5                                                # vendor halves the history
    cw2["SPY"] *= 0.9
    factors = led.rebase(cw2)
    assert factors == {"AAA": pytest.approx(2.0), "SPY(bench)": pytest.approx(1 / 0.9)}
    assert led.positions["AAA"] * cw2.loc["2026-08-28", "AAA"] == pytest.approx(value)
    assert led.rebase(cw2) == {}                                     # idempotent


def test_fill_price_falls_back_for_a_delisted_name():
    cw, ow = market()
    ow.loc["2026-08-31":, "BBB"] = np.nan                           # BBB stops trading
    cw.loc["2026-08-31":, "BBB"] = np.nan
    cw = cw.ffill()
    p, when = r5._fill_price(cw, ow, "BBB", "2026-08-31")
    assert (p, when) == (20.0, D("2026-08-31"))                      # last print, carried forward
    p, when = r5._fill_price(cw, ow, "AAA", "2026-08-29")           # weekend -> Monday
    assert when == D("2026-08-31")
    assert np.isnan(r5._fill_price(cw, ow, "ZZZ", "2026-08-31")[0])


def test_ledger_roundtrip(tmp_path):
    led = r5.R5Ledger.new(100.0, "2026-08-28")
    led.sleeves = {"0": {"names": ["AAA"], "since": "2026-08-28"}}
    led.pending = {"decision_date": "2026-08-28", "weights": {"AAA": 1.0}}
    path = tmp_path / "ledger_r5.json"
    led.save(path)
    again = r5.R5Ledger.load(path)
    assert again == led
    assert json.loads(path.read_text())["started"] == "2026-08-28"
    assert r5.R5Ledger.load(tmp_path / "missing.json") is None


def test_rank_members_requires_recent_prices(monkeypatch):
    preds = pd.Series({"AAA": 0.5, "BBB": 0.9, "OLD": 0.7})
    prices = pd.DataFrame({"ticker": ["AAA", "BBB", "OLD"],
                           "date": [D("2026-08-28"), D("2026-08-27"), D("2026-08-01")],
                           "close": [1.0, 2.0, 3.0]})
    monkeypatch.setattr(r5, "MIN_UNIVERSE", 2)
    ranked = r5.rank_members(preds, prices, "2026-08-28")
    assert list(ranked.index) == ["BBB", "AAA"]
    monkeypatch.setattr(r5, "MIN_UNIVERSE", 3)
    with pytest.raises(RuntimeError, match="rankable"):
        r5.rank_members(preds, prices, "2026-08-28")


def test_render_markdown_lists_book_and_fills():
    sig = {"date": "2026-08-28", "sleeve_due": 2, "rotated": [0, 1, 2, 3],
           "sleeves": {"0": {"names": ["A1"], "since": "2026-08-28"}},
           "ballast": {"30": "SPY", "40": "IEF", "52": "SPY"},
           "weights": {"A1": 0.7, "SPY": 0.2, "IEF": 0.1}, "nav": 100.0, "spy_nav": 100.0,
           "cash": 100.0, "held_value": {}, "fills": [["2026-08-24", "A1", 1.5, 10.0, 0.0075]],
           "rebase_factors": {"A1": 0.5}, "top": [("A1", 0.1234)], "n_ranked": 480,
           "positions": {}, "freshness": {"panel": "2026-08-28"}, "elapsed_s": 12}
    md = r5.render_markdown(sig, SMAP)
    assert "| A1 | Tech | 1 | 70.00% | $70.00 | $0.00 | +70.00 |" in md
    assert "| SPY | ballast |  | 20.00% |" in md
    assert "| 2026-08-24 | A1 | +1.5000 | $10.00 | $0.0075 |" in md
    assert "A1 ×0.500000" in md and "Top-1 of 480" in md
