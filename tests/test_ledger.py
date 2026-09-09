"""stocks_ml.ledger: the book's rules and the paper ledger, shared by
selection.simulate and the live job."""
import json

import numpy as np
import pandas as pd
import pytest

from stocks_ml.ledger import (ANCHOR, Ledger, ballast_state, close_asof, due_sleeve,
                              fill_price, friday_of, rotate_sleeves, sleeve_counts,
                              target_weights, week_index)

D = pd.Timestamp
SMAP = {"A1": "Tech", "A2": "Tech", "A3": "Tech", "B1": "Fin", "B2": "Fin", "B3": "Fin",
        "C1": "Ind", "C2": "Ind", "D1": "Util", "E1": "Ener", "F1": "Cons", "G1": "Heal",
        "H1": "Mat", "I1": "Real", "J1": "Comm"}
RANKED = ["A1", "A2", "A3", "B1", "B2", "B3", "C1", "C2", "D1", "E1", "F1", "G1", "H1", "I1", "J1"]


# ---- schedule ----
def test_sleeve_schedule_cycles_from_the_anchor():
    assert week_index(ANCHOR) == 0 and due_sleeve(ANCHOR, 4) == 0
    assert due_sleeve(ANCHOR + pd.Timedelta(weeks=1), 4) == 1
    assert due_sleeve(ANCHOR + pd.Timedelta(weeks=4), 4) == 0
    assert due_sleeve(ANCHOR + pd.Timedelta(weeks=4, days=-1), 4) == 0   # a holiday Thursday: its own week
    assert friday_of("2026-09-01") == D("2026-09-04")
    assert friday_of("2026-09-04") == D("2026-09-04")


def test_rotation_fills_empty_sleeves_then_one_per_week():
    t0 = D("2026-08-28")                              # sleeve 2 due
    sleeves, rotated = rotate_sleeves({}, t0, RANKED, SMAP, 4, 6, 2)
    assert rotated == [0, 1, 2, 3]
    picked = sleeves["0"]["names"]
    assert picked == ["A1", "A2", "B1", "B2", "C1", "C2"]   # cap 2 per sector, top-6 by rank
    assert all(s["names"] == picked and s["since"] == "2026-08-28" for s in sleeves.values())

    t1 = t0 + pd.Timedelta(weeks=1)                   # sleeve 3 due
    ranked2 = list(reversed(RANKED))
    sleeves2, rotated2 = rotate_sleeves(sleeves, t1, ranked2, SMAP, 4, 6, 2)
    assert rotated2 == [3]
    assert sleeves2["3"]["names"] == ["J1", "I1", "H1", "G1", "F1", "E1"]
    assert sleeves2["3"]["since"] == "2026-09-04"
    for k in ("0", "1", "2"):
        assert sleeves2[k] == sleeves[k]


def test_rotation_on_a_holiday_thursday_rotates_the_weeks_sleeve():
    """The Friday before Good Friday 2026 rotated sleeve 0; the Thursday
    signal (2026-04-02) is the next week and must rotate sleeve 1, not be
    swallowed by the same-day guard."""
    fri, thu = D("2026-03-27"), D("2026-04-02")
    assert due_sleeve(fri, 4) == 0 and due_sleeve(thu, 4) == 1 == due_sleeve(D("2026-04-03"), 4)
    sleeves = {str(k): {"names": ["A1"], "since": "2026-03-27" if k == 0 else "2026-03-06"}
               for k in range(4)}
    out, rotated = rotate_sleeves(sleeves, thu, RANKED, SMAP, 4, 6, 2)
    assert rotated == [1] and out["1"]["since"] == "2026-04-02"
    assert out["0"]["since"] == "2026-03-27"


def test_rotation_catches_up_a_stale_sleeve():
    t = D("2026-08-28")
    sleeves = {str(k): {"names": ["A1"], "since": "2026-08-21"} for k in range(4)}
    sleeves["1"]["since"] = "2026-07-17"              # six weeks old: the job skipped its week
    out, rotated = rotate_sleeves(sleeves, t, RANKED, SMAP, 4, 6, 2)
    assert rotated == [1, 2]                          # stale + due
    assert out["0"]["since"] == "2026-08-21" and out["3"]["since"] == "2026-08-21"


def test_rotation_is_idempotent_on_the_same_day():
    """A rerun of the same Friday (manual, --as-of, a second launchd kick)
    must not rotate the due sleeve again, even if the refit ranks differently:
    simulate rotates each sleeve once per week. Seen live 2026-09-02."""
    t = D("2026-08-28")
    s1, _ = rotate_sleeves({}, t, RANKED, SMAP, 4, 6, 2)
    s2, rotated = rotate_sleeves(s1, t, list(reversed(RANKED)), SMAP, 4, 6, 2)
    assert s2 == s1 and rotated == []
    # the following week rotates as usual
    s3, rotated3 = rotate_sleeves(s2, t + pd.Timedelta(weeks=1), list(reversed(RANKED)), SMAP, 4, 6, 2)
    assert rotated3 == [3] and s3["3"]["names"] != s1["3"]["names"]


# ---- ballast + weights ----
def test_ballast_per_window():
    idx = pd.date_range("2024-01-05", periods=60, freq="W-FRI")
    rising = pd.Series(np.arange(60, dtype=float) + 100, index=idx)
    assert ballast_state(rising, idx[-1]) == {"30": "SPY", "40": "SPY", "52": "SPY"}
    falling = pd.Series(200 - np.arange(60, dtype=float), index=idx)
    assert ballast_state(falling, idx[-1]) == {"30": "IEF", "40": "IEF", "52": "IEF"}
    # too little history for the 52-week third: not "below", so SPY
    assert ballast_state(falling.iloc[-45:], idx[-1]) == {"30": "IEF", "40": "IEF", "52": "SPY"}
    # closes after t's Friday are not used
    assert ballast_state(falling, idx[-1] - pd.Timedelta(weeks=20))["30"] == "IEF"
    mixed = falling.copy()
    mixed.iloc[-1] = 300.0                            # last close above every mean
    assert ballast_state(mixed, idx[-1]) == {"30": "SPY", "40": "SPY", "52": "SPY"}


def test_target_weights_sum_to_one_with_double_counting():
    sleeves = {"0": {"names": ["A1", "A2", "B1", "B2", "C1", "C2"]},
               "1": {"names": ["A1", "A3", "B1", "B3", "C1", "D1"]},
               "2": {"names": ["E1", "F1", "G1", "H1", "I1", "J1"]},
               "3": {"names": ["E1", "F1", "G1", "H1", "I1", "J1"]}}
    w = target_weights(sleeves, {"30": "IEF", "40": "SPY", "52": "SPY"}, 0.7)
    assert sum(w.values()) == pytest.approx(1.0)
    assert w["A1"] == pytest.approx(2 * 0.7 / 24) and w["A2"] == pytest.approx(0.7 / 24)
    assert w["SPY"] == pytest.approx(0.2) and w["IEF"] == pytest.approx(0.1)
    assert list(w)[:2] == ["SPY", "IEF"]             # sorted by weight
    assert sleeve_counts(sleeves)["A1"] == 2


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
    led = Ledger.new(100.0, "2026-08-28")
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
    # references anchor at the fill date (two closes) and mark leaves them there
    assert led.refs["AAA"] == ["2026-08-31", 15.0, "2026-08-28", 14.0]
    assert led.nav_history == [["2026-09-04", nav, bench]]
    led.mark(cw, "2026-09-04")                                       # rerun: one row
    assert len(led.nav_history) == 1


def test_fill_sells_first_and_never_overdraws():
    cw, ow = market()
    led = Ledger.new(100.0, "2026-08-21")
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
    led = Ledger.new(100.0, "2026-08-21")
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
    led = Ledger.new(100.0, "2026-08-28")
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
    p, when = fill_price(cw, ow, "BBB", "2026-08-31")
    assert (p, when) == (20.0, D("2026-08-31"))                      # last print, carried forward
    p, when = fill_price(cw, ow, "AAA", "2026-08-29")           # weekend -> Monday
    assert when == D("2026-08-31")
    assert np.isnan(fill_price(cw, ow, "ZZZ", "2026-08-31")[0])


def test_fill_price_waits_at_most_five_sessions():
    cw, ow = market()
    ow.loc["2026-08-24":"2026-08-28", "BBB"] = np.nan                # halted Mon-Fri
    p, when = fill_price(cw, ow, "BBB", "2026-08-24")
    assert (p, when) == (20.0, D("2026-08-24"))                      # the last close, not next Monday's open
    ow.loc["2026-08-28", "BBB"] = 19.5                               # prints again on Friday
    assert fill_price(cw, ow, "BBB", "2026-08-24") == (19.5, D("2026-08-28"))


def test_close_asof_uses_the_last_print_on_or_before():
    cw, _ = market()
    assert close_asof(cw, "AAA", "2026-08-30") == (14.0, D("2026-08-28"))  # Sunday -> Friday
    assert np.isnan(close_asof(cw, "ZZZ", "2026-08-28")[0])


def test_target_weights_with_no_ballast_is_the_book_alone():
    sleeves = {"0": {"names": ["A1", "A2"]}, "1": {"names": ["A1"]}}
    assert target_weights(sleeves, {}, 1.0) == pytest.approx({"A1": 0.75, "A2": 0.25})


def test_ledger_roundtrip(tmp_path):
    led = Ledger.new(100.0, "2026-08-28")
    led.sleeves = {"0": {"names": ["AAA"], "since": "2026-08-28"}}
    led.pending = {"decision_date": "2026-08-28", "weights": {"AAA": 1.0}}
    path = tmp_path / "ledger_r5.json"
    led.save(path)
    again = Ledger.load(path)
    assert again == led
    assert json.loads(path.read_text())["started"] == "2026-08-28"
    assert Ledger.load(tmp_path / "missing.json") is None


# ---- safety: renames, rebase discipline, dead names (2026-09 bug hunt) ----
def test_rename_moves_the_book_to_the_new_symbol():
    """EQR -> VMRK (2026-09-01): the vendor rewrites every table, so the
    ledger must follow or the holding has no price column: worth $0 in NAV
    and unsellable."""
    led = Ledger.new(100.0, "2026-08-28")
    led.positions = {"EQR": 3.0, "BBB": 1.0}
    led.refs = {"EQR": ["2026-08-28", 10.0, "2026-08-27", 9.0]}
    led.sleeves = {"0": {"names": ["EQR", "BBB"], "since": "2026-08-28"}}
    led.pending = {"decision_date": "2026-08-28", "weights": {"EQR": 0.5, "SPY": 0.5}}
    hit = led.rename({"EQR": "VMRK", "ZZZ": "YYY"})
    assert hit == {"EQR": "VMRK"}
    assert led.positions == {"VMRK": 3.0, "BBB": 1.0}
    assert "EQR" not in led.refs and led.refs["VMRK"][1] == 10.0
    assert led.sleeves["0"]["names"] == ["VMRK", "BBB"]
    assert led.pending["weights"] == {"VMRK": 0.5, "SPY": 0.5}


def test_rebase_refuses_a_held_name_with_no_price_column():
    cw, _ = market()
    led = Ledger.new(100.0, "2026-08-28")
    led.positions = {"GONE": 1.0}
    with pytest.raises(RuntimeError, match="GONE"):
        led.rebase(cw)


def test_mark_refuses_a_held_name_with_no_print():
    cw, _ = market()
    cw["DEAD"] = np.nan
    led = Ledger.new(100.0, "2026-08-28")
    led.positions = {"DEAD": 1.0}
    with pytest.raises(RuntimeError, match="DEAD"):
        led.mark(cw, "2026-09-04")


def test_rebase_skips_a_transient_bad_print():
    """A reference stored off a stale carry (Friday's row was Thursday's
    close) implies a phantom 'adjustment' at one reference date but not the
    other: units must not be rescaled. A real adjustment moves both."""
    cw, _ = market()
    led = Ledger.new(100.0, "2026-08-28")
    led.positions = {"AAA": 10.0}
    led.refs = {"AAA": ["2026-08-28", 13.0, "2026-08-27", 13.0]}   # 08-28 really closed 14.0
    msgs = []
    assert led.rebase(cw, log=msgs.append) == {}
    assert led.positions["AAA"] == 10.0
    assert led.refs["AAA"] == ["2026-08-28", 13.0, "2026-08-27", 13.0]
    assert msgs and "AAA" in msgs[0]
    led.refs = {"AAA": ["2026-08-28", 14.0, "2026-08-27", 13.0]}   # true refs
    cw2 = cw.copy()
    cw2["AAA"] *= 0.5                                              # a real vendor readjustment
    factors = led.rebase(cw2)
    assert factors["AAA"] == pytest.approx(2.0)
    assert led.refs["AAA"] == ["2026-08-28", 7.0, "2026-08-27", 6.5]


def test_rebase_catches_an_adjustment_published_weeks_after_the_fill():
    """References anchor at the fill date and stay there through marks, so a
    dividend the vendor posts two marks after its ex-date still rescales
    units (the old mark-refreshed reference lost it for good)."""
    cw, _ = market()
    led = Ledger.new(100.0, "2026-08-24")
    led.positions = {"BBB": 5.0}
    led.refs = {"BBB": ["2026-08-25", 20.0, "2026-08-24", 20.0]}
    led.mark(cw, "2026-08-28")
    led.mark(cw, "2026-09-04")
    assert led.refs["BBB"] == ["2026-08-25", 20.0, "2026-08-24", 20.0]
    cw2 = cw.copy()
    cw2["BBB"] *= 0.98                                             # published after two marks
    factors = led.rebase(cw2)
    assert factors["BBB"] == pytest.approx(1 / 0.98)
    assert led.positions["BBB"] == pytest.approx(5.0 / 0.98)


def test_dead_names_are_never_bought():
    """A buy needs a live open. The close fallback used to 'buy' a name that
    had stopped trading at its final print: a phantom position and a fee."""
    cw, ow = market()
    cw.loc["2026-08-25":, "BBB"] = np.nan                          # last print 08-24
    ow.loc["2026-08-25":, "BBB"] = np.nan
    led = Ledger.new(100.0, "2026-08-28")
    led.pending = {"decision_date": "2026-08-28", "weights": {"BBB": 0.5, "AAA": 0.5}}
    fills = led.fill_pending(cw, ow, "2026-09-04")
    assert [f[1] for f in fills] == ["AAA"]
    assert "BBB" not in led.positions
    assert led.cash == pytest.approx(100.0 - 50.0 * (1 + 5.0 / 1e4) * 1.0, rel=1e-6)


def test_a_dead_name_still_exits_at_its_last_print():
    cw, ow = market()
    cw.loc["2026-08-25":, "BBB"] = np.nan
    ow.loc["2026-08-25":, "BBB"] = np.nan
    led = Ledger.new(100.0, "2026-08-24")
    led.cash, led.positions = 50.0, {"BBB": 2.0}
    led.pending = {"decision_date": "2026-08-24", "weights": {"AAA": 1.0}}
    fills = led.fill_pending(cw, ow, "2026-08-28")
    assert fills[0][1] == "BBB" and fills[0][2] == -2.0 and fills[0][3] == 20.0


def test_new_positions_get_fill_anchored_references():
    cw, ow = market()
    led = Ledger.new(100.0, "2026-08-28")
    led.pending = {"decision_date": "2026-08-28", "weights": {"AAA": 1.0}}
    led.fill_pending(cw, ow, "2026-09-04")                         # fills at 08-31
    assert led.refs["AAA"] == ["2026-08-31", 15.0, "2026-08-28", 14.0]
    assert led.bench["ref"] == ["2026-08-31", 500.0, "2026-08-28", 500.0]


def test_nav_history_stays_sorted():
    cw, _ = market()
    led = Ledger.new(100.0, "2026-08-24")
    led.mark(cw, "2026-09-04")
    led.mark(cw, "2026-08-28")
    assert [r[0] for r in led.nav_history] == ["2026-08-28", "2026-09-04"]


def test_rotation_ages_count_weeks_not_days():
    """Sleeve 3 catches up (empty fill) on Friday 2026-12-18; Thursday
    2026-12-24 (Christmas Friday) is the next rank week and sleeve 3 is due.
    Day arithmetic said age 0 (6 days // 7) and swallowed the rotation."""
    fri, thu = D("2026-12-18"), D("2026-12-24")
    assert due_sleeve(thu, 4) == 3
    sleeves = {str(k): {"names": ["A1"], "since": "2026-12-18"} for k in range(4)}
    out, rotated = rotate_sleeves(sleeves, thu, RANKED, SMAP, 4, 6, 2)
    assert rotated == [3]
    assert out["3"]["since"] == "2026-12-24"
    assert out["0"]["since"] == "2026-12-18"


def test_rebase_skips_when_the_store_lacks_the_reference_date():
    """A stale or vintage-mismatched store (no print on the reference date
    itself) must not book price drift against another day's close as a
    vendor adjustment."""
    cw, _ = market()
    led = Ledger.new(100.0, "2026-08-28")
    led.positions = {"AAA": 10.0}
    led.refs = {"AAA": ["2026-09-11", 25.0, "2026-09-10", 24.0]}   # beyond the store's last print
    msgs = []
    assert led.rebase(cw, log=msgs.append) == {}
    assert led.positions["AAA"] == 10.0
    assert msgs and "AAA" in msgs[0]
