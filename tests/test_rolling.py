"""The rolling procedure: windows end at label_end(t), a trailing rule waits
for its whole window, the followed rule uses the decision in force, and the
lookback is chosen by argmax on the common span."""
import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import stocks_ml.rolling as roll
import stocks_ml.selection as sel


def test_windows_end_at_label_end_and_a_trailing_rule_waits_for_its_window():
    weeks = pd.date_range("2002-01-04", "2010-12-31", freq="W-FRI")
    w = roll.windows(weeks, 3)
    t, lo, hi = w[0]
    assert t == pd.Timestamp("2005-01-07")                       # first t with t - 3y >= first week
    assert lo == t - pd.DateOffset(years=3)
    assert hi == t - pd.Timedelta(days=35)                       # label_end: the 4w return is realized
    assert all(hi_ == t_ - pd.Timedelta(days=35) for t_, _, hi_ in w)
    assert len(roll.windows(weeks, 3, cadence=4)) == (len(w) + 3) // 4
    e = roll.windows(weeks, None, min_years=3)                   # expanding: lo pinned to the start
    assert e[0][0] == pd.Timestamp("2005-01-07") and all(lo_ == weeks[0] for _, lo_, _ in e)
    assert roll.windows(weeks, 20) == []


def test_parse_lookback_and_variant_names():
    assert roll.parse_lookback("expanding") is None and roll.parse_lookback("8") == 8
    assert roll.variant_name(None, 1) == "expanding_c1" and roll.variant_name(5, 4) == "trailing_5_c4"
    with pytest.raises(ValueError):
        roll.parse_lookback("0")


def test_decisions_call_decide_strategy_on_each_window_in_process(monkeypatch):
    weeks = pd.date_range("2006-01-06", periods=52 * 5, freq="W-FRI")
    hold = pd.DataFrame({"week": weeks, "top3": 0.0, "top6": 0.0, "top10": 0.0,
                         "spy": 0.0, "rand_mean": 0.0, "top15": "A"})
    seen = []

    def fake(ctx, holdings, horizon, lo, hi, **kw):
        seen.append((lo, hi))
        book = 10 if hi.year >= 2010 else 6
        return {"book": book, "floor": "halfgate", "stop": None, "cap": None, "vol_cut": None,
                "evidence": {"book": {"6": 1.0}}}
    monkeypatch.setattr(sel, "decide_strategy", fake)
    dec = roll.decisions(sel, None, "store", hold, 3, cadence=13, workers=1, log=lambda m: None)
    assert list(dec.columns) == ["lo", "hi", "book", "floor", "stop", "cap", "vol_cut", "evidence"]
    assert dec.index[0] == weeks[weeks >= weeks[0] + pd.DateOffset(years=3)][0]
    assert seen[0] == (dec.index[0] - pd.DateOffset(years=3), dec.index[0] - pd.Timedelta(days=35))
    assert (dec.book == 6).sum() > 0 and (dec.book == 10).sum() > 0


def test_settings_at_uses_the_decision_in_force_and_reads_nan_as_none():
    dec = pd.DataFrame({"book": [6, 10], "floor": ["60/40", "halfgate"],
                        "stop": [-0.25, np.nan], "cap": [2, None], "vol_cut": [None, "abs"]},
                       index=pd.to_datetime(["2010-01-08", "2011-01-07"]))
    assert sel.settings_at(dec, "2010-06-04") == (6, "60/40", -0.25, 2, None)
    assert sel.settings_at(dec, "2011-01-07") == (10, "halfgate", None, None, "abs")
    assert sel.settings_at(dec.drop(columns=["vol_cut"]), "2011-01-07") == (10, "halfgate", None, None, None)
    with pytest.raises(ValueError):
        sel.settings_at(dec, "2009-12-31")


def test_summarize_counts_changes_and_the_share_equal_to_the_spec():
    dec = pd.DataFrame({"book": [6, 6, 10, 10], "floor": ["halfgate"] * 4,
                        "stop": [None] * 4, "cap": [None, 2, None, None], "vol_cut": [None] * 4},
                       index=pd.date_range("2010-01-08", periods=4, freq="W-FRI"))
    s = roll.summarize(dec, {"book": 10, "floor": "halfgate", "stop": None, "cap": None, "vol_cut": None})
    assert (s["decisions"], s["changes"], s["share_equal_to_spec"]) == (4, 2, 0.5)
    assert s["mode"]["book"] in ("6", "10") and s["share_by_layer"]["cap"]["2"] == 0.25


def _rolling_json(path, name, first, weekly_rule, weekly_fixed, weekly_spy, spec):
    idx = pd.date_range("2008-01-04", periods=len(weekly_rule), freq="W-FRI")
    rec = {"variant": name, "spec_settings": spec,
           "summary": {"first": first, "decisions": 10, "changes": 2, "share_equal_to_spec": 0.5},
           "decisions": [{"t": "2016-01-08", "lo": "2008-01-04", "hi": "2015-12-04",
                          "book": 10, "floor": "halfgate", "stop": None, "cap": None, "vol_cut": None}],
           "weekly": {"rule": {str(d.date()): float(v) for d, v in zip(idx, weekly_rule)},
                      "fixed": {str(d.date()): float(v) for d, v in zip(idx, weekly_fixed)},
                      "spy": {str(d.date()): float(v) for d, v in zip(idx, weekly_spy)}}}
    path.write_text(json.dumps(rec))
    return path


def test_choose_is_the_argmax_on_the_common_span_and_refuses_a_late_rule(tmp_path, monkeypatch):
    n = 52 * 17
    wig = 0.001 * (-1.0) ** np.arange(n)                          # so no series is constant
    spec = {"book": 10, "floor": "halfgate", "stop": None, "cap": None, "vol_cut": None}
    a = _rolling_json(tmp_path / "a.json", "trailing_3_c1", "2008-01-04",
                      0.003 + wig, 0.002 + wig, 0.001 + wig, spec)
    b = _rolling_json(tmp_path / "b.json", "trailing_5_c1", "2008-01-04",
                      0.001 + wig, 0.002 + wig, 0.001 + wig, spec)
    led = []
    import stocks_ml.models.trials as trials
    monkeypatch.setattr(trials, "record_trials", lambda rows, **k: led.extend(rows))
    out = roll.choose([a, b], "2010-01-01", "2015-12-31", log=lambda m: None)
    assert out["choice"] == "trailing_3_c1" and out["graded_on"] == ["2010-01-01", "2015-12-31"]
    assert set(out["candidates"]) == {"trailing_3_c1", "trailing_5_c1"}
    assert set(out["one_look"]["rows"]) == {"trailing_3_c1 (rolling)", "spec settings, fixed", "sp500"}
    assert led and led[0]["kind"] == "rolling_lookback"
    assert (tmp_path / "lookback_choice_2010-2015.json").exists()
    late = _rolling_json(tmp_path / "c.json", "trailing_8_c1", "2011-01-07",
                         0.003 + wig, 0.002 + wig, 0.001 + wig, spec)
    with pytest.raises(SystemExit):
        roll.choose([a, late], "2010-01-01", "2015-12-31", log=lambda m: None)


# ---- the live side (2026-09-25) ----
def _walk_file(path, weeks, k, base=1.0):
    rows = [{"week": w, "ticker": t, **{f"c{c}": base * c * (i + 1) for c in range(1, k + 1)}}
            for w in weeks for i, t in enumerate("AB")]
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_history_from_walks_stores_the_ensemble_mean_and_a_later_file_wins(tmp_path):
    weeks = list(pd.date_range("2006-01-06", periods=4, freq="W-FRI"))
    a = _walk_file(tmp_path / "a.parquet", weeks[:3], 16)
    b = _walk_file(tmp_path / "b.parquet", weeks[2:], 4, base=10.0)      # overlaps week 3
    out = roll.history_from_walks([a, b], tmp_path / "h" / "preds.parquet", log=lambda m: None)
    h = pd.read_parquet(out)
    assert list(h.columns) == ["week", "ticker", "c1"] and h.week.nunique() == 4
    assert h[(h.week == weeks[0]) & (h.ticker == "A")].c1.iloc[0] == pytest.approx(np.mean(range(1, 17)))
    assert h[(h.week == weeks[2]) & (h.ticker == "A")].c1.iloc[0] == pytest.approx(10 * np.mean(range(1, 5)))   # b won


def test_append_history_replaces_a_rerun_week(tmp_path):
    p = tmp_path / "preds.parquet"
    t = pd.Timestamp("2026-09-18")
    roll.append_history(p, t, pd.Series({"A": 1.0, "B": 2.0}))
    h = roll.append_history(p, t, pd.Series({"A": 5.0, "B": 6.0}))
    assert len(h) == 2 and h.c1.tolist() == [5.0, 6.0]
    h = roll.append_history(p, t + pd.Timedelta(days=7), pd.Series({"A": 0.0}))
    assert h.week.nunique() == 2 and len(h) == 3


def test_is_due_counts_rank_weeks():
    assert roll.is_due(None, "2026-09-18", 4)
    assert not roll.is_due("2026-09-04", "2026-09-18", 4)               # 2 weeks
    assert roll.is_due("2026-08-21", "2026-09-18", 4)                   # 4 weeks
    assert roll.is_due("2026-08-21", "2026-09-17", 4)                   # a holiday Thursday is its own week


def test_decide_live_reads_the_trailing_window_and_the_rolling_stop_menu(monkeypatch):
    from stocks_ml.selection import label_end
    weeks = pd.date_range("2016-01-08", "2026-09-18", freq="W-FRI")
    hist = pd.DataFrame({"week": np.repeat(weeks, 2), "ticker": ["A", "B"] * len(weeks), "c1": 1.0})
    seen = {}
    def fake_hold(ctx, preds, copies, horizon, filt=None):
        seen["copies"], seen["weeks"] = list(copies), sorted(preds.week.unique())
        return pd.DataFrame({"week": sorted(preds.week.unique()), "top3": 0.01, "top6": 0.01, "top10": 0.01}), {}
    def fake_decide(ctx, hold, horizon, lo, hi, book_band=None, stop_menu=None):
        seen["lo"], seen["hi"], seen["stop_menu"] = lo, hi, stop_menu
        return {"book": 10, "floor": "60/40", "stop": None, "cap": None, "vol_cut": None, "evidence": {"book": {}}}
    fake_sel = SimpleNamespace(ensemble_holdings=fake_hold, decide_strategy=fake_decide)
    d = roll.decide_live(fake_sel, None, hist, "2026-09-18", 3)
    assert seen["copies"] == [1] and seen["stop_menu"] == roll.STOP_MENU_ROLLING == (None,)
    assert seen["lo"] == pd.Timestamp("2023-09-18") and seen["hi"] == label_end(pd.Timestamp("2026-09-18"), 4)
    assert seen["weeks"][0] >= seen["lo"] and seen["weeks"][-1] <= seen["hi"]     # nothing past label_end(t)
    assert d["book"] == 10 and d["decided"] == "2026-09-18" and d["weeks"] == len(seen["weeks"])
    with pytest.raises(SystemExit, match="shorter than the 3-year window"):
        roll.decide_live(fake_sel, None, hist[hist.week >= "2025-01-01"], "2026-09-18", 3)


def test_decide_live_refuses_a_window_with_a_hole_and_passes_the_filter(monkeypatch):
    weeks = pd.date_range("2016-01-08", "2026-09-18", freq="W-FRI")
    hist = pd.DataFrame({"week": np.repeat(weeks, 2), "ticker": ["A", "B"] * len(weeks), "c1": 1.0})
    seen = {}
    def fake_hold(ctx, preds, copies, horizon, filt=None):
        seen["filt"] = filt
        return pd.DataFrame({"week": sorted(preds.week.unique()), "top3": 0.01, "top6": 0.01, "top10": 0.01}), {}
    def fake_decide(ctx, hold, horizon, lo, hi, book_band=None, stop_menu=None):
        return {"book": 10, "floor": "60/40", "stop": None, "cap": None, "vol_cut": None, "evidence": {}}
    fake_sel = SimpleNamespace(ensemble_holdings=fake_hold, decide_strategy=fake_decide)
    ctx = SimpleNamespace(weeks=list(weeks))
    d = roll.decide_live(fake_sel, ctx, hist, "2026-09-18", 3, filt="x_dollar_vol:0.2:1")
    assert d["book"] == 10 and seen["filt"] == "x_dollar_vol:0.2:1"
    hole = hist[(hist.week < "2024-07-19") | (hist.week > "2026-06-01")]     # a skipped holdout walk
    with pytest.raises(SystemExit, match="below 90%"):
        roll.decide_live(fake_sel, ctx, hole, "2026-09-18", 3)


def test_adopt_writes_the_registered_choice_into_the_spec_and_check_sees_drift(tmp_path):
    import json
    rec = {"variant": "trailing_3_c4", "lookback_years": 3, "cadence": 4, "min_years": 3}
    (tmp_path / "trailing_3_c4.json").write_text(json.dumps(rec))
    choice = {"choice": "trailing_3_c4", "graded_on": ["2010-01-01", "2015-12-31"], "metric": "x", "common_weeks": 313,
              "candidates": {"trailing_3_c4": {"cagr_pct": 9.85, "sharpe": 0.53, "max_dd": 0.38, "decisions": 203, "changes": 42, "first": "2009-01-09"},
                             "expanding_c4": {"cagr_pct": 4.41, "sharpe": 0.3, "max_dd": 0.43, "decisions": 203, "changes": 21, "first": "2009-01-09"}},
              "one_look": {"paired_t_rolling_vs_fixed": 0.95, "decisions_2016_2024": {"changes": 18},
                           "rows": {"trailing_3_c4 (rolling)": {"2016-2024": {"terminal_100": 336.8, "cagr_pct": 15.21, "sharpe": 0.668, "max_dd": 0.4}},
                                    "sp500": {"2016-2024": {"terminal_100": 315.7, "cagr_pct": 14.34, "sharpe": 0.87, "max_dd": 0.32}}}},
              "inputs": {"trailing_3_c4": {"path": str(tmp_path / "trailing_3_c4.json")}}}
    cp = tmp_path / "lookback_choice_2010-2015.json"; cp.write_text(json.dumps(choice))
    spec_path = tmp_path / "spec.json"; spec_path.write_text(json.dumps({"strategy": {"book_size": 3}}))
    block = roll.adopt(cp, spec_path, log=lambda m: None)
    spec = json.loads(spec_path.read_text())
    assert spec["strategy"] == {"book_size": 3}                       # the fallback is untouched
    assert spec["rolling"]["variant"] == "trailing_3_c4" and spec["rolling"]["cadence"] == 4
    assert spec["rolling"]["stop_menu"] == [None] and spec["rolling"]["history"] == roll.HISTORY_FILE
    assert spec["rolling"]["one_look"]["rows"]["sp500"]["cagr_pct"] == 14.34
    assert roll.check(spec) == {}
    spec["rolling"]["cadence"] = 1                                      # a hand edit
    assert roll.check(spec) == {"cadence": (1, 4)}
    from stocks_ml.procedure_card import rolling_words
    w = rolling_words(spec)
    assert "trailing 3 years" in w and "$337" in w and "S&P 500 $316" in w
