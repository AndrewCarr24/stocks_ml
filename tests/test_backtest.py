"""stocks_ml.backtest: loading walks, the settings, the owner's table."""
import json

import numpy as np
import pandas as pd
import pytest

import stocks_ml.backtest as bt
import stocks_ml.selection as sel


def _walk(weeks, k=2, tickers=("A", "B", "C")):
    rows = [{"week": t, "ticker": tk, **{f"c{c}": float(i + c) for c in range(1, k + 1)}}
            for t in weeks for i, tk in enumerate(tickers)]
    return pd.DataFrame(rows)


def test_load_preds_joins_segments_sorted_and_refuses_the_holdout(tmp_path):
    a = list(pd.date_range("2006-01-06", periods=3, freq="W-FRI"))
    b = list(pd.date_range("2016-01-08", periods=3, freq="W-FRI"))
    (tmp_path / "select").mkdir()
    (tmp_path / "extend").mkdir()
    _walk(b).to_parquet(tmp_path / "extend/preds.parquet", index=False)
    _walk(a).to_parquet(tmp_path / "select/preds.parquet", index=False)
    df = bt.load_preds([tmp_path / "extend/preds.parquet", tmp_path / "select/preds.parquet"])
    assert list(df.week.unique()) == [np.datetime64(w) for w in a + b]
    assert bt.copies_in(df) == 2
    assert bt.walk_records([tmp_path / "select/preds.parquet"]) == [{}]
    (tmp_path / "select/spec.json").write_text(json.dumps({"recipe": {"label": "label_4w", "train_years": 5}}))
    assert bt.walk_records([tmp_path / "select/preds.parquet"])[0]["recipe"]["train_years"] == 5
    _walk(a + [sel.HOLDOUT_START]).to_parquet(tmp_path / "select/preds.parquet", index=False)
    with pytest.raises(RuntimeError, match="holdout"):
        bt.load_preds([tmp_path / "select/preds.parquet"])


def test_spec_settings_are_the_procedure_decision(tmp_path):
    st = bt.spec_settings()
    spec = json.loads(bt.SPEC_PATH.read_text())
    d = spec["procedure"]["decision"]
    assert st == dict(book=d["book_size"], floor=d["floor"], stop=d["stop_loss"], cap=d["sector_cap"])
    assert st["book"] == spec["strategy"]["book_size"]
    assert bt.settings_label(dict(book=10, floor="halfgate", stop=None, cap=None)) == \
        "top-10 / halfgate / stop None / cap None"


def test_spans_are_the_three_windows_ending_at_the_holdout():
    assert bt.SPANS["2006-2015"] == (pd.Timestamp("2006-01-01"), pd.Timestamp("2016-01-01"))
    assert bt.SPANS["2016-2024"] == (pd.Timestamp("2016-01-01"), sel.HOLDOUT_START)
    assert bt.SPANS["2006-2024"] == (pd.Timestamp("2006-01-01"), sel.HOLDOUT_START)


def test_paired_t_and_row_vs_spy():
    weeks = pd.date_range("2006-01-06", periods=52 * 20, freq="W-FRI")
    rng = np.random.default_rng(0)
    spy = pd.Series(rng.normal(0.002, 0.02, len(weeks)), index=weeks)
    r = spy + 0.001
    assert bt.paired_t(r - spy) > 1e6                                   # a constant edge: huge t
    assert np.isnan(bt.paired_t(pd.Series([0.1, 0.2])))
    row = bt.row_vs_spy(sel, r, spy)
    assert set(row) == {"2006-2015", "2016-2024", "2006-2024", "paired_t_vs_sp500"}
    assert row["2006-2024"]["terminal_100"] > 100 and set(row["paired_t_vs_sp500"]) == set(bt.SPANS)


def test_table_md_always_carries_the_sp500_row():
    m = {"terminal_100": 1994.0, "cagr_pct": 17.5, "sharpe": 0.70, "max_dd": 0.63}
    rows = {"champion": {"2006-2024": m, "2016-2024": m, "2006-2015": m,
                         "paired_t_vs_sp500": {"2006-2024": 1.9, "2016-2024": 1.52, "2006-2015": 0.9}},
            "sp500": {"2006-2024": m, "2016-2024": m, "2006-2015": m}}
    md = bt.table_md(rows)
    assert md[0].startswith("| model | 2006-2024: $100, %/yr | SR, DD | 2016-2024")
    assert md[0].endswith("| weekly t vs sp500 (06-24 / 16-24) |")
    assert md[2] == "| champion | $1,994, +17.5% | 0.70, 63% | $1,994, +17.5% | 0.70, 63% | $1,994, +17.5% | 0.70, 63% | +1.90 / +1.52 |"
    assert md[3].endswith("| — |")
    with pytest.raises(ValueError, match="sp500"):
        bt.table_md({"champion": rows["champion"]})
    two = bt.table_md(rows, spans=("2006-2015",))
    assert two[0].count("|") == 5 and "(06-15" in two[0]


def test_selection_metric_is_decide_book_on_the_selection_window_only():
    weeks = pd.date_range("2006-01-06", "2016-12-30", freq="W-FRI")
    rng = np.random.default_rng(0)
    hold = pd.DataFrame({"week": weeks, "spy": 0.0, "rand_mean": 0.0,
                         "top3": rng.normal(0.01, 0.02, len(weeks)),
                         "top6": rng.normal(0.02, 0.02, len(weeks)),
                         "top10": rng.normal(0.03, 0.02, len(weeks))})
    m = bt.selection_metric(sel, hold)
    assert set(m) == set(sel.BOOKS)
    inside = hold[(hold.week >= bt.SELECT_START) & (hold.week <= bt.SELECT_END)]
    _, res = sel.decide_book(inside, "4w", bt.SELECT_START, bt.SELECT_END)
    assert m == {int(k): round(float(v), 2) for k, v in res.items()}
    # the 2016 weeks change nothing; a walk that misses the window has no metric
    assert bt.selection_metric(sel, inside) == m
    assert bt.selection_metric(sel, hold[hold.week >= bt.EXTEND_START]) == {}


def test_only_the_champions_walk_gets_the_spec_settings(tmp_path):
    import stocks_ml.selection as sel
    spec = {"horizon": {"label": "label_4w_sector"}, "training_window_years": 8, "features": [],
            "model": {"params": dict(sel.MODEL_PARAMS)}}
    sp = tmp_path / "spec.json"
    sp.write_text(json.dumps(spec))
    champion = [{"recipe": {"label": "label_4w_sector", "train_years": 8}}] * 2
    assert bt.is_champion_walk(champion, sp)
    assert not bt.is_champion_walk([{"recipe": {"label": "label_4w_sector_rank", "train_years": 8}}], sp)
    assert not bt.is_champion_walk([{"recipe": {"label": "label_4w_sector", "train_years": 5}}], sp)
    assert not bt.is_champion_walk([{"recipe": {"label": "label_4w_sector", "train_years": 8,
                                                "params": {"max_depth": 4}}}], sp)
    assert not bt.is_champion_walk([{}], sp) and not bt.is_champion_walk([], sp)


def test_own_settings_refuse_a_walk_outside_the_selection_window():
    hold = pd.DataFrame({"week": pd.date_range("2016-01-08", periods=100, freq="W-FRI"), "top3": 0.0})
    with pytest.raises(SystemExit):
        bt.own_settings(None, None, hold)
