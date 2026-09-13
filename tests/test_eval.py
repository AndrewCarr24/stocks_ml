"""stocks_ml.eval: a walk's settings come from the procedure's code; the
falsification rule; the record the run writes."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import stocks_ml.eval as ev
import stocks_ml.selection as sel


def test_spec_walk_is_the_procedure_walks_parent():
    spec = json.loads(ev.SPEC_PATH.read_text())
    walk = ev.spec_walk()
    assert Path(spec["procedure"]["preds"]["path"]) == walk / "select" / "preds.parquet"


def test_segments_require_both_halves(tmp_path):
    with pytest.raises(SystemExit, match="lacks"):
        ev.segments(tmp_path)
    for seg in ("select", "extend"):
        (tmp_path / seg).mkdir()
        (tmp_path / seg / "preds.parquet").write_bytes(b"")
    assert ev.segments(tmp_path) == [tmp_path / "select/preds.parquet", tmp_path / "extend/preds.parquet"]


def test_walk_settings_are_decided_once_and_cached_by_the_files_hash(tmp_path, monkeypatch):
    (tmp_path / "select").mkdir()
    p = tmp_path / "select/preds.parquet"
    p.write_bytes(b"walk-1")
    sha = hashlib.sha256(b"walk-1").hexdigest()
    calls = []

    def fake_decide(path, store, k, log=None):
        calls.append(path)
        return {"decision": {"book_size": 10, "floor": "halfgate", "stop_loss": None, "sector_cap": None},
                "evidence": {}, "preds": {"sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()},
                "decided_at": "now", "k_copies": k or 16}
    import stocks_ml.procedure as proc
    monkeypatch.setattr(proc, "decide", fake_decide)
    got = ev.walk_settings(tmp_path, "w", 16, log=lambda m: None)
    assert got["preds"]["sha256"] == sha and (tmp_path / "procedure.json").exists()
    ev.walk_settings(tmp_path, "w", 16, log=lambda m: None)
    assert len(calls) == 1                                              # cached
    ev.walk_settings(tmp_path, "w", 4, log=lambda m: None)              # another K decides again
    assert len(calls) == 2
    p.write_bytes(b"walk-2")                                            # the file changed: decide again
    ev.walk_settings(tmp_path, "w", 4, log=lambda m: None)
    assert len(calls) == 3
    assert ev.settings_of(got) == dict(book=10, floor="halfgate", stop=None, cap=None)
    assert ev.who({"recipe": {"label": "label_4w_sector", "train_years": 8}}, ev.settings_of(got)) == \
        "sector-relative 4w label / 8y / top-10 / halfgate / cap none"


def test_falsification_reads_2016_2024_at_minus_two():
    weeks = pd.date_range("2006-01-06", periods=52 * 19, freq="W-FRI")
    inc = pd.Series(0.0, index=weeks)
    pkg = pd.Series(np.where(weeks >= "2016-01-01", -0.01, 0.01), index=weeks)
    pkg.iloc[-1] += 1e-6                                                # a hair of variance
    f = ev.falsification(pkg, inc)
    assert f["2016-2024"]["t"] < -2 and f["verdict"] == "REJECTED"
    assert f["2006-2015"]["t"] > 2
    assert f["2016-2024"]["weeks"] == int(((weeks >= "2016-01-01") & (weeks < sel.HOLDOUT_START)).sum())
    assert f["rule"] == "paired weekly excess vs the incumbent on 2016-2024, t < -2.0 rejects"
    assert ev.falsification(-pkg, inc)["verdict"] == "not rejected"


def test_check_complete_wants_every_rank_week_and_k_copies():
    from types import SimpleNamespace
    weeks = list(pd.date_range("2006-01-06", periods=4, freq="W-FRI"))
    ctx = SimpleNamespace(weeks=weeks)
    preds = pd.DataFrame({"week": weeks * 2, "ticker": ["A"] * 4 + ["B"] * 4, "c1": 1.0, "c2": 2.0})
    ev.check_complete(ctx, preds, 2, weeks[0], weeks[-1] + pd.Timedelta(days=1))
    with pytest.raises(RuntimeError, match="copies"):
        ev.check_complete(ctx, preds, 3, weeks[0], weeks[-1] + pd.Timedelta(days=1))
    with pytest.raises(RuntimeError, match="rank weeks"):
        ev.check_complete(ctx, preds[preds.week != weeks[2]], 2, weeks[0], weeks[-1] + pd.Timedelta(days=1))


def test_report_md_carries_the_table_the_verdicts_and_the_intervals():
    m = {"terminal_100": 660.0, "cagr_pct": 24.6, "sharpe": 0.88, "max_dd": 0.33}
    ci_w = {"weeks": 446, "p_excess_positive_nested": 0.91,
            **{nm: {"point": 9.0, "seed_95": [7.3, 10.5], "history_95": [-3.9, 24.8], "nested_95": [-3.9, 24.2]}
               for nm in ("terminal_100", "cagr_pct", "excess_cagr_vs_sp500_pct")}}
    res = {"label": "champion", "walk": "w", "who": "x", "k": 16, "evaluated_at": "t",
           "settings": dict(book=10, floor="halfgate", stop=None, cap=None),
           "table": {"champion": {"2006-2024": m, "2016-2024": m, "2006-2015": m,
                                  "paired_t_vs_sp500": {"2006-2024": 1.9, "2016-2024": 1.52}},
                     "sp500": {"2006-2024": m, "2016-2024": m, "2006-2015": m}},
           "falsification": {"2016-2024": {"t": 1.32, "weeks": 446}, "rule": "r", "verdict": "not rejected"},
           "edge_over_incumbent_cagr_pp": {"2006-2015": 2.1, "2016-2024": 6.4, "2006-2024": 4.5},
           "leak_audit": {"VERDICT": "PASS", "segments": {}},
           "confidence": {"2016-2024": ci_w, "2006-2024": ci_w, "2006-2015": ci_w,
                          "method": {"seed_draws": 200, "history_draws": 4000, "nested_per_seed": 20,
                                     "block_weeks": 8, "rng_seed": 1}},
           "charts": ["reports/champion_vs_sp500_2006_2024.png"]}
    md = "\n".join(ev.report_md(res))
    assert md.startswith("# champion: the one look")
    assert "| sp500 |" in md and "t +1.32 on 446 weeks -> **not rejected**" in md
    assert "Leak audit: PASS." in md
    assert "| 2016-2024 | excess CAGR vs sp500 %/yr | +9.0 | +7.3 … +10.5 | -3.9 … +24.8 | -3.9 … +24.2 |" in md
    assert "| 2016-2024 | P(excess > 0), nested | 0.91 |" in md
    assert "![champion vs sp500](champion_vs_sp500_2006_2024.png)" in md


def test_run_is_the_champion_only_when_the_walk_is_the_specs(tmp_path, monkeypatch):
    """A challenger's label is its directory name; the champion's settings
    must equal the spec's, else the spec is stale."""
    walk = tmp_path / "ch"
    for seg in ("select", "extend"):
        (walk / seg).mkdir(parents=True)
        (walk / seg / "preds.parquet").write_bytes(b"x")
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps({"procedure": {"preds": {"path": str(walk / "select/preds.parquet")},
                                              "decision": {"book_size": 6, "floor": "60/40",
                                                           "stop_loss": None, "sector_cap": 2}}}))
    monkeypatch.setattr(ev, "walk_records", lambda paths: [{"recipe": {"label": "label_4w", "train_years": 5}}])
    monkeypatch.setattr(ev, "walk_settings", lambda *a, **k: {
        "decision": {"book_size": 10, "floor": "halfgate", "stop_loss": None, "sector_cap": None}})
    with pytest.raises(SystemExit, match="run stocks-ml procedure"):
        ev.run(walk, store="w", spec_path=spec, log=lambda m: None)
