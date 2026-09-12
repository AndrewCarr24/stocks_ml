"""The backtest procedure writes the deployed strategy; nothing else may.

models/champion_spec.json's strategy layers (book, floor, stop, cap) must
equal the decision recorded by `stocks-ml procedure`, and the live job's
SPEC must be read from that decision. A hand edit of either side fails here.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import stocks_ml.procedure as proc
import stocks_ml.selection as sel
from stocks_ml.procedure import (DECISION_KEYS, apply, drift, live_floor, live_strategy,
                                 load_walk, mix_label)

SPEC = json.loads((Path(__file__).resolve().parents[1] / "models/champion_spec.json").read_text())


def _spec():
    return copy.deepcopy(SPEC)


# ---- the spec on disk is what the procedure wrote ----
def test_spec_strategy_is_the_procedure_decision():
    dec = SPEC["procedure"]["decision"]
    assert set(dec) == set(DECISION_KEYS)
    for k in ("book_size", "stop_loss", "sector_cap"):
        assert SPEC["strategy"][k] == dec[k], f"strategy.{k} was edited by hand"
    assert SPEC["ballast"]["mix"] == mix_label(dec["floor"])
    assert dec["floor"] in SPEC["name"]
    assert dec["book_size"] in sel.BOOKS and dec["floor"] in sel.FLOORS
    assert dec["stop_loss"] in (None, -0.25) and dec["sector_cap"] in (None, 2)


def test_procedure_record_names_what_it_read():
    p = SPEC["procedure"]
    assert p["selection_window"] == ["2006-01-01", "2015-12-31"]          # selection only
    assert pd.Timestamp(p["preds"]["last"]) < sel.HOLDOUT_START
    assert pd.Timestamp(p["preds"]["last"]) <= pd.Timestamp(p["selection_window"][1])
    assert p["k_copies"] == sel.K_COPIES and p["horizon"] == "4w"
    # the model fields were read from the walk's own record, never typed
    assert p["model"]["label"] == SPEC["horizon"]["label"] in sel.LABELS_4W
    assert p["model"]["train_years"] == SPEC["training_window_years"]
    assert p["model"]["record"] == str(Path(p["preds"]["path"]).parent / "spec.json")
    assert SPEC["horizon"]["purge_days"] == sel.HORIZONS["4w"]["purge"]
    if Path(p["model"]["record"]).exists():                              # data/ is git-ignored
        rec = json.loads(Path(p["model"]["record"]).read_text())["recipe"]
        assert (rec["label"], rec["train_years"]) == (p["model"]["label"], p["model"]["train_years"])
    assert p["price_basis"] == SPEC["price_basis"] and p["delist_labels"] == SPEC["delist_labels"]
    assert p["world"] in SPEC["panel"]
    assert len(p["preds"]["sha256"]) == 64
    assert p["preds"]["rank_weeks"] >= 500                                # every week of 2006-2015
    ev = p["evidence"]
    dec = p["decision"]
    assert max(ev["book"], key=ev["book"].get) == str(dec["book_size"])   # the argmax, no override
    assert max(ev["floor"], key=ev["floor"].get) == dec["floor"]
    assert (ev["stop"]["None"] >= ev["stop"]["-0.25"]) == (dec["stop_loss"] is None)
    assert (ev["cap"]["None"] >= ev["cap"]["2"]) == (dec["sector_cap"] is None)


# ---- the live job reads the decision ----
def test_live_spec_is_read_from_the_procedure():
    from stocks_ml.features.bundle import FEATURES
    from stocks_ml.live import r5
    dec = SPEC["procedure"]["decision"]
    assert r5.SPEC["book"] == dec["book_size"] == SPEC["strategy"]["book_size"]
    assert r5.SPEC["cap"] == dec["sector_cap"] == SPEC["strategy"]["sector_cap"]
    assert r5.SPEC["floor"] == dec["floor"] and dec["floor"] in sel.FLOORS
    assert r5.SPEC["train_years"] == SPEC["training_window_years"] == SPEC["procedure"]["model"]["train_years"]
    assert r5.SPEC["horizon"] == "4w"
    assert r5.SPEC["label"] == SPEC["horizon"]["label"] == SPEC["procedure"]["model"]["label"]
    assert list(r5.SPEC["features"]) == list(FEATURES) == list(SPEC["features"])
    assert r5.load_spec(r5.spec_path()) == r5.SPEC


def test_hand_edits_fail_loudly():
    s = _spec()
    s["strategy"]["book_size"] = 3 if s["strategy"]["book_size"] != 3 else 10   # typed over the decision
    with pytest.raises(RuntimeError, match="edited by hand"):
        live_strategy(s)
    s = _spec()
    s["ballast"]["mix"] = "70% book / 30% ballast" if s["procedure"]["decision"]["floor"] != "70/30" \
        else "60% book / 40% ballast"
    with pytest.raises(AssertionError, match="ballast.mix"):
        live_floor(s)
    s = _spec()
    s["procedure"]["decision"]["floor"] = "halfgate"      # live runs it since 2026-09-12
    s["ballast"]["mix"] = mix_label("halfgate")
    assert live_floor(s) == "halfgate"
    s = _spec()
    s["procedure"]["decision"]["floor"] = "50/50"         # not on the menu
    s["ballast"]["mix"] = "50% book / 50% ballast"
    with pytest.raises(RuntimeError, match="cannot express"):
        live_floor(s)
    s = _spec()
    s["procedure"]["decision"]["stop_loss"] = s["strategy"]["stop_loss"] = -0.25
    with pytest.raises(RuntimeError, match="stop_loss"):
        live_strategy(s)
    s = _spec()
    s["training_window_years"] = s["procedure"]["model"]["train_years"] + 1   # typed over the record
    with pytest.raises(RuntimeError, match="train_years.*edited by hand"):
        live_strategy(s)
    s = _spec()
    s["horizon"]["label"] = [l for l in sel.LABELS_4W if l != s["horizon"]["label"]][0]
    with pytest.raises(RuntimeError, match="label.*edited by hand"):
        live_strategy(s)
    s = _spec()
    s["horizon"]["label"] = s["procedure"]["model"]["label"] = "label_4w_gauss"   # not a live target
    with pytest.raises(RuntimeError, match="cannot train"):
        live_strategy(s)
    s = _spec()
    del s["procedure"]["model"]
    with pytest.raises(RuntimeError, match="names no model"):
        live_strategy(s)


def test_mix_label_and_floor_fraction_agree():
    for floor, frac in sel.FLOOR_FRACTION.items():
        assert mix_label(floor) == f"{round(frac * 100)}% book / {100 - round(frac * 100)}% ballast"
        assert sel.floor_split(floor, {"30w": "SPY"})[0] == frac
    assert mix_label("halfgate").startswith("halfgate") and "IEF" in mix_label("halfgate")
    assert mix_label("none").startswith("100% book")
    assert len({mix_label(f) for f in sel.FLOORS}) == len(sel.FLOORS)
    with pytest.raises(KeyError):
        mix_label("50/50")                                # not on the menu


# ---- apply / drift ----
def test_apply_writes_every_decision_field_and_nothing_prose():
    s = _spec()
    rec = {"decision": {"book_size": 3, "floor": "80/20", "stop_loss": -0.25, "sector_cap": None},
           "model": {"label": "label_4w", "train_years": 2, "record": "w/spec.json"},
           "evidence": {}, "decided_at": "x"}
    before = s["provenance"], s["strategy"]["settings_note"], s["training_window_rationale"]
    out = apply(s, rec)
    assert out["strategy"]["book_size"] == 3 and out["strategy"]["stop_loss"] == -0.25
    assert out["strategy"]["sector_cap"] is None
    assert out["ballast"]["mix"] == "80% book / 20% ballast" and "80/20" in out["name"]
    assert out["horizon"]["label"] == "label_4w" and out["training_window_years"] == 2
    assert out["horizon"]["purge_days"] == sel.HORIZONS["4w"]["purge"]
    assert out["procedure"] is rec
    assert (out["provenance"], out["strategy"]["settings_note"],
            out["training_window_rationale"]) == before
    assert drift(out, rec) == {}
    want = {k: (SPEC["procedure"]["decision"][k], v) for k, v in rec["decision"].items()
            if SPEC["procedure"]["decision"][k] != v}
    if SPEC["horizon"]["label"] != "label_4w":
        want["label"] = (SPEC["horizon"]["label"], "label_4w")
    if SPEC["training_window_years"] != 2:
        want["train_years"] = (SPEC["training_window_years"], 2)
    assert drift(SPEC, rec) == want


# ---- the walk's record names the model ----
def test_walk_recipe_is_read_from_the_record_beside_the_walk(tmp_path):
    from stocks_ml.procedure import walk_recipe
    preds = tmp_path / "preds.parquet"
    with pytest.raises(RuntimeError, match="missing"):
        walk_recipe(preds)
    rec = tmp_path / "spec.json"
    rec.write_text(json.dumps({"k": 16, "store": "w"}))                 # a walk before recipes
    with pytest.raises(RuntimeError, match="names no recipe"):
        walk_recipe(preds)
    rec.write_text(json.dumps({"k": 16, "recipe": {"label": "label_4w_gauss", "train_years": 8}}))
    with pytest.raises(RuntimeError, match="cannot train on"):
        walk_recipe(preds)
    rec.write_text(json.dumps({"k": 16, "recipe": {"label": "label_4w_sector", "train_years": 8}}))
    assert walk_recipe(preds) == {"label": "label_4w_sector", "train_years": 8, "record": str(rec)}


# ---- the walk the procedure may read ----
def _walk(weeks, k=2, tickers=("A", "B", "C")):
    rows = [{"week": t, "ticker": tk, **{f"c{c}": float(i + c) for c in range(1, k + 1)}}
            for t in weeks for i, tk in enumerate(tickers)]
    return pd.DataFrame(rows)


def test_load_walk_refuses_samples_missing_copies_and_the_holdout(tmp_path):
    weeks = list(pd.date_range("2006-01-06", periods=8, freq="W-FRI"))
    lo, hi = weeks[0], weeks[-1]
    p = tmp_path / "preds.parquet"
    _walk(weeks).to_parquet(p, index=False)
    got = load_walk(p, 2, weeks, lo, hi)
    assert list(got.columns) == ["week", "ticker", "c1", "c2"] and got.week.nunique() == 8
    with pytest.raises(RuntimeError, match="lacks copies"):
        load_walk(p, 3, weeks, lo, hi)
    _walk(weeks[::2]).to_parquet(p, index=False)                     # a spaced sample
    with pytest.raises(RuntimeError, match="no frozen decision reads a sample"):
        load_walk(p, 2, weeks, lo, hi)
    _walk(weeks + [sel.HOLDOUT_START]).to_parquet(p, index=False)
    with pytest.raises(RuntimeError, match="holdout"):
        load_walk(p, 2, weeks, lo, hi)
    _walk(weeks + [hi + pd.Timedelta(days=7)]).to_parquet(p, index=False)
    assert load_walk(p, 2, weeks, lo, hi).week.max() == hi           # past the window: dropped


# ---- the layers are the argmax, stop and cap only if higher ----
def test_decide_strategy_is_the_argmax_of_its_menus(monkeypatch):
    weeks = pd.date_range("2006-01-06", periods=120, freq="W-FRI")
    hold = pd.DataFrame({"week": weeks, "top3": 0.010, "top6": 0.020, "top10": 0.015,
                         "spy": 0.0, "rand_mean": 0.0, "top15": "A"})
    sr = {("none", None, None): 0.30, ("60/40", None, None): 0.50, ("70/30", None, None): 0.45,
          ("80/20", None, None): 0.40, ("halfgate", None, None): 0.35,
          ("60/40", -0.25, None): 0.50, ("60/40", None, 2): 0.49}
    calls = []

    def fake_sim(ctx, holdings, horizon, book, cap, stop, floor, trace=None):
        calls.append((book, floor, stop, cap))
        s = sr[(floor, stop, cap)]
        return pd.Series(np.full(len(weeks), s / np.sqrt(52)), index=weeks)  # mean, sd 1 -> SR s
    monkeypatch.setattr(sel, "simulate", fake_sim)
    monkeypatch.setattr(sel, "sharpe", lambda series, lo, hi: float(series.mean() * np.sqrt(52)))
    got = sel.decide_strategy(None, hold, "4w", weeks[0], weeks[-1])
    assert (got["book"], got["floor"], got["stop"], got["cap"]) == (6, "60/40", None, None)
    assert all(b == 6 for b, *_ in calls)                                  # layers below read the book
    assert got["evidence"]["floor"]["60/40"] == 0.5 and got["evidence"]["cap"]["2"] == 0.49
    assert set(got["evidence"]) == {"book", "floor", "stop", "cap"}


def test_run_cascade_layers_come_from_decide_strategy(monkeypatch):
    """run_cascade's book-down block is decide_strategy: same fields, same evidence."""
    seen = {}

    def fake(ctx, holdings, horizon, lo, hi):
        seen["args"] = (horizon, lo, hi)
        return {"book": 3, "floor": "80/20", "stop": -0.25, "cap": 2,
                "evidence": {"book": {"3": 1.0}, "floor": {}, "stop": {}, "cap": {}}}
    monkeypatch.setattr(sel, "decide_strategy", fake)
    monkeypatch.setattr(sel, "decide_horizon", lambda grids, lo, hi: ("4w", {}))
    monkeypatch.setattr(sel, "load_windows", lambda *a, **k: {})
    monkeypatch.setattr(sel, "decide_window", lambda sweeps, lo, hi: (5, {}))
    monkeypatch.setattr(sel, "_load", lambda out, pat: pd.DataFrame({"week": []}))
    import stocks_ml.models.trials as trials
    monkeypatch.setattr(trials, "record_trials", lambda rows: None)
    cfg = sel.run_cascade(None, Path("."), pd.Timestamp("2006-01-01"), pd.Timestamp("2015-12-31"))
    assert (cfg["book"], cfg["floor"], cfg["stop"], cfg["cap"]) == (3, "80/20", -0.25, 2)
    assert cfg["evidence"]["book"] == {"3": 1.0} and seen["args"][0] == "4w"


def test_procedure_run_writes_spec_card_and_ledger(tmp_path, monkeypatch):
    spec_path = tmp_path / "champion_spec.json"
    spec_path.write_text(json.dumps(_spec()))
    rec = {"decision": {"book_size": 3, "floor": "80/20", "stop_loss": None, "sector_cap": 2},
           "model": {"label": "label_4w_sector", "train_years": 8, "record": "w/spec.json"},
           "evidence": {"book": {"3": 2.0}}, "preds": {"sha256": "ab" * 32, "path": "p",
                                                       "rank_weeks": 522, "first": "2006-01-06",
                                                       "last": "2015-12-25"},
           "selection_window": ["2006-01-01", "2015-12-31"], "world": "w", "k_copies": 16,
           "decided_at": "2026-09-11 14:00:00",
           "selection_window_record": {"config": {"sharpe": 0.5}, "sp500": {"sharpe": 0.4}}}
    monkeypatch.setattr(proc, "decide", lambda *a, **k: rec)
    led = []
    import stocks_ml.models.trials as trials
    monkeypatch.setattr(trials, "record_trials", lambda rows: led.extend(rows))
    card = tmp_path / "PROCEDURE.md"
    with pytest.raises(SystemExit, match="drifted"):                      # --check refuses to write
        proc.run("p", check=True, spec_path=spec_path, card_path=card, log=lambda m: None)
    assert json.loads(spec_path.read_text())["strategy"]["book_size"] == SPEC["strategy"]["book_size"]
    assert not card.exists()
    proc.run("p", spec_path=spec_path, card_path=card, log=lambda m: None)
    s = json.loads(spec_path.read_text())
    assert s["strategy"]["book_size"] == 3 and s["ballast"]["mix"] == "80% book / 20% ballast"
    assert s["horizon"]["label"] == "label_4w_sector" and s["training_window_years"] == 8
    assert s["procedure"] == rec and live_strategy(s)["floor"] == "80/20"
    assert live_strategy(s)["label"] == "label_4w_sector" and live_strategy(s)["train_years"] == 8
    text = card.read_text()
    assert "top-3" in text and "book 3 / floor 80/20" in text
    assert "trailing 8 years" in text and "label_4w_sector / 8-year window" in text
    assert "median of its sector" in text
    assert led[0]["kind"] == "procedure" and led[0]["pre_holdout_sharpe"] == 0.5
    proc.run("p", check=True, spec_path=spec_path, card_path=card, log=lambda m: None)  # matches now
