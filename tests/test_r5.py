"""live/r5.py: the champion's weekly job around the shared ledger."""
import pandas as pd
import pytest

from stocks_ml.live import r5

D = pd.Timestamp
SMAP = {"A1": "Tech", "A2": "Tech", "A3": "Tech", "B1": "Fin", "B2": "Fin", "B3": "Fin",
        "C1": "Ind", "C2": "Ind", "D1": "Util", "E1": "Ener", "F1": "Cons", "G1": "Heal",
        "H1": "Mat", "I1": "Real", "J1": "Comm"}


def test_last_friday():
    assert r5.last_friday("2026-09-05") == D("2026-09-04")
    assert r5.last_friday("2026-09-04") == D("2026-09-04")
    assert r5.last_friday("2026-09-03") == D("2026-08-28")


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


def test_render_markdown_orders_equal_weights_by_ticker():
    """Rerun-is-a-no-op depends on a byte-identical report; set iteration
    order is hash-seeded, so equal-weight rows need a deterministic tie-break."""
    names = ["C1", "A1", "B1", "A2", "B2", "C2"]
    sig = {"date": "2026-08-28", "sleeve_due": 0, "rotated": [0],
           "sleeves": {"0": {"names": names, "since": "2026-08-28"}},
           "ballast": {"30": "SPY", "40": "SPY", "52": "SPY"},
           "weights": {**{n: 0.7 / 6 for n in names}, "SPY": 0.3},
           "nav": 100.0, "spy_nav": 100.0, "cash": 100.0, "held_value": {},
           "fills": [], "rebase_factors": {}, "top": [], "n_ranked": 6,
           "positions": {}, "freshness": {"panel": "2026-08-28"}, "elapsed_s": 1}
    md = r5.render_markdown(sig, SMAP)
    rows = [ln.split(" | ")[0].strip("| ") for ln in md.splitlines()
            if ln.startswith("| ") and ln.split(" | ")[0].strip("| ") in {*names, "SPY"}]
    assert rows == ["SPY", "A1", "A2", "B1", "B2", "C1", "C2"]


def test_commit_outputs_skips_when_nothing_changed():
    """A rerun on the same Friday regenerates identical files: no empty
    commit, no push; a real change is committed, rebased and pushed."""
    from types import SimpleNamespace

    from stocks_ml.cli import commit_outputs

    def fake_run(staged_changes):
        calls = []

        def run(cmd, **kw):
            calls.append(cmd[:2])
            rc = 1 if (cmd[:2] == ["git", "diff"] and staged_changes) else 0
            return SimpleNamespace(returncode=rc)
        return run, calls

    run, calls = fake_run(staged_changes=False)
    assert commit_outputs("2026-08-28", ["signals_r5", "ledger_r5.json"], run=run) is False
    assert calls == [["git", "add"], ["git", "diff"]]

    run, calls = fake_run(staged_changes=True)
    assert commit_outputs("2026-08-28", ["signals_r5", "ledger_r5.json"], run=run) is True
    assert calls == [["git", "add"], ["git", "diff"], ["git", "commit"], ["git", "pull"],
                     ["git", "push"]]


def test_live_spec_mirrors_the_champion_spec():
    # r5.SPEC is the live job's hard-coded copy of models/champion_spec.json:
    # the engine settings and the feature bundle must not drift apart
    import json
    from pathlib import Path

    from stocks_ml.features.bundle import FEATURES, FORMULAS
    from stocks_ml.selection import K_COPIES
    spec = json.loads((Path(__file__).resolve().parents[1] / "models/champion_spec.json").read_text())
    assert spec["ensemble"]["k_copies"] == K_COPIES == 16       # the live job averages K_COPIES copies
    assert list(r5.SPEC["features"]) == list(FEATURES) and spec["formulas"] == FORMULAS
    assert r5.SPEC["train_years"] == spec["training_window_years"]
    assert r5.SPEC["book"] == spec["strategy"]["book_size"]
    assert r5.SPEC["cap"] == spec["strategy"]["sector_cap"]
    assert r5.SPEC["horizon"] == "4w" and spec["horizon"]["label"] == "label_4w"
    assert r5.SPEC["floor"] == int(spec["ballast"]["mix"].split("%")[0]) / 100
    assert list(r5.SPEC["features"]) == list(spec["features"])
