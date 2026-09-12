"""live/r5.py: the champion's weekly job around the shared ledger."""
import pandas as pd
import pytest

from stocks_ml.live import r5
from stocks_ml.procedure import mix_label

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


def test_live_weights_follow_the_floor_menu(monkeypatch):
    """The live job parks the money exactly as the backtest's floor_split
    does, for every floor on the menu — halfgate included (2026-09-12)."""
    from stocks_ml.ledger import floor_split, target_weights
    sleeves = {"0": {"names": ["A1", "B1"]}, "1": {"names": ["C1", "D1"]}}
    gates = {"30": "IEF", "40": "IEF", "52": "SPY"}
    for floor, book, spy, ief in (("halfgate", 2 / 3, 0.0, 1 / 3), ("60/40", 0.6, 0.4 / 3, 0.8 / 3),
                                  ("none", 1.0, 0.0, 0.0)):
        monkeypatch.setitem(r5.SPEC, "floor", floor)
        frac, weights = r5.book_weights(sleeves, gates)
        assert frac == pytest.approx(book)
        assert weights == pytest.approx(target_weights(sleeves, *reversed(floor_split(floor, gates))))
        assert weights.get("SPY", 0.0) == pytest.approx(spy) and weights.get("IEF", 0.0) == pytest.approx(ief)
        assert sum(weights.values()) == pytest.approx(1.0)


def test_render_markdown_names_the_floor_and_this_weeks_fraction(monkeypatch):
    sig = {"date": "2026-08-28", "sleeve_due": 2, "rotated": [0],
           "sleeves": {"0": {"names": ["A1"], "since": "2026-08-28"}},
           "ballast": {"30": "IEF", "40": "SPY", "52": "SPY"}, "book_fraction": 5 / 6,
           "weights": {"A1": 5 / 6, "IEF": 1 / 6}, "nav": 100.0, "spy_nav": 100.0,
           "cash": 100.0, "held_value": {}, "fills": [], "rebase_factors": {}, "top": [],
           "n_ranked": 480, "positions": {}, "freshness": {"panel": "2026-08-28"}, "elapsed_s": 1}
    monkeypatch.setitem(r5.SPEC, "floor", "halfgate")
    md = r5.render_markdown(sig, SMAP)
    assert "halfgate" in md and "book 83% of NAV this week" in md and "30w→IEF" in md
    monkeypatch.setitem(r5.SPEC, "floor", "60/40")
    sig["book_fraction"] = 0.6
    assert "60/40" in r5.render_markdown(sig, SMAP)


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
    # r5.SPEC is read from models/champion_spec.json (r5.load_spec, the procedure.s decision):
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
    assert r5.SPEC["horizon"] == "4w" and r5.SPEC["label"] == spec["horizon"]["label"]
    assert spec["horizon"]["label"] == spec["procedure"]["model"]["label"]
    assert spec["training_window_years"] == spec["procedure"]["model"]["train_years"]
    assert r5.SPEC["floor"] == spec["procedure"]["decision"]["floor"]
    assert spec["ballast"]["mix"] == mix_label(r5.SPEC["floor"])
    assert list(r5.SPEC["features"]) == list(spec["features"])


# ---- freshness: holiday Fridays and --as-of (2026-09 bug hunt) ----
def test_nyse_friday_holiday_calendar():
    cases = {"2026-12-25": True,    # Christmas on a Friday
             "2027-01-01": True,    # New Year on a Friday
             "2027-03-26": True,    # Good Friday
             "2026-04-03": True,    # Good Friday
             "2021-12-24": True,    # Christmas Saturday, observed Friday
             "2020-07-03": True,    # July 4 Saturday, observed Friday
             "2026-06-19": True,    # Juneteenth on a Friday
             "2015-06-19": False,   # Juneteenth before it was a holiday
             "2021-12-31": False,   # Jan 1 Saturday: NYSE stays open over year end
             "2020-11-27": False,   # day after Thanksgiving: open (half day)
             "2026-12-24": False,   # a Thursday, whatever else it is
             "2026-09-04": False}
    for d, want in cases.items():
        assert r5.nyse_friday_holiday(d) is want, d


def test_latest_complete_week_accepts_a_holiday_thursday(monkeypatch):
    monkeypatch.setattr(r5, "last_friday", lambda today=None: D("2026-12-25"))
    weeks = pd.DatetimeIndex(["2026-12-11", "2026-12-18", "2026-12-24"])
    assert r5._latest_complete_week(weeks) == D("2026-12-24")


def test_latest_complete_week_rejects_a_stale_store(monkeypatch):
    monkeypatch.setattr(r5, "last_friday", lambda today=None: D("2026-12-11"))
    with pytest.raises(RuntimeError, match="not refreshed"):
        r5._latest_complete_week(pd.DatetimeIndex(["2026-12-04", "2026-12-10"]))
    # a Thursday row on an ordinary week is stale, not a holiday
    monkeypatch.setattr(r5, "last_friday", lambda today=None: D("2026-12-18"))
    with pytest.raises(RuntimeError, match="not refreshed"):
        r5._latest_complete_week(pd.DatetimeIndex(["2026-12-11", "2026-12-17"]))


def test_latest_complete_week_skips_a_partial_midweek_row(monkeypatch):
    monkeypatch.setattr(r5, "last_friday", lambda today=None: D("2026-12-11"))
    weeks = pd.DatetimeIndex(["2026-12-04", "2026-12-11", "2026-12-15"])   # Tuesday: partial week
    assert r5._latest_complete_week(weeks) == D("2026-12-11")


def test_guard_as_of_refuses_history_rewrites():
    from stocks_ml.ledger import Ledger
    led = Ledger.new(100.0, "2026-08-28")
    led.nav_history = [["2026-09-04", 100.0, 100.0]]
    led.pending = {"decision_date": "2026-09-04", "weights": {}}
    with pytest.raises(RuntimeError, match="as-of"):
        r5._guard_as_of(led, D("2026-08-28"), "2026-08-28", False)
    r5._guard_as_of(led, D("2026-08-28"), "2026-08-28", True)      # dry runs may look back
    r5._guard_as_of(led, D("2026-09-04"), "2026-09-04", False)     # same-week rerun is fine
    r5._guard_as_of(led, D("2026-09-11"), None, False)             # the scheduled run
