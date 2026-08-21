"""The formal leaderboard (owner-specified 2026-08-21).

Format contract:
  * Ranking metric: PRE-TAX EARNINGS (terminal $ from 100), SR as tiebreak.
    (Supersedes the SR-ranked rule of iron rule 6; owner: "SR weights risk
    too heavily.")
  * Window: always 2001-01-05 -> 2024-07-19 (the extended pre-holdout).
    Holdout columns only when explicitly requested.
  * Rows: top THREE configs of the #1 model, top TWO of the #2 model, ONE of
    the #3 model (models ranked by their best config's earnings), plus the
    sp500 benchmark row, always.
  * Entries must be K-copy standardized ensemble books (replication.py);
    nothing else is rankable.

Books live in data/leaderboard_books/ with manifest.json rows:
  {"model": ..., "config": ..., "book": <filename>}
Run via `stocks-ml leaderboard [--holdout]`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REGISTRY = Path("data/leaderboard_books")
WINDOW_START = pd.Timestamp("2001-01-05")
HOLDOUT_START = pd.Timestamp("2024-07-19")


def rank_select(entries: list[dict]) -> list[dict]:
    """Owner's 3-2-1 selection: entries need model/config/earnings/sr keys."""
    by_model: dict = {}
    for e in entries:
        by_model.setdefault(e["model"], []).append(e)
    for rows in by_model.values():
        rows.sort(key=lambda r: (-r["earnings"], -r["sr"]))
    models = sorted(by_model, key=lambda m: (-by_model[m][0]["earnings"],
                                             -by_model[m][0]["sr"]))
    take = [3, 2, 1]
    out = []
    for i, m in enumerate(models[:3]):
        out.extend(by_model[m][:take[i]])
    return out


def _grade(nav: pd.Series, lo, hi) -> dict:
    from stocks_ml.backtest.metrics import summarize

    s = nav[(nav.index >= lo) & (nav.index <= hi)]
    s = 100 * s / s.iloc[0]
    m = summarize(s)
    return {"earnings": float(m["terminal_100"]), "sr": float(m["sharpe"]),
            "dd": float(m["max_drawdown"])}


def build_leaderboard(holdout: bool = False, registry: Path = REGISTRY) -> str:
    from stocks_ml.backtest.blend import run_weights_backtest
    from stocks_ml.config import load_config
    from stocks_ml.data.store import DataStore

    cfg = load_config()
    prices = DataStore("data/sharadar_world2000").read("prices")
    manifest = json.loads((registry / "manifest.json").read_text())
    entries = []
    for row in manifest:
        book = pd.read_parquet(registry / row["book"])
        nav, _ = run_weights_backtest(prices, book, cfg)
        e = {"model": row["model"], "config": row["config"],
             **_grade(nav, WINDOW_START, HOLDOUT_START)}
        if holdout:
            h = _grade(nav, HOLDOUT_START, pd.Timestamp("2100-01-01"))
            e.update({f"h_{k}": v for k, v in h.items()})
        entries.append(e)

    spy = prices[prices.ticker == "SPY"].set_index("date")["close"].sort_index()
    bench = {"model": "benchmark", "config": "buy & hold sp500",
             **_grade(spy, WINDOW_START, HOLDOUT_START)}
    if holdout:
        bench.update({f"h_{k}": v
                      for k, v in _grade(spy, HOLDOUT_START,
                                         pd.Timestamp("2100-01-01")).items()})

    rows = rank_select(entries) + [bench]
    hdr = f"{'#':>2} {'model':14} {'config':42} {'earnings$':>9} {'SR':>6} {'DD':>5}"
    if holdout:
        hdr += f" | {'hold$':>6} {'hSR':>6} {'hDD':>5}"
    lines = [hdr]
    for i, r in enumerate(rows):
        tag = "--" if r["model"] == "benchmark" else f"{i + 1:>2}"
        line = (f"{tag} {r['model']:14} {r['config']:42} "
                f"{r['earnings']:>9,.0f} {r['sr']:>6.3f} {r['dd']:>5.0%}")
        if holdout:
            line += (f" | {r['h_earnings']:>6,.0f} {r['h_sr']:>6.2f} "
                     f"{r['h_dd']:>5.0%}")
        lines.append(line)
    return "\n".join(lines)
