"""`stocks-ml backtest`: a walk's predictions run through the strategy.

A backtest has two parts, both in selection.py and both shared with the
live job:

    ensemble_holdings   at each rank week, average the K copies' scores and
                        keep the top names (the same ranking r5.py trades)
    simulate            the ledger: next-open fills, 5 bp a side, four
                        staggered sleeves, the floor, the book

This module runs them over a whole walk and prints the owner's table:
$100 grown, %/yr, Sharpe, worst drawdown, on 2006-2015 (where the model was
chosen), 2016-2024 (read once) and pre-holdout 2006-2024, beside the S&P 500
on the same weeks, with the paired weekly t. Settings default to the deployed
spec's decision, which `stocks-ml procedure` wrote; `--book/--floor/--stop/
--cap` try others for exploration only — the procedure decides.

    stocks-ml backtest --preds <walk>/select/preds.parquet <walk>/extend/preds.parquet
    stocks-ml backtest --preds ... --book 6 --floor none --k 4

Nothing at or past the holdout (selection.HOLDOUT_START) is ever loaded.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from stocks_ml.selection import HOLDOUT_START

SPEC_PATH = Path("models/champion_spec.json")
SELECT_START = pd.Timestamp("2006-01-01")     # the selection window opens
EXTEND_START = pd.Timestamp("2016-01-01")     # the one-look years open
SPANS = {"2006-2015": (SELECT_START, EXTEND_START),
         "2016-2024": (EXTEND_START, HOLDOUT_START),
         "2006-2024": (SELECT_START, HOLDOUT_START)}


def load_preds(paths) -> pd.DataFrame:
    """The walk segments as one frame (week, ticker, c1..cK), sorted;
    refused if any week is at or past the holdout."""
    frames = []
    for p in paths:
        df = pd.read_parquet(p)
        df["week"] = pd.to_datetime(df["week"])
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    if (df.week >= HOLDOUT_START).any():
        raise RuntimeError(f"{list(map(str, paths))} reach the holdout ({HOLDOUT_START.date()})")
    return df.sort_values(["week", "ticker"]).reset_index(drop=True)


def walk_records(paths) -> list[dict]:
    """Each segment's record (spec.json beside preds.parquet), or {}."""
    out = []
    for p in paths:
        rec = Path(p).parent / "spec.json"
        out.append(json.loads(rec.read_text()) if rec.exists() else {})
    return out


def copies_in(preds: pd.DataFrame) -> int:
    return sum(1 for c in preds.columns if c.startswith("c") and c[1:].isdigit())


def spec_settings(spec_path: Path = SPEC_PATH) -> dict:
    """The deployed strategy layers as the spec carries them — the
    procedure's decision, read, never typed."""
    d = json.loads(Path(spec_path).read_text())["procedure"]["decision"]
    return dict(book=d["book_size"], floor=d["floor"], stop=d["stop_loss"], cap=d["sector_cap"])


def weekly_returns(sel, ctx, preds: pd.DataFrame, copies, st: dict) -> pd.Series:
    """The strategy's weekly returns for the mean of `copies` at settings
    st = {book, floor, stop, cap}."""
    hold, _ = sel.ensemble_holdings(ctx, preds, copies)
    hold = hold.sort_values("week").reset_index(drop=True)
    return sel.simulate(ctx, hold, "4w", st["book"], st["cap"], st["stop"], st["floor"])


def paired_t(x: pd.Series) -> float:
    x = x.dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x))) if len(x) > 2 else float("nan")


def row_vs_spy(sel, series: pd.Series, spy: pd.Series, spans: dict = SPANS) -> dict:
    """metrics per span plus the paired weekly t vs sp500 — one table row."""
    row = {w: sel.metrics(series, a, b) for w, (a, b) in spans.items()}
    d = (series - spy.reindex(series.index)).dropna()
    row["paired_t_vs_sp500"] = {w: round(paired_t(d[(d.index >= a) & (d.index < b)]), 2)
                                for w, (a, b) in spans.items()}
    return row


def settings_label(st: dict) -> str:
    return f"top-{st['book']} / {st['floor']} / stop {st['stop']} / cap {st['cap']}"


def table_md(rows: dict, spans=("2006-2024", "2016-2024", "2006-2015")) -> list[str]:
    """The owner's table. `rows` maps a name to row_vs_spy's dict (the sp500
    row has no paired t); every table carries the sp500 row."""
    if "sp500" not in rows:
        raise ValueError("a performance table always carries the sp500 row")
    head = " | ".join(f"{w}: $100, %/yr | SR, DD" for w in spans)
    md = [f"| model | {head} | weekly t vs sp500 ({' / '.join(w[2:4] + '-' + w[-2:] for w in spans[:2])}) |",
          "|---|" + "---|" * (2 * len(spans) + 1)]
    for name, r in rows.items():
        cells = []
        for w in spans:
            cells += [f"${r[w]['terminal_100']:,.0f}, {r[w]['cagr_pct']:+.1f}%",
                      f"{r[w]['sharpe']:.2f}, {r[w]['max_dd']:.0%}"]
        t = ("—" if "paired_t_vs_sp500" not in r else
             " / ".join(f"{r['paired_t_vs_sp500'][w]:+.2f}" for w in spans[:2]))
        md.append(f"| {name} | " + " | ".join(cells) + f" | {t} |")
    return md


def run(preds_paths, store: str, st: dict | None = None, k: int | None = None,
        name: str = "walk", log=print) -> dict:
    """Backtest the walk at the settings (default: the spec's decision) for
    the mean of copies 1..k (default: every copy the walk holds). Prints the
    table; returns {settings, k, rank_weeks, records, table}."""
    from stocks_ml.train import context
    st = st or spec_settings()
    preds = load_preds(preds_paths)
    k = k or copies_in(preds)
    if k > copies_in(preds):
        raise SystemExit(f"the walk holds {copies_in(preds)} copies, --k {k} asked")
    sel, ctx, _ = context(store)
    log(f"backtest: {name} — {preds.week.nunique()} rank weeks {preds.week.min().date()} -> "
        f"{preds.week.max().date()}, K={k}, {settings_label(st)}")
    r = weekly_returns(sel, ctx, preds, range(1, k + 1), st)
    spy = ctx.wret["SPY"]
    spans = {w: (a, b) for w, (a, b) in SPANS.items()
             if ((r.index >= a) & (r.index < b)).sum() > 52}
    rows = {name: row_vs_spy(sel, r, spy, spans),
            "sp500": {w: sel.metrics(spy.reindex(r.index), a, b) for w, (a, b) in spans.items()}}
    md = table_md(rows, tuple(w for w in ("2006-2024", "2016-2024", "2006-2015") if w in spans))
    log("\n".join(md))
    return {"settings": st, "k": k, "rank_weeks": int(preds.week.nunique()),
            "records": walk_records(preds_paths), "table": rows, "md": md}
