"""The rolling procedure: the strategy layers re-decided at every rank week
on the trailing window, and the lookback chosen on the selection window only.

Today's procedure decides the layers once, on 2006-2015. The rolling rule
decides them at rank week t on the walk's own out-of-sample predictions over
[t - lookback, label_end(t)] — the last week read is the last whose 4-week
return is realized by t, the same purge the model trains under — and the
book in force phases in as sleeves rotate (selection.simulate `settings`).
`expanding` reads everything since the walk's first week. The lookback is
itself a configuration, so it is chosen the way every layer is: each
candidate rule is followed across the selection window and the registered
metric's argmax on a common span (every rule deciding from its start)
names it. 2016-2024 is read once, for the chosen rule only.

    stocks-ml backtest --preds <pre> <select> <extend> --rolling 8        one rule, graded on
                       --sel-start 2010-01-01 --sel-end 2015-12-31       the selection window
    stocks-ml procedure --lookback <walk>/rolling/*.json                 the choice + one look
                        --sel-start 2010-01-01 --sel-end 2015-12-31
"""
from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

HORIZON = "4w"
MIN_YEARS = 3                 # expanding: the first decision reads at least this much
COLS = ["book", "floor", "stop", "cap", "vol_cut"]


def parse_lookback(v: str):
    """'expanding' -> None; '8' -> 8 (years)."""
    if str(v).lower() == "expanding":
        return None
    n = int(v)
    if n < 1:
        raise ValueError("lookback must be a positive number of years or 'expanding'")
    return n


def variant_name(lookback, cadence: int) -> str:
    return ("expanding" if lookback is None else f"trailing_{lookback}") + f"_c{cadence}"


def windows(weeks, lookback, cadence: int = 1, min_years: int = MIN_YEARS) -> list[tuple]:
    """(t, lo, hi) for every decision week t: hi = label_end(t) (the last
    rank week whose 4-week return is realized by t); lo = t - lookback years,
    or the first week for expanding. A trailing rule first decides when its
    whole window exists; expanding when min_years do. Every cadence-th
    eligible week decides; the rest hold the last decision."""
    from stocks_ml.selection import label_end
    weeks = sorted(pd.Timestamp(w) for w in set(weeks))
    if not weeks:
        return []
    first, need = weeks[0], (min_years if lookback is None else lookback)
    out = []
    for t in weeks:
        if t - pd.DateOffset(years=need) < first:
            continue
        lo = first if lookback is None else t - pd.DateOffset(years=lookback)
        out.append((t, lo, label_end(t, 4)))
    return out[::cadence]


_G: dict = {}


def _init(store, hold):
    from stocks_ml.train import context
    sel, ctx, _ = context(store)
    _G.update(sel=sel, ctx=ctx, hold=hold)


def _decide(args):
    t, lo, hi = args
    d = _G["sel"].decide_strategy(_G["ctx"], _G["hold"], HORIZON, lo, hi)
    return {"t": t, "lo": lo, "hi": hi, **{c: d[c] for c in COLS}, "evidence": d["evidence"]}


def decisions(sel, ctx, store: str, hold: pd.DataFrame, lookback, cadence: int = 1,
              min_years: int = MIN_YEARS, workers: int | None = None, log=print) -> pd.DataFrame:
    """The rule followed: selection.decide_strategy at every decision week
    of `windows`, on the holdings frame the backtest grades. Indexed by
    decision week; columns book, floor, stop, cap, lo, hi, evidence. Decision
    weeks are independent, so they run in parallel (workers=1 runs in-process)."""
    w = windows(hold.week, lookback, cadence, min_years)
    if not w:
        raise SystemExit(f"the walk ({hold.week.min().date()} -> {hold.week.max().date()}) is "
                         f"shorter than the lookback: no decision week")
    log(f"rolling: {variant_name(lookback, cadence)} — {len(w)} decisions "
        f"{w[0][0].date()} -> {w[-1][0].date()}, window {'expanding' if lookback is None else f'{lookback}y'} "
        f"ending label_end(t)")
    workers = workers or max(1, (os.cpu_count() or 2) - 2)
    rows = []
    if workers == 1:
        _G.update(sel=sel, ctx=ctx, hold=hold)
        for i, a in enumerate(w, 1):
            rows.append(_decide(a))
            if i % 50 == 0 or i == len(w):
                log(f"  {i}/{len(w)} decisions")
    else:
        mp = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(workers, mp_context=mp, initializer=_init,
                                 initargs=(store, hold)) as ex:
            futs = [ex.submit(_decide, a) for a in w]
            for i, f in enumerate(as_completed(futs), 1):
                rows.append(f.result())
                if i % 50 == 0 or i == len(w):
                    log(f"  {i}/{len(w)} decisions")
    return pd.DataFrame(rows).set_index("t").sort_index()


def follow(sel, ctx, hold: pd.DataFrame, dec: pd.DataFrame) -> pd.Series:
    """Weekly returns of the rule followed from its first decision week."""
    h = hold[hold.week >= dec.index[0]]
    return sel.simulate(ctx, h, HORIZON, None, None, None, None, settings=dec[COLS])


def _cell(col: str, v) -> str:
    """One layer's value as text; NaN and None both 'none' (a frame column
    of optional ints turns None into NaN)."""
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "none"
    return {"book": lambda x: str(int(x)), "cap": lambda x: str(int(x)),
            "stop": lambda x: str(float(x))}.get(col, str)(v)


def _key(d) -> str:
    """One decision as text."""
    return (f"top-{_cell('book', d['book'])} / {_cell('floor', d['floor'])} / "
            f"stop {_cell('stop', d['stop'])} / cap {_cell('cap', d['cap'])} / vol cut {_cell('vol_cut', d.get('vol_cut'))}")


def summarize(dec: pd.DataFrame, spec: dict | None = None) -> dict:
    """How often the rule changed its mind, and how often it landed on the
    spec's decision {book, floor, stop, cap}."""
    tup = pd.Series([_key(r) for _, r in dec[COLS].iterrows()], index=dec.index)
    changes = int((tup != tup.shift()).sum() - 1)
    out = {"decisions": int(len(dec)), "changes": changes,
           "first": str(dec.index[0].date()), "last": str(dec.index[-1].date()),
           "mode": {c: dec[c].map(lambda v: _cell(c, v)).mode().iloc[0] for c in COLS},
           "share_by_layer": {c: {k: round(float(v), 3) for k, v in
                                  dec[c].map(lambda v: _cell(c, v)).value_counts(normalize=True).items()}
                              for c in COLS}}
    if spec:
        same = tup == _key(spec)
        out["share_equal_to_spec"] = round(float(same.mean()), 3)
    return out


def run(preds_paths, store: str, lookback, cadence: int = 1, min_years: int = MIN_YEARS,
        sel_lo="2006-01-01", sel_hi="2015-12-31", out: Path | None = None, k: int | None = None,
        workers: int | None = None, log=print) -> dict:
    """`stocks-ml backtest --rolling`: one rule followed across the whole
    walk, graded on [sel_lo, sel_hi] only (the selection window; nothing
    later is printed — `procedure --lookback` reads 2016-2024 for the chosen
    rule alone). Writes the decisions, the rule's weekly series, the spec's
    fixed settings on the same weeks and SPY to `out` for that choice."""
    from stocks_ml.backtest import (copies_in, holdings, load_preds, row_vs_spy,
                                    settings_label, simulate_holdings, spec_settings, table_md)
    from stocks_ml.train import context
    sel_lo, sel_hi = pd.Timestamp(sel_lo), pd.Timestamp(sel_hi)
    preds = load_preds(preds_paths)
    k = k or copies_in(preds)
    sel, ctx, _ = context(store)
    hold = holdings(sel, ctx, preds, range(1, k + 1))
    name = variant_name(lookback, cadence)
    log(f"rolling: {name} on {hold.week.nunique()} rank weeks {hold.week.min().date()} -> "
        f"{hold.week.max().date()}, K={k}")
    dec = decisions(sel, ctx, store, hold, lookback, cadence, min_years, workers, log)
    if dec.index[0] > sel_lo:
        raise SystemExit(f"{name} first decides {dec.index[0].date()}, after --sel-start "
                         f"{sel_lo.date()}: the rule cannot be graded on the whole span")
    r = follow(sel, ctx, hold, dec)
    st = spec_settings()
    fixed = simulate_holdings(sel, ctx, hold[hold.week >= dec.index[0]], st)
    spy = ctx.wret["SPY"].reindex(r.index)
    span = f"{sel_lo.year}-{sel_hi.year}"
    spans = {span: (sel_lo, sel_hi + pd.Timedelta(days=1))}
    summ = summarize(dec, st)
    rows = {name: row_vs_spy(sel, r, spy, spans),
            "sp500": {span: sel.metrics(spy, *spans[span])}}
    md = table_md(rows, (span,))
    log(f"rolling: {summ['decisions']} decisions, {summ['changes']} changes; "
        f"share equal to the spec's ({settings_label(st)}): {summ['share_equal_to_spec']:.0%}; "
        f"mode {summ['mode']}")
    log("\n".join(md))
    rec = {"variant": name, "lookback_years": lookback, "cadence": cadence, "min_years": min_years,
           "k": k, "store": store,
           "preds": [{"path": str(p), "sha256": hashlib.sha256(Path(p).read_bytes()).hexdigest()}
                     for p in preds_paths],
           "rank_weeks": [str(hold.week.min().date()), str(hold.week.max().date())],
           "spec_settings": st, "summary": summ, "graded_on": [str(sel_lo.date()), str(sel_hi.date())],
           "table": rows, "md": md,
           "decisions": [{"t": str(t.date()), "lo": str(d.lo.date()), "hi": str(d.hi.date()),
                          **{c: d[c] for c in COLS}, "evidence": d["evidence"]}
                         for t, d in dec.iterrows()],
           "weekly": {"rule": {str(i.date()): float(v) for i, v in r.items()},
                      "fixed": {str(i.date()): float(v) for i, v in fixed.reindex(r.index).items()},
                      "spy": {str(i.date()): float(v) for i, v in spy.items()}}}
    if out is None:
        out = Path(preds_paths[0]).parent.parent / "rolling" / f"{name}.json"
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, default=str))
    log(f"rolling -> {out}")
    return rec


def _series(d: dict) -> pd.Series:
    s = pd.Series(d, dtype=float)
    s.index = pd.to_datetime(s.index)
    return s.sort_index()


def choose(paths, lo, hi, log=print, spec_path=None) -> dict:
    """`stocks-ml procedure --lookback`: the registered choice among rolling
    rules — cost-adjusted compounded %/yr of each rule followed on the common
    span [lo, hi] (inside the selection window; every rule must decide from
    lo or earlier), argmax. Then the one look: the chosen rule on 2016-2024
    against the spec's fixed settings on the same weeks and the S&P 500,
    paired weekly t. Writes nothing to the spec; records a ledger row."""
    import stocks_ml.selection as sel
    from stocks_ml.backtest import EXTEND_START, paired_t, row_vs_spy, table_md
    from stocks_ml.models.trials import record_trials
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi) + pd.Timedelta(days=1)
    recs = {}
    for p in paths:
        rec = json.loads(Path(p).read_text())
        first = pd.Timestamp(rec["summary"]["first"])
        if first > lo:
            raise SystemExit(f"{rec['variant']} first decides {first.date()}, after {lo.date()}: "
                             "not the same span as the others")
        recs[rec["variant"]] = rec
    span = f"{lo.year}-{(hi - pd.Timedelta(days=1)).year}"
    series = {name: _series(rec["weekly"]["rule"]) for name, rec in recs.items()}
    common = None                          # the same weeks for every rule (same-basis rule)
    for r in series.values():
        idx = r[(r.index >= lo) & (r.index < hi)].index
        common = idx if common is None else common.intersection(idx)
    table, metric = {}, {}
    for name, rec in recs.items():
        r = series[name].reindex(common)
        m = sel.metrics(r, lo, hi)
        metric[name] = m["cagr_pct"]
        s = rec["summary"]
        table[name] = {**m, "decisions": s["decisions"], "changes": s["changes"],
                       "share_equal_to_spec": s.get("share_equal_to_spec"), "first": s["first"]}
    best = max(metric, key=metric.get)
    md = [f"| rule | first decision | decisions, changes | share = spec | {span}: $100, %/yr | SR, DD |",
          "|---|---|---|---|---|---|"]
    for name, t in table.items():
        md.append(f"| {'**' + name + '**' if name == best else name} | {t['first']} | "
                  f"{t['decisions']}, {t['changes']} | {t['share_equal_to_spec']:.0%} | "
                  f"${t['terminal_100']:,.0f}, {t['cagr_pct']:+.1f}% | {t['sharpe']:.2f}, {t['max_dd']:.0%} |")
    log(f"lookback choice on {span} ({len(common)} common weeks) by cost-adjusted compounded %/yr "
        f"of the rule followed (the same convention as the training window): argmax {best} "
        f"({metric[best]:+.2f})")
    log("\n".join(md))
    # the one look: the chosen rule alone
    rec = recs[best]
    rule, fixed, spy = (_series(rec["weekly"][k]) for k in ("rule", "fixed", "spy"))
    ol = {"2016-2024": (EXTEND_START, sel.HOLDOUT_START)}
    rows = {f"{best} (rolling)": row_vs_spy(sel, rule, spy, ol),
            "spec settings, fixed": row_vs_spy(sel, fixed, spy, ol),
            "sp500": {"2016-2024": sel.metrics(spy, *ol["2016-2024"])}}
    d = (rule - fixed).dropna()
    d = d[(d.index >= EXTEND_START) & (d.index < sel.HOLDOUT_START)]
    t_vs_fixed = round(paired_t(d), 2)
    md2 = table_md(rows, ("2016-2024",))
    dec = pd.DataFrame(rec["decisions"]); dec["t"] = pd.to_datetime(dec["t"])
    late = dec[dec.t >= EXTEND_START]
    summ_late = summarize(late.set_index("t"), rec["spec_settings"]) if len(late) else {}
    log(f"the one look — {best} on 2016-2024 vs the spec's fixed settings on the same weeks: "
        f"paired weekly t {t_vs_fixed:+.2f}; decisions on 2016-2024: "
        f"{summ_late.get('changes')} changes, share equal to spec {summ_late.get('share_equal_to_spec')}")
    log("\n".join(md2))
    out = {"graded_on": [str(lo.date()), str((hi - pd.Timedelta(days=1)).date())],
           "common_weeks": int(len(common)),
           "metric": "cost-adjusted compounded %/yr of the rule followed",
           "candidates": table, "choice": best, "md": md,
           "one_look": {"rows": rows, "paired_t_rolling_vs_fixed": t_vs_fixed,
                        "decisions_2016_2024": summ_late, "md": md2},
           "inputs": {n: {"path": str(p)} for n, p in zip(recs, paths)}}
    out_path = Path(paths[0]).parent / f"lookback_choice_{span}.json"
    out_path.write_text(json.dumps(out, indent=1, default=str))
    record_trials([{"kind": "rolling_lookback", "name": f"rolling_lookback_{span}_{best}",
                    "pre_holdout_sharpe": rows[f"{best} (rolling)"]["2016-2024"]["sharpe"],
                    "notes": json.dumps({"choice": best, "metric": metric, "graded_on": out["graded_on"],
                                         "one_look_2016_2024": {k: v["2016-2024"] for k, v in rows.items()},
                                         "paired_t_rolling_vs_fixed": t_vs_fixed})}])
    log(f"lookback choice -> {out_path}; ledger row rolling_lookback_{span}_{best}")
    return out
