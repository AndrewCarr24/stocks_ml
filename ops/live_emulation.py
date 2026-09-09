"""The champion's grade as deployed, and what fill timing is worth.

selection.simulate runs the live job's own ledger (stocks_ml.ledger): each
rank date's target weights are filled at the next session's open, 5 bp a
side on every trade, rebalances under 0.5% of NAV are skipped and NAV is
marked at Friday's close — so the backtest number is the live-rules number
by construction. This script grades the champion's cached picks twice, as
deployed and with fills priced at the decision date's close (an `opens`
frame that carries the previous close), and reports the ingredient behind
the difference: the weekend gap of incoming vs outgoing names at rotations.

  .venv/bin/python ops/live_emulation.py     # reports/live_emulation.md + ledger row
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import stocks_ml.selection as sel
from stocks_ml.ledger import COST_BPS, close_asof, fill_price
from stocks_ml.models.trials import record_trials
from stocks_ml.selection import HOLDOUT_START

HOLD = Path("data/experiments/champion_2006_2024/holdings_4w_5y_s0.parquet")
REPORT = Path("reports/live_emulation.md")
LO, HI = pd.Timestamp("2006-01-01"), HOLDOUT_START - pd.Timedelta(days=1)
WINDOWS = (("2006-01 -> 2024-06 (pre-holdout)", "2006", HOLDOUT_START),
           ("2006 -> 2012", "2006", "2013"),
           ("2013 -> 2024-06", "2013", HOLDOUT_START),
           ("2021 -> 2024-06", "2021", HOLDOUT_START))
CHAMPION = dict(horizon="4w", book=6, cap=2, stop=None, floor="70/30")


def grade_line(m):
    return (f"${m['terminal_100']:,.0f} ({m['cagr_pct']:+.1f}%/yr, "
            f"SR {m['sharpe']:.2f}, DD {m['max_dd']:.0%})")


def fills_at_close(ctx):
    """The same world with every open replaced by the previous session's
    close: the ledger then fills an order decided on d at d's close."""
    return SimpleNamespace(smap=ctx.smap, members=ctx.members,
                           **sel.price_frames(ctx.closes, ctx.closes.shift(1)))


def rotations(trace):
    """(t, outgoing names, incoming names) per rotation with something to sell."""
    out, prev = [], None
    for x in trace:
        if x["rotated"] and prev is not None:
            for k in x["rotated"]:
                old, new = set(prev["sleeves"][k]), set(x["sleeves"][k])
                if old:
                    out.append({"t": x["t"], "out": sorted(old - new), "in": sorted(new - old)})
        prev = x
    return out


def weekend_gaps(ctx, events):
    """Per rotation: mean open(next session) / close(decision date) - 1 of
    the incoming and of the outgoing names. A fill at the decision close
    credits the incoming gap to the book; the deployed book earns the
    outgoing one."""
    rows = []
    for e in events:
        gaps = {}
        for side in ("in", "out"):
            vals = []
            for n in e[side]:
                c, _ = close_asof(ctx.closes, n, e["t"])
                o, when = fill_price(ctx.closes, ctx.opens, n,
                                     ctx.opens.index[ctx.opens.index > e["t"]][0])
                if np.isfinite(c) and np.isfinite(o) and c > 0 and when is not None:
                    vals.append(o / c - 1.0)
            gaps[side] = float(np.mean(vals)) if vals else np.nan
        rows.append({"t": e["t"], "gap_in": gaps["in"], "gap_out": gaps["out"]})
    return pd.DataFrame(rows)


def window_metrics(series, lo, hi):
    return sel.metrics(series, pd.Timestamp(lo), pd.Timestamp(hi))


def run():
    ctx = sel.Ctx()
    # the whole holdings file (from 2001-06), as the official grade uses it:
    # the book is fully formed before the window opens and every 2006 week
    # counts; metrics() windows the returns
    hold = pd.read_parquet(HOLD)
    hold["week"] = pd.to_datetime(hold["week"])
    hold = hold[hold.week < HI].sort_values("week")

    trace = []
    deployed = sel.simulate(ctx, hold, trace=trace, **CHAMPION)
    at_close = sel.simulate(fills_at_close(ctx), hold, **CHAMPION)
    idx = deployed.index[(deployed.index >= LO) & (deployed.index < HI)]
    series = {"as deployed (fills at the next open)": deployed.reindex(idx),
              "fills at the decision close": at_close.reindex(idx),
              "sp500 (close-to-close)": ctx.wret["SPY"].reindex(idx)}
    events = [e for e in rotations(trace) if e["t"] >= LO]
    gaps = weekend_gaps(ctx, events)
    fills = [f for x in trace if x["nxt"] >= LO for f in x["fills"]]

    lines = ["# Champion as deployed", "",
             "Same 2006-2024 picks (5y window, 4w label, K=4) graded by `selection.simulate`,",
             "which runs the live job's ledger (`stocks_ml.ledger`: target weights filled at",
             "the next session's open, 5 bp a side on every trade, 0.5%-of-NAV dust threshold,",
             "NAV marked at Friday's close) — the official record — and once more with fills",
             "priced at the decision date's close. The two differ only in fill timing.", ""]
    for title, lo, hi in WINDOWS:
        lines += [f"## {title}", "", "| series | grade |", "|---|---|"]
        for name, s in series.items():
            lines.append(f"| {name} | {grade_line(window_metrics(s, lo, hi))} |")
        lines.append("")
    full = {name: window_metrics(s, "2006", HOLDOUT_START) for name, s in series.items()}
    dep, cl = full["as deployed (fills at the next open)"], full["fills at the decision close"]
    g_in, g_out = gaps["gap_in"].mean(), gaps["gap_out"].mean()
    lines += ["## Fill timing, 2006-01 -> 2024-06", "",
              f"- decision close -> next open: {dep['cagr_pct'] - cl['cagr_pct']:+.2f}%/yr "
              f"(${cl['terminal_100']:,.0f} -> ${dep['terminal_100']:,.0f}), "
              f"{dep['sharpe'] - cl['sharpe']:+.2f} on Sharpe", "",
              f"Weekend gap at the {len(gaps)} rotations (mean open-after-signal / "
              f"close-at-signal - 1): incoming names {g_in * 1e4:+.1f} bp, outgoing names "
              f"{g_out * 1e4:+.1f} bp, difference {(g_in - g_out) * 1e4:+.1f} bp per "
              f"rotation. A fill at the decision close credits the incoming gap to the book; "
              f"the deployed book earns the outgoing one. One sleeve of four rotates per week "
              f"at 70% book weight, so the difference is worth about "
              f"{(g_in - g_out) * 0.7 / 4 * 52 * 100:+.2f}%/yr before compounding.", "",
              f"Trades as deployed: {len(fills)} fills over {len(idx)} weeks; fees "
              f"${sum(f[4] for f in fills):,.2f} on a $100 start.", "",
              "## Notes", "",
              "- Training labels are open-to-open (first session after the rank date to 20 "
              "sessions later): the model learns the returns a Monday-open fill earns, and "
              "the engine now grades on the same basis.",
              "- Before 2026-09 the engine bought at the rank date's close, rebalanced the "
              "whole book for free every week and charged 10 bp on the rotated sleeve only; "
              "it graded these picks $1,644 (+16.3%/yr, SR 0.72; ledger "
              "champion_r5_7030_cap2_regraded before this change). The move to the live "
              "rules is the champion's headline change from that number.",
              "- Holdings weeks with no cached prediction are held through; a sleeve that "
              "missed its rotation catches up (STALE_WEEKS), as in the live job.", ""]
    REPORT.write_text("\n".join(lines))
    record_trials([{"kind": "r4w_strategy", "name": "champion_live_rules_2006_2024",
                    "config": {**CHAMPION, "train_years": 5, "fills": "next open",
                               "cost_bps_per_side": COST_BPS},
                    "terminal_100": dep["terminal_100"], "cagr_pct": dep["cagr_pct"],
                    "sharpe": dep["sharpe"], "max_dd": dep["max_dd"], "n_weeks": dep["n_weeks"],
                    "notes": f"2006-01 -> 2024-06, selection.simulate on the live ledger rules "
                             f"(the official engine since 2026-09): {grade_line(dep)} | same "
                             f"rules, fills at the decision close: {grade_line(cl)}. Weekend "
                             f"gap in {g_in * 1e4:+.1f} bp vs out {g_out * 1e4:+.1f} bp per "
                             f"rotation. reports/live_emulation.md"}])
    print("\n".join(lines))
    return series, gaps


if __name__ == "__main__":
    run()
