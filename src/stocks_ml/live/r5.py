"""The r5 champion's weekly signal job (`stocks-ml r5-weekly`).

Runs on the owner's Mac (launchd, Saturday morning): it needs the Sharadar
key and a fresh world store, neither of which belongs in Actions. Steps:

  1. data/world.py refreshes the live world and rebuilds panel_sf.parquet
  2. selection.ensemble_preds ranks this Friday's members exactly as the
     research pipeline did (K=16 week-bootstrap copies, 4w label, 5y window,
     the panel's f_ columns plus the champion's generated bundle, the g_
     columns build_world_panel computes from features/bundle.py's formulas;
     SPEC["features"])
  3. the sleeve schedule rotates one of four 6-name sleeves (sector cap 2)
  4. the 70/30 trend ballast decides SPY vs IEF per moving-average third
  5. a paper ledger fills LAST week's orders at Monday's open, marks NAV at
     Friday's close and stores this week's target weights as pending orders

Everything the job decides is written to signals_r5/<friday>.md (+ .json)
and ledger_r5.json. The rules and the ledger are stocks_ml.ledger — the same
code selection.simulate grades the champion with — plus one live-only
requirement: a name must have traded within the last few sessions to be
rankable. Units in the ledger are on Sharadar's total-return price basis;
see ledger.py for the rebase that keeps them right when the vendor
re-adjusts a history.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from stocks_ml.features.bundle import FEATURES as BUNDLE
from stocks_ml.ledger import (FUNDS, Ledger, ballast_state, due_sleeve, friday_of,
                              rotate_sleeves, sleeve_counts, target_weights)
from stocks_ml.selection import HORIZONS, K_COPIES, Ctx, ensemble_preds

SPEC = {"horizon": "4w", "train_years": 5, "book": 6, "cap": 2, "floor": 0.7,
        "top_n": 15,                       # models/champion_spec.json (tests keep them equal)
        "features": list(BUNDLE)}          # the generated bundle (features/bundle.py), adopted 2026-09-06
N_SLEEVES = HORIZONS[SPEC["horizon"]]["kweeks"]
MIN_UNIVERSE = 100                         # rankable names needed for a signal
TRADABLE_DAYS = 7                          # a name must have a close this recent


def _log(msg):
    print(msg, flush=True)


# ---- the weekly run ----
def rank_members(preds: pd.Series, prices: pd.DataFrame, t) -> pd.Series:
    """Predictions for names that traded within the last few sessions,
    highest first."""
    t = pd.Timestamp(t)
    recent = prices[(prices["date"] > t - pd.Timedelta(days=TRADABLE_DAYS))
                    & (prices["date"] <= t)]
    tradable = set(recent.dropna(subset=["close"])["ticker"])
    p = preds[preds.index.isin(tradable)]
    if len(p) < MIN_UNIVERSE:
        raise RuntimeError(f"only {len(p)} rankable names at {t.date()} (need {MIN_UNIVERSE})")
    return p.sort_values(ascending=False)


def last_friday(today=None) -> pd.Timestamp:
    d = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    return d - pd.Timedelta(days=(d.weekday() - 4) % 7)


def run_weekly(live_dir, cfg, as_of=None, refresh=True, sec=True, dry_run=False,
               capital=100.0, out_dir="signals_r5", ledger_path="ledger_r5.json",
               log=_log) -> dict:
    from stocks_ml.data.world import build_world_panel, refresh_world
    t0 = time.time()
    report = {"run_at": str(pd.Timestamp.now()), "spec": SPEC, "dry_run": dry_run}
    if refresh:
        report["refresh"] = refresh_world(live_dir, cfg, sec=sec, log=log)
    if refresh or not (Path(live_dir) / "panel_sf.parquet").exists():
        build_world_panel(live_dir, cfg, log=log)

    ctx = Ctx(str(live_dir))
    ctx.extra = list(SPEC.get("features", ()))     # the champion's bundle beyond feature_cols
    missing = [c for c in ctx.extra if c not in ctx.pan.columns]
    if missing:
        raise RuntimeError(f"panel_sf lacks the champion's bundle columns {missing[:3]}...: "
                           "rebuild it (build_world_panel computes the g_ columns)")
    t = pd.Timestamp(as_of) if as_of else ctx.weeks[-1]
    if t not in ctx.members:
        raise RuntimeError(f"{t.date()} is not a panel date; latest is {ctx.weeks[-1].date()}")
    if as_of is None and friday_of(t) < last_friday():       # holiday Fridays: Thursday is fine
        raise RuntimeError(f"panel ends {t.date()} but the last Friday was "
                           f"{last_friday().date()}: prices are not refreshed yet")
    log(f"signal date {t.date()} (sleeve {due_sleeve(t, N_SLEEVES)} due); fitting {SPEC}")
    t1 = time.time()
    preds = ensemble_preds(ctx, t, SPEC["horizon"], SPEC["train_years"])
    if preds is None:
        raise RuntimeError(f"no ensemble prediction for {t.date()}")
    ranked = rank_members(preds, ctx.prices, t)
    log(f"ranked {len(ranked)} names in {time.time() - t1:.0f}s; "
        f"top-{SPEC['top_n']}: {', '.join(ranked.index[:SPEC['top_n']])}")

    ledger = Ledger.load(ledger_path) or Ledger.new(capital, t)
    factors = ledger.rebase(ctx.closes)
    fills = ledger.fill_pending(ctx.closes, ctx.opens, t)
    nav, bench = ledger.mark(ctx.closes, t)
    sleeves, rotated = rotate_sleeves(ledger.sleeves, t, list(ranked.index), ctx.smap,
                                      N_SLEEVES, SPEC["book"], SPEC["cap"], SPEC["top_n"])
    ballast = ballast_state(ctx.spy_w, t)
    weights = target_weights(sleeves, ballast, SPEC["floor"])
    ledger.sleeves = sleeves
    ledger.pending = {"decision_date": str(t.date()), "weights": weights}

    held = ledger.value_of(ctx.closes, t)
    signal = {
        "date": str(t.date()), "sleeve_due": due_sleeve(t, N_SLEEVES), "rotated": rotated,
        "sleeves": sleeves, "ballast": ballast, "weights": weights,
        "nav": nav, "spy_nav": bench, "cash": ledger.cash,
        "held_value": held, "fills": fills, "rebase_factors": factors,
        "top": [(tk, float(v)) for tk, v in ranked.iloc[:SPEC["top_n"]].items()],
        "n_ranked": int(len(ranked)), "positions": ledger.positions,
        "freshness": _freshness(ctx, report.get("refresh")),
        "elapsed_s": round(time.time() - t0),
    }
    report["signal"] = signal
    md = render_markdown(signal, ctx.smap)
    if dry_run:
        log("dry run: ledger and signal files not written")
    else:
        out = Path(out_dir)
        out.mkdir(exist_ok=True)
        (out / f"{t.date()}.md").write_text(md)
        (out / f"{t.date()}.json").write_text(json.dumps(signal, indent=2, default=str))
        ledger.save(ledger_path)
        log(f"wrote {out / f'{t.date()}.md'} and {ledger_path}")
    log(md)
    return report


def _freshness(ctx: Ctx, refresh: dict | None) -> dict:
    out = {"panel": str(ctx.weeks[-1].date()),
           "prices": str(ctx.prices["date"].max().date())}
    if refresh:
        for src, rep in refresh.get("sharadar", {}).items():
            for k in ("through", "filed_through", "events_through"):
                if k in rep:
                    out[src] = rep[k]
        for src, rep in refresh.get("sec", {}).items():
            for k in ("filed_through", "coverage_end", "last_date"):
                if k in rep:
                    out[src] = rep[k]
    return out


def render_markdown(sig: dict, smap: dict) -> str:
    nav, bench = sig["nav"], sig["spy_nav"]
    counts = sleeve_counts(sig["sleeves"])
    lines = [f"# r5 signal — {sig['date']}", "",
             "Champion r5 (PROCEDURE.md): 70/30 trend ballast, top-6 four-sleeve "
             f"stagger, sector cap 2, 4-week label, 5-year window, K={K_COPIES}.", "",
             f"Paper NAV **${nav:,.2f}** · SPY buy-and-hold ${bench:,.2f} · "
             f"cash ${sig['cash']:,.2f}", "",
             "## This week", "",
             f"Sleeve {sig['sleeve_due']} due; rotated {sig['rotated']}.",
             "Ballast: " + ", ".join(f"{w}w→{f}" for w, f in sig["ballast"].items()), "",
             "## Target book (execute at Monday's open)", "",
             "| ticker | sector | sleeves | weight | target $ | held $ | Δ $ |",
             "|---|---|---|---|---|---|---|"]
    held = sig["held_value"]
    # ties (equal-weight names) break on the ticker: a rerun must reproduce
    # the file byte for byte, and set order varies with the hash seed
    for tk in sorted(set(sig["weights"]) | set(held),
                     key=lambda x: (-sig["weights"].get(x, 0.0), x)):
        w = sig["weights"].get(tk, 0.0)
        cur = held.get(tk, 0.0)
        sector = "ballast" if tk in FUNDS else smap.get(tk, "?")
        lines.append(f"| {tk} | {sector} | {counts.get(tk, '')} | {w:.2%} | "
                     f"${w * nav:,.2f} | ${cur:,.2f} | {w * nav - cur:+,.2f} |")
    lines += ["", "## Sleeves", ""]
    for k, s in sig["sleeves"].items():
        lines.append(f"- sleeve {k} (since {s['since']}): {', '.join(s['names'])}")
    lines += ["", f"## Top-{len(sig['top'])} of {sig['n_ranked']} ranked", "",
              "| rank | ticker | sector | score |", "|---|---|---|---|"]
    for i, (tk, v) in enumerate(sig["top"], 1):
        lines.append(f"| {i} | {tk} | {smap.get(tk, '?')} | {v:+.4f} |")
    if sig["fills"]:
        lines += ["", "## Fills since the last signal", "",
                  "| date | ticker | units | price | fee |", "|---|---|---|---|---|"]
        for d, tk, u, p, f in sig["fills"]:
            lines.append(f"| {d} | {tk} | {u:+.4f} | ${p:,.2f} | ${f:.4f} |")
    if sig["rebase_factors"]:
        lines += ["", "Re-adjusted histories (units rescaled): " +
                  ", ".join(f"{k} ×{v:.6f}" for k, v in sig["rebase_factors"].items())]
    lines += ["", "## Data", "",
              ", ".join(f"{k} {v}" for k, v in sig["freshness"].items()),
              "", f"Run time {sig['elapsed_s']}s."]
    return "\n".join(lines)
