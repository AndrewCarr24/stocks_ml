"""The r5 champion's weekly signal job (`stocks-ml r5-weekly`).

Runs on the owner's Mac (launchd, Saturday morning): it needs the Sharadar
key and a fresh world store, neither of which belongs in Actions. Steps:

  1. data/world.py refreshes the live world and rebuilds panel_sf.parquet
  2. selection.ensemble_preds ranks this Friday's members exactly as the
     research pipeline did (K=16 week-bootstrap copies; the label and the
     training window are the spec's, written by the procedure from the
     champion walk's own record -- the sector-centred 4-week label on an
     8-year window since 2026-09-12, Stage E of the clean program;
     the panel's f_ columns on the nominal price basis; SPEC["features"] is
     the adopted bundle beyond them — empty since 2026-09-11, when the
     split-leak bundle was retired, features/bundle.py)
  3. the sleeve schedule rotates one of four sleeves of the spec's book size
     (top-10, no sector cap since 2026-09-12; top-6 with cap 2 before)
  4. the trend ballast: the spec's floor (one of ledger.FLOORS, decided by
     the procedure) sets this week's book fraction and parks the rest —
     a fixed split shifts SPY to IEF one third per breached moving average;
     halfgate cuts the book from 100% to 50% as the three gates go down,
     the rest in IEF (ledger.floor_split, the rule selection.simulate grades)
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
from stocks_ml.ledger import (FUNDS, Ledger, ballast_state, due_sleeve, floor_split, friday_of,
                              rotate_sleeves, sleeve_counts, target_weights)
from stocks_ml.selection import HORIZONS, K_COPIES, Ctx, ensemble_preds



def spec_path() -> Path:
    """models/champion_spec.json: beside this package's source tree (uv's
    editable install, the checkout GitHub Actions and ops/r5_weekly.sh run
    from) or under the working directory."""
    for p in (Path(__file__).resolve().parents[3] / "models/champion_spec.json",
              Path("models/champion_spec.json")):
        if p.exists():
            return p
    raise FileNotFoundError("models/champion_spec.json: run the job from the repository root")


def load_spec(path: Path | None = None) -> dict:
    """The live job's settings from the spec the procedure wrote
    (stocks_ml.procedure): horizon, label, window, book, cap and floor are
    read, never typed here. top_n (rotation candidates) and the feature bundle are
    the job's own; tests/test_procedure.py holds them to the spec."""
    from stocks_ml.procedure import live_strategy
    spec = json.loads((path or spec_path()).read_text())
    return {**live_strategy(spec), "top_n": 15, "features": list(BUNDLE)}


SPEC = load_spec()                         # features/bundle.py: empty since 2026-09-11 (clean line)
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


def _easter(year: int) -> pd.Timestamp:
    """Easter Sunday (anonymous Gregorian computus)."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    g = (8 * b + 13) // 25
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month, day = divmod(h + l - 7 * m + 114, 31)
    return pd.Timestamp(year, month, day + 1)


def nyse_friday_holiday(d) -> bool:
    """Fridays the NYSE is closed: Good Friday, and holidays that fall on
    (or are observed on) a Friday. A Saturday holiday moves to Friday except
    over a year end (Rule 7.2: Dec 31 stays open, as on 2021-12-31)."""
    d = pd.Timestamp(d).normalize()
    if d.weekday() != 4:
        return False
    if d == _easter(d.year) - pd.Timedelta(days=2):
        return True
    md = (d.month, d.day)
    if md in {(1, 1), (7, 4), (12, 25)}:                  # the holiday itself on a Friday
        return True
    if md in {(7, 3), (12, 24)}:                          # Saturday holiday observed Friday
        return True
    if md in {(6, 19), (6, 18)} and d.year >= 2022:       # Juneteenth (and its observance)
        return True
    return False


def _latest_complete_week(weeks) -> pd.Timestamp:
    """The newest panel week that is ready to trade: this week's Friday row,
    or its Thursday when that Friday is a market holiday. A partial week
    (store refreshed mid-week) is skipped; a store a session behind on an
    ordinary week fails loudly rather than signalling on stale prices."""
    lf = last_friday()
    done = [w for w in weeks if w <= lf]
    if not done:
        raise RuntimeError("panel has no completed week yet")
    t = done[-1]
    if t == lf or (friday_of(t) == lf and nyse_friday_holiday(lf) and (lf - t).days == 1):
        return t
    raise RuntimeError(f"panel ends {t.date()} but the last trading Friday was "
                       f"{lf.date()}: prices are not refreshed yet")


def _guard_as_of(ledger: Ledger, t, as_of, dry_run: bool) -> None:
    """Refuse a signal dated before the book's last mark or pending decision
    unless dry_run: it would rewrite NAV history out of order and re-date the
    pending orders so the next run fills them a week early."""
    if as_of is None or dry_run:
        return
    marks = [r[0] for r in ledger.nav_history]
    if ledger.pending:
        marks.append(ledger.pending["decision_date"])
    last = max(marks, default=None)
    if last and str(pd.Timestamp(t).date()) < last:
        raise RuntimeError(f"--as-of {pd.Timestamp(t).date()} is before the ledger's last "
                           f"mark/decision {last}; use --dry-run to look back")


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
    t = pd.Timestamp(as_of) if as_of else _latest_complete_week(ctx.weeks)
    if t not in ctx.members:
        raise RuntimeError(f"{t.date()} is not a panel date; latest is {ctx.weeks[-1].date()}")
    log(f"signal date {t.date()} (sleeve {due_sleeve(t, N_SLEEVES)} due); fitting {SPEC}")
    t1 = time.time()
    preds = ensemble_preds(ctx, t, SPEC["horizon"], SPEC["train_years"], label=SPEC["label"])
    if preds is None:
        raise RuntimeError(f"no ensemble prediction for {t.date()}")
    ranked = rank_members(preds, ctx.prices, t)
    log(f"ranked {len(ranked)} names in {time.time() - t1:.0f}s; "
        f"top-{SPEC['top_n']}: {', '.join(ranked.index[:SPEC['top_n']])}")

    ledger = Ledger.load(ledger_path) or Ledger.new(capital, t)
    renames = ((report.get("refresh") or {}).get("sharadar") or {}).get("renames") or {}
    if renames:
        hit = ledger.rename(renames)
        if hit:
            log(f"vendor renames applied to the ledger: {hit}")
    _guard_as_of(ledger, t, as_of, dry_run)
    factors = ledger.rebase(ctx.closes, log=log)
    fills = ledger.fill_pending(ctx.closes, ctx.opens, t)
    nav, bench = ledger.mark(ctx.closes, t)
    sleeves, rotated = rotate_sleeves(ledger.sleeves, t, list(ranked.index), ctx.smap,
                                      N_SLEEVES, SPEC["book"], SPEC["cap"], SPEC["top_n"])
    ballast = ballast_state(ctx.spy_w, t)
    frac, weights = book_weights(sleeves, ballast)
    ledger.sleeves = sleeves
    ledger.pending = {"decision_date": str(t.date()), "weights": weights}

    held = ledger.value_of(ctx.closes, t)
    signal = {
        "date": str(t.date()), "sleeve_due": due_sleeve(t, N_SLEEVES), "rotated": rotated,
        "sleeves": sleeves, "ballast": ballast, "book_fraction": frac, "weights": weights,
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


def book_weights(sleeves: dict, gates: dict) -> tuple[float, dict[str, float]]:
    """This week's book fraction and target weights under the spec's floor —
    ledger.floor_split then ledger.target_weights, exactly as
    selection.simulate graded the champion."""
    frac, ballast = floor_split(SPEC["floor"], gates)
    return frac, target_weights(sleeves, ballast, frac)


def label_text(label: str) -> str:
    """The target as the report names it."""
    return {"label_4w": "4-week label", "label_4w_sector": "sector-relative 4-week label"}[label]


def render_markdown(sig: dict, smap: dict) -> str:
    nav, bench = sig["nav"], sig["spy_nav"]
    counts = sleeve_counts(sig["sleeves"])
    frac = sig.get("book_fraction")
    frac_text = f" (book {frac:.0%} of NAV this week)" if frac is not None else ""
    lines = [f"# r5 signal — {sig['date']}", "",
             f"Champion r5 (PROCEDURE.md): {SPEC['floor']} trend ballast{frac_text}, "
             f"top-{SPEC['book']} four-sleeve stagger, sector cap {SPEC['cap']}, "
             f"{label_text(SPEC['label'])}, {SPEC['train_years']}-year window, K={K_COPIES}.", "",
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
