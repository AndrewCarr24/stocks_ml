"""The r5 champion's weekly signal job (`stocks-ml r5-weekly`).

Runs on the owner's Mac (launchd, Saturday morning): it needs the Sharadar
key and a fresh world store, neither of which belongs in Actions. Steps:

  1. data/world.py refreshes the live world and rebuilds panel_sf.parquet
  2. selection.ensemble_preds ranks this Friday's members exactly as the
     research pipeline did (K=16 week-bootstrap copies; the label and the
     training window are the spec's, written by the procedure from the
     champion walk's own record -- the sector-centred 4-week label on an
     8-year window since 2026-09-12, Stage E of the clean program;
     the panel's f_ columns on the nominal price basis; SPEC["features"]
     names any panel columns the model gets beyond them — none)
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

from stocks_ml.ledger import (FUNDS, Ledger, ballast_state, due_sleeve, floor_split, friday_of,
                              rotate_sleeves, sleeve_counts, target_weights, vol_cut_pool, VOL_POOL)
from stocks_ml.selection import HORIZONS, K_COPIES, Ctx, apply_filter, ensemble_preds, vol_context



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
    read, never typed here; `features` is the spec's list of panel columns the
    model gets beyond the panel's f_ columns (empty). top_n (rotation
    candidates) is the job's own; tests/test_procedure.py holds it to the spec."""
    from stocks_ml.procedure import live_strategy
    spec = json.loads((path or spec_path()).read_text())
    return {**live_strategy(spec), "top_n": 15, "features": list(spec.get("features") or []),
            "drop": list(spec.get("drop_features") or []), "params": dict(spec["model"]["params"]),
            "train_top": spec.get("train_top"), "filter": spec.get("universe_filter"),
            "rolling": spec.get("rolling")}          # the live rule (rolling.adopt): settings re-decided on the trailing window


SPEC = load_spec()
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
    if SPEC.get("train_top"):
        raise SystemExit("the spec's train_top needs market-cap snapshots the live world does not carry yet")
    ctx.extra = list(SPEC.get("features", ()))     # panel columns beyond feature_cols (none)
    missing = [c for c in ctx.extra if c not in ctx.pan.columns]
    if missing:
        raise RuntimeError(f"panel_sf lacks the spec's extra feature columns {missing[:3]}")
    t = pd.Timestamp(as_of) if as_of else _latest_complete_week(ctx.weeks)
    if t not in ctx.members:
        raise RuntimeError(f"{t.date()} is not a panel date; latest is {ctx.weeks[-1].date()}")
    log(f"signal date {t.date()} (sleeve {due_sleeve(t, N_SLEEVES)} due); fitting {SPEC}")
    t1 = time.time()
    preds = ensemble_preds(ctx, t, SPEC["horizon"], SPEC["train_years"], label=SPEC["label"],
                           features=SPEC["features"], params=SPEC["params"], drop=SPEC["drop"],
                           train_top=SPEC["train_top"])
    if preds is None:
        raise RuntimeError(f"no ensemble prediction for {t.date()}")
    ranked = rank_members(preds, ctx.prices, t)
    if SPEC.get("filter"):                      # the recipe's universe filter, the backtest's rule (selection.apply_filter)
        keep = apply_filter(ctx, t, list(ranked.index), SPEC["filter"])
        log(f"universe filter ({SPEC['filter']}): {len(keep)} of {len(ranked)} rankable names pass")
        ranked = ranked.reindex(keep)
        if len(ranked) < MIN_UNIVERSE:
            raise RuntimeError(f"only {len(ranked)} names pass the universe filter at {t.date()} (need {MIN_UNIVERSE})")
    from stocks_ml.leak_audit import archive_live_rows
    archive_live_rows(live_dir, ctx, t)          # this week's rows, as computed this week (leak_audit.live_vs_rebuilt)
    log(f"ranked {len(ranked)} names in {time.time() - t1:.0f}s; "
        f"top-{SPEC['top_n']}: {', '.join(ranked.index[:SPEC['top_n']])}")

    ledger = Ledger.load(ledger_path) or Ledger.new(capital, t)
    if SPEC.get("rolling"):
        rolling_decision(ledger, ctx, live_dir, t, preds, log=log)
    st = settings_in_force(ledger)
    log(f"settings in force: {settings_text(st)}" + (f" — the rolling rule's decision of {st['decided']} "
        f"(window {st['lo']} -> {st['hi']}); spec fallback {settings_text(SPEC)}" if st.get("decided") else " (the spec's fixed decision)"))
    renames = ((report.get("refresh") or {}).get("sharadar") or {}).get("renames") or {}
    if renames:
        hit = ledger.rename(renames)
        if hit:
            log(f"vendor renames applied to the ledger: {hit}")
    _guard_as_of(ledger, t, as_of, dry_run)
    factors = ledger.rebase(ctx.closes, log=log)
    fills = ledger.fill_pending(ctx.closes, ctx.opens, t)
    nav, bench = ledger.mark(ctx.closes, t)
    pool = list(ranked.index)
    if st.get("vol_cut"):
        vol, vs = vol_context(ctx, t, pool[:VOL_POOL])
        pool = vol_cut_pool(pool, vol, vs, st["vol_cut"], keep=SPEC["top_n"])
        log(f"volatility cut ({st['vol_cut']}): the sleeve picks from {', '.join(pool)}")
    sleeves, rotated = rotate_sleeves(ledger.sleeves, t, pool, ctx.smap,
                                      N_SLEEVES, st["book"], st["cap"], SPEC["top_n"])
    ballast = ballast_state(ctx.spy_w, t)
    frac, weights = book_weights(sleeves, ballast, st["floor"])
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
        "settings": st,
        "freshness": _freshness(ctx, report.get("refresh")),
        "elapsed_s": round(time.time() - t0),
    }
    report["signal"] = signal
    md = render_markdown(signal, ctx.smap, st)
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


def book_weights(sleeves: dict, gates: dict, floor: str | None = None) -> tuple[float, dict[str, float]]:
    """This week's book fraction and target weights under the floor in force
    (the spec's unless given) — ledger.floor_split then ledger.target_weights,
    exactly as selection.simulate graded the champion."""
    frac, ballast = floor_split(floor or SPEC["floor"], gates)
    return frac, target_weights(sleeves, ballast, frac)


# ---- the rolling rule (the spec's `rolling` block, stocks_ml.rolling): the settings re-decided live ----
SETTINGS_KEYS = ("book", "floor", "stop", "cap", "vol_cut")


def settings_in_force(ledger: Ledger) -> dict:
    """The strategy settings this week trades: the rolling rule's decision
    held in the ledger, else the spec's fixed decision (the fallback)."""
    if ledger.settings:
        return dict(ledger.settings)
    return {k: SPEC.get(k) for k in SETTINGS_KEYS}


def settings_text(st: dict) -> str:
    return (f"top-{st['book']} / floor {st['floor']} / cap {st.get('cap') or 'none'} / "
            f"volatility cut {st.get('vol_cut') or 'none'}")


def rolling_decision(ledger: Ledger, ctx: Ctx, live_dir, t, preds: pd.Series, log=_log) -> dict | None:
    """The live side of the rolling rule (rolling.py): this week's ensemble
    scores go into the store's prediction history; when a decision is due
    (every `cadence` rank weeks), the rule decides on its trailing window as
    the backtest did (rolling.decide_live) and the ledger holds the decision
    — book, floor, stop, cap, vol cut, the window — never the evidence (the
    ledger is committed to a public repo). Returns the new decision, if any."""
    import stocks_ml.selection as selmod
    from stocks_ml.rolling import append_history, decide_live, is_due
    rb = SPEC["rolling"]
    hist = append_history(Path(live_dir) / rb["history"], t, preds)
    last = (ledger.settings or {}).get("decided")
    if not is_due(last, t, int(rb["cadence"])):
        log(f"rolling rule: decision of {last} stands ({int(rb['cadence'])}-week cadence); history {hist.week.nunique()} weeks")
        return None
    d = decide_live(selmod, ctx, hist, t, rb["lookback_years"], min_years=int(rb["min_years"]), filt=SPEC.get("filter"))
    new = {k: d[k] for k in ("decided", "lo", "hi", "weeks", *SETTINGS_KEYS)}
    if new["stop"] is not None:
        raise RuntimeError(f"the rolling rule decided a stop ({new['stop']}), which this job does not implement")
    old = ledger.settings
    ledger.settings = new
    log(f"rolling rule: decided {settings_text(new)} on {new['weeks']} weeks {new['lo']} -> {new['hi']}"
        + (f"; was {settings_text(old)} (decided {old['decided']})" if old else "; first decision"))
    return new


def label_text(label: str) -> str:
    """The target as the report names it."""
    return {"label_4w": "4-week label", "label_4w_sector": "sector-relative 4-week label",
            "label_4w_sector_log": "sector-relative 4-week log label",
            "label_4w_sector_clip": "sector-relative 4-week label, clipped",
            "label_4w_sector_rank": "sector-relative 4-week rank label",
            "label_4w_rank": "4-week rank label",
            "label_13w_sector_rank": "sector-relative 13-week rank label (monthly rotation)",
            "label_blend_rank": "blended 4-week + 13-week rank label",
            "label_4w_sector11_rank": "Sharadar-sector-relative 4-week rank label"}[label]


def render_markdown(sig: dict, smap: dict, st: dict | None = None) -> str:
    nav, bench = sig["nav"], sig["spy_nav"]
    counts = sleeve_counts(sig["sleeves"])
    frac = sig.get("book_fraction")
    frac_text = f" (book {frac:.0%} of NAV this week)" if frac is not None else ""
    st = st or sig.get("settings") or {k: SPEC.get(k) for k in SETTINGS_KEYS}
    rule = (f" Settings re-decided by the rolling rule every {SPEC['rolling']['cadence']} rank weeks on the trailing "
            f"{SPEC['rolling']['lookback_years']} years; this decision {st.get('decided', '—')}." if SPEC.get("rolling") else "")
    lines = [f"# r5 signal — {sig['date']}", "",
             f"Champion r5 (PROCEDURE.md): {st['floor']} trend ballast{frac_text}, "
             f"top-{st['book']} four-sleeve stagger, sector cap {st.get('cap')}, volatility cut {st.get('vol_cut') or 'none'}, "
             f"{label_text(SPEC['label'])}, {SPEC['train_years']}-year window, K={K_COPIES}.{rule}", "",
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
