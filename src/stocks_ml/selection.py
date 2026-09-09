"""The full selection procedure as a programmatic pipeline.

`stocks-ml select --sel-start A --sel-end B [--eval-start C --eval-end D]`
runs, in order, with per-stage caching under data/experiments/<name>/:

  1. grid      population per-week top-k returns, horizons {1w,4w}, reference
               2y training window, K=K_COPIES ensembles
  2. wsweep    population holdings at every training window {1..5}y on the
               selection window, at the chosen horizon
  3. holdings  the chosen (horizon, window)'s holdings extended to the eval end
  4. screen    (--screen) the feature screen (feature_screen.py): probe the
               panel's candidates on the window, examine the keepers as one
               bundle in a second holdings run WITH them; admitted features
               enter every stage below
  5. cascade   the documented decisions (PROCEDURE.md "Selection procedure")
  6. grade     frozen config on the eval window (if given) vs sp500

Every decision reads every week of the selection window — no stage samples
(owner mandate 2026-09-04; until then the window and book layers read a
spaced sample of ~116 weeks, the book layer thinned to 29, and its argmax
flipped between sample generations). Compounded statistics average the
kweeks phases so every week counts. No decision reads a rank week whose
forward label ends after the window (label_end): the last five weeks of a
window that ends at the holdout's edge would otherwise be graded on holdout
prices (the screen stage had this rule from the start; the cascade's
horizon, window and book layers got it 2026-09-05, before the full-window
run, with nested3_v1's choice unchanged).
Model config is fixed (MODEL_PARAMS, depth-3 XGBoost) per the procedure card
— never searched.
Every stage appends to models/trials_ledger.json. Stages resume from cache;
`--shard i/n` lets several processes split a stage's weeks.
"""
from __future__ import annotations

import copy
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from stocks_ml.feature_screen import label_span_days
from stocks_ml.ledger import (COST_BPS, Ledger, ballast_state, close_asof,  # noqa: F401
                              pick_capped, rotate_sleeves, target_weights, week_index)

MODEL_PARAMS = dict(max_depth=3, learning_rate=0.02, n_estimators=1500,
                    min_child_weight=20, subsample=0.85, colsample_bytree=0.8,
                    reg_alpha=0.5, reg_lambda=1.0, gamma=0.01, max_bin=256,
                    tree_method="hist")
HORIZONS = {"1w": dict(label="label", purge=10, kweeks=1),
            "4w": dict(label="label_4w", purge=35, kweeks=4)}
WINDOWS = (1, 2, 3, 4, 5)
BOOKS = (3, 6, 10)
FLOORS = ("none", "halfgate", "80/20", "70/30", "60/40")
COST = 0.0010
K_COPIES = 16          # 4 until 2026-09-07 (ledger k16_champion_2006_2015_verdict)
# The holdout's first session. Every pre-holdout grade window ends here as an
# EXCLUSIVE bound (metrics slices index < hi), so the label credited at
# 2024-07-19 — the first holdout close — is never counted. Before 2026-09-09
# some graders used Timestamp("2025") (967/447 weeks, crediting that label)
# and others 07-19-exclusive (966/446): one convention now, this one.
HOLDOUT_START = pd.Timestamp("2024-07-19")
REF_WINDOW = 2


def fixed(purge):
    return dict(n_jobs=-1, random_state=0, eval_fraction=0.1,
                early_stopping_rounds=75, early_stop_purge_days=purge,
                early_stop_metric="weekly_spearman")


class Ctx:
    def __init__(self, data_dir="data/sharadar_world2000",
                 panel_file="panel_sf.parquet"):
        from stocks_ml.config import load_config
        from stocks_ml.data.store import DataStore
        self.cfg = load_config()
        self.world = DataStore(data_dir)
        world = self.world
        p = Path(data_dir) / panel_file
        self.pan = pd.read_parquet(p) if p.exists() else world.read("panel")
        if "yr" in self.pan.columns:
            self.pan = self.pan.drop(columns=["yr"])
        self.prices = world.read("prices")
        mem = world.read("membership")
        self.smap = dict(mem.dropna(subset=["sector"])
                         .drop_duplicates("ticker")[["ticker", "sector"]].values)
        daily = self.prices.sort_values("date")
        self.__dict__.update(price_frames(
            daily.pivot(index="date", columns="ticker", values="close").sort_index(),
            daily.pivot(index="date", columns="ticker", values="open").sort_index()))
        self.members = {d: list(g["ticker"])
                        for d, g in self.pan[["date", "ticker"]].groupby("date")}
        self.weeks = sorted(self.members)
        self.extra: list[str] = []      # panel columns the model gets beyond feature_cols

    def world_cfg(self, train_years):
        c = copy.copy(self.cfg)
        for k, v in (("data_dir", "data/sharadar_world2000"),
                     ("backtest_start", pd.Timestamp("2001-01-05")),
                     ("cv_train_years", train_years)):
            try:
                setattr(c, k, v)
            except AttributeError:
                object.__setattr__(c, k, v)
        return c


def ensemble_preds(ctx, t, horizon, train_years):
    from stocks_ml.models.walk import walk_forward_predictions
    from stocks_ml.models.xgb import TimeTailEarlyStopXGB
    from stocks_ml.models.replication import WeekBootstrapEstimator
    h = HORIZONS[horizon]
    cfg2 = ctx.world_cfg(train_years)
    copies = []
    for c in range(1, K_COPIES + 1):
        est = WeekBootstrapEstimator(
            TimeTailEarlyStopXGB(**MODEL_PARAMS, **fixed(h["purge"])),
            bootstrap_seed=c)
        wf = walk_forward_predictions(ctx.pan, est, cfg2, start=t, end=t,
                                      label_col=h["label"],
                                      purge_days=h["purge"],
                                      extra_features=tuple(ctx.extra))
        p = wf.preds.get(t)
        if p is not None:
            copies.append(p)
    if not copies:
        return None
    p = pd.concat(copies, axis=1).mean(axis=1)
    return p if p.nunique() >= 20 else None


def price_frames(closes, opens):
    """Everything the engine reads from daily closes and opens: the frames
    themselves (closes carried forward), weekly closes on the W-FRI grid,
    weekly returns, SPY's weekly closes, and per horizon the forward return
    on the fill basis — a pick dated in week t is bought at the first open
    after t and sold at the first open k weeks later (the training labels'
    basis, and the live job's)."""
    closes = closes.ffill()
    cw = closes.resample("W-FRI").last()
    fills = next_open(opens, cw.index)
    return {"closes": closes, "opens": opens, "cw": cw,
            "wret": cw.pct_change(fill_method=None), "spy_w": cw["SPY"],
            "fwd": {h: fills.pct_change(c["kweeks"], fill_method=None).shift(-c["kweeks"])
                    for h, c in HORIZONS.items()}}


def next_open(opens, labels):
    """Fill price after each week label: the open of the first session after
    the label, or the first open within five sessions of it (ledger.fill_price);
    NaN where there is none."""
    pos = opens.index.searchsorted(labels, side="right")
    ok = pos < len(opens.index)
    out = opens.bfill(limit=4).iloc[pos[ok]]
    out.index = labels[ok]
    return out.reindex(labels)


def week_slot(index, t):
    """W-FRI label of the week a rank date t belongs to (first label >= t),
    or None past the grid. Rank dates are the week's last trading day, so a
    Thursday before a Friday holiday still maps to its own week; `asof`
    would snap it to the previous Friday and pay the pick for a week that
    had already happened."""
    i = int(index.searchsorted(t))
    return index[i] if i < len(index) else None


def slice_row(ctx, t, horizon, preds):
    wk = week_slot(ctx.fwd[horizon].index, t)
    if wk is None:
        return None
    r = ctx.fwd[horizon].loc[wk]
    uni = [x for x in ctx.members[t] if x in r.index and not pd.isna(r[x])]
    if len(uni) < 100 or pd.isna(r.get("SPY")):
        return None
    p = preds.loc[preds.index.intersection(pd.Index(uni))]
    order = p.sort_values(ascending=False).index
    row = {"week": t, "spy": float(r["SPY"]),
           "rand_mean": float(r.loc[uni].mean()),
           "top15": ",".join(order[:15])}
    for k in BOOKS:
        row[f"top{k}"] = float(r.loc[order[:k]].mean())
    return row


def _stage_spec(extra=()):
    """What a stage cache's rows depend on beyond the file name: the
    ensemble recipe and the feature set. Written beside each cache file so a
    resume after a recipe change refuses instead of mixing rows."""
    return {"k_copies": K_COPIES, "model_params": MODEL_PARAMS,
            "features": sorted(extra)}


def _stage_loop(ctx, todo, out_path, fn, checkpoint=25, spec=None):
    done, rows = set(), []
    sp = Path(str(out_path) + ".spec.json")
    if Path(out_path).exists():
        if spec is not None:
            if not sp.exists():
                raise RuntimeError(
                    f"{out_path} predates recipe stamping: verify it was written under "
                    f"{spec} and write that to {sp.name}, or move the file aside")
            old = json.loads(sp.read_text())
            if old != json.loads(json.dumps(spec)):
                raise RuntimeError(f"{out_path} was written under {old}; the current recipe "
                                   f"is {spec} — resuming would mix rows. Use a fresh out dir.")
        old = pd.read_parquet(out_path)
        rows = old.to_dict("records")
        done = set(pd.to_datetime(old["week"]))
    if spec is not None and not sp.exists():
        sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(spec, indent=1, sort_keys=True))
    todo = [t for t in todo if t not in done]
    for i, t in enumerate(todo):
        row = fn(t)
        if row is not None:
            rows.append(row)
        if (i + 1) % checkpoint == 0:
            pd.DataFrame(rows).to_parquet(out_path)
            print(f"  {Path(out_path).stem}: {i+1}/{len(todo)}", flush=True)
    pd.DataFrame(rows).to_parquet(out_path)
    return len(rows)


def stage_grid(ctx, out, lo, hi, shard=(0, 1)):
    weeks = [t for t in ctx.weeks if lo <= t <= hi]
    weeks = [t for i, t in enumerate(weeks) if i % shard[1] == shard[0]]
    for h in HORIZONS:
        _stage_loop(ctx, weeks, f"{out}/grid_{h}_s{shard[0]}.parquet",
                    lambda t, h=h: (lambda p: slice_row(ctx, t, h, p)
                                    if p is not None else None)(
                        ensemble_preds(ctx, t, h, REF_WINDOW)),
                    spec=_stage_spec(ctx.extra))


def sample_weeks(weeks, lo, hi, spacing=28, seed=11):
    """Spaced random weeks for the cheap paired exams under ops/ (sample-first
    compute). No selection decision reads a sample."""
    rng = np.random.default_rng(seed)
    out, last = [], pd.Timestamp("1900-01-01")
    for t in weeks:
        if lo <= t <= hi and (t - last).days >= spacing and rng.random() < 0.9:
            out.append(t)
            last = t
    return out


def stage_wsweep(ctx, out, horizon, lo, hi, shard=(0, 1)):
    """Population holdings at every training window on the selection window:
    the window decision's evidence, one file per window (the chosen window's
    file is what stage holdings extends and the cascade grades)."""
    for yrs in WINDOWS:
        print(f"  wsweep: {holdings_name(horizon, yrs, ctx.extra)} on {lo.date()} -> {hi.date()}", flush=True)
        stage_holdings(ctx, out, horizon, yrs, lo, hi, shard)


def holdings_name(horizon, train_years, features=()):
    """Stem of the holdings files: `_x` plus a short hash of the bundle's
    names marks a run WITH extra features, so a walk resumed from cache can
    never mix two bundles."""
    stem = f"holdings_{horizon}_{train_years}y"
    if features:
        stem += "_x" + hashlib.sha1(",".join(sorted(features)).encode()).hexdigest()[:6]
    return stem


def stage_holdings(ctx, out, horizon, train_years, lo, hi, shard=(0, 1)):
    weeks = [t for t in ctx.weeks if lo <= t <= hi]
    weeks = [t for i, t in enumerate(weeks) if i % shard[1] == shard[0]]
    _stage_loop(ctx, weeks,
                f"{out}/{holdings_name(horizon, train_years, ctx.extra)}_s{shard[0]}.parquet",
                lambda t: (lambda p: slice_row(ctx, t, horizon, p)
                           if p is not None else None)(
                    ensemble_preds(ctx, t, horizon, train_years)),
                spec=_stage_spec(ctx.extra))


def stage_screen(ctx, out, horizon, train_years, lo, hi, end, shard=(0, 1),
                 name=None, report_path=None):
    """The feature screen (feature_screen.py): probe the panel's candidates on
    the selection window; run the holdings stage again WITH the keepers
    (ctx.extra) over the same weeks; verdict from the paired difference.
    Writes screen.json / screen.md under `out` and one ledger row."""
    import stocks_ml.feature_screen as fs
    from stocks_ml.models.trials import record_trials
    out = Path(out)
    kweeks, label = HORIZONS[horizon]["kweeks"], HORIZONS[horizon]["label"]
    probe_path = out / "screen_probe.parquet"
    if probe_path.exists():
        res = pd.read_parquet(probe_path)
    else:
        print("stage screen: probe", flush=True)
        frame = fs.probe_frame(ctx.pan, ctx.world, lo, hi, kweeks, label,
                               log=lambda m: print(f"  {m}", flush=True))
        res = fs.probe(frame, lo, hi, kweeks, label)
        res.to_parquet(probe_path)
    keep = fs.keepers(res)
    print(f"stage screen: keepers {keep or 'none'}", flush=True)
    ex = df = None
    if keep:
        ctx.extra = list(keep)
        try:
            stage_holdings(ctx, out, horizon, train_years, lo, end, shard)
        finally:
            ctx.extra = []
        without = _load(out, f"{holdings_name(horizon, train_years)}_s*.parquet")
        with_ = _load(out, f"{holdings_name(horizon, train_years, keep)}_s*.parquet")
        assert without is not None, "holdings stage not run — run stage holdings first"
        need = set(without.week[(without.week >= lo) & (without.week <= hi)])
        if not need <= set(with_.week):
            print(f"stage screen: with-arm holdings cover {len(need & set(with_.week))}/{len(need)} "
                  f"window weeks — rerun --stage screen once every shard has finished", flush=True)
            return None
        df = fs.paired_rows(without, with_, lo, hi, kweeks)
        ex = fs.exam(df, kweeks)
    name = name or out.name
    summary = fs.write_screen(out, name, (lo, hi), kweeks, label, res, keep, ex, df, report_path)
    stats = None if ex is None else ex["stats"][fs.PRIMARY]
    record_trials([{"kind": "feature_screen", "name": name,
                    "config": {"window": summary["window"], "horizon": horizon, "train_years": int(train_years),
                               "candidates": len(res), "keepers": keep,
                               "exam_weeks": None if df is None else int(len(df)),
                               "hac_bandwidth_days": None if ex is None else ex["hac_bandwidth_days"]},
                    **({} if stats is None else {"top6_diff": stats["diff"], "top6_t": stats["t"],
                                                 "top6_cmp": ex["compounded_pct"]}),
                    "passed": None if ex is None else ex["passed"], "admitted": summary["admitted"],
                    "notes": ("no keeper cleared the probe" if ex is None else
                              f"{'ADMITTED' if ex['passed'] else 'NOT ADMITTED'} ({ex['rule']}): top-6 "
                              f"compounded {ex['compounded_pct']['without']:.2f} -> "
                              f"{ex['compounded_pct']['with']:.2f} %/yr, paired diff {stats['diff']:+.2%}, "
                              f"HAC t {stats['t']:+.2f} on {len(df)} weeks") + f"; {out / 'screen.md'}"}])
    print(f"SCREEN: admitted {summary['admitted'] or 'nothing'}", flush=True)
    return summary


def _load(out, pattern):
    fs = glob.glob(f"{out}/{pattern}")
    if not fs:
        return None
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["week"] = pd.to_datetime(df["week"])
    return df.drop_duplicates("week").sort_values("week")


# ---- pure decision functions (unit-testable) ----
def label_end(hi, kweeks=max(c["kweeks"] for c in HORIZONS.values())):
    """Last rank week whose k-week forward label (fill basis) ends inside a
    window closing at `hi` — the screen stage's rule (feature_screen). The
    default is the longest horizon's span, so layers that compare horizons
    read the same weeks."""
    return pd.Timestamp(hi) - pd.Timedelta(days=label_span_days(kweeks))


def compounded_pct(df, col, kweeks, lo, hi):
    """Cost-adjusted compounded %/yr of a book held kweeks, on every week of
    [lo, hi]: the non-overlapping chain from each of the kweeks phases,
    averaged (a single phase would leave three weeks in four unread)."""
    g = df[(df.week >= lo) & (df.week <= hi)].sort_values("week")
    slots = g["week"].map(week_index)      # calendar phase: a missing week
    out = []                               # cannot re-phase the chains
    for phase in range(kweeks):
        r = g.loc[slots % kweeks == phase, col] - COST
        if not len(r):
            continue
        yrs = len(r) * kweeks / 52
        out.append((float(np.prod(1 + r)) ** (1 / yrs) - 1) * 100)
    return float(np.mean(out))


def decide_horizon(grids: dict, lo, hi) -> str:
    """Cost-adjusted compounded %/yr of the top-6 book decides, both horizons
    read on the weeks whose labels end inside the window."""
    last = label_end(hi)
    res = {h: compounded_pct(df, "top6", HORIZONS[h]["kweeks"], lo, last)
           for h, df in grids.items()}
    return max(res, key=res.get), res


def decide_window(sweeps: dict, lo, hi):
    """top-6 edge vs random, paired on the weeks every window has, decides.
    `sweeps` holds each window's population holdings (load_windows)."""
    common, last = None, label_end(hi)
    for df in sweeps.values():
        w = set(df[(df.week >= lo) & (df.week <= last)]["week"])
        common = w if common is None else common & w
    res = {}
    for yrs, df in sweeps.items():
        g = df[df.week.isin(common)]
        res[yrs] = float((g["top6"] - g["rand_mean"]).mean()) * 13 * 100
    return max(res, key=res.get), res


def decide_book(df, horizon, lo, hi):
    """Cost-adjusted compounded %/yr of each book on the chosen window's
    population holdings (the frame the cascade grades) decides."""
    kw = HORIZONS[horizon]["kweeks"]
    res = {k: compounded_pct(df, f"top{k}", kw, lo, label_end(hi, kw)) for k in BOOKS}
    return max(res, key=res.get), res


def load_windows(out, horizon, lo, hi, min_coverage=0.95):
    """Every window's population holdings on [lo, hi]; a missing or
    unfinished window (fewer weeks than 95% of the fullest) is an error,
    so no decision reads a partial stage."""
    frames = {y: d for y in WINDOWS
              if (d := _load(out, f"{holdings_name(horizon, y)}_s*.parquet")) is not None}
    missing = [y for y in WINDOWS if y not in frames]
    n = {y: int(((d.week >= lo) & (d.week <= hi)).sum()) for y, d in frames.items()}
    short = [y for y, k in n.items() if k < min_coverage * max(n.values(), default=0)]
    if missing or short:
        raise RuntimeError(f"window sweep incomplete on {lo.date()} -> {hi.date()}: "
                           f"missing {missing}, short {short} of {n} weeks — rerun --stage wsweep")
    return frames


def floor_split(floor, gates):
    """The floor menu as (book fraction, ballast thirds) for ledger.target_weights.
    `gates` is ballast_state's per-window SPY/IEF reading."""
    if floor == "none":
        return 1.0, {}
    if floor == "halfgate":
        # book exposure 1 - g/2, g the share of windows below their mean; the
        # rest in IEF
        below = {w: f for w, f in gates.items() if f == "IEF"}
        return 1.0 - 0.5 * len(below) / len(gates), below
    return {"80/20": 0.8, "70/30": 0.7, "60/40": 0.6}[floor], dict(gates)


def simulate(ctx, holdings, horizon, book, cap, stop, floor, trace=None):
    """Weekly returns of the configured book under the live job's rules
    (stocks_ml.ledger): each rank date's target weights are filled at the
    next session's open at COST_BPS a side, rebalances under 0.5% of NAV are
    skipped, and NAV is marked at each week's last close. Indexed by the
    week-ending label each return is credited to; a week with no pick holds
    the book. `stop` moves a name that has fallen that far from its rotation
    close into SPY until its sleeve rotates. Pass a list as `trace` to
    receive one record per credited week (pick date, sleeves, fills, per-name
    $ and price returns, gate)."""
    ranked = {r.week: r.top15.split(",") for r in holdings.itertuples()}
    grid = ctx.wret.index
    by_label = {}
    for t in sorted(ranked):
        lab = week_slot(grid, t)
        if lab is not None:
            by_label[lab] = t
    if not by_label:
        return pd.Series(dtype=float)
    n_sleeves = HORIZONS[horizon]["kweeks"]
    labels = grid[(grid >= min(by_label)) & (grid <= max(by_label) + pd.Timedelta(days=7))]
    led = Ledger.new(100.0, min(by_label))
    entry, stopped = {}, {}                  # sleeve -> {name: rotation close} / stopped names
    nav, prev, t, rotated, g, weights = {}, None, None, [], 0.0, {}
    for lab in labels:
        v_prev = led.value_of(ctx.closes, prev) if trace is not None and prev is not None else {}
        fills = led.fill_pending(ctx.closes, ctx.opens, lab)
        nav[lab], _ = led.mark(ctx.closes, lab)
        if trace is not None and prev is not None:
            trace.append({"t": t, "wk": prev, "nxt": lab, "r": nav[lab] / nav[prev] - 1.0,
                          "rotated": rotated, "g": g, "nav": nav[lab], "weights": weights,
                          "fills": fills,
                          "sleeves": [list(s["names"]) for s in led.sleeves.values()],
                          **_attribute(ctx, led, v_prev, fills, prev, lab)})
        prev = lab
        if lab not in by_label:                # no signal this week: hold the book
            rotated, weights = [], {}
            continue
        t = by_label[lab]
        sleeves, rotated = rotate_sleeves(led.sleeves, t, ranked[t], ctx.smap,
                                          n_sleeves, book, cap)
        for k in rotated:
            entry[k] = {n: close_asof(ctx.closes, n, t)[0] for n in sleeves[str(k)]["names"]}
            stopped[k] = set()
        held = sleeves
        if stop is not None:
            held = {}
            for k, s in sleeves.items():
                for n in s["names"]:
                    px, e = close_asof(ctx.closes, n, t)[0], entry[int(k)].get(n)
                    if e and np.isfinite(px) and np.isfinite(e) and px / e - 1 <= stop:
                        stopped[int(k)].add(n)
                held[k] = {"names": ["SPY" if n in stopped[int(k)] else n for n in s["names"]]}
        gates = ballast_state(ctx.spy_w, t)
        g = sum(f == "IEF" for f in gates.values()) / len(gates)
        frac, ballast = floor_split(floor, gates)
        weights = target_weights(held, ballast, frac)
        led.sleeves = sleeves
        led.pending = {"decision_date": str(t.date()), "weights": weights}
    return pd.Series(nav).sort_index().pct_change().iloc[1:]


def _attribute(ctx, led, v_prev, fills, wk, nxt):
    """Per name over the week wk -> nxt: `pnl` in $ (value change less the
    cash paid for it, fees included; sums to the NAV change) and `vals`, the
    name's price return over the part of the week the book held it."""
    v_now = led.value_of(ctx.closes, nxt)
    pnl, vals = {}, {}
    paid, first_px = {}, {}
    for _, tk, units, price, fee in fills:
        paid[tk] = paid.get(tk, 0.0) + units * price + fee
        first_px.setdefault(tk, price)
    for tk in set(v_prev) | set(v_now) | set(paid):
        pnl[tk] = v_now.get(tk, 0.0) - v_prev.get(tk, 0.0) - paid.get(tk, 0.0)
        c1, c0 = close_asof(ctx.closes, tk, nxt)[0], close_asof(ctx.closes, tk, wk)[0]
        if tk in v_prev and tk in v_now:
            v = c1 / c0 - 1.0
        elif tk in v_now:                                  # bought this week
            v = c1 / first_px[tk] - 1.0
        else:                                              # sold out this week
            v = first_px[tk] / c0 - 1.0 if tk in first_px else 0.0
        vals[tk] = float(v) if np.isfinite(v) else 0.0
    return {"pnl": pnl, "vals": vals}


def sharpe(series, lo, hi):
    x = series[(series.index >= lo) & (series.index < hi)].dropna()
    return float(x.mean() / x.std(ddof=1) * np.sqrt(52))


def metrics(series, lo, hi):
    x = series[(series.index >= lo) & (series.index < hi)].dropna()
    nav = np.cumprod(1 + x.values)
    yrs = len(x) / 52
    return {"terminal_100": round(float(nav[-1] * 100), 1),
            "cagr_pct": round((float(nav[-1]) ** (1 / yrs) - 1) * 100, 2),
            "sharpe": round(sharpe(series, lo, hi), 3),
            "max_dd": round(float((1 - nav / np.maximum.accumulate(nav)).max()), 3),
            "n_weeks": len(x)}


def run_cascade(ctx, out, lo, hi, features=(), name=None):
    """Horizon and window are decided on the standing features; from the book
    down every decision reads the holdings the run grades (the `_x` file of a
    screened run). The ledger row is named after the run (`name`, `_x` for
    the with-features arm), so two runs on one window keep separate rows."""
    from stocks_ml.models.trials import record_trials
    grids = {h: _load(out, f"grid_{h}_s*.parquet") for h in HORIZONS}
    horizon, hres = decide_horizon(grids, lo, hi)
    if horizon == "4w":
        window, wres = decide_window(load_windows(out, horizon, lo, hi), lo, hi)
    else:
        window, wres = REF_WINDOW, {"fixed": "1w keeps reference window"}
    holdings = _load(out, f"{holdings_name(horizon, window, features)}_s*.parquet")
    assert holdings is not None, \
        f"holdings stage not run for {horizon}/{window}y — run stage holdings"
    book, bres = decide_book(holdings, horizon, lo, hi)
    fres = {f: sharpe(simulate(ctx, holdings, horizon, book, None, None, f), lo, hi)
            for f in FLOORS}
    floor = max(fres, key=fres.get)
    sres = {str(s): sharpe(simulate(ctx, holdings, horizon, book, None, s, floor), lo, hi)
            for s in (None, -0.25)}
    stop = None if sres["None"] >= sres["-0.25"] else -0.25
    cres = {str(c): sharpe(simulate(ctx, holdings, horizon, book, c, stop, floor), lo, hi)
            for c in (None, 2)}
    cap = None if cres["None"] >= cres["2"] else 2
    config = {"horizon": horizon, "train_years": int(window), "book": int(book),
              "floor": floor, "stop": stop, "cap": cap, "features": list(features),
              "evidence": {"horizon": hres, "window": {str(k): round(v, 2) if isinstance(v, float) else v for k, v in wres.items()},
                           "book": {str(k): round(v, 2) for k, v in bres.items()},
                           "floor": {k: round(v, 3) for k, v in fres.items()},
                           "stop": {k: round(v, 3) for k, v in sres.items()},
                           "cap": {k: round(v, 3) for k, v in cres.items()}}}
    record_trials([{"kind": "select_pipeline",
                    "name": f"{name or f'select_{lo.date()}_{hi.date()}'}{'_x' if features else ''}",
                    "notes": json.dumps({k: config[k] for k in
                                         ("horizon", "train_years", "book",
                                          "floor", "stop", "cap", "features")})}])
    return config


def decide_engine(out, lo, hi):
    """(horizon, window) from the cached grid and wsweep stages."""
    grids = {h: _load(out, f"grid_{h}_s*.parquet") for h in HORIZONS}
    h, _ = decide_horizon(grids, lo, hi)
    w = decide_window(load_windows(out, h, lo, hi), lo, hi)[0] if h == "4w" else REF_WINDOW
    return h, w


def run_select(sel_start, sel_end, eval_start=None, eval_end=None,
               name=None, stage="all", shard=(0, 1), screen=False):
    from stocks_ml.feature_screen import load_screen
    lo, hi = pd.Timestamp(sel_start), pd.Timestamp(sel_end)
    name = name or f"select_{lo.date()}_{hi.date()}"
    out = Path("data/experiments") / name
    out.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp(eval_end) if eval_end else hi
    ctx = Ctx()
    if stage in ("all", "grid"):
        print("stage grid", flush=True)
        stage_grid(ctx, out, lo, hi, shard)
    if stage in ("all", "wsweep"):
        grids = {h: _load(out, f"grid_{h}_s*.parquet") for h in HORIZONS}
        if all(g is not None for g in grids.values()):
            h, _ = decide_horizon(grids, lo, hi)
            if h == "4w":
                print("stage wsweep", flush=True)
                stage_wsweep(ctx, out, h, lo, hi, shard)
    if stage in ("all", "holdings"):
        h, w = decide_engine(out, lo, hi)
        print(f"stage holdings ({h}/{w}y)", flush=True)
        stage_holdings(ctx, out, h, w, lo, end, shard)
    if stage == "screen" or (screen and stage == "all"):
        h, w = decide_engine(out, lo, hi)
        print(f"stage screen ({h}/{w}y)", flush=True)
        stage_screen(ctx, out, h, w, lo, hi, end, shard, name=name,
                     report_path=Path("reports") / f"{name}_screen.md")
    if stage in ("all", "cascade"):
        # --screen: cascade on the screen stage's admitted features (if any);
        # its outputs carry the `_x` suffix so both arms live in one experiment
        features = (load_screen(out) or {}).get("admitted", []) if screen else []
        sfx = "_x" if features else ""
        config = run_cascade(ctx, out, lo, hi, features, name=name)
        (out / f"frozen_config{sfx}.json").write_text(json.dumps(config, indent=2))
        print("FROZEN:", {k: config[k] for k in
                          ("horizon", "train_years", "book", "floor", "stop", "cap", "features")})
        if eval_start:
            holdings = _load(out, f"{holdings_name(config['horizon'], config['train_years'], features)}"
                                  f"_s*.parquet")
            s = simulate(ctx, holdings, config["horizon"], config["book"],
                         config["cap"], config["stop"], config["floor"])
            elo, ehi = pd.Timestamp(eval_start), pd.Timestamp(eval_end)
            rep = {"config": metrics(s, elo, ehi),
                   "sp500": metrics(ctx.wret["SPY"].reindex(s.index), elo, ehi)}
            (out / f"eval{sfx}.json").write_text(json.dumps(rep, indent=2))
            print("EVAL:", json.dumps(rep))
    return out
