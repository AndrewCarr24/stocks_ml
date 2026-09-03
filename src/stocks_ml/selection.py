"""The full selection procedure as a programmatic pipeline.

`stocks-ml select --sel-start A --sel-end B [--eval-start C --eval-end D]`
runs, in order, with per-stage caching under data/experiments/<name>/:

  1. grid      population per-week top-k returns, horizons {1w,4w}, reference
               2y training window, K=4 ensembles
  2. wsweep    sampled-week training-window sweep {1..5}y at the chosen horizon
  3. holdings  full-population ranked top-15 for the chosen (horizon, window)
  4. cascade   the documented decisions (PROCEDURE.md "Selection procedure")
  5. grade     frozen config on the eval window (if given) vs sp500

Model config is fixed (MODEL_PARAMS, depth-3 XGBoost) per the procedure card
— never searched.
Every stage appends to models/trials_ledger.json. Stages resume from cache;
`--shard i/n` lets several processes split a stage's weeks.
"""
from __future__ import annotations

import copy
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd

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
K_COPIES = 4
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
        world = DataStore(data_dir)
        p = Path(data_dir) / panel_file
        self.pan = pd.read_parquet(p) if p.exists() else world.read("panel")
        if "yr" in self.pan.columns:
            self.pan = self.pan.drop(columns=["yr"])
        self.prices = world.read("prices")
        mem = world.read("membership")
        self.smap = dict(mem.dropna(subset=["sector"])
                         .drop_duplicates("ticker")[["ticker", "sector"]].values)
        self.cw = (self.prices.pivot(index="date", columns="ticker", values="close")
                   .sort_index().ffill().resample("W-FRI").last())
        self.wret = self.cw.pct_change(fill_method=None)
        self.spy_w = self.cw["SPY"]
        self.fwd = {h: self.cw.pct_change(c["kweeks"], fill_method=None)
                    .shift(-c["kweeks"]) for h, c in HORIZONS.items()}
        self.members = {d: list(g["ticker"])
                        for d, g in self.pan[["date", "ticker"]].groupby("date")}
        self.weeks = sorted(self.members)

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
                                      purge_days=h["purge"])
        p = wf.preds.get(t)
        if p is not None:
            copies.append(p)
    if not copies:
        return None
    p = pd.concat(copies, axis=1).mean(axis=1)
    return p if p.nunique() >= 20 else None


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


def _stage_loop(ctx, todo, out_path, fn, checkpoint=25):
    done, rows = set(), []
    if Path(out_path).exists():
        old = pd.read_parquet(out_path)
        rows = old.to_dict("records")
        done = set(pd.to_datetime(old["week"]))
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
                        ensemble_preds(ctx, t, h, REF_WINDOW)))


def sample_weeks(weeks, lo, hi, spacing=28, seed=11):
    rng = np.random.default_rng(seed)
    out, last = [], pd.Timestamp("1900-01-01")
    for t in weeks:
        if lo <= t <= hi and (t - last).days >= spacing and rng.random() < 0.9:
            out.append(t)
            last = t
    return out


def stage_wsweep(ctx, out, horizon, lo, hi, shard=(0, 1)):
    weeks = sample_weeks(ctx.weeks, lo, hi)
    weeks = [t for i, t in enumerate(weeks) if i % shard[1] == shard[0]]
    for yrs in WINDOWS:
        _stage_loop(ctx, weeks, f"{out}/wsweep_{yrs}y_s{shard[0]}.parquet",
                    lambda t, y=yrs: (lambda p: slice_row(ctx, t, horizon, p)
                                      if p is not None else None)(
                        ensemble_preds(ctx, t, horizon, y)), checkpoint=10)


def stage_holdings(ctx, out, horizon, train_years, lo, hi, shard=(0, 1)):
    weeks = [t for t in ctx.weeks if lo <= t <= hi]
    weeks = [t for i, t in enumerate(weeks) if i % shard[1] == shard[0]]
    _stage_loop(ctx, weeks,
                f"{out}/holdings_{horizon}_{train_years}y_s{shard[0]}.parquet",
                lambda t: (lambda p: slice_row(ctx, t, horizon, p)
                           if p is not None else None)(
                    ensemble_preds(ctx, t, horizon, train_years)))


def _load(out, pattern):
    fs = glob.glob(f"{out}/{pattern}")
    if not fs:
        return None
    df = pd.concat([pd.read_parquet(f) for f in fs], ignore_index=True)
    df["week"] = pd.to_datetime(df["week"])
    return df.drop_duplicates("week").sort_values("week")


# ---- pure decision functions (unit-testable) ----
def decide_horizon(grids: dict, lo, hi) -> str:
    """Cost-adjusted compounded %/yr of the top-6 book decides."""
    res = {}
    for h, df in grids.items():
        kw = HORIZONS[h]["kweeks"]
        g = df[(df.week >= lo) & (df.week <= hi)].iloc[::kw]
        r = g["top6"] - COST
        yrs = len(g) * kw / 52
        res[h] = (float(np.prod(1 + r)) ** (1 / yrs) - 1) * 100
    return max(res, key=res.get), res


def decide_window(sweeps: dict, lo, hi):
    """top-6 edge vs random on paired sampled weeks decides."""
    common = None
    for df in sweeps.values():
        w = set(df[(df.week >= lo) & (df.week <= hi)]["week"])
        common = w if common is None else common & w
    res = {}
    for yrs, df in sweeps.items():
        g = df[df.week.isin(common)]
        res[yrs] = float((g["top6"] - g["rand_mean"]).mean()) * 13 * 100
    return max(res, key=res.get), res


def decide_book(df, horizon, lo, hi):
    kw = HORIZONS[horizon]["kweeks"]
    g = df[(df.week >= lo) & (df.week <= hi)].iloc[::kw]
    res = {}
    for k in BOOKS:
        r = g[f"top{k}"] - COST
        yrs = len(g) * kw / 52
        res[k] = (float(np.prod(1 + r)) ** (1 / yrs) - 1) * 100
    return max(res, key=res.get), res


def pick_capped(names, cap, k, smap):
    if cap is None:
        return names[:k]
    out, cnt = [], {}
    for n in names:
        s = smap.get(n, "UNK")
        if cnt.get(s, 0) < cap:
            out.append(n)
            cnt[s] = cnt.get(s, 0) + 1
        if len(out) == k:
            break
    return out if len(out) == k else (out + [n for n in names if n not in out])[:k]


def simulate(ctx, holdings, horizon, book, cap, stop, floor):
    ranked = {r.week: r.top15.split(",") for r in holdings.itertuples()}
    weeks = sorted(ranked)
    period = HORIZONS[horizon]["kweeks"]
    ncoh = period
    cohorts = {c: {"names": [], "entry": {}} for c in range(max(ncoh, 1))}
    rets, exp_prev = {}, 1.0
    grid = ctx.wret.index
    for i, t in enumerate(weeks):
        wk = week_slot(grid, t)
        if wk is None:
            break
        loc = grid.get_loc(wk)
        if loc + 1 >= len(grid):
            break
        # the book decided at t is held until the next pick's week: a week
        # with no pick keeps the previous book and still counts
        if i + 1 < len(weeks):
            nxt_slot = week_slot(grid, weeks[i + 1])
            end_loc = len(grid) - 1 if nxt_slot is None else grid.get_loc(nxt_slot)
            end_loc = max(end_loc, loc + 1)
        else:
            end_loc = loc + 1
        c_ = 0.0
        for c, st in cohorts.items():
            if (i % max(period, 1) == c) or not st["names"]:
                st["names"] = pick_capped(ranked[t], cap, book, ctx.smap)
                st["entry"] = {n: float(ctx.cw[n].asof(wk))
                               if n in ctx.cw.columns else np.nan
                               for n in st["names"]}
                c_ += COST / len(cohorts)
        for j in range(loc, end_loc):
            wk, nxt = grid[j], grid[j + 1]
            rets[nxt], exp_prev = _credit_week(ctx, cohorts, wk, nxt, c_, book,
                                               stop, floor, exp_prev)
            c_ = 0.0
    return pd.Series(rets).sort_index()


def _credit_week(ctx, cohorts, wk, nxt, c_, book, stop, floor, exp_prev):
    """Return of the book held at wk's close over the week ending nxt, after
    the rotation cost c_ already paid this week, plus the floor's exposure
    state for the next call."""
    rs = []
    for st in cohorts.values():
        vals = []
        for n in st["names"]:
            if n not in ctx.wret.columns:
                vals.append(0.0)
                continue
            if stop is not None:
                e, px = st["entry"].get(n), ctx.cw[n].asof(wk)
                if e and not pd.isna(px) and not pd.isna(e) and px / e - 1 <= stop:
                    v = ctx.wret.loc[nxt, "SPY"]
                    c_ += COST / (len(cohorts) * book)
                    vals.append(0.0 if pd.isna(v) else float(v))
                    continue
            v = ctx.wret.loc[nxt, n]
            vals.append(0.0 if pd.isna(v) else float(v))
        rs.append(float(np.mean(vals)))
    book_r = float(np.mean(rs)) - c_
    hist = ctx.spy_w[ctx.spy_w.index <= wk]
    g = float(np.mean([1.0 if len(hist) >= W and hist.iloc[-1] < hist.iloc[-W:].mean()
                       else 0.0 for W in (30, 40, 52)]))
    fvals = []
    for W in (30, 40, 52):
        below = len(hist) >= W and hist.iloc[-1] < hist.iloc[-W:].mean()
        v = (ctx.wret.loc[nxt, "IEF"]
             if (below and "IEF" in ctx.wret.columns) else ctx.wret.loc[nxt, "SPY"])
        fvals.append(0.0 if pd.isna(v) else float(v))
    fr = float(np.mean(fvals))
    if floor == "halfgate":
        exp = 1.0 - 0.5 * g
        ir = ctx.wret.loc[nxt].get("IEF")
        ir = 0.0 if pd.isna(ir) else float(ir)
        r = exp * book_r + (1 - exp) * ir - COST * abs(exp - exp_prev)
        exp_prev = exp
    elif floor in ("80/20", "70/30", "60/40"):
        a = {"80/20": 0.8, "70/30": 0.7, "60/40": 0.6}[floor]
        r = a * book_r + (1 - a) * fr
    else:
        r = book_r
    return r, exp_prev


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


def run_cascade(ctx, out, lo, hi):
    from stocks_ml.models.trials import record_trials
    grids = {h: _load(out, f"grid_{h}_s*.parquet") for h in HORIZONS}
    horizon, hres = decide_horizon(grids, lo, hi)
    if horizon == "4w":
        sweeps = {y: _load(out, f"wsweep_{y}y_s*.parquet") for y in WINDOWS}
        sweeps = {y: d for y, d in sweeps.items() if d is not None}
        window, wres = decide_window(sweeps, lo, hi)
    else:
        window, wres = REF_WINDOW, {"fixed": "1w keeps reference window"}
    src = _load(out, f"wsweep_{window}y_s*.parquet") if window != REF_WINDOW \
        else grids[horizon]
    book, bres = decide_book(src, horizon, lo, hi)
    holdings = _load(out, f"holdings_{horizon}_{window}y_s*.parquet")
    assert holdings is not None, \
        f"holdings stage not run for {horizon}/{window}y — run stage holdings"
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
              "floor": floor, "stop": stop, "cap": cap,
              "evidence": {"horizon": hres, "window": {str(k): round(v, 2) if isinstance(v, float) else v for k, v in wres.items()},
                           "book": {str(k): round(v, 2) for k, v in bres.items()},
                           "floor": {k: round(v, 3) for k, v in fres.items()},
                           "stop": {k: round(v, 3) for k, v in sres.items()},
                           "cap": {k: round(v, 3) for k, v in cres.items()}}}
    record_trials([{"kind": "select_pipeline", "name": f"select_{lo.date()}_{hi.date()}",
                    "notes": json.dumps({k: config[k] for k in
                                         ("horizon", "train_years", "book",
                                          "floor", "stop", "cap")})}])
    return config


def run_select(sel_start, sel_end, eval_start=None, eval_end=None,
               name=None, stage="all", shard=(0, 1)):
    lo, hi = pd.Timestamp(sel_start), pd.Timestamp(sel_end)
    name = name or f"select_{lo.date()}_{hi.date()}"
    out = Path("data/experiments") / name
    out.mkdir(parents=True, exist_ok=True)
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
        grids = {h: _load(out, f"grid_{h}_s*.parquet") for h in HORIZONS}
        h, _ = decide_horizon(grids, lo, hi)
        if h == "4w":
            sweeps = {y: d for y in WINDOWS
                      if (d := _load(out, f"wsweep_{y}y_s*.parquet")) is not None}
            w, _ = decide_window(sweeps, lo, hi)
        else:
            w = REF_WINDOW
        print(f"stage holdings ({h}/{w}y)", flush=True)
        stage_holdings(ctx, out, h, w, lo,
                       pd.Timestamp(eval_end) if eval_end else hi, shard)
    if stage in ("all", "cascade"):
        config = run_cascade(ctx, out, lo, hi)
        (out / "frozen_config.json").write_text(json.dumps(config, indent=2))
        print("FROZEN:", {k: config[k] for k in
                          ("horizon", "train_years", "book", "floor", "stop", "cap")})
        if eval_start:
            holdings = _load(out, f"holdings_{config['horizon']}_"
                                  f"{config['train_years']}y_s*.parquet")
            s = simulate(ctx, holdings, config["horizon"], config["book"],
                         config["cap"], config["stop"], config["floor"])
            elo, ehi = pd.Timestamp(eval_start), pd.Timestamp(eval_end)
            rep = {"config": metrics(s, elo, ehi),
                   "sp500": metrics(ctx.wret["SPY"].reindex(s.index), elo, ehi)}
            (out / "eval.json").write_text(json.dumps(rep, indent=2))
            print("EVAL:", json.dumps(rep))
    return out
