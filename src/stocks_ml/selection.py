"""The engine research and the live job share.

    Ctx                 a world store in memory: the panel, prices, members,
                        weekly returns, SPY, the forward returns on the fill
                        basis (price_frames), the sector map
    ensemble_preds      the model call: K_COPIES copies of the champion
                        model (MODEL_PARAMS, depth-3 XGBoost, time-tail early
                        stopping) refit on the trailing window at one rank
                        week, scores averaged — what `stocks-ml train` runs
                        every week and what live/r5.py runs on Saturday
    ensemble_holdings   a saved walk's scores -> the top names each week
    simulate            the ledger's rules over those holdings: next-open
                        fills at 5 bp a side, four staggered sleeves, the
                        floor, an optional stop and sector cap (trace hook
                        for the explorer)
    decide_strategy     the strategy layers, book down, on the selection
                        window: book by cost-adjusted compounded %/yr
                        (decide_book / compounded_pct), floor, stop and cap
                        by Sharpe — what `stocks-ml procedure` writes
    metrics             $100 grown, %/yr, Sharpe, max drawdown on a window

Model config is fixed (MODEL_PARAMS) — never searched. No decision reads a
rank week whose forward label ends after the window (label_end). Every
pre-holdout window ends at HOLDOUT_START, exclusive.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd

from stocks_ml.ledger import (COST_BPS, FLOOR_FRACTION, FLOORS, Ledger, ballast_state,  # noqa: F401
                              close_asof, floor_split, pick_capped, rotate_sleeves,
                              target_weights, week_index)

MODEL_PARAMS = dict(max_depth=3, learning_rate=0.02, n_estimators=1500,
                    min_child_weight=20, subsample=0.85, colsample_bytree=0.8,
                    reg_alpha=0.5, reg_lambda=1.0, gamma=0.01, max_bin=256,
                    tree_method="hist")
HORIZONS = {"1w": dict(label="label", purge=10, kweeks=1),
            "4w": dict(label="label_4w", purge=35, kweeks=4)}
# The 4-week targets a walk (and the live job) can train on: the horizon's
# own week-centred label and the sector-centred one (Stage E of the clean
# program, adopted 2026-09-12; build_panel stores both, Ctx recomputes the
# sector one for older panels). Same hold, same purge; only the recentring
# differs. HORIZONS is what price_frames keys its forward returns on.
LABELS_4W = {"label_4w": "the stock's 4-week return minus that week's median member's",
             "label_4w_sector": "the stock's 4-week return minus the same-week median of its "
                                "sector (the week's median where the sector is unknown)",
             "label_4w_sector_log": "log(1 + the stock's 4-week return) minus log(1 + the same-week "
                                    "median of its sector): the upside tempered, the downside stretched",
             "label_4w_sector_clip": "the stock's 4-week return minus the same-week median of its "
                                     "sector, capped at +/-20%",
             "label_4w_sector_rank": "the stock's 4-week return minus the same-week median of its "
                                     "sector, replaced by its within-week rank as a normal score",
             "label_4w_rank": "the stock's 4-week return replaced by its within-week rank as a normal "
                              "score (no sector centring)",
             "label_13w_sector_rank": "the stock's 13-week return minus the same-week median of its "
                                      "sector, replaced by its within-week rank as a normal score "
                                      "(ranks by the quarter, rotates monthly)",
             "label_blend_rank": "the mean of the 4-week and 13-week sector-relative rank scores, "
                                 "re-ranked within the week as a normal score",
             "label_4w_sector11_rank": "the stock's 4-week return minus the same-week median of its "
                                       "Sharadar sector (11 groups), replaced by its within-week rank "
                                       "as a normal score"}
# Labels whose forward span exceeds the 4-week hold's purge train with their
# own (13 weeks + the fill week); the hold and the ledger are unchanged.
LABEL_PURGE = {"label_13w_sector_rank": 13 * 7 + 7, "label_blend_rank": 13 * 7 + 7}


def label_purge(label: str, horizon: str = "4w") -> int:
    """Days between the last training label and the rank week: the hold's
    purge or the label's own, whichever is longer."""
    return max(HORIZONS[horizon]["purge"], LABEL_PURGE.get(label, 0))
BOOKS = (3, 6, 10)
# FLOORS / FLOOR_FRACTION / floor_split: stocks_ml.ledger, the one rule the
# backtest and the live job share (halfgate runs live since 2026-09-12).
COST = 0.0010
K_COPIES = 16          # 4 until 2026-09-07 (ledger k16_champion_2006_2015_verdict)
# The holdout's first session. Every pre-holdout grade window ends here as an
# EXCLUSIVE bound (metrics slices index < hi), so the label credited at
# 2024-07-19 — the first holdout close — is never counted. Before 2026-09-09
# some graders used Timestamp("2025") (967/447 weeks, crediting that label)
# and others 07-19-exclusive (966/446): one convention now, this one.
HOLDOUT_START = pd.Timestamp("2024-07-19")


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
        # A stored panel carries label_4w (and label_4w_sector since
        # 2026-09-12); every other 4-week target is the same fwd_ret_4w
        # recentred on build_panel's sector map and transformed here, so the
        # research walk and the live job train on identical columns.
        from stocks_ml.features.panel import LABEL_TRANSFORMS, fwd_ret_13w
        if "fwd_ret_13w" not in self.pan.columns:
            # Panels built before 2026-09-16 lack the 13-week forward return;
            # it is a function of the store's prices alone, cached beside the panel.
            cache = Path(data_dir) / "fwd_ret_13w.parquet"
            if cache.exists():
                f13 = pd.read_parquet(cache)
            else:
                f13 = fwd_ret_13w(self.prices, self.pan["date"].unique(), self.cfg.horizon_days,
                                  getattr(self.cfg, "delist_labels", "drop"))
                f13.to_parquet(cache, index=False)
            f13["date"] = pd.to_datetime(f13["date"])
            self.pan = self.pan.merge(f13[["date", "ticker", "fwd_ret_13w"]], on=["date", "ticker"], how="left")
        # the Sharadar 11-sector map, when the store carries the tickers table
        self.smap11 = {}
        if world.exists("sharadar_tickers"):
            tk = world.read("sharadar_tickers")
            if "sector" in tk.columns:
                self.smap11 = dict(tk.dropna(subset=["sector"]).drop_duplicates("ticker")[["ticker", "sector"]].values)
        import inspect
        extra = {"fwd13": self.pan["fwd_ret_13w"],
                 "sector11": self.pan["ticker"].map(self.smap11) if self.smap11 else None}
        for name, fn in LABEL_TRANSFORMS.items():
            if name in self.pan.columns:
                continue
            kw = {k: v for k, v in extra.items() if k in inspect.signature(fn).parameters}
            if "sector11" in kw and kw["sector11"] is None:
                continue                       # no tickers table: the label stays unavailable
            self.pan[name] = fn(self.pan["fwd_ret_4w"], self.pan["date"], self.pan["ticker"].map(self.smap), **kw)
        daily = self.prices.sort_values("date")
        self.delist_labels = getattr(self.cfg, "delist_labels", "drop")
        self.__dict__.update(price_frames(
            daily.pivot(index="date", columns="ticker", values="close").sort_index(),
            daily.pivot(index="date", columns="ticker", values="open").sort_index(),
            delist=self.delist_labels))
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


def ensemble_preds(ctx, t, horizon, train_years, label=None, features=None, params=None):
    """The K_COPIES-copy ensemble score at rank week t: copy c is the champion
    model (MODEL_PARAMS, `params` overriding) on the trailing `train_years`
    under whole-week bootstrap seed c, trained on `label` (the horizon's own
    label unless given; LABELS_4W), on the panel's f_ columns plus `features`
    (ctx.extra unless given) — the spec's recipe, as the live job passes it."""
    from stocks_ml.models.walk import walk_forward_predictions
    from stocks_ml.models.xgb import TimeTailEarlyStopXGB
    from stocks_ml.models.replication import WeekBootstrapEstimator
    h = HORIZONS[horizon]
    label = label or h["label"]
    purge = label_purge(label, horizon)
    cfg2 = ctx.world_cfg(train_years)
    copies = []
    for c in range(1, K_COPIES + 1):
        est = WeekBootstrapEstimator(
            TimeTailEarlyStopXGB(**{**MODEL_PARAMS, **(params or {})}, **fixed(purge)),
            bootstrap_seed=c)
        wf = walk_forward_predictions(ctx.pan, est, cfg2, start=t, end=t,
                                      label_col=label,
                                      purge_days=purge,
                                      extra_features=tuple(ctx.extra if features is None else features))
        p = wf.preds.get(t)
        if p is not None:
            copies.append(p)
    if not copies:
        return None
    p = pd.concat(copies, axis=1).mean(axis=1)
    return p if p.nunique() >= 20 else None


def price_frames(closes, opens, delist="drop"):
    """Everything the engine reads from daily closes and opens: the frames
    themselves (closes carried forward), weekly closes on the W-FRI grid,
    weekly returns, SPY's weekly closes, and per horizon the forward return
    on the fill basis — a pick dated in week t is bought at the first open
    after t and sold at the first open k weeks later (the training labels'
    basis, and the live job's).

    delist="last_print": a name whose series ends inside the k-week window
    grades to its final close (the live ledger's exit fallback) instead of
    NaN, so a held delisting is priced. Weeks entirely after the end stay
    NaN; a series alive within a week of the data's edge is right-censored."""
    last_date = {tk: closes[tk].last_valid_index() for tk in closes.columns}
    last_close = {tk: (float(closes[tk].loc[d]) if d is not None else np.nan)
                  for tk, d in last_date.items()}
    closes = closes.ffill()
    cw = closes.resample("W-FRI").last()
    fills = next_open(opens, cw.index)
    fwd = {h: fills.pct_change(c["kweeks"], fill_method=None).shift(-c["kweeks"])
           for h, c in HORIZONS.items()}
    if delist == "last_print":
        edge = closes.index[-1] - pd.Timedelta(days=7)
        for h, c in HORIZONS.items():
            k, f = c["kweeks"], fwd[h]
            for tk, ld in last_date.items():
                if ld is None or ld >= edge:
                    continue
                pos = int(f.index.searchsorted(ld))       # first label >= last print
                for w in f.index[max(0, pos - k):pos]:    # windows crossing the end
                    entry = fills.at[w, tk]
                    if np.isfinite(entry) and entry > 0 and pd.isna(f.at[w, tk]):
                        f.at[w, tk] = last_close[tk] / entry - 1.0
    return {"closes": closes, "opens": opens, "cw": cw, "last_print": pd.Series(last_date),
            "wret": cw.pct_change(fill_method=None), "spy_w": cw["SPY"], "fwd": fwd}


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
    if getattr(ctx, "delist_labels", "drop") == "last_print":
        # live's universe rule (r5.rank_members): a name must have traded
        # within the last 7 days of t. With last-print labels this — not
        # label finiteness — is what keeps dead names out, so the backtest
        # can rank a name that will delist mid-hold, exactly as live can.
        lp = ctx.last_print
        cut = pd.Timestamp(t) - pd.Timedelta(days=7)
        uni = [x for x in uni if lp.get(x) is not None and lp[x] > cut]
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


def ensemble_holdings(ctx, preds, copies, horizon="4w"):
    """slice_row at every week of a saved walk (columns week, ticker, c1..cK)
    for the mean of the given copies, exactly as ensemble_preds ranks them
    (mean over copies; a week with fewer than 20 distinct scores is skipped).
    Returns the holdings frame the cascade grades and the per-week scores."""
    cols = [f"c{c}" for c in copies]
    rows, means = [], {}
    for t, g in preds.groupby("week"):
        p = g.set_index("ticker")[[c for c in cols if c in g.columns]].mean(axis=1)
        if p.nunique() < 20:
            continue
        row = slice_row(ctx, t, horizon, p)
        if row is not None:
            rows.append(row)
            means[t] = p
    return pd.DataFrame(rows), means


# ---- pure decision functions (unit-testable) ----
def label_span_days(kweeks: int) -> int:
    """Calendar days a k-week label spans on the fill basis (label week + the
    next open): the purge the walk uses for that horizon."""
    return kweeks * 7 + 7


def label_end(hi, kweeks=max(c["kweeks"] for c in HORIZONS.values())):
    """Last rank week whose k-week forward label (fill basis) ends inside a
    window closing at `hi`: no decision reads a week graded on prices past
    the window. The default is the longest horizon's span."""
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


def decide_book(df, horizon, lo, hi):
    """Cost-adjusted compounded %/yr of each book on the chosen window's
    population holdings (the frame the cascade grades) decides."""
    kw = HORIZONS[horizon]["kweeks"]
    res = {k: compounded_pct(df, f"top{k}", kw, lo, label_end(hi, kw)) for k in BOOKS}
    return max(res, key=res.get), res


def simulate(ctx, holdings, horizon, book, cap, stop, floor, trace=None, settings=None):
    """Weekly returns of the configured book under the live job's rules
    (stocks_ml.ledger): each rank date's target weights are filled at the
    next session's open at COST_BPS a side, rebalances under 0.5% of NAV are
    skipped, and NAV is marked at each week's last close. Indexed by the
    week-ending label each return is credited to; a week with no pick holds
    the book. `stop` moves a name that has fallen that far from its rotation
    close into SPY until its sleeve rotates. Pass a list as `trace` to
    receive one record per credited week (pick date, sleeves, fills, per-name
    $ and price returns, gate). `settings`, when given, is a frame indexed by
    decision week with columns book, floor, stop, cap (rolling.decisions): at
    each rank week the row decided at or before it is in force and the four
    scalar arguments are ignored — a book-size change phases in as sleeves
    rotate, as it would for the live job."""
    ranked = {r.week: r.top15.split(",") for r in holdings.itertuples()}
    if settings is not None:
        settings = settings.sort_index()
        if ranked and settings.index[0] > min(ranked):
            raise ValueError(f"no decision in force at the first rank week {min(ranked).date()} "
                             f"(the first decision is {settings.index[0].date()})")
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
        if settings is not None:
            book, floor, stop, cap = settings_at(settings, t)
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


def settings_at(settings, t):
    """(book, floor, stop, cap) in force at rank week t: the last row of
    `settings` decided at or before t. NaN stop/cap read as none."""
    i = settings.index.searchsorted(pd.Timestamp(t), side="right") - 1
    if i < 0:
        raise ValueError(f"no decision in force at {pd.Timestamp(t).date()}")
    r = settings.iloc[i]
    none = lambda v: None if v is None or (isinstance(v, float) and np.isnan(v)) else v
    stop, cap = none(r["stop"]), none(r["cap"])
    return (int(r["book"]), str(r["floor"]), None if stop is None else float(stop),
            None if cap is None else int(cap))


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


def decide_strategy(ctx, holdings, horizon, lo, hi):
    """The strategy layers, book down, on the holdings frame the run grades,
    over the selection window [lo, hi] only: book by cost-adjusted compounded
    %/yr (decide_book); floor, stop and cap by Sharpe of the simulated weekly
    series, each at the picks above it, a stop or cap adopted only if higher.
    This is the whole of what decides the deployed strategy settings —
    `stocks-ml procedure` writes its result into models/champion_spec.json."""
    lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
    hold = holdings[(holdings.week >= lo) & (holdings.week <= hi)]
    book, bres = decide_book(hold, horizon, lo, hi)
    fres = {f: sharpe(simulate(ctx, hold, horizon, book, None, None, f), lo, hi)
            for f in FLOORS}
    floor = max(fres, key=fres.get)
    sres = {str(s): sharpe(simulate(ctx, hold, horizon, book, None, s, floor), lo, hi)
            for s in (None, -0.25)}
    stop = None if sres["None"] >= sres["-0.25"] else -0.25
    cres = {str(c): sharpe(simulate(ctx, hold, horizon, book, c, stop, floor), lo, hi)
            for c in (None, 2)}
    cap = None if cres["None"] >= cres["2"] else 2
    return {"book": int(book), "floor": floor, "stop": stop, "cap": cap,
            "evidence": {"book": {str(k): round(v, 2) for k, v in bres.items()},
                         "floor": {k: round(v, 3) for k, v in fres.items()},
                         "stop": {k: round(v, 3) for k, v in sres.items()},
                         "cap": {k: round(v, 3) for k, v in cres.items()}}}


