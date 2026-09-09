"""Generated features: a mechanical generator's formulas over raw inputs.

The screen's candidates (features/candidates.py) are thirty human ideas written
after reading the whole record — hence their asterisk. This module is the
other arm: an enumerated space of formulas (OpenFE's order-1 numeric
operators; Zhang et al., NeurIPS 2023) over inputs written down as a rule
before any number, scored on the selection window only (ops/openfe_arm.py),
and the winners computed here for every panel row as ranked ``g_`` columns. A
walk uses them through ``extra_features`` like any admitted bundle.

Inputs (``raw_inputs``), the panel's point-in-time conventions throughout:
the champion's model features before their per-week ranking (build_panel
rerun with the ranking step intercepted and the store left untouched), every
numeric field of the store's fundamentals table (trailing-12-month dimension)
as its latest filed level and its year-over-year change (SF1 facts usable at
filing + 1 day, sharadar_fundamentals._asof), every 8-K item code of the
post-2004 numbering with at least ``MIN_FILINGS`` filings as a 26-week count
(the day after SEC acceptance, the filing date when acceptance is missing),
and the nominal close as of the rank date.

Formulas use OpenFE's syntax and operator semantics — ``(a+b)``, ``(a-b)``,
``(a*b)``, ``(a/b)`` with a zero divisor -> NaN, ``min(a,b)``, ``max(a,b)``,
``abs(a)``, a bare input name — so formulas.json round-trips through OpenFE's
own parser. Every generated column is ranked per week to (-1, 1] and
neutral-filled like every ranked feature (features/ranking.py), so the
generator scores ranked values (the model's representation) and a formula
that is constant within a week is never a candidate.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from stocks_ml.features.candidates import _asof_wide, _lookup, _rows_by_ticker
from stocks_ml.features.ranking import rank_normalize
from stocks_ml.features.sharadar_fundamentals import _asof

PREFIX = "g_"
STORE_PREFIX = "r_"
MIN_FILINGS = 100
WINDOW_DAYS = 182
PAIR_OPS = ("+", "-", "*", "/", "min", "max")     # symmetric up to sign: one order per pair
UNARY_OPS = ("abs",)
ID_COLS = ("ticker", "dimension", "calendardate", "date", "reportperiod", "lastupdated")
ITEM_CODE = re.compile(r"^\d\.\d\d$")               # 8-K items of the post-2004 numbering

# OpenFE's LightGBM settings (openfe.py): the base model it calls feature
# boosting, the stage-1 'predictive' metric, the stage-2 gain ranking.
BOOST_PARAMS = {"n_estimators": 10000, "learning_rate": 0.1, "metric": "rmse", "verbose": -1}
BOOST_STOP = 200
STAGE1_PARAMS = {"n_estimators": 100, "importance_type": "gain", "num_leaves": 16, "seed": 1,
                 "deterministic": True, "n_jobs": 1, "metric": "rmse", "verbose": -1}
STAGE1_STOP = 3
STAGE2_PARAMS = {"n_estimators": 1000, "importance_type": "gain", "num_leaves": 16, "seed": 1,
                 "metric": "rmse", "verbose": -1}
STAGE2_STOP = 50


# ----------------------------------------------------------------------------- inputs
def fundamental_fields(fund: pd.DataFrame) -> list[str]:
    return sorted(c for c in fund.columns if c not in ID_COLS and pd.api.types.is_numeric_dtype(fund[c]))


PER_SHARE_FIELDS = ("bvps", "dps", "eps", "epsdil", "epsusd", "sps", "fcfps",
                    "ncfps", "tbvps", "sharesbas", "shareswa", "shareswadil")


def sf1_inputs(fund: pd.DataFrame, base: pd.DataFrame,
               split_factor: pd.Series | None = None) -> pd.DataFrame:
    """Every numeric SF1 field, trailing-12-month dimension: the latest filed
    level (``r_sf_<field>``) and its year-over-year change (``r_sf_<field>_yoy``:
    x / prev - 1 against the row four quarters earlier, report-period gap
    330-400 days, prior value > 0).

    `split_factor` (nominal basis): closeunadj/close_split per base row —
    the cumulative splits between the row's date and the download. Sharadar
    restates per-share fields through the download date, so the stored LEVEL
    at t encodes future splits; multiplying by the factor restores the
    as-of-t value. YoY fields are same-basis ratios and stay untouched."""
    from stocks_ml.features.sharadar_fundamentals import _monotone
    fields = fundamental_fields(fund)
    art = (_monotone(fund[fund["dimension"] == "ART"])
           .sort_values(["ticker", "date", "reportperiod"])
           .drop_duplicates(["ticker", "reportperiod"], keep="last")   # last filed wins
           .sort_values(["ticker", "reportperiod"]).copy())
    g = art.groupby("ticker")
    gap = g["reportperiod"].diff(4).dt.days
    ok = (gap > 330) & (gap < 400)
    yoy = {}
    for f in fields:
        prev = g[f].shift(4)
        yoy[f + "_yoy"] = (art[f] / prev - 1).where(ok & (prev > 0))
    art = pd.concat([art, pd.DataFrame(yoy, index=art.index)], axis=1)
    cols = fields + [f + "_yoy" for f in fields]
    out = _asof(base, art, cols)
    out.columns = [f"{STORE_PREFIX}sf_{c}" for c in cols]
    if split_factor is not None:
        for f in PER_SHARE_FIELDS:
            col = f"{STORE_PREFIX}sf_{f}"
            if col in out.columns:
                out[col] = out[col] * split_factor
    return out


def item_codes(sec8k: pd.DataFrame, min_filings: int = MIN_FILINGS) -> list[str]:
    """8-K item codes (post-2004 numbering) carried by at least min_filings filings."""
    codes = sec8k["items"].fillna("").astype(str).str.split(r",\s*").explode().str.strip()
    counts = codes[codes.str.match(ITEM_CODE)].value_counts()
    return sorted(c for c, n in counts.items() if n >= min_filings)


def sec8k_inputs(sec8k: pd.DataFrame, base: pd.DataFrame, min_filings: int = MIN_FILINGS) -> pd.DataFrame:
    """``r_8k_<code>``: filings carrying the item in the last WINDOW_DAYS days,
    available the day after SEC acceptance (filing date + 1 when missing)."""
    f = sec8k.copy()
    accepted = pd.to_datetime(f["accepted"], utc=True, errors="coerce").dt.tz_convert(None).dt.normalize()
    anchor = accepted.fillna(pd.to_datetime(f["filed"]).dt.normalize())
    f["available"] = anchor + pd.Timedelta(days=1)
    f = f.dropna(subset=["available"]).sort_values("available")
    items = f["items"].fillna("").astype(str)
    by_ticker = _rows_by_ticker(base)
    dates = base["date"].values
    out = pd.DataFrame(index=base.index)
    for code in item_codes(sec8k, min_filings):
        mask = items.str.contains(rf"(?:^|,\s*){code.replace('.', r'\.')}(?:,|$)", regex=True)
        cnt = np.zeros(len(base))
        for tk, grp in f.loc[mask, ["ticker", "available"]].groupby("ticker"):
            rows = by_ticker.get(tk)
            if rows is None:
                continue
            av, t = grp["available"].values, dates[rows]
            cnt[rows] = (np.searchsorted(av, t, side="right")
                         - np.searchsorted(av, t - np.timedelta64(WINDOW_DAYS, "D"), side="right"))
        out[f"{STORE_PREFIX}8k_{code.replace('.', '_')}"] = cnt
    return out


def close_input(prices: pd.DataFrame, base: pd.DataFrame,
                field: str = "close") -> pd.Series:
    """``r_close``: the close level as of the rank date (non-positive -> NaN).
    field='close' is the store's closeadj basis (the pre-2026-09 behavior,
    which encodes future splits); 'closeunadj' is the true nominal close."""
    close = prices.pivot(index="date", columns="ticker", values=field).sort_index()
    dates = pd.DatetimeIndex(sorted(base["date"].unique()))
    px = _lookup(_asof_wide(close, dates), base)
    return pd.Series(np.where(px > 0, px, np.nan), index=base.index, name=f"{STORE_PREFIX}close")


def split_factor_input(prices: pd.DataFrame, base: pd.DataFrame) -> pd.Series:
    """closeunadj/close_split as of each base row: the cumulative split
    restatement Sharadar applied between the row's date and the download."""
    p = prices.copy()
    p["_sf"] = p["closeunadj"] / p["close_split"]
    wide = p.pivot(index="date", columns="ticker", values="_sf").sort_index()
    dates = pd.DatetimeIndex(sorted(base["date"].unique()))
    return pd.Series(_lookup(_asof_wide(wide, dates), base), index=base.index)


def store_inputs(store, base: pd.DataFrame, log=None,
                 price_basis: str = "closeadj") -> pd.DataFrame:
    """The store-derived inputs on base rows (date, ticker); base.index kept."""
    log = log or (lambda msg: None)
    prices = store.read("prices")
    nominal = price_basis == "nominal"
    if nominal and not {"closeunadj", "close_split"} <= set(prices.columns):
        raise RuntimeError("price_basis='nominal' needs closeunadj/close_split in the "
                           "prices table (world.prices_from_sep)")
    factor = split_factor_input(prices, base) if nominal else None
    parts = [sf1_inputs(store.read("fundamentals"), base, split_factor=factor)]
    log("inputs: sf1")
    parts.append(sec8k_inputs(store.read("sec8k"), base))
    log("inputs: 8-K")
    parts.append(close_input(prices, base, field="closeunadj" if nominal else "close").to_frame())
    log("inputs: close")
    return pd.concat(parts, axis=1)


def base_raw_features(live_dir, cfg, log=None) -> pd.DataFrame:
    """The champion's model features before per-week ranking, on the panel's
    rows: build_panel rerun on the world store with the ranking step
    intercepted and every write turned off, plus the Sharadar blocks as
    data/world.py computes them. Columns: date, ticker, feature_cols(...)."""
    import copy
    from unittest import mock

    from stocks_ml.data.world import BACKTEST_START, _PanelStore
    from stocks_ml.features import ranking
    from stocks_ml.features.panel import build_panel, feature_cols
    from stocks_ml.features.sharadar_fundamentals import (
        sharadar_fundamental_features, sharadar_insider_features)

    log = log or (lambda msg: None)

    class _CaptureStore(_PanelStore):
        def write(self, name, df):
            pass

        def set_manifest(self, key, value):
            pass

    cfg2 = copy.copy(cfg)
    for k, v in (("data_dir", str(live_dir)), ("backtest_start", BACKTEST_START)):
        object.__setattr__(cfg2, k, v)
    store = _CaptureStore(live_dir)
    captured = {}
    real = ranking.rank_normalize

    def capture(df, cols, neutral_fill=True):
        captured["panel"] = df
        return real(df, cols, neutral_fill)

    with mock.patch.object(ranking, "rank_normalize", capture):
        build_panel(store, cfg2)
    raw = captured["panel"]
    log(f"inputs: panel rebuilt raw, {len(raw):,} rows")

    prices, fund, ins = store.read("prices"), store.read("fundamentals"), store.read("insiders")
    cw = prices.pivot(index="date", columns="ticker", values="close").sort_index().ffill()
    wk = cw.reindex(pd.Index(sorted(raw["date"].unique())), method="ffill")
    close = pd.Series(wk.stack().reindex(
        pd.MultiIndex.from_frame(raw[["date", "ticker"]])).values, index=raw.index)
    ff = sharadar_fundamental_features(fund, raw, close)
    shares = _asof(raw, fund[fund["dimension"] == "ARQ"], ["sharesbas"])["sharesbas"]
    fi = sharadar_insider_features(ins, raw, mktcap=close * shares)
    full = pd.concat([raw, ff, fi], axis=1)
    cols = feature_cols(full)
    log(f"inputs: {len(cols)} base features raw")
    return full[["date", "ticker"] + cols].reset_index(drop=True)


def raw_inputs(live_dir, cfg, log=None) -> pd.DataFrame:
    """date, ticker, the raw base features, the store inputs — every panel row."""
    from stocks_ml.data.world import _PanelStore
    base = base_raw_features(live_dir, cfg, log)
    extra = store_inputs(_PanelStore(live_dir), base[["date", "ticker"]], log,
                         price_basis=getattr(cfg, "price_basis", "closeadj"))
    return pd.concat([base, extra], axis=1)


def input_names(inputs: pd.DataFrame) -> list[str]:
    return [c for c in inputs.columns if c not in ("date", "ticker")]


def week_constant(inputs: pd.DataFrame, names=None) -> set[str]:
    """Inputs with one value per week (market and macro columns): their
    formulas with each other rank to zero."""
    names = list(names or input_names(inputs))
    n = inputs.groupby("date")[names].nunique(dropna=False)
    return set(n.columns[(n <= 1).all(axis=0)])


# ----------------------------------------------------------------------------- formulas
def enumerate_candidates(names, constant=frozenset(), identity=()) -> list[str]:
    """OpenFE's order-1 numeric space over `names`: every unordered pair under
    PAIR_OPS plus abs(x), plus the bare `identity` inputs; without the
    formulas a per-week ranking makes trivial — both inputs week-constant,
    abs of a week-constant, and x +/- a week-constant (ranks as x)."""
    names = sorted(names)
    out = [n for n in identity if n not in constant]
    for i, a in enumerate(names):
        if a not in constant:
            out.append(f"abs({a})")
        for b in names[i + 1:]:
            ca, cb = a in constant, b in constant
            if ca and cb:
                continue
            for op in PAIR_OPS:
                if (ca or cb) and op in ("+", "-"):
                    continue
                out.append(f"({a}{op}{b})" if op in "+-*/" else f"{op}({a},{b})")
    return out


_PAIR = re.compile(r"^\((?P<a>[^()+\-*/,]+)(?P<op>[+\-*/])(?P<b>[^()+\-*/,]+)\)$")
_CALL = re.compile(r"^(?P<fn>abs|min|max)\((?P<args>[^()]+)\)$")


def formula_inputs(formula: str) -> list[str]:
    m = _PAIR.match(formula)
    if m:
        return [m["a"], m["b"]]
    m = _CALL.match(formula)
    if m:
        return [s.strip() for s in m["args"].split(",")]
    return [formula]


def evaluate(formula: str, inputs: pd.DataFrame) -> np.ndarray:
    """One formula on inputs' rows with OpenFE's operator semantics; +-inf -> NaN."""
    m = _PAIR.match(formula)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        if m:
            a, b = inputs[m["a"]].to_numpy(float), inputs[m["b"]].to_numpy(float)
            op = m["op"]
            if op == "+":
                out = a + b
            elif op == "-":
                out = a - b
            elif op == "*":
                out = a * b
            else:
                out = a / np.where(b == 0, np.nan, b)
        else:
            m = _CALL.match(formula)
            if m:
                args = [inputs[s.strip()].to_numpy(float) for s in m["args"].split(",")]
                fn = m["fn"]
                if fn == "abs" and len(args) == 1:
                    out = np.abs(args[0])
                elif fn == "min" and len(args) == 2:
                    out = np.minimum(args[0], args[1])
                elif fn == "max" and len(args) == 2:
                    out = np.maximum(args[0], args[1])
                else:
                    raise ValueError(f"cannot parse formula {formula!r}")
            elif formula in inputs.columns:
                out = inputs[formula].to_numpy(float).copy()
            else:
                raise ValueError(f"cannot parse formula {formula!r}")
    out[~np.isfinite(out)] = np.nan
    return out


def rank_week(x: np.ndarray, dates) -> np.ndarray:
    """rank_normalize for one column: per-date pct rank to (-1, 1], NaN -> 0."""
    r = pd.Series(np.asarray(x, float)).groupby(np.asarray(dates)).rank(pct=True)
    return (r.mul(2).sub(1.0).fillna(0.0)).to_numpy()


def generated_features(inputs: pd.DataFrame, formulas: dict[str, str]) -> pd.DataFrame:
    """Raw values of the named formulas on inputs' rows; inputs.index kept."""
    return pd.DataFrame({name: evaluate(f, inputs) for name, f in formulas.items()}, index=inputs.index)


def add_generated(panel: pd.DataFrame, inputs: pd.DataFrame, formulas: dict[str, str]) -> pd.DataFrame:
    """The panel plus every formula as a ranked, neutral-filled ``g_`` column,
    matched on (date, ticker); existing ``g_`` columns are rebuilt. Row order
    and the other columns are untouched."""
    feats = generated_features(inputs, formulas)
    feats.index = pd.MultiIndex.from_frame(inputs[["date", "ticker"]])
    feats = feats.reindex(pd.MultiIndex.from_frame(panel[["date", "ticker"]]))
    feats.index = panel.index
    out = panel.drop(columns=[c for c in panel.columns if c.startswith(PREFIX)])
    out = pd.concat([out, feats], axis=1)
    return rank_normalize(out, list(formulas))


# ----------------------------------------------------------------------------- scoring
def week_blocks(dates, block_weeks: int = 26, folds: int = 5) -> np.ndarray:
    """Fold id per row: contiguous blocks of `block_weeks` distinct dates,
    blocks dealt to `folds` folds in turn."""
    weeks = np.array(sorted(pd.unique(np.asarray(dates))))
    block = {w: (i // block_weeks) % folds for i, w in enumerate(weeks)}
    return np.array([block[w] for w in np.asarray(dates)])


def purged_train(dates, fold: np.ndarray, k: int, purge_days: int = 35) -> np.ndarray:
    """Mask of the rows that may train against fold k: not in it, and not
    within purge_days of any of its blocks on either side."""
    d = pd.to_datetime(pd.Series(np.asarray(dates))).to_numpy()
    val = fold == k
    train = ~val
    vd = np.sort(np.unique(d[val]))
    if len(vd) == 0:
        return train
    # a fold's blocks are runs of consecutive dates; purge around each run
    gaps = np.where(np.diff(vd) > np.timedelta64(21, "D"))[0]
    starts = np.concatenate([[0], gaps + 1])
    ends = np.concatenate([gaps, [len(vd) - 1]])
    pad = np.timedelta64(purge_days, "D")
    for s, e in zip(starts, ends):
        train &= (d < vd[s] - pad) | (d > vd[e] + pad)
    return train


def _fit(gbm, X, y, Xv, yv, stop, init=None, init_v=None):
    """LightGBM fit with early stopping on (Xv, yv), the base model's scores
    as init_score when given (lightgbm >= 4.7's eval_X / eval_y form)."""
    import lightgbm as lgb
    kw = {} if init is None else dict(init_score=init, eval_init_score=[init_v])
    gbm.fit(X, y, eval_X=Xv, eval_y=yv, callbacks=[lgb.early_stopping(stop, verbose=False)], **kw)
    return gbm


def init_scores(X: np.ndarray, y: np.ndarray, dates, fold: np.ndarray, purge_days: int = 35,
                n_jobs: int = 1, seed: int = 1) -> np.ndarray:
    """Out-of-fold predictions of OpenFE's base model (feature boosting:
    LightGBM, lr 0.1, early stop BOOST_STOP on the held-out fold)."""
    import lightgbm as lgb
    out = np.zeros(len(y))
    for k in sorted(set(fold)):
        tr, va = purged_train(dates, fold, k, purge_days), fold == k
        gbm = _fit(lgb.LGBMRegressor(**BOOST_PARAMS, seed=seed, n_jobs=n_jobs),
                   X[tr], y[tr], X[va], y[va], BOOST_STOP)
        out[va] = gbm.predict(X[va])
    return out


def init_metric(init: np.ndarray, y: np.ndarray, val: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y[val] - init[val]) ** 2)))


def predictive_score(x: np.ndarray, y: np.ndarray, init: np.ndarray, train: np.ndarray,
                     val: np.ndarray, base_rmse: float) -> float:
    """OpenFE's stage-1 'predictive' metric: the validation RMSE a 100-tree
    LightGBM on the candidate alone takes off the base model's residual."""
    import lightgbm as lgb
    gbm = _fit(lgb.LGBMRegressor(**STAGE1_PARAMS), x[train].reshape(-1, 1), y[train],
               x[val].reshape(-1, 1), y[val], STAGE1_STOP, init[train], init[val])
    return base_rmse - float(gbm.best_score_["valid_0"]["rmse"])


def gain_ranking(X: np.ndarray, names: list[str], y: np.ndarray, init: np.ndarray, train: np.ndarray,
                 val: np.ndarray, n_jobs: int = 1) -> pd.Series:
    """OpenFE's stage 2: one LightGBM on every column with the base model's
    init score; gain importance per column, descending."""
    import lightgbm as lgb
    gbm = _fit(lgb.LGBMRegressor(**STAGE2_PARAMS, n_jobs=n_jobs), X[train], y[train], X[val], y[val],
               STAGE2_STOP, init[train], init[val])
    out = pd.Series(gbm.feature_importances_, index=names, dtype=float).sort_values(ascending=False)
    out.attrs["best_iteration"] = int(gbm.best_iteration_ or 0)
    out.attrs["val_rmse"] = float(gbm.best_score_["valid_0"]["rmse"])
    return out


# ----------------------------------------------------------------------------- ranking yardstick (v2)
def week_index(dates) -> np.ndarray:
    """0-based id of each row's week (distinct dates in order)."""
    return pd.factorize(pd.Series(np.asarray(dates)), sort=True)[0]


def standardize_week(x: np.ndarray, week: np.ndarray, n_weeks: int):
    """x centred and scaled within each week (population sd). Returns the
    z scores, the mask of weeks where x varies, and the rows per week."""
    x = np.asarray(x, float)
    cnt = np.bincount(week, minlength=n_weeks)
    mean = np.bincount(week, weights=x, minlength=n_weeks) / np.maximum(cnt, 1)
    xc = x - mean[week]
    sd = np.sqrt(np.bincount(week, weights=xc * xc, minlength=n_weeks) / np.maximum(cnt, 1))
    ok = sd > 0
    return xc / np.where(ok, sd, 1.0)[week], ok, cnt


class WeeklyIC:
    """The champion's yardstick as a score: the mean weekly Spearman IC of a
    per-week ranked column with the label and its t statistic
    (mean / sd * sqrt(weeks)) over the weeks where the column varies.
    Fewer than min_weeks valid weeks -> NaN score."""

    def __init__(self, label: np.ndarray, dates, min_weeks: int = 1):
        self.week = week_index(dates)
        self.n_weeks = int(self.week.max()) + 1
        self.lz, self.lok, self.cnt = standardize_week(rank_week(label, dates), self.week, self.n_weeks)
        self.min_weeks = min_weeks

    def __call__(self, x_rank: np.ndarray) -> tuple[float, float, int]:
        z, ok, _ = standardize_week(x_rank, self.week, self.n_weeks)
        ic = np.bincount(self.week, weights=z * self.lz, minlength=self.n_weeks) / np.maximum(self.cnt, 1)
        ic = ic[ok & self.lok]
        n = int(len(ic))
        if n < max(self.min_weeks, 2):
            return float("nan"), float("nan"), n
        m = float(ic.mean())
        sd = float(ic.std(ddof=1))
        return m, (m / sd * np.sqrt(n) if sd > 0 else float("inf") * np.sign(m)), n


def dedup(candidates, held: np.ndarray, column, n: int, max_abs: float = 0.9) -> list[tuple[str, float]]:
    """Greedy: candidates in the given order; one is kept when the absolute
    pooled correlation of its column with every column already held (the
    `held` matrix, then each kept candidate) is at most max_abs; stops at n
    kept. `column(name)` returns the candidate's per-week ranked column.
    Returns (name, its largest |correlation| with what was held)."""
    def std(v):
        v = np.asarray(v, np.float32) - np.float32(np.mean(v))
        norm = float(np.sqrt(np.sum(v.astype(np.float64) ** 2)))
        return v / np.float32(norm) if norm > 0 else v
    H = np.column_stack([std(held[:, j]) for j in range(held.shape[1])]) if held.size else np.empty((len(held), 0), np.float32)
    kept = []
    for name in candidates:
        if len(kept) >= n:
            break
        c = std(column(name))
        r = float(np.max(np.abs(H.T @ c))) if H.shape[1] else 0.0
        if r <= max_abs:
            kept.append((name, r))
            H = np.column_stack([H, c])
    return kept
