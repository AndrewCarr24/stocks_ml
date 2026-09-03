"""Walk-forward prediction: fit on the trailing window, score one cross-section.

The champion (selection.ensemble_preds) calls this one rebalance date at a
time — K bootstrapped copies of the model, each fitted on the trailing
``cfg.cv_train_years`` of labeled weeks ending ``purge_days`` before the
date, predictions averaged downstream. The staggered multi-member ensemble
(``cfg.retrain_weeks`` members kept across a multi-date walk) is the same
code path and stays exercised by the tests.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
from sklearn.base import clone

from stocks_ml.features.panel import feature_cols
from stocks_ml.models.xgb import dated_features

MIN_TRAIN_ROWS = 50  # low floor: one real week has ~500 rows; small value keeps synthetic tests tradeable
MIN_TRAIN_WEEKS = 12  # enough complete dates for a purged time-tail stopping split


@dataclass
class WalkForwardPredictions:
    """Per-rebalance-date cross-sectional predictions plus the fit count behind them."""
    preds: dict = field(default_factory=dict)   # {rebalance date: pd.Series by ticker}
    n_fits: int = 0


def rebalance_calendar(panel, start=None, end=None, rebalance_every: int = 1) -> pd.DatetimeIndex:
    """Rebalance dates: every `rebalance_every`-th panel date within [start, end].

    Sliced from the panel's full date sequence BEFORE the start filter, so the
    cadence phase is a property of the panel, not of the walk's start — the
    staggered ensemble's anchor-independence carries over to slower cadences."""
    rdates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    rdates = rdates[::rebalance_every]
    if start:
        rdates = rdates[rdates >= pd.Timestamp(start)]
    if end:
        rdates = rdates[rdates <= pd.Timestamp(end)]
    return rdates


def walk_forward_predictions(panel, estimator, cfg, start=None, end=None,
                             label_col: str = "label", purge_days: int | None = None,
                             rebalance_every: int = 1,
                             cache_path=None) -> WalkForwardPredictions:
    """Staggered-refit ensemble walk: refresh one member per rebalance period.

    At each rebalance the newest member trains on data ending purge_days
    earlier; the ensemble keeps the last ``cfg.retrain_weeks`` members, so a
    date is always scored by models whose training cutoffs are 0..retrain_weeks-1
    rebalance periods stale, and the prediction is their plain mean. This makes
    predictions independent of when the walk started (no refit-anchor luck: a
    2-week shift in the old single-model schedule swung the June-2026 holdout
    between a degenerate abstention and a semiconductor bet). Averaging raw
    values also degrades gracefully: a degenerate member adds a near-constant
    offset that leaves healthy members' ranking intact.

    Other targets/cadences pass `label_col` (e.g. "label_4w"), `purge_days`
    exceeding that label's calendar span, and `rebalance_every` (panel dates
    per rebalance)."""
    # Walks cost hours of fits; cache_path (under the data dir, NOT tmp — the
    # OS purges tmp and has eaten these before) lets studies reuse them. The
    # caller owns invalidation: pass a new path when estimator/panel change.
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            stored = pd.read_parquet(cache_path)
            return WalkForwardPredictions(
                preds={pd.Timestamp(c): stored[c].dropna() for c in stored.columns},
                n_fits=int(stored.attrs.get("n_fits", len(stored.columns))))
    purge = cfg.purge_days if purge_days is None else purge_days
    fcols = feature_cols(panel)
    rdates = rebalance_calendar(panel, start, end, rebalance_every)
    labeled = panel[panel[label_col].notna()]

    members: list = []          # (fit_date, model), oldest first
    out = WalkForwardPredictions()
    for t in rdates:
        train_end = t - pd.Timedelta(days=purge)
        train_start = train_end - pd.DateOffset(years=cfg.cv_train_years)
        train = labeled[labeled["date"].between(train_start, train_end)]
        if cfg.train_sample_rows:
            train = train.sort_values("date").tail(cfg.train_sample_rows)
        if len(train) >= MIN_TRAIN_ROWS and train["date"].nunique() >= MIN_TRAIN_WEEKS:
            members.append((t, clone(estimator).fit(dated_features(train, fcols),
                                                    train[label_col])))
            out.n_fits += 1
            if len(members) > cfg.retrain_weeks:
                members.pop(0)
        if not members:
            continue
        rows = panel[panel["date"] == t]
        member_preds = [pd.Series(m.predict(rows[fcols]), index=rows["ticker"].values)
                        for _, m in members]
        out.preds[t] = pd.concat(member_preds, axis=1).mean(axis=1).dropna()
    if cache_path is not None:
        frame = pd.DataFrame(out.preds)
        frame.columns = [str(c) for c in frame.columns]
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(cache_path)
    return out
