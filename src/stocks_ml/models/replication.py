"""The standardized K-copy ensembling protocol (iron rule 6 annex).

Any finalist on a leaderboard is promoted only in its replicated-ensemble
form, produced by exactly this procedure (owner-approved 2026-08-20):

  * K = 4 copies, identical for every finalist and every model class.
  * Copy c (c = 1..K) re-rolls two dice, both seeded by c:
      - the model's internal randomness (``random_state = c``), applied when
        the estimator exposes that parameter and inert otherwise;
      - a week-level bootstrap of every training window (same-size resample
        of whole weeks, with replacement) — the dice that deterministic
        models are sensitive to.
  * The deployable/graded object is the equal-weight average of the K
    copies' portfolios. Its exam score is what enters the leader leaderboard.

The procedure is config-preserving by design: it never alters a
hyperparameter (the t12 jitter study proved even small config changes can
carry real signal), only the realization of training. Rationale and the
seed-replication episode that produced this rule: AGENTS.md hard-won
history #9 and iron rule 6.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone

K_COPIES = 4


class WeekBootstrapEstimator(BaseEstimator, RegressorMixin):
    """Wraps any estimator; fits it on a whole-week bootstrap of the window.

    Requires the feature frame produced by ``dated_features`` (dates in
    ``X.attrs``): weeks are the panel's unique dates, resampled with
    replacement to the original week count and re-sorted chronologically so
    time-ordered internals (the time-tail early stop) still see a valid
    ordering. Duplicated weeks appear as adjacent blocks — the bootstrap
    analogue of sample weighting, not a shuffle."""

    def __init__(self, base, bootstrap_seed: int = 1):
        self.base = base
        self.bootstrap_seed = bootstrap_seed

    def fit(self, X, y):
        dates = X.attrs.get("dates") if hasattr(X, "attrs") else None
        if dates is None:
            raise ValueError("WeekBootstrapEstimator needs dated_features frames "
                             "(X.attrs['dates']); refusing to fit undated data")
        dates = np.asarray(dates)
        uniq = np.unique(dates)
        rng = np.random.default_rng(self.bootstrap_seed)
        sampled = np.sort(rng.choice(uniq, size=len(uniq), replace=True))
        idx = np.concatenate([np.flatnonzero(dates == d) for d in sampled])
        Xb = X.iloc[idx].reset_index(drop=True)
        Xb.attrs["dates"] = dates[idx]
        yb = pd.Series(np.asarray(y)[idx])   # inner estimators expect a Series
        self.model_ = clone(self.base)
        if "random_state" in self.model_.get_params():
            self.model_.set_params(random_state=self.bootstrap_seed)
        self.model_.fit(Xb, yb)
        return self

    def predict(self, X):
        return self.model_.predict(X)


def average_books(books: list[pd.DataFrame]) -> pd.DataFrame:
    """Equal-weight average of portfolio books on the union of their columns."""
    if not books:
        raise ValueError("average_books needs at least one book")
    cols = sorted(set().union(*[b.columns for b in books]))
    idx = books[0].index
    return sum(b.reindex(index=idx, columns=cols).fillna(0.0) for b in books) / len(books)


def replicated_ensemble(panel, prices, base_estimator, cfg, strategy_factory,
                        start, k: int = K_COPIES, cache_prefix: str | None = None,
                        walk_kwargs: dict | None = None):
    """Run the full protocol for one finalist: K bootstrapped walks -> K books
    -> one averaged ensemble book.

    ``strategy_factory`` must return a FRESH strategy instance per call
    (stateful strategies carry holdings memory). Returns (ensemble_book,
    per_copy_books). Scoring is the caller's job, via the standard exam."""
    from stocks_ml.backtest.simulator import run_backtest, walk_forward_predictions

    books = []
    for c in range(1, k + 1):
        est = WeekBootstrapEstimator(base_estimator, bootstrap_seed=c)
        cache = f"{cache_prefix}_copy{c}.parquet" if cache_prefix else None
        wf = walk_forward_predictions(panel, est, cfg, start=start,
                                     cache_path=cache, **(walk_kwargs or {}))
        r = run_backtest(panel, prices, strategy_factory(), est, cfg,
                         start=start, predictions=wf)
        books.append(r.weights)
    return average_books(books), books
