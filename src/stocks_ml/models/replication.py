"""The standardized K-copy ensembling protocol (iron rule 6 annex).

Every configuration the selection procedure grades — and the champion the
weekly job runs — is the replicated-ensemble form produced by exactly this
procedure (owner-approved 2026-08-20):

  * K = 4 copies, identical for every finalist and every model class.
  * Copy c (c = 1..K) re-rolls two dice, both seeded by c:
      - the model's internal randomness (``random_state = c``), applied when
        the estimator exposes that parameter and inert otherwise;
      - a week-level bootstrap of every training window (same-size resample
        of whole weeks, with replacement) — the dice that deterministic
        models are sensitive to.
  * The deployable/graded object averages the K copies' predictions
    (selection.ensemble_preds) before the book is formed.

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
