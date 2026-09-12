"""Derived feature families for the clean-improvement program's stage D
(reports/clean_improvement_registration.md): two point-in-time transforms
of the panel's own ranked features, computed from the panel alone.

  sector-relative   d_sec_<f>     f minus the median of f across the stock's
                                  sector that same week (the sector map the
                                  label_4w_sector label and the sector cap
                                  already use)
  rank momentum     d_mom<L>_<f>  f now minus f L panel weeks earlier for the
                                  same ticker (L in LAGS); NaN when the
                                  ticker had no row then

Both read only rows dated at or before the row they describe. The panel's
features are cross-sectional percentile ranks in [-1, 1] (features/ranking),
so the differences are already scale-free and are not re-ranked. Only the
per-stock ranked features qualify: within-week constants (market, macro,
calendar) give an identical zero under both transforms and the binary event
flags are not ranks (features/ranking.RANK_EXEMPT_PREFIXES).
"""
from __future__ import annotations

import pandas as pd

from stocks_ml.features.panel import feature_cols
from stocks_ml.features.ranking import RANK_EXEMPT_PREFIXES

LAGS = (4, 13, 26)
SEC_PREFIX = "d_sec_"
MOM_PREFIX = "d_mom"


def ranked_features(pan: pd.DataFrame) -> list[str]:
    """The per-stock cross-sectionally ranked members of feature_cols."""
    return [c for c in feature_cols(pan) if not c.startswith(RANK_EXEMPT_PREFIXES)]


def sector_relative(pan: pd.DataFrame, cols: list[str], smap: dict) -> pd.DataFrame:
    """d_sec_<f> = f - median(f | same date, same sector). A stock without a
    sector gets the week's median instead (as label_4w_sector does)."""
    sec = pan["ticker"].map(smap)
    keys = [pan["date"], sec]
    sec_med = pan[cols].groupby(keys).transform("median")      # NaN where no sector
    wk_med = pan[cols].groupby(pan["date"]).transform("median")
    out = pan[cols] - sec_med.fillna(wk_med)
    out.columns = [f"{SEC_PREFIX}{c}" for c in cols]
    return out


def rank_momentum(pan: pd.DataFrame, cols: list[str], lags=LAGS) -> pd.DataFrame:
    """d_mom<L>_<f> = f(t) - f(t - L panel weeks) per ticker, aligned on the
    panel's own date calendar (row position, not calendar days)."""
    dates = pd.Index(sorted(pan["date"].unique()))
    pos = pan["date"].map({d: i for i, d in enumerate(dates)}).to_numpy()
    idx = pd.MultiIndex.from_arrays([pos, pan["ticker"].to_numpy()], names=["pos", "ticker"])
    cur = pd.DataFrame(pan[cols].to_numpy(), index=idx, columns=cols)
    frames = []
    for lag in lags:
        past = cur.copy()
        past.index = pd.MultiIndex.from_arrays([past.index.get_level_values("pos") + lag,
                                                past.index.get_level_values("ticker")],
                                               names=["pos", "ticker"])
        past = past.reindex(idx)
        diff = pd.DataFrame(cur.to_numpy() - past.to_numpy(), index=pan.index,
                            columns=[f"{MOM_PREFIX}{lag}_{c}" for c in cols])
        frames.append(diff)
    return pd.concat(frames, axis=1)


def add_derived(pan: pd.DataFrame, smap: dict, lags=LAGS) -> tuple[pd.DataFrame, list[str]]:
    """The panel with both families appended; returns (panel, new columns)."""
    cols = ranked_features(pan)
    sec = sector_relative(pan, cols, smap)
    mom = rank_momentum(pan, cols, lags)
    new = pd.concat([sec, mom], axis=1)
    return pd.concat([pan, new], axis=1), list(new.columns)
