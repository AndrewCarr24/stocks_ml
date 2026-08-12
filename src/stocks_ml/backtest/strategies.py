from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52

# Correlation cap: a candidate is skipped only when it would pair with an
# already-selected pick above this quantile of ALL pairwise trailing
# correlations in that week's universe. Deliberately lenient (owner-specified):
# ~99% of possible pairs are allowed, so the cap is a no-op in typical weeks
# and binds only on statistical outliers — share-class twins, one sector
# crowding the whole book (the June-2026 semiconductor bet averaged 0.45
# pairwise vs a 0.12 random-basket mean). Self-calibrating: the threshold is
# recomputed weekly from the cross-section, so high-correlation regimes raise
# the bar instead of freezing the strategy.
CORR_PAIR_QUANTILE = 0.99
MIN_CORR_OBS = 60  # trading days of overlapping returns needed to trust a correlation


@dataclass
class RiskState:
    drawdown: float = 0.0
    # trailing daily returns (rows = days, cols = tickers), known at decision
    # time; None disables the correlation cap (e.g. unit tests, missing data)
    trailing_returns: pd.DataFrame | None = None


class Strategy:
    name = "base"

    def propose_weights(self, preds: pd.Series, vols: pd.Series, risk: RiskState) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    def _clean(preds: pd.Series, vols: pd.Series) -> tuple[pd.Series, pd.Series]:
        common = preds.dropna().index.intersection(vols.dropna().index)
        return preds[common], vols[common]


def _pair_corr_cap(pos_index, trailing_returns):
    """Correlation matrix over candidates + that week's outlier threshold.

    Returns (corr, cap) or (None, None) when there isn't enough data to
    calibrate — in which case the cap is disabled rather than guessed."""
    cols = [t for t in pos_index if t in trailing_returns.columns]
    r = trailing_returns[cols]
    r = r.loc[:, r.notna().sum() >= MIN_CORR_OBS]
    if r.shape[1] < 3:
        return None, None
    corr = r.corr()
    vals = corr.values[np.triu_indices(len(corr), 1)]
    vals = vals[np.isfinite(vals)]
    if not len(vals):
        return None, None
    return corr, float(np.quantile(vals, CORR_PAIR_QUANTILE))


def select_top_k(preds: pd.Series, k: int,
                 trailing_returns: pd.DataFrame | None = None) -> pd.Index:
    """Top-k selection that never splits a tie and refuses crowded pairs.

    Slots fill with whole groups of equally-predicted stocks, best value first.
    A group larger than the remaining slots is refused outright: sampling
    inside a tie would select by row order (alphabetical), turning "the model
    can't tell these apart" into fake conviction. Near-constant predictions —
    the degenerate refits where early stopping kept ~no trees — therefore
    select nothing, and the unfilled slots fall through to the strategy's
    floor (cash, or SPY under SpyFloor).

    With ``trailing_returns``, a candidate is additionally skipped when its
    trailing correlation with ANY already-selected pick exceeds the week's
    CORR_PAIR_QUANTILE pairwise threshold; its slot passes to the next-ranked
    name, so the ranking concedes only the minimum needed to avoid outlier
    crowding. Skipping is judged against prior selections only (never within
    a tie group), keeping selection independent of row order. Tickers with
    insufficient return history are treated as uncorrelated."""
    pos = preds[preds > 0]
    corr, cap = (None, None)
    if trailing_returns is not None and len(pos):
        corr, cap = _pair_corr_cap(pos.index, trailing_returns)
    selected: list = []
    remaining = k
    for value in np.sort(pos.unique())[::-1]:
        members = pos.index[pos == value]
        if len(members) > remaining:
            break
        if cap is not None and selected:
            prior = [s for s in selected if s in corr.columns]
            members = [t for t in members
                       if t not in corr.index or not prior
                       or not (corr.loc[t, prior] > cap + 1e-12).any()]
        selected.extend(members)
        remaining -= len(members)
        if remaining == 0:
            break
    return pd.Index(selected)


class EqualWeightTopK(Strategy):
    name = "equal_topk"

    def __init__(self, k: int):
        self.k = k

    def propose_weights(self, preds, vols, risk):
        preds, vols = self._clean(preds, vols)
        picks = select_top_k(preds, self.k, risk.trailing_returns)
        return pd.Series(1.0 / self.k, index=picks)


class VolScaledTopK(Strategy):
    name = "vol_scaled"

    def __init__(self, k: int, vol_target: float, rho: float,
                 dd_derisk: float, dd_full: float):
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must be in [0, 1) for a valid equicorrelation matrix, got {rho}")
        self.k, self.vol_target, self.rho = k, vol_target, rho
        self.dd_derisk, self.dd_full = dd_derisk, dd_full
        self._guarded = False

    def restore_guard(self, guarded: bool) -> None:
        """Warm-start the hysteresis state (live runs replay it from NAV history)."""
        self._guarded = bool(guarded)

    def _exposure(self, dd: float) -> float:
        if dd >= self.dd_full:
            self._guarded = True
            return 0.0
        if dd >= self.dd_derisk:
            self._guarded = True
            return 0.5
        if self._guarded and dd >= self.dd_derisk / 2:
            return 0.5
        self._guarded = False
        return 1.0

    def propose_weights(self, preds, vols, risk):
        exposure = self._exposure(risk.drawdown)
        preds, vols = self._clean(preds, vols)
        picks = select_top_k(preds, self.k, risk.trailing_returns)
        if len(picks) == 0 or exposure == 0.0:
            return pd.Series(dtype=float)
        v = vols[picks].clip(lower=1e-4)
        w = (1.0 / v) / (1.0 / v).sum()
        var = (w**2 * v**2).sum()
        cross = np.outer(w * v, w * v)
        cov = self.rho * (cross.sum() - np.trace(cross))
        port_vol = float(np.sqrt(max(var + cov, 0.0)))
        if port_vol > self.vol_target:
            w = w * (self.vol_target / port_vol)
        return w * exposure


class FractionalKelly(Strategy):
    name = "kelly"

    def __init__(self, fraction: float, cap: float):
        self.fraction, self.cap = fraction, cap

    def propose_weights(self, preds, vols, risk):
        preds, vols = self._clean(preds, vols)
        weekly_var = (vols.clip(lower=1e-4) ** 2) / WEEKS_PER_YEAR
        w = (self.fraction * preds / weekly_var).clip(lower=0.0).clip(upper=self.cap)
        w = w[w > 0]
        if w.sum() > 1.0:
            w = w / w.sum()
        return w


class SpyFloor(Strategy):
    """Wraps a strategy so its UNALLOCATED fraction goes into SPY instead of cash.

    The inner strategy's stock sleeve is untouched; whatever it leaves on the
    table (1 - sum of its weights) buys the index. This changes the floor of the
    strategy from "cash earning nothing" to "the market": the model's picks then
    only need to beat SPY — not zero — to justify their allocation. The Kelly
    sleeve is confidence-scaled by construction (weights ∝ predicted edge /
    variance), so SpyFloor(FractionalKelly) directly implements "invest part by
    model confidence, rest in the S&P 500."

    Deliberately NOT applied to VolScaledTopK: its drawdown guard's whole point
    is a cash refuge during crashes, and routing guard-freed money into SPY
    would re-expose it to the very drawdown it is fleeing."""

    def __init__(self, inner: Strategy, spy_ticker: str = "SPY"):
        self.inner = inner
        self.spy_ticker = spy_ticker
        self.name = f"{inner.name}_spy"

    def propose_weights(self, preds, vols, risk):
        w = self.inner.propose_weights(preds, vols, risk).copy()
        # Guard against the inner strategy ever emitting SPY itself (it never
        # should — SPY has no panel predictions — but summing twice would breach
        # the no-leverage invariant).
        w = w.drop(self.spy_ticker, errors="ignore")
        remainder = 1.0 - float(w.sum())
        if remainder > 1e-9:
            w[self.spy_ticker] = remainder
        return w


def make_strategies(cfg) -> dict[str, Strategy]:
    return {
        "equal_topk": EqualWeightTopK(cfg.top_k),
        "vol_scaled": VolScaledTopK(cfg.top_k, cfg.vol_target, cfg.avg_correlation,
                                    cfg.dd_derisk, cfg.dd_full),
        "kelly": FractionalKelly(cfg.kelly_fraction, cfg.kelly_cap),
        # SPY-floor variants: unallocated money holds the index instead of cash.
        "kelly_spy": SpyFloor(FractionalKelly(cfg.kelly_fraction, cfg.kelly_cap)),
        "topk_spy": SpyFloor(EqualWeightTopK(cfg.top_k)),
    }
