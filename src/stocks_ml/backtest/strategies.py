from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

WEEKS_PER_YEAR = 52


@dataclass
class RiskState:
    drawdown: float = 0.0


class Strategy:
    name = "base"

    def propose_weights(self, preds: pd.Series, vols: pd.Series, risk: RiskState) -> pd.Series:
        raise NotImplementedError

    @staticmethod
    def _clean(preds: pd.Series, vols: pd.Series) -> tuple[pd.Series, pd.Series]:
        common = preds.dropna().index.intersection(vols.dropna().index)
        return preds[common], vols[common]


def select_top_k(preds: pd.Series, k: int) -> pd.Index:
    """Top-k selection that never splits a tie.

    Slots fill with whole groups of equally-predicted stocks, best value first.
    A group larger than the remaining slots is refused outright: sampling
    inside a tie would select by row order (alphabetical), turning "the model
    can't tell these apart" into fake conviction. Near-constant predictions —
    the degenerate refits where early stopping kept ~no trees — therefore
    select nothing, and the unfilled slots fall through to the strategy's
    floor (cash, or SPY under SpyFloor)."""
    pos = preds[preds > 0]
    selected: list = []
    remaining = k
    for value in np.sort(pos.unique())[::-1]:
        members = pos.index[pos == value]
        if len(members) > remaining:
            break
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
        picks = select_top_k(preds, self.k)
        return pd.Series(1.0 / self.k, index=picks)


class BandedTopK(Strategy):
    """Top-k with an exit band: enter at the top, leave only on real decay.

    The breadth study (2026-08-18) showed the champion's weekly signal is
    noisy and that damping turnover was worth +0.11 Sharpe; this is the
    band-rule version deployable at feasible position counts. A stock enters
    only from the top-k of the ranking (via select_top_k — tie guard applies),
    but once held it keeps its slot until its rank decays past `exit_rank`
    (or its prediction turns non-positive). Classic hysteresis: churn in the
    16-40 rank zone — pure noise trading — is eliminated.

    Stateful across a walk (holdings memory); use a fresh instance per
    backtest, same convention as VolScaledTopK's guard state."""

    name = "banded_topk"

    def __init__(self, k: int, exit_rank: int):
        if exit_rank < k:
            raise ValueError("exit_rank must be >= k (the band cannot be inverted)")
        self.k, self.exit_rank = k, exit_rank
        self._held: list = []

    def propose_weights(self, preds, vols, risk):
        preds, vols = self._clean(preds, vols)
        ranks = preds.rank(ascending=False, method="min")
        keep = [t for t in self._held
                if t in ranks.index and ranks[t] <= self.exit_rank and preds[t] > 0]
        free = self.k - len(keep)
        if free > 0:
            pool = preds.drop(index=keep, errors="ignore")
            keep = keep + list(select_top_k(pool, free))
        self._held = keep
        return pd.Series(1.0 / self.k, index=pd.Index(keep))


class ConfidenceTopK(Strategy):
    """Top-k by score, where only scores above `floor` count as conviction.

    Built for calibrated-probability models: with floor=0.5, a stock qualifies
    only when the model says it is more likely than not to be a top-quintile
    name. Reuses select_top_k on (score - floor), so the tie guard applies and
    low-confidence weeks fill fewer slots — the shortfall lands in cash, or in
    SPY under SpyFloor. Slots are 1/k each regardless of how many fill, same
    conviction-budget semantics as EqualWeightTopK."""

    name = "conf_topk"

    def __init__(self, k: int, floor: float):
        self.k, self.floor = k, floor

    def propose_weights(self, preds, vols, risk):
        preds, vols = self._clean(preds, vols)
        picks = select_top_k(preds - self.floor, self.k)
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
        picks = select_top_k(preds, self.k)
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
