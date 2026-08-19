from __future__ import annotations

from collections import Counter
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


class EliteEntryBanded(BandedTopK):
    """BandedTopK whose entries must come from the very top of the ranking.

    Slots fill only from rank <= entry_rank (entering demands conviction);
    exits keep the wide band (rank > exit_rank). With entry_rank < k, slots
    the elite pool cannot fill stay empty and fall through to the floor —
    under SpyFloor the index ballast grows exactly when top-of-book
    conviction is scarce. 2026-08 selection study: entry 8 / slots 16 /
    exit 32 beat plain banded on every column."""

    name = "elite_banded"

    def __init__(self, k: int, entry_rank: int, exit_rank: int):
        super().__init__(k, exit_rank)
        if not 1 <= entry_rank <= exit_rank:
            raise ValueError("entry_rank must be in [1, exit_rank]")
        self.entry_rank = entry_rank

    def propose_weights(self, preds, vols, risk):
        preds, vols = self._clean(preds, vols)
        ranks = preds.rank(ascending=False, method="min")
        keep = [t for t in self._held
                if t in ranks.index and ranks[t] <= self.exit_rank and preds[t] > 0]
        free = self.k - len(keep)
        if free > 0:
            pool = preds[ranks <= self.entry_rank].drop(index=keep, errors="ignore")
            keep = keep + list(select_top_k(pool, free))
        self._held = keep
        return pd.Series(1.0 / self.k, index=pd.Index(keep))


class SectorCapElite(EliteEntryBanded):
    """EliteEntryBanded with a per-sector cap on new entries.

    A candidate is refused while `cap` held names already share its sector;
    held names are grandfathered (exits stay rank/sign-driven), and blocked
    slots stay unfilled -> SPY under SpyFloor. The cap converts sector
    crowding into index ballast at exactly the moments the model wants to
    concentrate: the 2026-08 study cut the semiconductor-slump drawdown from
    -32% to -12% at unchanged pre-holdout Sharpe, with a plateau across
    cap 3/4/5 (ridge, not spike). A trailing-correlation gate was tested as
    the statistical generalization and rejected — correlations spike during
    a bust, too late for an entry gate; sector labels carry the shared bet
    ex ante."""

    name = "seccap_banded"

    def __init__(self, k: int, entry_rank: int, exit_rank: int, cap: int,
                 sector_map: dict):
        super().__init__(k, entry_rank, exit_rank)
        if cap < 1:
            raise ValueError("cap must be >= 1")
        self.cap = cap
        self.sector_map = dict(sector_map)

    def propose_weights(self, preds, vols, risk):
        preds, vols = self._clean(preds, vols)
        ranks = preds.rank(ascending=False, method="min")
        keep = [t for t in self._held
                if t in ranks.index and ranks[t] <= self.exit_rank and preds[t] > 0]
        counts = Counter(self.sector_map.get(t) for t in keep)
        free = self.k - len(keep)
        pool = preds[ranks <= self.entry_rank].drop(index=keep, errors="ignore")
        while free > 0 and len(pool):
            pick = select_top_k(pool, 1)   # tie guard: tied top group > 1 -> abstain
            if len(pick) == 0:
                break
            t = pick[0]
            pool = pool.drop(index=[t])
            if counts[self.sector_map.get(t)] >= self.cap:
                continue
            keep.append(t)
            counts[self.sector_map.get(t)] += 1
            free -= 1
        self._held = keep
        return pd.Series(1.0 / self.k, index=pd.Index(keep))


class EasedWeights(Strategy):
    """Move halfway toward the inner strategy's book instead of jumping.

    Each week the proposed book is lam * last week's book + (1-lam) * the
    inner target: same average holdings, roughly half the trade sizes, so
    costs and variance drag fall with no measured return give-up (2026-08
    turnover study: paired weekly t vs the raw book ~0.15; flat across
    lam 0.25-0.75). Sub-0.01% dust positions are dropped; the freed weight
    falls through to the floor. Stateful (previous book) — fresh instance
    per backtest, same convention as BandedTopK."""

    def __init__(self, inner: Strategy, lam: float = 0.5):
        if not 0.0 <= lam < 1.0:
            raise ValueError(f"lam must be in [0, 1), got {lam}")
        self.inner, self.lam = inner, lam
        self.name = f"eased_{inner.name}"
        self._prev = pd.Series(dtype=float)

    def propose_weights(self, preds, vols, risk):
        target = self.inner.propose_weights(preds, vols, risk)
        allidx = target.index.union(self._prev.index)
        w = (self.lam * self._prev.reindex(allidx).fillna(0.0)
             + (1 - self.lam) * target.reindex(allidx).fillna(0.0))
        w = w[w > 1e-4]
        if w.sum() > 1.0:
            w = w / w.sum()
        self._prev = w
        return w


class BookAverage(Strategy):
    """Equal-weight average of several strategies' proposed books.

    Diversifies across our own parameter choices instead of betting on one
    cell: where member configs agree a position is full-size, where they
    disagree it is automatically partial. The 2026-08 nine-member average
    matched the best single cell's Sharpe with lower costs and better
    holdout risk — parameter selection risk removed for free. Members keep
    their own state, so fresh instances per backtest apply to them too."""

    def __init__(self, members: list):
        if not members:
            raise ValueError("BookAverage needs at least one member")
        self.members = list(members)
        self.name = f"book_avg{len(members)}"

    def propose_weights(self, preds, vols, risk):
        books = [m.propose_weights(preds, vols, risk) for m in self.members]
        allidx = pd.Index([])
        for b in books:
            allidx = allidx.union(b.index)
        w = sum(b.reindex(allidx).fillna(0.0) for b in books) / len(books)
        return w[w > 0]


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

    # The full stop (exposure 0) freezes the NAV, and drawdown is measured
    # against the all-time peak, so from inside the cash state the drawdown
    # can never improve: without a time limit the full stop is absorbing and
    # the strategy would sit in cash forever. Cap it at one quarter, then
    # resume at half exposure so the NAV can trade back to the release
    # threshold. The cool-off budget refills only on full recovery.
    REENTRY_WEEKS = 13

    def __init__(self, k: int, vol_target: float, rho: float,
                 dd_derisk: float, dd_full: float):
        if not 0.0 <= rho < 1.0:
            raise ValueError(f"rho must be in [0, 1) for a valid equicorrelation matrix, got {rho}")
        self.k, self.vol_target, self.rho = k, vol_target, rho
        self.dd_derisk, self.dd_full = dd_derisk, dd_full
        self._guarded = False
        self._cash_weeks = 0

    def restore_guard(self, guarded: bool) -> None:
        """Warm-start the hysteresis state (live runs replay it from NAV history)."""
        self._guarded = bool(guarded)

    def _exposure(self, dd: float) -> float:
        if dd >= self.dd_full and self._cash_weeks < self.REENTRY_WEEKS:
            self._cash_weeks += 1
            self._guarded = True
            return 0.0
        if dd >= self.dd_derisk:
            self._guarded = True
            return 0.5
        if self._guarded and dd >= self.dd_derisk / 2:
            return 0.5
        self._guarded = False
        self._cash_weeks = 0
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
