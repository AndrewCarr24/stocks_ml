"""Strategy blending: a pre-registered meta-allocation over sleeve strategies.

Implements the four practitioner principles as mechanical rules (owner request,
2026-08-13). This is stacking with the capacity turned way down: the meta-layer
sees only each sleeve's OUT-OF-SAMPLE track record (walk-forward NAVs, the
analogue of out-of-fold predictions) and combines sleeves with equal weights
plus a small, bounded performance tilt — the Timmermann result says estimated
"optimal" combination weights lose to (near-)equal weights, so the combiner is
deliberately near-zero-capacity.

PRE-REGISTERED RULES (v1, fixed before the first backtest; constants are round
numbers chosen a priori and MUST NOT be tuned against backtest results —
changing any of them is a new trial for the ledger):

1. Blend-don't-switch: base allocation is EQUAL across active sleeves, tilted
   by the rank of trailing 3-year (156-week) Sharpe mapped linearly onto
   [0.75, 1.25], renormalized. Bounded tilt = shrinkage toward equality.
2. Change budget: weights update only at quarterly reviews (every 13 weeks),
   and each sleeve moves at most 25% of the way from its current weight to its
   target per review. No mid-quarter changes, no binary switching.
3. Promotion pipeline (probation): a sleeve with under 52 weeks of live track
   record receives HALF its computed share (renormalized across the roster).
4. Kill criteria (suspension with hysteresis, per sleeve, mechanical):
   - drawdown: suspended (weight 0) while drawdown from its all-time high
     exceeds 40%; reinstated only after recovery to under 20% drawdown.
   - dead money: trailing 3-year total return below zero halves its share.
   Suspension decisions use only NAV data available at the review date.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REVIEW_WEEKS = 13
TILT_LO, TILT_HI = 0.75, 1.25
MAX_MOVE = 0.25
PROBATION_WEEKS = 52
SUSPEND_DD, REINSTATE_DD = 0.40, 0.20
TRAIL_WEEKS = 156


@dataclass
class MetaAllocator:
    """Walk-forward sleeve weights under the pre-registered v1 rules."""

    sleeve_names: list
    suspended: set = field(default_factory=set)
    current: dict = field(default_factory=dict)

    def review(self, navs_to_date: dict) -> dict:
        """New sleeve weights from track records ending at the review date.

        navs_to_date: {sleeve: pd.Series of NAV strictly up to the review}."""
        # --- kill criteria (rule 4): suspension with hysteresis -------------
        active = []
        for s in self.sleeve_names:
            nav = navs_to_date[s].dropna()
            if len(nav) < 2:
                active.append(s)          # newborn: probation handles caution
                continue
            dd = 1.0 - nav.iloc[-1] / nav.cummax().iloc[-1]
            if s in self.suspended:
                if dd < REINSTATE_DD:
                    self.suspended.discard(s)
                    active.append(s)
            elif dd > SUSPEND_DD:
                self.suspended.add(s)
            else:
                active.append(s)
        if not active:                     # everything suspended: hold cash
            target = {s: 0.0 for s in self.sleeve_names}
            return self._bounded_move(target)

        # --- base equal weights with bounded trailing-Sharpe tilt (rule 1) --
        sharpe = {}
        for s in active:
            wk = (navs_to_date[s].dropna().resample("W-FRI").last()
                  .pct_change().dropna().iloc[-TRAIL_WEEKS:])
            sharpe[s] = (wk.mean() / wk.std() if len(wk) > 26 and wk.std() > 0
                         else 0.0)
        order = pd.Series(sharpe).rank(method="average")
        if len(order) > 1:
            tilt = TILT_LO + (order - 1) / (len(order) - 1) * (TILT_HI - TILT_LO)
        else:
            tilt = pd.Series(1.0, index=order.index)

        raw = {s: tilt[s] for s in active}

        # --- dead-money halving (rule 4) ------------------------------------
        for s in active:
            nav = navs_to_date[s].dropna()
            trail = nav.iloc[-min(len(nav), TRAIL_WEEKS * 5):]
            if len(trail) > 260 and trail.iloc[-1] / trail.iloc[0] - 1.0 < 0.0:
                raw[s] *= 0.5

        # --- probation (rule 3) ---------------------------------------------
        for s in active:
            if navs_to_date[s].dropna().resample("W-FRI").last().shape[0] < PROBATION_WEEKS:
                raw[s] *= 0.5

        total = sum(raw.values())
        target = {s: (raw.get(s, 0.0) / total if total > 0 else 0.0)
                  for s in self.sleeve_names}
        return self._bounded_move(target)

    def _bounded_move(self, target: dict) -> dict:
        """Rule 2: move each sleeve at most MAX_MOVE of the gap per review."""
        if not self.current:               # first review: adopt target directly
            self.current = dict(target)
            return dict(self.current)
        moved = {s: self.current.get(s, 0.0)
                 + MAX_MOVE * (target[s] - self.current.get(s, 0.0))
                 for s in target}
        total = sum(moved.values())
        if total > 1.0:                    # renormalize only if over-allocated
            moved = {s: w / total for s, w in moved.items()}
        self.current = moved
        return dict(moved)


def blended_weights(sleeve_navs: dict, sleeve_weights: dict,
                    rdates: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk the review calendar; return (stock-level targets, meta-weight path).

    sleeve_navs: {name: NAV Series}; sleeve_weights: {name: DataFrame of the
    sleeve's own stock targets on its rebalance grid (forward-filled here for
    slower cadences)}. Meta decisions at review t use NAVs strictly <= t."""
    allocator = MetaAllocator(sleeve_names=list(sleeve_navs))
    filled = {s: w.reindex(rdates).ffill().fillna(0.0)
              for s, w in sleeve_weights.items()}
    meta_path, combined = {}, {}
    meta = None
    for i, t in enumerate(rdates):
        if i % REVIEW_WEEKS == 0:
            meta = allocator.review({s: nav[nav.index <= t]
                                     for s, nav in sleeve_navs.items()})
        meta_path[t] = dict(meta)
        rows = [filled[s].loc[t] * w for s, w in meta.items() if w > 0]
        combined[t] = (pd.concat(rows, axis=1).sum(axis=1) if rows
                       else pd.Series(dtype=float))
    targets = pd.DataFrame(combined).T.fillna(0.0)
    targets = targets.loc[:, (targets != 0).any()]
    return targets, pd.DataFrame(meta_path).T


def run_weights_backtest(prices: pd.DataFrame, targets: pd.DataFrame, cfg,
                         initial: float = 100.0):
    """Trade a precomputed weekly stock-target path with the simulator's
    mechanics: execute at next open, 5bps one-way costs netted from capital,
    long-only/no-leverage invariant, daily NAV marks."""
    open_w = prices.pivot(index="date", columns="ticker", values="open").sort_index().ffill()
    close_w = prices.pivot(index="date", columns="ticker", values="close").sort_index().ffill()
    cal = close_w.index
    cash, shares, navs, total_costs = initial, {}, {}, 0.0
    rdates = targets.index
    for i, t in enumerate(rdates):
        w = targets.loc[t]
        w = w[w > 0]
        if (w < -1e-9).any() or w.sum() > 1 + 1e-9:
            raise ValueError("blend violated long-only/no-leverage invariant")
        ei = cal.searchsorted(t, side="right")
        if ei >= len(cal):
            break
        exec_day = cal[ei]
        opens = open_w.loc[exec_day]
        tradable = {tk: x for tk, x in w.items() if opens.get(tk, 0) > 0}
        port_val = cash + sum(s * opens.get(tk, 0.0) for tk, s in shares.items())
        if not navs:
            navs[t] = port_val
        current = {tk: s * opens.get(tk, 0.0) for tk, s in shares.items()}
        est_traded = sum(abs(tradable.get(tk, 0.0) * port_val - current.get(tk, 0.0))
                         for tk in set(tradable) | set(current))
        cost = est_traded * cfg.cost_bps / 1e4
        investable = port_val - cost
        total_costs += cost
        dollars = {tk: x * investable for tk, x in tradable.items()}
        shares = {tk: d / opens[tk] for tk, d in dollars.items()}
        cash = investable - sum(dollars.values())
        span_end = rdates[i + 1] if i + 1 < len(rdates) else cal[-1]
        for day in cal[(cal >= exec_day) & (cal <= span_end)]:
            px = close_w.loc[day]
            navs[day] = cash + sum(s * px.get(tk, 0.0) for tk, s in shares.items())
    return pd.Series(navs).sort_index(), total_costs
