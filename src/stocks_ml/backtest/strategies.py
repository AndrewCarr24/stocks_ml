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


class EqualWeightTopK(Strategy):
    name = "equal_topk"

    def __init__(self, k: int):
        self.k = k

    def propose_weights(self, preds, vols, risk):
        preds, vols = self._clean(preds, vols)
        picks = preds[preds > 0].nlargest(self.k)
        return pd.Series(1.0 / self.k, index=picks.index)


class VolScaledTopK(Strategy):
    name = "vol_scaled"

    def __init__(self, k: int, vol_target: float, rho: float,
                 dd_derisk: float, dd_full: float):
        self.k, self.vol_target, self.rho = k, vol_target, rho
        self.dd_derisk, self.dd_full = dd_derisk, dd_full
        self._guarded = False

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
        picks = preds[preds > 0].nlargest(self.k)
        if picks.empty or exposure == 0.0:
            return pd.Series(dtype=float)
        v = vols[picks.index].clip(lower=1e-4)
        w = (1.0 / v) / (1.0 / v).sum()
        var = (w**2 * v**2).sum()
        cross = np.outer(w * v, w * v)
        cov = self.rho * (cross.sum() - np.trace(cross))
        port_vol = float(np.sqrt(var + cov))
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


def make_strategies(cfg) -> dict[str, Strategy]:
    return {
        "equal_topk": EqualWeightTopK(cfg.top_k),
        "vol_scaled": VolScaledTopK(cfg.top_k, cfg.vol_target, cfg.avg_correlation,
                                    cfg.dd_derisk, cfg.dd_full),
        "kelly": FractionalKelly(cfg.kelly_fraction, cfg.kelly_cap),
    }
