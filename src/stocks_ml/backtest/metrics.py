from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

EULER_GAMMA = 0.5772156649015329


def nav_to_returns(nav: pd.Series) -> pd.Series:
    return nav.pct_change().dropna()


def cagr(nav: pd.Series) -> float:
    days = (nav.index[-1] - nav.index[0]).days
    if days <= 0 or nav.iloc[0] <= 0:
        return float("nan")
    return float((nav.iloc[-1] / nav.iloc[0]) ** (365.25 / days) - 1)


def ann_sharpe(returns: pd.Series, periods: int = 252) -> float:
    if returns.std() == 0 or returns.empty:
        return float("nan")
    return float(returns.mean() / returns.std() * np.sqrt(periods))


def ann_sortino(returns: pd.Series, periods: int = 252) -> float:
    downside = returns.clip(upper=0).std()
    if downside == 0 or returns.empty:
        return float("nan")
    return float(returns.mean() / downside * np.sqrt(periods))


def max_drawdown(nav: pd.Series) -> float:
    return float((1 - nav / nav.cummax()).max())


def longest_underwater(nav: pd.Series) -> int:
    underwater = nav < nav.cummax()
    if not underwater.any():
        return 0
    idx = nav.index
    groups = (~underwater).cumsum()[underwater]
    dates = idx.to_series()[underwater]
    longest = 0
    for _, span in dates.groupby(groups):
        start_pos = idx.get_loc(span.iloc[0])
        peak_date = idx[start_pos - 1]  # row 0 is always its own cummax, so start_pos >= 1
        end_pos = idx.get_loc(span.iloc[-1])
        recovery_date = idx[end_pos + 1] if end_pos + 1 < len(idx) else idx[-1]
        longest = max(longest, (recovery_date - peak_date).days)
    return int(longest)


def worst_week(nav: pd.Series) -> float:
    weekly = nav.resample("W-FRI").last().pct_change().dropna()
    return float(weekly.min()) if not weekly.empty else float("nan")


def deflated_sharpe(returns: pd.Series, n_trials: int, periods: int = 252,
                    cross_trial_var: float | None = None) -> float:
    """Bailey & Lopez de Prado (2014). Returns P(true SR > 0 | n_trials tried).

    `cross_trial_var` is the variance of ANNUALIZED Sharpes across the actual
    trials (from the trials ledger) — the paper's prescribed input for the
    expected-max benchmark. When None (ledger too thin), falls back to the
    single-path estimator-variance proxy, which under-deflates when trials
    were diverse; reports should state which input was used."""
    T = len(returns)
    if T < 10 or returns.std() == 0:
        return float("nan")
    sr = float(returns.mean() / returns.std())  # per-period SR
    skew = float(stats.skew(returns))
    kurt = float(stats.kurtosis(returns, fisher=False))
    if n_trials <= 1:
        sr0 = 0.0
    else:
        if cross_trial_var is not None:
            var_sr = cross_trial_var / periods   # annualized -> per-period scale
        else:
            var_sr = (1 - skew * sr + (kurt - 1) / 4 * sr**2) / (T - 1)
        z1 = stats.norm.ppf(1 - 1.0 / n_trials)
        z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
        sr0 = np.sqrt(max(var_sr, 0)) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)
    denom = np.sqrt(max(1 - skew * sr + (kurt - 1) / 4 * sr**2, 1e-12))
    return float(stats.norm.cdf((sr - sr0) * np.sqrt(T - 1) / denom))


def expected_max_sr(n_trials: int, var_sr: float) -> float:
    """Bailey-LdP expected maximum SR among n_trials of true-zero strategies
    with cross-trial SR variance var_sr (same periodicity as the SR in use)."""
    if n_trials <= 1 or var_sr <= 0:
        return 0.0
    z1 = stats.norm.ppf(1 - 1.0 / n_trials)
    z2 = stats.norm.ppf(1 - 1.0 / (n_trials * np.e))
    return float(np.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2))


def min_track_record(sr: float, sr_benchmark: float, skew: float = 0.0,
                     kurt: float = 3.0, alpha: float = 0.05) -> float:
    """Bailey-LdP Minimum Track Record Length, in observations of the SR's
    periodicity: how long a live record must run before an observed SR clears
    the expected-max benchmark at confidence 1-alpha. inf when sr <= benchmark
    (no record length suffices)."""
    if not np.isfinite(sr) or sr <= sr_benchmark:
        return float("inf")
    z = stats.norm.ppf(1 - alpha)
    return float(1 + (1 - skew * sr + (kurt - 1) / 4 * sr**2)
                 * (z / (sr - sr_benchmark)) ** 2)


def regime_flags(spy_close: pd.Series, vix: pd.Series) -> pd.DataFrame:
    sma200 = spy_close.rolling(200, min_periods=50).mean()
    flags = pd.DataFrame(index=spy_close.index)
    flags["bull"] = spy_close > sma200
    flags["high_vol"] = vix.reindex(spy_close.index).ffill() > vix.median()
    return flags


def summarize(nav: pd.Series, n_trials: int = 1,
              cross_trial_var: float | None = None) -> dict:
    rets = nav_to_returns(nav)
    return {
        "terminal_100": float(100.0 * nav.iloc[-1] / nav.iloc[0]),
        "cagr": cagr(nav),
        "sharpe": ann_sharpe(rets),
        "sortino": ann_sortino(rets),
        "max_drawdown": max_drawdown(nav),
        "worst_week": worst_week(nav),
        "longest_underwater_days": longest_underwater(nav),
        "deflated_sharpe": deflated_sharpe(rets, n_trials,
                                           cross_trial_var=cross_trial_var),
    }


def regime_summaries(nav: pd.Series, flags: pd.DataFrame) -> dict:
    rets = nav_to_returns(nav)
    flags = flags.reindex(rets.index).ffill().fillna(False)
    out = {}
    for name, mask in {"bull": flags["bull"], "bear": ~flags["bull"],
                       "high_vol": flags["high_vol"], "low_vol": ~flags["high_vol"]}.items():
        sub = rets[mask.astype(bool)]
        out[name] = {
            "n_days": int(len(sub)),
            "ann_mean": float(sub.mean() * 252) if len(sub) else float("nan"),
            "ann_vol": float(sub.std() * np.sqrt(252)) if len(sub) > 1 else float("nan"),
            "sharpe": ann_sharpe(sub),
            "hit_rate": float((sub > 0).mean()) if len(sub) else float("nan"),
        }
    return out
