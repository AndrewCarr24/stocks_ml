from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd
import yaml


@dataclass(frozen=True)
class Config:
    data_dir: Path
    user_agent: str
    horizon_days: int
    purge_days: int
    rebalance_weekday: int
    retrain_weeks: int
    backtest_start: pd.Timestamp
    eval_start: pd.Timestamp
    holdout_years: int
    n_cv_folds: int
    train_sample_rows: int | None
    top_k: int
    vol_target: float
    avg_correlation: float
    dd_derisk: float
    dd_full: float
    kelly_fraction: float
    kelly_cap: float
    cost_bps: float
    live_strategy: str
    membership_floor: pd.Timestamp
    fred_series: dict = field(default_factory=dict)
    edgar_concepts: dict = field(default_factory=dict)


def load_config(path: str | Path = "config/config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text())
    return Config(
        data_dir=Path(raw["data_dir"]),
        user_agent=raw["user_agent"],
        horizon_days=int(raw["horizon_days"]),
        purge_days=int(raw["purge_days"]),
        rebalance_weekday=int(raw["rebalance_weekday"]),
        retrain_weeks=int(raw["retrain_weeks"]),
        backtest_start=pd.Timestamp(raw["backtest_start"]),
        eval_start=pd.Timestamp(raw.get("eval_start", raw["backtest_start"])),
        holdout_years=int(raw["holdout_years"]),
        n_cv_folds=int(raw["n_cv_folds"]),
        train_sample_rows=raw.get("train_sample_rows"),
        top_k=int(raw["top_k"]),
        vol_target=float(raw["vol_target"]),
        avg_correlation=float(raw["avg_correlation"]),
        dd_derisk=float(raw["dd_derisk"]),
        dd_full=float(raw["dd_full"]),
        kelly_fraction=float(raw["kelly_fraction"]),
        kelly_cap=float(raw["kelly_cap"]),
        cost_bps=float(raw["cost_bps"]),
        live_strategy=raw["live_strategy"],
        membership_floor=pd.Timestamp(raw["membership_floor"]),
        fred_series=dict(raw["fred_series"]),
        edgar_concepts={k: list(v) for k, v in raw["edgar_concepts"].items()},
    )
