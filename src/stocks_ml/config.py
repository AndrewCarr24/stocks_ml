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
    cv_train_years: int
    train_sample_rows: int | None
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
        cv_train_years=int(raw.get("cv_train_years", 2)),
        train_sample_rows=raw.get("train_sample_rows"),
        fred_series=dict(raw["fred_series"]),
        edgar_concepts={k: list(v) for k, v in raw["edgar_concepts"].items()},
    )
