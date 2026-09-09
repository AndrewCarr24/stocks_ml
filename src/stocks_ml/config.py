from __future__ import annotations

import os
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
    # "closeadj" (status quo: level features on the total-return basis) or
    # "nominal" (levels from closeunadj/close_split; the split-leak fix,
    # reports/nominal_basis_registration.md). Returns are closeadj either way.
    price_basis: str = "closeadj"
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
        # precedence: explicit yaml > STOCKS_ML_PRICE_BASIS env (how the
        # nominal research drivers select the basis without touching the
        # global config the live job reads) > closeadj
        price_basis=str(raw.get("price_basis",
                                os.environ.get("STOCKS_ML_PRICE_BASIS", "closeadj"))),
        fred_series=dict(raw["fred_series"]),
        edgar_concepts={k: list(v) for k, v in raw["edgar_concepts"].items()},
    )
