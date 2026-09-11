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
    # "drop" or "last_print": how labels and the backtest universe treat a
    # name whose price series ends inside the label window (see load_config)
    delist_labels: str = "drop"
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
        # precedence: STOCKS_ML_PRICE_BASIS env (a research driver's explicit
        # per-process choice — how the cross-grade drivers read the other
        # world) > the yaml (the standing value; what the live job runs, which
        # sets no STOCKS_ML_* env) > closeadj. Flipped from yaml-first on
        # 2026-09-11 when config.yaml took the nominal basis.
        price_basis=str(os.environ.get("STOCKS_ML_PRICE_BASIS",
                                       raw.get("price_basis", "closeadj"))),
        # "drop" (a name leaving the tape inside the label window has no
        # label and is excluded from the backtest universe) or "last_print"
        # (delisting-honest: labels grade to the final print — the live
        # ledger's exit — and the backtest universe uses live's
        # traded-within-7-days rule). Same precedence pattern.
        delist_labels=str(os.environ.get("STOCKS_ML_DELIST_LABELS",
                                         raw.get("delist_labels", "drop"))),
        fred_series=dict(raw["fred_series"]),
        edgar_concepts={k: list(v) for k, v in raw["edgar_concepts"].items()},
    )
