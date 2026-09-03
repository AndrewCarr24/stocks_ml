from pathlib import Path

import pandas as pd

from stocks_ml.config import load_config


def test_load_config_defaults():
    cfg = load_config("config/config.yaml")
    assert cfg.horizon_days == 5
    assert cfg.purge_days == 10
    assert cfg.retrain_weeks == 4
    assert cfg.fred_series["VIXCLS"] == 1
    assert "net_income" in cfg.edgar_concepts
    assert cfg.data_dir == Path("data")
    assert cfg.backtest_start.year == 2005
    assert cfg.cv_train_years == 2
    assert cfg.train_sample_rows is None


def test_config_is_frozen():
    cfg = load_config("config/config.yaml")
    raised = False
    try:
        cfg.purge_days = 99
    except Exception:
        raised = True
    assert raised
