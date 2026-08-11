from pathlib import Path

import pandas as pd

from stocks_ml.config import load_config


def test_load_config_defaults():
    cfg = load_config("config/config.yaml")
    assert cfg.horizon_days == 5
    assert cfg.top_k == 8
    assert cfg.fred_series["VIXCLS"] == 1
    assert "net_income" in cfg.edgar_concepts
    assert cfg.data_dir == Path("data")
    assert cfg.backtest_start.year == 2005
    assert cfg.eval_start == pd.Timestamp("2015-03-01")
    assert cfg.n_cv_folds == 4
    assert cfg.cv_train_years == 2
    assert cfg.live_strategy == "vol_scaled"


def test_config_is_frozen():
    cfg = load_config("config/config.yaml")
    try:
        cfg.top_k = 99
        raised = False
    except Exception:
        raised = True
    assert raised
