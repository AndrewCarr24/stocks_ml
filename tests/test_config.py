from pathlib import Path

from stocks_ml.config import load_config


def test_load_config_defaults():
    cfg = load_config("config/config.yaml")
    assert cfg.horizon_days == 5
    assert cfg.top_k == 8
    assert cfg.fred_series["VIXCLS"] == 1
    assert "net_income" in cfg.edgar_concepts
    assert cfg.data_dir == Path("data")
    assert cfg.backtest_start.year == 2005


def test_config_is_frozen():
    cfg = load_config("config/config.yaml")
    try:
        cfg.top_k = 99
        raised = False
    except Exception:
        raised = True
    assert raised
