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


def test_price_basis_precedence(tmp_path, monkeypatch):
    """yaml explicit > STOCKS_ML_PRICE_BASIS env > closeadj default. The env
    hook is how the nominal research drivers pick the basis without touching
    the global config the live job reads."""
    import shutil
    from stocks_ml.config import load_config
    p = tmp_path / "config.yaml"
    shutil.copy("config/config.yaml", p)
    monkeypatch.delenv("STOCKS_ML_PRICE_BASIS", raising=False)
    assert load_config(p).price_basis == "closeadj"
    monkeypatch.setenv("STOCKS_ML_PRICE_BASIS", "nominal")
    assert load_config(p).price_basis == "nominal"
    p.write_text(p.read_text() + "\nprice_basis: closeadj\n")
    assert load_config(p).price_basis == "closeadj"     # explicit yaml wins
