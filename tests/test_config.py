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
    """STOCKS_ML_PRICE_BASIS env > yaml > closeadj default (env-first since
    2026-09-11: the yaml carries the standing nominal basis the live job runs,
    and a research driver reads the other world by setting the env)."""
    from stocks_ml.config import load_config
    p = tmp_path / "config.yaml"
    text = Path("config/config.yaml").read_text()
    bare = "\n".join(l for l in text.splitlines()
                     if not l.startswith(("price_basis:", "delist_labels:"))) + "\n"
    p.write_text(bare)
    monkeypatch.delenv("STOCKS_ML_PRICE_BASIS", raising=False)
    monkeypatch.delenv("STOCKS_ML_DELIST_LABELS", raising=False)
    assert load_config(p).price_basis == "closeadj" and load_config(p).delist_labels == "drop"
    p.write_text(bare + "price_basis: nominal\ndelist_labels: last_print\n")
    assert load_config(p).price_basis == "nominal" and load_config(p).delist_labels == "last_print"
    monkeypatch.setenv("STOCKS_ML_PRICE_BASIS", "closeadj")
    monkeypatch.setenv("STOCKS_ML_DELIST_LABELS", "drop")
    assert load_config(p).price_basis == "closeadj"       # env wins over the yaml
    assert load_config(p).delist_labels == "drop"
    # the checked-in config is the live job's: nominal basis, last-print labels
    monkeypatch.delenv("STOCKS_ML_PRICE_BASIS")
    monkeypatch.delenv("STOCKS_ML_DELIST_LABELS")
    live = load_config("config/config.yaml")
    assert (live.price_basis, live.delist_labels) == ("nominal", "last_print")
