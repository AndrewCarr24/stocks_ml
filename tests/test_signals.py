from stocks_ml.live.ledger import Ledger
from stocks_ml.live.signals import generate_signals


def test_generate_signals(synthetic_store, tiny_cfg, tmp_path):
    from stocks_ml.features.panel import build_panel
    from stocks_ml.models.champion import run_training
    from stocks_ml.models.candidates import MomentumRank, ZeroForecast

    build_panel(synthetic_store, tiny_cfg)
    models_dir = tmp_path / "models"
    run_training(synthetic_store, tiny_cfg,
                 candidates={"zero": ZeroForecast(), "momentum": MomentumRank()},
                 out_dir=models_dir)
    ledger = Ledger.load(tmp_path / "ledger.json")
    ledger.cash = 100.0
    md, trades = generate_signals(synthetic_store, tiny_cfg, ledger, models_dir=models_dir)
    assert "Target portfolio" in md
    assert isinstance(trades, list)
    for ticker, delta_shares, price in trades:
        assert isinstance(ticker, str) and price > 0
