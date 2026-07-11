from stocks_ml.live.ledger import Ledger
from stocks_ml.live.signals import generate_signals, replay_guard_state


def test_replay_guard_state_stays_guarded_until_half_derisk_dd():
    # 100 -> 82 is a 18% dd (>= 15% dd_derisk) -> guard trips; 82 -> 90 is a 10% dd
    # from the 100 high-water mark, still >= 7.5% (dd_derisk / 2) -> stays guarded
    nav_history = [("2026-01-01", 100.0), ("2026-01-02", 82.0), ("2026-01-03", 90.0)]
    assert replay_guard_state(nav_history, dd_derisk=0.15, dd_full=0.30) is True


def test_replay_guard_state_clears_below_half_derisk_dd():
    # 100 -> 82 trips the guard; 82 -> 95.1 recovers to a 4.9% dd (< 7.5%) -> clears
    nav_history = [("2026-01-01", 100.0), ("2026-01-02", 82.0), ("2026-01-03", 95.1)]
    assert replay_guard_state(nav_history, dd_derisk=0.15, dd_full=0.30) is False


def test_replay_guard_state_empty_history_is_unguarded():
    assert replay_guard_state([], dd_derisk=0.15, dd_full=0.30) is False


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
