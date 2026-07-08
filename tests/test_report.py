import pandas as pd

from stocks_ml.backtest.report import benchmark_navs, build_report, run_all_backtests


def test_benchmark_navs(synthetic_store, tiny_cfg):
    prices = synthetic_store.read("prices")
    fred = synthetic_store.read("fred")
    idx = pd.DatetimeIndex(sorted(prices[prices.ticker == "SPY"].date.unique()))[100:200]
    bench = benchmark_navs(prices, fred, idx)
    assert set(bench) == {"spy_hold", "cash"}
    assert bench["spy_hold"].iloc[0] == 100.0
    assert (bench["cash"].diff().dropna() >= 0).all()   # cash never loses


def test_spy_hold_survives_leading_gap(synthetic_store):
    prices = synthetic_store.read("prices")
    fred = synthetic_store.read("fred")
    spy_dates = pd.DatetimeIndex(sorted(prices[prices.ticker == "SPY"].date.unique()))
    # index starts 10 business days BEFORE SPY's first price
    idx = pd.bdate_range(end=spy_dates[0], periods=11)[:-1].append(spy_dates[:50])
    bench = benchmark_navs(prices, fred, pd.DatetimeIndex(idx))
    spy_nav = bench["spy_hold"]
    assert spy_nav.iloc[:10].isna().all()          # honest NaN before SPY exists
    assert spy_nav.loc[spy_dates[0]] == 100.0      # rescaled at first valid date
    assert spy_nav.dropna().notna().all()


def test_run_all_backtests_writes_report(synthetic_store, tiny_cfg, tmp_path):
    from stocks_ml.features.panel import build_panel
    from stocks_ml.models.champion import run_training
    from stocks_ml.models.candidates import MomentumRank, ZeroForecast

    build_panel(synthetic_store, tiny_cfg)
    models_dir = tmp_path / "models"
    run_training(synthetic_store, tiny_cfg,
                 candidates={"zero": ZeroForecast(), "momentum": MomentumRank()},
                 out_dir=models_dir)
    out = run_all_backtests(synthetic_store, tiny_cfg, models_dir=models_dir,
                            out_dir=tmp_path / "reports")
    text = out.read_text()
    assert "$100 →" in text
    assert "equal_topk" in text and "vol_scaled" in text and "kelly" in text
    assert "spy_hold" in text
    assert "Deflated Sharpe" in text or "deflated_sharpe" in text
    assert "bull" in text
    assert (tmp_path / "reports" / "equity.png").exists()
