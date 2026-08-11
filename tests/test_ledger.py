import json

import pandas as pd
import pytest

from stocks_ml.live.ledger import Ledger, latest_closes


def test_ledger_roundtrip_and_mark(tmp_path):
    path = tmp_path / "ledger.json"
    led = Ledger.load(path)          # missing -> empty
    assert led.cash == 0.0
    led.cash = 100.0
    led.apply_trades([("AAA", 2.0, 10.0)], "2026-07-06")   # buy 2 shares @ $10
    assert led.cash == 80.0
    assert led.positions["AAA"] == 2.0
    nav = led.mark(pd.Series({"AAA": 12.0}), "2026-07-07")
    assert nav == 80.0 + 24.0
    led.save(path)
    again = Ledger.load(path)
    assert again.positions == {"AAA": 2.0}
    assert again.nav_history[-1][1] == 104.0


def test_applied_files_roundtrip(tmp_path):
    path = tmp_path / "ledger.json"
    led = Ledger.load(path)
    led.applied_files.append("2026-07-06-trades.json")
    led.save(path)
    again = Ledger.load(path)
    assert again.applied_files == ["2026-07-06-trades.json"]


def test_load_tolerates_old_format_without_applied_files(tmp_path):
    path = tmp_path / "ledger.json"
    path.write_text(json.dumps({"cash": 5.0, "positions": {}, "nav_history": [], "trades": []}))
    led = Ledger.load(path)
    assert led.applied_files == []


def test_sell_more_than_held_is_capped(tmp_path):
    led = Ledger.load(tmp_path / "x.json")
    led.cash = 100.0
    led.apply_trades([("AAA", 1.0, 10.0)], "2026-07-06")
    led.apply_trades([("AAA", -5.0, 10.0)], "2026-07-07")  # only 1 held
    assert led.positions.get("AAA", 0.0) == 0.0
    assert led.cash == 100.0


def test_apply_trades_sells_fund_buys_and_overdraft_raises(tmp_path):
    led = Ledger.load(tmp_path / "x.json")
    led.cash = 10.0
    led.apply_trades([("AAA", 2.0, 5.0)], "2026-07-06")           # cash 0, 2 AAA
    # one call: sell AAA (proceeds 10) then buy BBB for 10 -> fine despite cash starting at 0
    led.apply_trades([("BBB", 1.0, 10.0), ("AAA", -2.0, 5.0)], "2026-07-07")
    assert led.positions == {"BBB": 1.0}
    assert led.cash == pytest.approx(0.0)
    with pytest.raises(ValueError):
        led.apply_trades([("CCC", 5.0, 10.0)], "2026-07-08")      # $50 buy with $0 cash


def test_apply_trades_deducts_one_way_costs(tmp_path):
    led = Ledger(cash=100.0)
    led.apply_trades([("AAA", 9.9, 10.0)], "2026-07-06", cost_bps=10.0)
    assert led.cash == pytest.approx(100.0 - 99.0 - 0.099)
    assert led.trades[-1][-1] == pytest.approx(0.099)


def test_ledger_rejects_nonfinite_trade(tmp_path):
    led = Ledger(cash=100.0)
    with pytest.raises(ValueError, match="finite"):
        led.apply_trades([("AAA", float("nan"), 10.0)], "2026-07-06")


def test_latest_closes_uses_last_available_price():
    prices = pd.DataFrame({
        "date": pd.to_datetime(["2026-07-01", "2026-07-02", "2026-07-01"]),
        "ticker": ["AAA", "AAA", "GONE"],
        "close": [10.0, 11.0, 5.0],
    })
    closes = latest_closes(prices)
    assert closes["AAA"] == 11.0
    assert closes["GONE"] == 5.0   # not zero — last available close
