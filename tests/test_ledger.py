import pandas as pd

from stocks_ml.live.ledger import Ledger


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


def test_sell_more_than_held_is_capped(tmp_path):
    led = Ledger.load(tmp_path / "x.json")
    led.cash = 100.0
    led.apply_trades([("AAA", 1.0, 10.0)], "2026-07-06")
    led.apply_trades([("AAA", -5.0, 10.0)], "2026-07-07")  # only 1 held
    assert led.positions.get("AAA", 0.0) == 0.0
    assert led.cash == 100.0
