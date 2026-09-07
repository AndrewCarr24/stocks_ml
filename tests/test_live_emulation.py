"""ops/live_emulation.py: the champion as deployed vs fills at the decision close."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from stocks_ml.ledger import Ledger
from stocks_ml.selection import price_frames

spec = importlib.util.spec_from_file_location(
    "live_emulation", Path(__file__).resolve().parents[1] / "ops/live_emulation.py")
emu = importlib.util.module_from_spec(spec)
spec.loader.exec_module(emu)


def _ctx():
    days = pd.to_datetime(["2024-06-27", "2024-06-28", "2024-07-01", "2024-07-02"])
    cw = pd.DataFrame({"AAA": [10.0, 10.0, 11.0, 11.5], "SPY": [50.0, 50.0, 50.0, 50.0]}, index=days)
    ow = pd.DataFrame({"AAA": [9.9, 10.1, 10.6, 11.2], "SPY": [50.0, 50.0, 50.0, 50.0]}, index=days)
    return SimpleNamespace(smap={}, members={}, **price_frames(cw, ow))


def test_fills_at_close_prices_a_friday_decision_at_fridays_close():
    ctx = _ctx()
    fri, mon = pd.Timestamp("2024-06-28"), pd.Timestamp("2024-07-01")
    for world, price in ((ctx, 10.6), (emu.fills_at_close(ctx), 10.0)):
        ledger = Ledger.new(100.0, fri)
        ledger.pending = {"decision_date": str(fri.date()), "weights": {"AAA": 1.0}}
        fills = ledger.fill_pending(world.closes, world.opens, mon + pd.Timedelta(days=4))
        assert [f[1] for f in fills] == ["AAA"] and fills[0][0] == str(mon.date())
        assert fills[0][3] == pytest.approx(price)


def test_rotations_list_what_each_rotation_sold_and_bought():
    t0, t1 = pd.Timestamp("2024-06-21"), pd.Timestamp("2024-06-28")
    trace = [{"t": t0, "rotated": [0, 1], "sleeves": [["AAA"], ["AAA"]]},   # first fill: nothing sold
             {"t": t1, "rotated": [1], "sleeves": [["AAA"], ["BBB"]]}]
    assert emu.rotations(trace) == [{"t": t1, "out": ["AAA"], "in": ["BBB"]}]


def test_weekend_gaps_use_incoming_and_outgoing_names():
    ctx = _ctx()
    events = [{"t": pd.Timestamp("2024-06-28"), "in": ["AAA"], "out": ["SPY"]}]
    g = emu.weekend_gaps(ctx, events)
    assert g.loc[0, "gap_in"] == pytest.approx(10.6 / 10.0 - 1)
    assert g.loc[0, "gap_out"] == pytest.approx(0.0)
