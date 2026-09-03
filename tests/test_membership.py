import pandas as pd

from stocks_ml.data.membership import members_asof, normalize_symbol

FLOOR = pd.Timestamp("1996-01-01")


def test_normalize_symbol():
    assert normalize_symbol(" brk.b ") == "BRK-B"


def test_normalize_symbol_strips_formatting_junk():
    assert normalize_symbol("ITT |") == "ITT"
    assert normalize_symbol(" BRK.B ") == "BRK-B"
    assert normalize_symbol("JCP |") == "JCP"


def test_members_asof():
    # AAA a member throughout; ZZZ left 2010-02-01 and rejoined 2020-05-01
    mem = pd.DataFrame({
        "ticker": ["AAA", "ZZZ", "ZZZ"],
        "start_date": [FLOOR, FLOOR, pd.Timestamp("2020-05-01")],
        "end_date": [pd.NaT, pd.Timestamp("2010-02-01"), pd.NaT],
        "sector": ["Tech", "Retail", "Retail"],
    })
    assert set(members_asof(mem, "2005-01-01")) == {"AAA", "ZZZ"}
    assert set(members_asof(mem, "2015-01-01")) == {"AAA"}
    assert set(members_asof(mem, "2021-01-01")) == {"AAA", "ZZZ"}
    # the leave date itself is exclusive; the join date inclusive
    assert set(members_asof(mem, "2010-02-01")) == {"AAA"}
    assert set(members_asof(mem, "2020-05-01")) == {"AAA", "ZZZ"}
