import pandas as pd

from stocks_ml.data.membership import build_membership, members_asof, normalize_symbol

FLOOR = pd.Timestamp("1996-01-01")


def _changes(rows):
    return pd.DataFrame(rows, columns=["date", "added", "removed"]).assign(
        date=lambda d: pd.to_datetime(d["date"])
    )


def test_normalize_symbol():
    assert normalize_symbol(" brk.b ") == "BRK-B"


def test_current_member_never_changed_gets_floor_start():
    current = pd.DataFrame({"ticker": ["AAA"], "sector": ["Tech"]})
    mem = build_membership(current, _changes([]), FLOOR)
    row = mem[mem.ticker == "AAA"].iloc[0]
    assert row.start_date == FLOOR
    assert pd.isna(row.end_date)


def test_added_member_gets_start_date():
    current = pd.DataFrame({"ticker": ["AAA", "BBB"], "sector": ["Tech", "Energy"]})
    changes = _changes([("2015-06-01", "BBB", None)])
    mem = build_membership(current, changes, FLOOR)
    assert mem[mem.ticker == "BBB"].iloc[0].start_date == pd.Timestamp("2015-06-01")


def test_removed_member_gets_closed_stint():
    current = pd.DataFrame({"ticker": ["AAA"], "sector": ["Tech"]})
    changes = _changes([("2018-03-05", "AAA", "OLD")])
    mem = build_membership(current, changes, FLOOR)
    old = mem[mem.ticker == "OLD"].iloc[0]
    assert old.start_date == FLOOR
    assert old.end_date == pd.Timestamp("2018-03-05")


def test_multi_stint_ticker():
    # ZZZ removed 2010, re-added 2020: two stints
    current = pd.DataFrame({"ticker": ["AAA", "ZZZ"], "sector": ["Tech", "Retail"]})
    changes = _changes([("2020-05-01", "ZZZ", None), ("2010-02-01", None, "ZZZ")])
    mem = build_membership(current, changes, FLOOR)
    stints = mem[mem.ticker == "ZZZ"].sort_values("start_date")
    assert len(stints) == 2
    assert stints.iloc[0].start_date == FLOOR
    assert stints.iloc[0].end_date == pd.Timestamp("2010-02-01")
    assert stints.iloc[1].start_date == pd.Timestamp("2020-05-01")
    assert pd.isna(stints.iloc[1].end_date)


def test_same_date_add_and_remove_in_separate_rows_is_order_independent():
    current = pd.DataFrame({"ticker": ["AAA"], "sector": ["Tech"]})
    d = "2020-01-06"
    for rows in ([(d, None, "AAA"), (d, "AAA", None)],
                 [(d, "AAA", None), (d, None, "AAA")]):
        mem = build_membership(current, _changes(rows), FLOOR)
        stints = mem[mem.ticker == "AAA"].sort_values("start_date")
        assert len(stints) == 2
        assert stints.iloc[0].start_date == FLOOR
        assert stints.iloc[0].end_date == pd.Timestamp(d)
        assert stints.iloc[1].start_date == pd.Timestamp(d)
        assert pd.isna(stints.iloc[1].end_date)


def test_members_asof():
    current = pd.DataFrame({"ticker": ["AAA", "ZZZ"], "sector": ["Tech", "Retail"]})
    changes = _changes([("2020-05-01", "ZZZ", None), ("2010-02-01", None, "ZZZ")])
    mem = build_membership(current, changes, FLOOR)
    assert set(members_asof(mem, "2005-01-01")) == {"AAA", "ZZZ"}
    assert set(members_asof(mem, "2015-01-01")) == {"AAA"}
    assert set(members_asof(mem, "2021-01-01")) == {"AAA", "ZZZ"}
