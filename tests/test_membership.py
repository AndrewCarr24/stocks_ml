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


def test_current_table_date_added_is_authoritative_for_active_stint():
    current = pd.DataFrame({"ticker": ["AAA", "NEW"], "sector": ["Tech", "Energy"],
                            "date_added": ["1957-03-04", "2019-06-07"]})
    changes = _changes([("2019-06-11", "NEW", "OLD")])
    mem = build_membership(current, changes, FLOOR)
    assert mem[(mem.ticker == "AAA") & mem.end_date.isna()].iloc[0].start_date == pd.Timestamp("1957-03-04")
    assert mem[(mem.ticker == "NEW") & mem.end_date.isna()].iloc[0].start_date == pd.Timestamp("2019-06-07")


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


def test_clean_wiki_tables_real_structure():
    from stocks_ml.data.membership import _clean_wiki_tables

    current_raw = pd.DataFrame({"Symbol": ["AAPL", "BRK.B"], "GICS Sector": ["Tech", "Financials"],
                                "Date added": ["1982-11-30", "2010-02-16"]})
    cols = pd.MultiIndex.from_tuples([
        ("Effective Date", "Effective Date"), ("Added", "Ticker"), ("Added", "Security"),
        ("Removed", "Ticker"), ("Removed", "Security"), ("Reason", "Reason"),
    ])
    changes_raw = pd.DataFrame(
        [["June 30, 2025", "NEWCO", "New Co", "OLDCO", "Old Co", "Index change"]], columns=cols)
    current, changes = _clean_wiki_tables(current_raw, changes_raw)
    assert list(current.ticker) == ["AAPL", "BRK-B"]
    assert list(current.date_added) == [pd.Timestamp("1982-11-30"), pd.Timestamp("2010-02-16")]
    assert changes.iloc[0]["added"] == "NEWCO"
    assert changes.iloc[0]["removed"] == "OLDCO"
    assert changes.iloc[0]["date"] == pd.Timestamp("2025-06-30")
    assert changes.iloc[0]["reason"] == "Index change"


def test_clean_wiki_tables_reason_kept_nan_where_absent():
    from stocks_ml.data.membership import _clean_wiki_tables

    current_raw = pd.DataFrame({"Symbol": ["AAPL"], "GICS Sector": ["Tech"]})
    cols = pd.MultiIndex.from_tuples([
        ("Effective Date", "Effective Date"), ("Added", "Ticker"), ("Added", "Security"),
        ("Removed", "Ticker"), ("Removed", "Security"), ("Reason", "Reason"),
    ])
    changes_raw = pd.DataFrame(
        [["June 30, 2025", "NEWCO", "New Co", None, None, None]], columns=cols)
    _, changes = _clean_wiki_tables(current_raw, changes_raw)
    assert pd.isna(changes.iloc[0]["reason"])


def test_ingest_membership_writes_removals_dataset(tmp_path):
    from types import SimpleNamespace

    from stocks_ml.data.membership import ingest_membership
    from stocks_ml.data.store import DataStore

    current = pd.DataFrame({"ticker": ["AAA"], "sector": ["Tech"]})
    changes = _changes([
        ("2018-03-05", "AAA", "old"),      # same-date add+remove event -> a removal row for OLD
        ("2019-01-10", "NEWCO2", None),    # add-only event -> must NOT appear in removals
    ])
    changes["reason"] = ["Acquired by BigCo", None]
    cfg = SimpleNamespace(membership_floor=FLOOR)

    store = DataStore(tmp_path)
    ingest_membership(store, cfg, fetch_fn=lambda: (current, changes))
    removals = store.read("removals")

    assert list(removals.columns) == ["ticker", "date", "reason"]
    assert removals.iloc[0]["ticker"] == "OLD"          # normalized
    assert removals.iloc[0]["date"] == pd.Timestamp("2018-03-05")
    assert removals.iloc[0]["reason"] == "Acquired by BigCo"
    assert len(removals) == 1                           # add-only event excluded


def test_members_asof():
    current = pd.DataFrame({"ticker": ["AAA", "ZZZ"], "sector": ["Tech", "Retail"]})
    changes = _changes([("2020-05-01", "ZZZ", None), ("2010-02-01", None, "ZZZ")])
    mem = build_membership(current, changes, FLOOR)
    assert set(members_asof(mem, "2005-01-01")) == {"AAA", "ZZZ"}
    assert set(members_asof(mem, "2015-01-01")) == {"AAA"}
    assert set(members_asof(mem, "2021-01-01")) == {"AAA", "ZZZ"}


def test_locate_table_by_columns_not_position():
    import pytest
    from stocks_ml.data.membership import _locate_table

    navbox = pd.DataFrame({"vteS&P 500 companies": [1], "vteS&P 500 companies.1": [2]})
    changes = pd.DataFrame(columns=pd.MultiIndex.from_tuples([
        ("Effective Date", "Effective Date"), ("Added", "Ticker"),
        ("Removed", "Ticker"), ("Reason", "Reason")]))
    # order scrambled + navbox first: located by content, not position
    found = _locate_table([navbox, changes], ["date", "added", "removed", "reason"], "x")
    assert found is changes
    with pytest.raises(ValueError, match="structure changed"):
        _locate_table([navbox], ["date", "added"], "x")


def test_normalize_symbol_strips_wiki_formatting_junk():
    from stocks_ml.data.membership import normalize_symbol

    assert normalize_symbol("ITT |") == "ITT"
    assert normalize_symbol(" BRK.B ") == "BRK-B"
    assert normalize_symbol("JCP |") == "JCP"


def _fixture_tables():
    current = pd.DataFrame({"ticker": ["AAA", "BBB"], "sector": ["T", "F"],
                            "date_added": pd.to_datetime(["2010-01-01", "2011-01-01"])})
    changes = pd.DataFrame({"date": pd.to_datetime(["2011-01-01"]),
                            "added": ["BBB"], "removed": [None], "reason": ["x"]})
    return current, changes


def test_membership_falls_back_to_stored_on_fetch_failure(synthetic_store, tiny_cfg):
    import pytest
    from stocks_ml.data.membership import ingest_membership

    def boom():
        raise ValueError("wikipedia moved the table again")

    # no stored membership -> hard failure
    for f in [synthetic_store.root / "membership.parquet"]:
        if f.exists():
            f.unlink()
    synthetic_store.manifest.pop("membership_fallback_weeks", None)
    good = lambda: (_fixture_tables()[0], _fixture_tables()[1])
    mem = ingest_membership(synthetic_store, tiny_cfg, fetch_fn=good)   # seed store
    assert synthetic_store.manifest.get("membership_fallback_weeks") == 0

    # failures 1..3 -> stored data with counter; 4th -> raises
    for wk in (1, 2, 3):
        out = ingest_membership(synthetic_store, tiny_cfg, fetch_fn=boom)
        assert synthetic_store.manifest["membership_fallback_weeks"] == wk
        assert set(out["ticker"]) == set(mem["ticker"])
    with pytest.raises(RuntimeError, match="staleness"):
        ingest_membership(synthetic_store, tiny_cfg, fetch_fn=boom)

    # a successful fetch resets the counter
    synthetic_store.set_manifest("membership_fallback_weeks", 2)
    ingest_membership(synthetic_store, tiny_cfg, fetch_fn=good)
    assert synthetic_store.manifest["membership_fallback_weeks"] == 0


def test_membership_rejects_implausible_swing(synthetic_store, tiny_cfg):
    from stocks_ml.data.membership import ingest_membership

    cur, chg = _fixture_tables()
    ingest_membership(synthetic_store, tiny_cfg,
                      fetch_fn=lambda: (cur, chg))
    # vandalized page: 20 brand-new members appear at once
    vandal = pd.DataFrame({"ticker": [f"Z{i:02d}" for i in range(20)],
                           "sector": ["T"] * 20,
                           "date_added": pd.to_datetime(["2020-01-01"] * 20)})
    out = ingest_membership(synthetic_store, tiny_cfg,
                            fetch_fn=lambda: (vandal, chg))
    assert synthetic_store.manifest["membership_fallback_weeks"] == 1
    assert "implausible" in synthetic_store.manifest["membership_fallback_reason"]
    assert set(out["ticker"]) == {"AAA", "BBB"}          # stored data served
