"""data/world.py: the live world refresh, exercised against a fake Sharadar
Direct endpoint (house rule: no network in tests)."""
import json

import numpy as np
import pandas as pd
import pytest

from stocks_ml.data import world
from stocks_ml.data.store import DataStore

D = pd.Timestamp
NAT = pd.NaT


# ---- fixtures: a five-name world ----
def sp500_table():
    rows = [
        ("1998-03-31", "historical", "AAA", "N/A"),
        ("1998-03-31", "historical", "BBB", "N/A"),
        ("1998-03-31", "historical", "LEHMQ", "N/A"),
        ("1998-01-09", "removed", "BBI1", "LEHMQ"),
        ("1998-01-09", "added", "LEHMQ", "BBI1"),       # pre-snapshot add (phantom case)
        ("2008-09-15", "removed", "LEHMQ", "N/A"),
        ("2010-05-03", "removed", "BBB", "CCC"),
        ("2010-05-03", "added", "CCC", "BBB"),
        ("2026-08-20", "added", "NEW", "N/A"),
    ]
    return pd.DataFrame(rows, columns=["date", "action", "ticker", "contraticker"])


def research_membership():
    """What the research stint builder produced: LEHMQ's snapshot stint never
    closed (the phantom), AAA carrying its research sector."""
    rows = [("AAA", D("1998-03-31"), NAT, "Old Sector"),
            ("BBB", D("1998-03-31"), D("2010-05-03"), "Finance"),
            ("CCC", D("2010-05-03"), NAT, "Services"),
            ("LEHMQ", D("1998-01-09"), D("2008-09-15"), "Finance"),
            ("LEHMQ", D("1998-03-31"), NAT, "Finance")]
    return pd.DataFrame(rows, columns=["ticker", "start_date", "end_date", "sector"])


def sep_rows(ticker, dates, base, lastupdated, adj=1.0):
    out = []
    for i, d in enumerate(dates):
        close = base + i
        out.append({"ticker": ticker, "date": d, "open": close - 0.5, "high": close + 1,
                    "low": close - 1, "close": close, "volume": 1000 + i,
                    "closeadj": close * adj, "closeunadj": close,
                    "lastupdated": lastupdated})
    return out


OLD_DATES = ["2026-08-17", "2026-08-18", "2026-08-19", "2026-08-20", "2026-08-21",
             "2026-08-24", "2026-08-25", "2026-08-26", "2026-08-27", "2026-08-28"]
NEW_DATES = ["2026-08-31", "2026-09-01", "2026-09-02", "2026-09-03", "2026-09-04"]


def stored_sep():
    rows = []
    for tk, base in (("AAA", 10), ("BBB", 20), ("CCC", 30), ("LEHMQ", 1), ("ZZZ", 99),
                     ("SPY", 500), ("IEF", 90)):
        rows += sep_rows(tk, OLD_DATES, base, "2026-08-28")
    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df["lastupdated"] = pd.to_datetime(df["lastupdated"])
    return df


def api_tables():
    """The endpoint's view after a week: new bars for everyone, AAA's whole
    history re-adjusted by a dividend (closeadj rescaled, lastupdated bumped),
    NEW's full history, SF1 with an MRQ row that must never be ingested, SF2
    with derivative and non-open-market rows."""
    stocks = []
    stocks += sep_rows("AAA", OLD_DATES + NEW_DATES, 10, "2026-09-04", adj=0.5)
    for tk, base in (("BBB", 20), ("CCC", 30), ("LEHMQ", 1)):
        stocks += sep_rows(tk, OLD_DATES, base, "2026-08-28")
        stocks += sep_rows(tk, NEW_DATES, base + len(OLD_DATES), "2026-09-04")
    stocks += sep_rows("NEW", OLD_DATES + NEW_DATES, 40, "2026-09-04")
    funds = []
    for tk, base in (("SPY", 500), ("IEF", 90)):
        funds += sep_rows(tk, OLD_DATES, base, "2026-08-28")
        funds += sep_rows(tk, NEW_DATES, base + len(OLD_DATES), "2026-09-04")
    tickers = [{"ticker": t, "table": "stocks", "permaticker": str(i), "sicsector": s,
                "relatedtickers": None} for i, (t, s) in enumerate(
               (("AAA", "Manufacturing"), ("BBB", "Finance"), ("CCC", "Services"),
                ("LEHMQ", "Finance"), ("NEW", "Technology")), 1)]
    tickers.append({"ticker": "AAA", "table": "fundamentals", "permaticker": "1",
                    "sicsector": "Manufacturing", "relatedtickers": None})
    fund_cols = {c: 1.0 for c in world.FUND_COLS
                 if c not in ("ticker", "dimension", "calendardate", "date", "reportperiod")}
    fundamentals = []
    for tk in ("AAA", "BBB", "CCC", "LEHMQ", "NEW"):
        for dim in ("ARQ", "ART", "MRQ"):
            for cd, filed in (("2026-03-31", "2026-05-01"), ("2026-06-30", "2026-08-01")):
                fundamentals.append({"ticker": tk, "dimension": dim, "calendardate": cd,
                                     "date": filed, "reportperiod": cd, **fund_cols})
    fundamentals.append({"ticker": "AAA", "dimension": "ARQ", "calendardate": "2026-06-30",
                         "date": "2026-09-02", "reportperiod": "2026-06-30", **fund_cols})
    insiders = [
        # (date, ticker, code, adcode, shares, price, value, transactiondate)
        ("2026-03-20", "AAA", "P", "N", 100, 10.0, 1000.0, "2026-03-18"),
        ("2026-08-20", "AAA", "S", "N", -50, 12.0, 600.0, "2026-08-19"),   # replaces stored
        ("2026-09-02", "BBB", "P", "NA", 10, 20.0, 200.0, None),           # NaN trans date
        ("2026-09-02", "BBB", "M", "N", 500, 1.0, 500.0, "2026-09-01"),    # option exercise: dropped
        ("2026-09-03", "CCC", "S", "D", -40, 30.0, 1200.0, "2026-09-02"),  # derivative: insiders only
        ("2026-09-03", "ZZZ", "P", "N", 1, 1.0, 1.0, "2026-09-02"),        # not in universe
    ]
    insiders = [dict(date=d, ticker=t, transactioncode=c, securityadcode=a,
                     transactionshares=s, transactionpricepershare=p, transactionvalue=v,
                     transactiondate=td, ownername="Someone")
                for d, t, c, a, s, p, v, td in insiders]
    return {"sp500": sp500_table().to_dict("records"), "tickers": tickers,
            "stocks": stocks, "funds": funds, "fundamentals": fundamentals,
            "insiders": insiders}


def make_fake_fetch(tables, calls):
    def fake(url, params, headers):
        table = url.rstrip("/").rsplit("/", 1)[-1]
        calls.append((table, dict(params)))
        rows = tables[table]
        if "ticker" in params:
            want = set(params["ticker"].split(","))
            rows = [r for r in rows if r["ticker"] in want]
        if "dimension" in params:
            rows = [r for r in rows if r["dimension"] == params["dimension"]]
        for k in ("from", "date.gte"):
            if k in params:
                rows = [r for r in rows if r["date"] >= params[k]]
        if "to" in params:
            rows = [r for r in rows if r["date"] <= params["to"]]
        if "lastupdated.gte" in params:
            rows = [r for r in rows if r["lastupdated"] >= params["lastupdated.gte"]]
        off, lim = params["offset"], params["limit"]
        return {"count": len(rows), "data": rows[off:off + lim]}
    return fake


def stored_world(tmp_path):
    """A live store as bootstrap_live_store would leave it."""
    store = DataStore(tmp_path)
    raw = stored_sep()
    store.write("sharadar_prices", raw)
    store.write("sharadar_tickers", pd.DataFrame(
        [{"ticker": t, "table": "stocks", "permaticker": str(i), "sicsector": "x",
          "relatedtickers": None}
         for i, t in enumerate(("AAA", "BBB", "CCC", "LEHMQ"), 1)]))
    store.write("prices", world.prices_from_sep(raw))
    store.write("membership", research_membership())
    fund_cols = {c: 1.0 for c in world.FUND_COLS
                 if c not in ("ticker", "dimension", "calendardate", "date", "reportperiod")}
    fund = pd.DataFrame([{"ticker": tk, "dimension": dim, "calendardate": D("2026-03-31"),
                          "date": D("2026-05-01"), "reportperiod": D("2026-03-31"), **fund_cols}
                         for tk in ("AAA", "BBB", "CCC", "LEHMQ") for dim in ("ARQ", "ART")])
    store.write("fundamentals", fund)
    ins = pd.DataFrame([
        {"ticker": "AAA", "date": D("2026-01-15"), "ownername": "Someone",
         "transactiondate": D("2026-01-14"), "transactioncode": "P",
         "transactionshares": 5.0, "transactionvalue": 50.0, "signed_value": 50.0},
        {"ticker": "AAA", "date": D("2026-08-20"), "ownername": "Someone",
         "transactiondate": D("2026-08-19"), "transactioncode": "S",
         "transactionshares": -999.0, "transactionvalue": 9.0, "signed_value": -9.0},
    ])
    store.write("insiders", ins)
    form4_sec = pd.DataFrame([{"ticker": "AAA", "filed": D("2026-03-31"),
                               "trans_date": D("2026-03-30"), "code": "P",
                               "shares": 7.0, "value": 70.0}])
    store.write("form4_sec", form4_sec)
    store.write("form4", form4_sec)
    store.set_manifest("sharadar", {"since": "2026-08-28", "form4_sec_through": "2026-03-31"})
    return store


# ---- pure transforms ----
def test_sp500_universe_includes_contratickers_and_drops_na():
    assert world.sp500_universe(sp500_table()) == ["AAA", "BBB", "BBI1", "CCC", "LEHMQ", "NEW"]


def test_membership_closes_pre_snapshot_stints():
    mem = world.membership_from_sp500(sp500_table(), {"AAA": "Manufacturing", "NEW": "Technology"})
    leh = mem[mem["ticker"] == "LEHMQ"]
    assert len(leh) == 1                                   # no phantom second stint
    assert leh.iloc[0]["start_date"] == D("1998-01-09")    # the add dates the start
    assert leh.iloc[0]["end_date"] == D("2008-09-15")
    bbb = mem[mem["ticker"] == "BBB"].iloc[0]
    assert (bbb["start_date"], bbb["end_date"]) == (D("1998-03-31"), D("2010-05-03"))
    assert mem[mem["ticker"] == "CCC"]["end_date"].isna().all()
    new = mem[mem["ticker"] == "NEW"].iloc[0]
    assert new["start_date"] == D("2026-08-20") and new["sector"] == "Technology"
    assert sorted(mem[mem["end_date"].isna()]["ticker"]) == ["AAA", "CCC", "NEW"]
    assert "BBI1" not in set(mem["ticker"])                # removed without a stint: none


def test_membership_readd_makes_a_second_stint():
    sp = pd.concat([sp500_table(), pd.DataFrame(
        [("2015-01-02", "removed", "CCC", "N/A"), ("2020-06-01", "added", "CCC", "N/A")],
        columns=["date", "action", "ticker", "contraticker"])])
    mem = world.membership_from_sp500(sp, {})
    ccc = mem[mem["ticker"] == "CCC"].reset_index(drop=True)
    assert list(ccc["start_date"]) == [D("2010-05-03"), D("2020-06-01")]
    assert ccc.iloc[0]["end_date"] == D("2015-01-02") and pd.isna(ccc.iloc[1]["end_date"])


def test_prices_from_sep_scales_open_by_the_adjustment_factor():
    raw = pd.DataFrame(sep_rows("AAA", ["2026-08-28"], 10, "2026-08-28", adj=0.5)
                       + sep_rows("AAA", ["2026-08-31"], 10, "2026-08-28"))
    raw.loc[1, "closeadj"] = np.nan
    px = world.prices_from_sep(raw)
    assert len(px) == 1                                    # NaN adjusted close dropped
    assert px.iloc[0]["close"] == 5.0 and px.iloc[0]["open"] == pytest.approx(9.5 * 0.5)
    assert list(px.columns) == ["date", "ticker", "open", "close", "volume"]


def test_prices_from_sep_drops_rows_sharadar_served_twice():
    raw = pd.DataFrame(sep_rows("AAA", ["2026-08-28"], 10, "2026-08-28") * 2)
    assert len(world.prices_from_sep(raw)) == 1


def test_refresh_removes_duplicate_sep_rows_from_the_store(tmp_path):
    store = stored_world(tmp_path)
    raw = store.read("sharadar_prices")
    store.write("sharadar_prices", pd.concat([raw, raw[raw["ticker"] == "BBB"].head(3)]))
    world.refresh_sharadar(store, "k", fetch_fn=make_fake_fetch(api_tables(), []),
                           log=lambda m: None, today="2026-09-05")
    for name in ("sharadar_prices", "prices"):
        assert not store.read(name).duplicated(["ticker", "date"]).any(), name


def test_upsert_replaces_on_key_and_appends_the_rest():
    old = pd.DataFrame({"ticker": ["A", "A", "B"], "date": [1, 2, 1], "v": [1, 2, 3]})
    new = pd.DataFrame({"date": [2, 3], "v": [20, 30], "ticker": ["A", "A"], "extra": [0, 0]})
    out = world.upsert(old, new, ["ticker", "date"])
    assert list(out.columns) == ["ticker", "date", "v"]
    assert out.set_index(["ticker", "date"])["v"].to_dict() == {("A", 1): 1, ("A", 2): 20,
                                                                ("A", 3): 30, ("B", 1): 3}
    assert world.upsert(None, new, ["ticker", "date"]).equals(new)
    assert world.upsert(old, new.iloc[:0], ["ticker", "date"]).equals(old)


def test_fundamentals_from_sf1_keeps_ar_dimensions_only():
    raw = pd.DataFrame(api_tables()["fundamentals"])
    fund = world.fundamentals_from_sf1(raw, {"AAA", "BBB"})
    assert set(fund["dimension"]) == {"ARQ", "ART"}
    assert set(fund["ticker"]) == {"AAA", "BBB"}
    assert list(fund.columns) == world.FUND_COLS
    assert fund["date"].dtype.kind == "M" and fund["revenue"].dtype.kind == "f"
    assert len(fund) == 2 * 2 * 2 + 1
    assert world.fundamentals_from_sf1(raw.iloc[:0], {"AAA"}).empty


def test_sf2_transforms():
    raw = pd.DataFrame(api_tables()["insiders"])
    uni = {"AAA", "BBB", "CCC"}
    ins = world.insiders_from_sf2(raw, uni)
    assert list(ins.columns) == world.INSIDER_COLS + ["signed_value"]
    assert list(ins["ticker"]) == ["AAA", "AAA", "BBB", "CCC"]        # M code and ZZZ dropped
    assert ins["signed_value"].tolist() == [1000.0, -600.0, 200.0, -1200.0]
    assert pd.isna(ins[ins["ticker"] == "BBB"]["transactiondate"]).all()
    f4 = world.form4_from_sf2(raw, uni)
    assert list(f4.columns) == world.FORM4_COLS
    assert list(f4["ticker"]) == ["AAA", "AAA", "BBB"]                 # CCC derivative dropped
    assert f4["shares"].tolist() == [100.0, 50.0, 10.0]                # abs
    assert f4["value"].tolist() == [1000.0, 600.0, 200.0]              # |shares x price|
    assert f4["code"].tolist() == ["P", "S", "P"]


def test_membership_diff_symmetric():
    old, new = research_membership(), world.membership_from_sp500(sp500_table(), {})
    diff = world.membership_diff(old, new)
    assert [(r.ticker, r.side) for r in diff.itertuples()] == [("LEHMQ", "old"), ("NEW", "new")]


# ---- store steps ----
def test_bootstrap_copies_research_world_and_seeds_manifest(tmp_path):
    research, data, live = tmp_path / "research", tmp_path / "data", tmp_path / "live"
    research.mkdir(), data.mkdir()
    for name in world.RESEARCH_TABLES:
        pd.DataFrame({"x": [1]}).to_parquet(research / f"{name}.parquet")
    pd.DataFrame({"ticker": ["AAA"], "filed": [D("2026-03-31")]}).to_parquet(research / "form4.parquet")
    raw = stored_sep()
    raw.to_parquet(data / "sharadar_prices.parquet")
    for name in ("sharadar_sp500", "sharadar_tickers"):
        pd.DataFrame({"x": [1]}).to_parquet(data / f"{name}.parquet")
    (research / "manifest.json").write_text(json.dumps({"edgar": {"n_ok": 3}, "other": 1}))
    store = world.bootstrap_live_store(live, research_dir=research, data_dir=data, log=lambda m: None)
    for name in world.RESEARCH_TABLES + ("form4", "form4_sec", "sharadar_prices"):
        assert store.exists(name)
    assert store.manifest["edgar"] == {"n_ok": 3} and "other" not in store.manifest
    assert store.manifest["sharadar"]["since"] == "2026-08-28"
    assert store.manifest["sharadar"]["form4_sec_through"] == "2026-03-31"
    # second call is a no-op
    (live / "prices.parquet").write_bytes(b"")
    world.bootstrap_live_store(live, research_dir=research, data_dir=data, log=lambda m: None)
    assert (live / "prices.parquet").read_bytes() == b""


def test_refresh_sharadar_end_to_end(tmp_path):
    store = stored_world(tmp_path)
    calls = []
    rep = world.refresh_sharadar(store, "k", fetch_fn=make_fake_fetch(api_tables(), calls),
                                 log=lambda m: None, today="2026-09-05")

    # membership: phantom closed, newcomer added, research sector kept
    mem = store.read("membership")
    assert len(mem[mem["ticker"] == "LEHMQ"]) == 1
    assert mem.set_index("ticker")["sector"]["AAA"] == "Old Sector"
    assert mem.set_index("ticker")["sector"]["NEW"] == "Technology"
    assert rep["membership"]["current"] == 3
    assert [c["ticker"] for c in rep["membership"]["changed_stints"]] == ["LEHMQ", "NEW"]

    # prices: incremental for known names, full history for NEW, orphan dropped
    stock_calls = [p for t, p in calls if t == "stocks"]
    known = [p for p in stock_calls if "NEW" not in p["ticker"] and "BBI1" not in p["ticker"]]
    assert known and all(p["lastupdated.gte"] == "2026-08-28" for p in known)
    assert [p for p in stock_calls if "NEW" in p["ticker"]][0].get("lastupdated.gte") is None
    px = store.read("prices")
    assert set(px["ticker"]) == {"AAA", "BBB", "CCC", "LEHMQ", "NEW", "SPY", "IEF"}
    assert px["date"].max() == D("2026-09-04")
    aaa = px[px["ticker"] == "AAA"].set_index("date")
    assert aaa.loc[D("2026-08-17"), "close"] == 5.0            # re-adjusted history replaced
    assert aaa.loc[D("2026-08-17"), "open"] == pytest.approx(9.5 * 0.5)
    assert len(aaa) == len(OLD_DATES) + len(NEW_DATES)
    assert len(px[px["ticker"] == "NEW"]) == len(OLD_DATES) + len(NEW_DATES)
    bbb = px[px["ticker"] == "BBB"].set_index("date")
    assert bbb.loc[D("2026-08-17"), "close"] == 20.0           # untouched history intact
    assert rep["prices"]["new_tickers"] == ["NEW"]
    assert rep["prices"]["no_history"] == ["BBI1"]              # nothing served, harmless
    assert rep["prices"]["through"] == "2026-09-04" and rep["prices"]["spy_through"] == "2026-09-04"
    assert store.manifest["sharadar"]["since"] == "2026-09-04"

    # fundamentals: full refetch, AR* only, universe only
    fund = store.read("fundamentals")
    assert set(fund["dimension"]) == {"ARQ", "ART"} and "MRQ" not in set(fund["dimension"])
    assert set(fund["ticker"]) == {"AAA", "BBB", "CCC", "LEHMQ", "NEW"}
    assert rep["fundamentals"]["filed_through"] == "2026-09-02"

    # insiders: window from min(stored max, sec_through+1) - 14d, replaced inside it
    assert rep["insiders"]["window_from"] == "2026-03-18"
    ins = store.read("insiders")
    aaa_ins = ins[ins["ticker"] == "AAA"].sort_values("date")
    assert aaa_ins["transactionshares"].tolist() == [5.0, 100.0, -50.0]   # -999 row replaced
    assert list(ins.columns) == world.INSIDER_COLS + ["signed_value"] or \
        set(ins.columns) == set(world.INSIDER_COLS + ["signed_value"])

    # form4: frozen SEC rows + bridge rows filed after sec_through
    bridge = store.read("form4_bridge")
    assert bridge["filed"].min() > D("2026-03-31")
    assert bridge[["ticker", "shares"]].values.tolist() == [["AAA", 50.0], ["BBB", 10.0]]
    form4 = store.read("form4")
    assert len(form4) == 1 + len(bridge)
    assert rep["form4"]["sec_through"] == "2026-03-31"
    assert store.manifest["sharadar"]["form4_sec_through"] == "2026-03-31"

    # a second run is idempotent on the tables
    before = {n: store.read(n) for n in ("prices", "membership", "fundamentals", "insiders", "form4")}
    world.refresh_sharadar(store, "k", fetch_fn=make_fake_fetch(api_tables(), []),
                           log=lambda m: None, today="2026-09-05")
    for n, df in before.items():
        pd.testing.assert_frame_equal(store.read(n), df)


def test_refresh_refuses_a_shrunken_fundamentals_table(tmp_path):
    store = stored_world(tmp_path)
    tables = api_tables()
    tables["fundamentals"] = tables["fundamentals"][:2]
    with pytest.raises(RuntimeError, match="refusing to shrink"):
        world.refresh_sharadar(store, "k", fetch_fn=make_fake_fetch(tables, []),
                               log=lambda m: None, today="2026-09-05")


def test_chunks_respect_the_ticker_parameter_length():
    tickers = [f"T{i:04d}" for i in range(300)]              # 5 chars each
    batches = world._chunks(tickers, 100)
    assert [t for b in batches for t in b] == tickers
    assert all(len(",".join(b)) <= world.TICKER_PARAM_MAX for b in batches)
    assert max(len(b) for b in batches) == 30
    assert max(len(b) for b in world._chunks(["AB"] * 100, 100)) == 30
    assert world._chunks(tickers[:5], 2) == [tickers[:2], tickers[2:4], tickers[4:5]]


def renamed_tables(old="CCC", new="CCX"):
    """The endpoint after Sharadar rewrites a symbol: every table carries the
    new one, `tickers` keeps the permaticker and lists the old symbol."""
    tables = api_tables()
    for name, rows in tables.items():
        for r in rows:
            for col in ("ticker", "contraticker"):
                if r.get(col) == old:
                    r[col] = new
            if name == "tickers" and r["ticker"] == new:
                r["relatedtickers"] = old
    return tables


def test_detect_renames_is_conservative():
    old_tk = pd.DataFrame({"ticker": ["AAA", "CCC", "DDD"], "permaticker": ["1", "3", "4"]})
    new_tk = pd.DataFrame({"ticker": ["AAA", "CCX", "DDD"], "permaticker": ["1", "3", "4"]})
    assert world.detect_renames(old_tk, new_tk, {"AAA", "CCC", "DDD"}) == {"CCC": "CCX"}
    # the new symbol is already held by someone else: refuse
    assert world.detect_renames(old_tk, new_tk, {"AAA", "CCC", "DDD", "CCX"}) == {}
    # the old symbol still exists: refuse
    both = pd.concat([new_tk, pd.DataFrame({"ticker": ["CCC"], "permaticker": ["9"]})])
    assert world.detect_renames(old_tk, both, {"AAA", "CCC", "DDD"}) == {}
    # ambiguous permaticker (two symbols): refuse
    amb = pd.concat([new_tk, pd.DataFrame({"ticker": ["CCY"], "permaticker": ["3"]})])
    assert world.detect_renames(old_tk, amb, {"AAA", "CCC", "DDD"}) == {}
    assert world.related_symbols(pd.DataFrame({"ticker": ["A", "B"],
                                               "relatedtickers": ["X Y", None]})) == {"A": ["X", "Y"]}


def test_refresh_applies_a_symbol_rename_everywhere(tmp_path):
    store = stored_world(tmp_path)
    store.write("edgar", pd.DataFrame({"ticker": ["CCC", "AAA"], "concept": ["x", "x"],
                                       "filed": [D("2026-01-01")] * 2, "val": [1.0, 2.0]}))
    rep = world.refresh_sharadar(store, "k", fetch_fn=make_fake_fetch(renamed_tables(), []),
                                 log=lambda m: None, today="2026-09-05")
    assert rep["renames"] == {"CCC": "CCX"}
    for name in ("prices", "sharadar_prices", "membership", "fundamentals", "insiders",
                 "edgar", "form4"):
        df = store.read(name)
        assert "CCC" not in set(df["ticker"]), name
    px = store.read("prices")
    ccx = px[px["ticker"] == "CCX"]
    assert len(ccx) == len(OLD_DATES) + len(NEW_DATES)            # history kept, not refetched as new
    assert rep["prices"]["new_tickers"] == ["NEW"]
    mem = store.read("membership").set_index("ticker")
    assert mem.loc["CCX", "sector"] == "Services" and pd.isna(mem.loc["CCX", "end_date"])
    assert [c["ticker"] for c in rep["membership"]["changed_stints"]] == ["LEHMQ", "NEW"]
    assert set(store.read("edgar")["ticker"]) == {"AAA", "CCX"}
    assert "CCX" in set(store.read("sharadar_tickers")["ticker"])


def test_cik_map_falls_back_to_previous_symbols():
    ciks = {"AAA": 1, "EQR": 2, "BRK-B": 3}
    out = world.sharadar_cik_map(["AAA", "VMRK", "BRK.B", "ZZZ"], "ua",
                                 related={"VMRK": ["EQR"]}, cik_map=ciks)
    assert out == {"AAA": 1, "VMRK": 2, "BRK.B": 3}
