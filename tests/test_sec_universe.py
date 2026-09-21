"""The EDGAR survivorship fix (2026-09-21): SEC data for every name ever in
the membership, keyed by Sharadar's CIK, and the corrected filing/8-K
features appended to a frozen panel as x_ columns."""
from types import SimpleNamespace

import numpy as np
import pandas as pd

from stocks_ml.data import world
from stocks_ml.data.store import DataStore

D = pd.Timestamp


def tickers_table():
    return pd.DataFrame([
        {"ticker": "AAA", "table": "stocks", "isdelisted": "N", "relatedtickers": None,
         "secfilings": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000000011"},
        {"ticker": "OLD", "table": "stocks", "isdelisted": "Y", "relatedtickers": None,
         "secfilings": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000000022"},
        {"ticker": "B", "table": "stocks", "isdelisted": "Y", "relatedtickers": None,     # a reused symbol
         "secfilings": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000000033"},
        {"ticker": "B", "table": "stocks", "isdelisted": "N", "relatedtickers": None,
         "secfilings": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000000044"},
        {"ticker": "NOSEC", "table": "stocks", "isdelisted": "Y", "relatedtickers": None, "secfilings": None},
    ])


def test_cik_from_tickers_reads_delisted_names_and_prefers_the_listed_holder():
    out = world.cik_from_tickers(tickers_table())
    assert out == {"AAA": 11, "OLD": 22, "B": 44}
    renamed = pd.concat([tickers_table(), pd.DataFrame([{"ticker": "VMRK", "table": "stocks", "isdelisted": "N",
                                                         "relatedtickers": "EQR, B",
                                                         "secfilings": "https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK=0000906107"}])])
    out = world.cik_from_tickers(renamed)
    assert out["VMRK"] == 906107 and out["EQR"] == 906107      # the old symbol takes the renamed row's CIK
    assert out["B"] == 44                                       # unless it has a row of its own
    assert world.cik_from_tickers(None) == {} and world.cik_from_tickers(pd.DataFrame({"ticker": ["A"]})) == {}


def test_sharadar_cik_map_prefers_sharadars_cik_then_the_sec_map():
    sec = {"AAA": 1, "NOSEC": 5, "PREV": 6}
    out = world.sharadar_cik_map(["AAA", "OLD", "NOSEC", "RENAMED", "ZZZ"], "ua", related={"RENAMED": ["PREV"]},
                                 cik_map=sec, tickers_table=tickers_table())
    assert out == {"AAA": 11, "OLD": 22, "NOSEC": 5, "RENAMED": 6}   # Sharadar first, SEC map, previous symbols


def _world(tmp_path):
    store = DataStore(tmp_path)
    days = pd.bdate_range("2024-01-01", "2024-06-28")
    px = pd.concat([pd.DataFrame({"ticker": t, "date": days, "close": 10.0 + np.arange(len(days)) * 0.01 * (i + 1),
                                  "closeadj": 10.0, "volume": 1000.0})
                    for i, t in enumerate(("AAA", "OLD"))], ignore_index=True)
    store.write("prices", px)
    store.write("membership", pd.DataFrame([("AAA", D("2020-01-01"), pd.NaT, "s"), ("OLD", D("2020-01-01"), D("2024-03-01"), "s")],
                                           columns=["ticker", "start_date", "end_date", "sector"]))
    store.write("sharadar_tickers", tickers_table())
    # the survivor-only tables: AAA (current) has facts and filings, OLD (departed) nothing
    store.write("edgar", pd.DataFrame([{"ticker": "AAA", "concept": "revenue", "start": D("2023-10-01"), "end": D("2023-12-31"),
                                        "filed": D("2024-02-15"), "val": 1.0, "form": "10-K"}]))
    store.write("sec8k", pd.DataFrame([{"ticker": "AAA", "accession": "a1", "accepted": "2024-02-14T16:05:00.000Z",
                                        "filed": D("2024-02-14"), "items": "2.02", "primary_document": "x", "is_amendment": False}]))
    return store, days


def facts(cik, filed="2024-02-15"):
    return {"facts": {"us-gaap": {"Revenues": {"units": {"USD": [
        {"start": "2023-10-01", "end": "2023-12-31", "filed": filed, "val": float(cik), "form": "10-K"}]}}}}}


def submissions(cik, filed="2024-02-14"):
    return {"filings": {"recent": {"accessionNumber": [f"acc-{cik}"], "acceptanceDateTime": [f"{filed}T16:05:00.000Z"],
                                   "filingDate": [filed], "form": ["8-K"], "items": ["2.02"], "primaryDocument": ["d.htm"]},
                        "files": []}}


def test_refetch_covers_every_past_member_and_keeps_the_survivor_tables(tmp_path):
    store, _ = _world(tmp_path)
    cfg = SimpleNamespace(user_agent="ua", edgar_concepts={"revenue": ["Revenues"]})
    fetched = []
    rep = world.refetch_sec_universe(tmp_path, cfg, log=lambda m: None,
                                     fetch_facts_fn=lambda cik, ua: (fetched.append(cik), facts(cik))[1],
                                     fetch_submissions_fn=lambda cik: submissions(cik),
                                     fetch_file_fn=lambda name: {}, data_dir=tmp_path / "no_project_pull")
    assert sorted(fetched) == [11, 22]                                   # the departed name too, by Sharadar's CIK
    assert rep["names"] == 2 and rep["with_cik"] == 2 and rep["no_cik"] == []
    edgar = store.read("edgar")
    assert set(edgar["ticker"]) == {"AAA", "OLD"}
    assert float(edgar.loc[edgar["ticker"] == "OLD", "val"].iloc[0]) == 22.0
    assert set(store.read("sec8k")["ticker"]) == {"AAA", "OLD"}
    backups = sorted(p.name for p in tmp_path.glob("*.survivors_*.parquet"))
    assert [b.split(".")[0] for b in backups] == ["edgar", "sec8k"]
    assert set(pd.read_parquet(tmp_path / backups[0])["ticker"]) == {"AAA"}
    assert store.manifest["sec_universe"]["edgar_names"] == 2


def test_append_sec_columns_matches_the_panels_own_features_and_fills_the_departed_name(tmp_path):
    from stocks_ml.features.events import filing_features, sec8k_features
    from stocks_ml.features.ranking import rank_normalize
    store, days = _world(tmp_path)
    dates = pd.DatetimeIndex([d for d in days if d.weekday() == 4][-8:])
    rows = pd.MultiIndex.from_product([dates, ["AAA", "OLD"]], names=["date", "ticker"]).to_frame(index=False)
    # the panel's f_ columns as build_panel makes them from the survivor-only tables
    prices = store.read("prices")
    f = filing_features(store.read("edgar"), prices, dates).merge(
        sec8k_features(store.read("sec8k"), ["AAA", "OLD"], dates), on=["date", "ticker"])
    panel = rank_normalize(rows.merge(f, on=["date", "ticker"], how="left"), [c for c in f.columns if c.startswith("f_")])
    panel["f_mom_1w"] = 0.0
    panel.to_parquet(tmp_path / "panel_sf.parquet", index=False)

    assert world.append_sec_columns(tmp_path, log=lambda m: None) == world.SEC_COLS
    out = pd.read_parquet(tmp_path / "panel_sf.parquet")
    for c in world.SEC_COLS:            # unchanged tables -> the x_ column IS the f_ column
        np.testing.assert_allclose(out[c].to_numpy(), out["f_" + c[2:]].to_numpy())
    assert world.append_sec_columns(tmp_path, log=lambda m: None) == []          # append-only, once

    # after the every-name refetch the departed name gets its own filing history
    (tmp_path / "panel_sf.parquet").unlink(); panel.to_parquet(tmp_path / "panel_sf.parquet", index=False)
    cfg = SimpleNamespace(user_agent="ua", edgar_concepts={"revenue": ["Revenues"]})
    world.refetch_sec_universe(tmp_path, cfg, log=lambda m: None,
                               fetch_facts_fn=lambda cik, ua: facts(cik, filed="2024-05-10"),
                               fetch_submissions_fn=lambda cik: submissions(cik, filed="2024-05-09"),
                               fetch_file_fn=lambda name: {}, data_dir=tmp_path / "no_project_pull")
    world.append_sec_columns(tmp_path, log=lambda m: None)
    out = pd.read_parquet(tmp_path / "panel_sf.parquet")
    old = out[out["ticker"] == "OLD"]
    assert (old["f_days_since_filing"] == 0.0).all()          # survivor-only: neutral-filled, no history
    assert old["x_days_since_filing"].abs().sum() > 0           # every-name: a real filing distance
    assert list(out.columns[:len(panel.columns)]) == list(panel.columns)     # existing columns untouched


def test_refetch_form4_from_sf2_covers_every_past_member_and_keeps_the_old_table(tmp_path):
    from stocks_ml.data.world import INSIDER_COLS
    store, _ = _world(tmp_path)
    store.write("form4", pd.DataFrame([{"ticker": "AAA", "filed": D("2024-02-01"), "trans_date": D("2024-01-31"),
                                        "code": "P", "shares": 10.0, "value": 100.0}]))
    calls = []
    def fake(url, params, headers):
        calls.append(params.get("ticker"))
        rows = [dict(date="2024-03-01", ticker=t, transactioncode="P", securityadcode="N", transactionshares=5, transactionpricepershare=2.0,
                     transactionvalue=10.0, transactiondate="2024-02-28", ownername="Someone") for t in params["ticker"].split(",")]
        return {"count": len(rows), "data": rows[params["offset"]:params["offset"] + params["limit"]]}
    rep = world.refetch_form4_from_sf2(tmp_path, "k", fetch_fn=fake, log=lambda m: None)
    assert rep["before"] == 1 and rep["after"] == 2 and set(calls[0].split(",")) == {"AAA", "OLD"}
    assert set(store.read("form4")["ticker"]) == {"AAA", "OLD"}
    assert set(pd.read_parquet(next(tmp_path.glob("form4.survivors_*.parquet")))["ticker"]) == {"AAA"}
