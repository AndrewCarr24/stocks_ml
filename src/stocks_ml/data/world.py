"""The live world store: weekly refresh of every input behind panel_sf.parquet.

The champion (r5, PROCEDURE.md) was selected on data/sharadar_world2000 — a
frozen snapshot. That snapshot stays frozen; the live job (live/r5.py) works
on its own copy (`bootstrap_live_store`) and brings it up to date each week:

  Sharadar Direct   sp500 -> membership (+ sicsector from `tickers`)
                    SEP/SFP -> prices  (incremental: `lastupdated.gte` + upsert
                    by (ticker, date); Sharadar bumps lastupdated on EVERY row
                    of a ticker when a dividend re-adjusts its history, so the
                    upsert is exact — verified 2026-09-01 on AAPL)
                    SF1 ARQ/ART -> fundamentals (full refetch: cheap, exact)
                    SF2 -> insiders + form4_bridge (window-replace by filing date)
  SEC/FINRA/FRED    edgar (companyfacts, every current member), sec8k,
                    shortint, fred — the free-data ingesters unchanged.

Then `build_world_panel` reruns the research recipe: build_panel with the
world's backtest_start, then the Sharadar fundamental/insider features and
rank_normalize — reproduced bit-for-bit against the research panel_sf on
2026-09-01. Two rules that fidelity check taught:
  * IEF (the ballast bond fund) must not be in the panel's price frame:
    f_mkt_dispersion is a cross-section over every price series, and the
    research panel was built before IEF was appended for ballast pricing.
  * Sharadar's `sp500` events start before its first `historical` snapshot;
    the research stint builder opened a second, never-closed stint for those
    tickers (LEHMQ, BIGGQ, MTL1, SUB1 stayed "members" with all-neutral
    features). `membership_from_sp500` closes them; no other row changes.
"""
from __future__ import annotations

import shutil
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from stocks_ml.data.sharadar import fetch_table
from stocks_ml.data.store import DataStore

UNIVERSE_START = "1998-01-01"        # SEP history pulled from here (research choice)
FUNDAMENTALS_START = "1990-01-01"    # SF1 filings; the bulk slice starts 1993-12
FUND_TICKERS = ("SPY", "IEF")        # SFP funds kept in prices (benchmark, ballast)
PANEL_EXCLUDED = ("IEF",)            # never in the panel's price frame (see module doc)
BACKTEST_START = pd.Timestamp("2001-01-05")
SEP_COLS = ["ticker", "date", "open", "high", "low", "close", "volume",
            "closeadj", "closeunadj", "lastupdated"]
RESEARCH_TABLES = ("prices", "membership", "fundamentals", "insiders", "edgar",
                   "sec8k", "shortint", "fred")
REQUEST_PAUSE_S = 0.2
# Table schemas, as the research world was built (Sharadar bulk CSVs filtered
# to the universe). fundamentals: AR* dimensions only — ARQ (as-reported
# quarterly) and ART (trailing twelve months); MR* rows are restated backward
# in time -> lookahead, never ingested. `date` is the SEC filing date, the
# point-in-time key downstream. insiders: open-market Form 3/4/5 rows keyed
# by filing date. form4: the SEC Form 4 schema features/insiders.py consumes.
FUND_COLS = [
    "ticker", "dimension", "calendardate", "date", "reportperiod",
    "revenue", "netinc", "gp", "assets", "equity", "debt", "ebitda", "ebit",
    "fcf", "ncfo", "capex", "currentratio", "de", "sharesbas", "shareswa",
    "bvps", "eps", "epsusd", "marketcap", "liabilities", "cashneq",
    "divyield", "dps", "grossmargin", "ebitdamargin", "netmargin", "roe",
]
INSIDER_COLS = ["ticker", "date", "transactiondate", "transactioncode",
                "transactionshares", "transactionvalue", "ownername"]
FORM4_COLS = ["ticker", "filed", "trans_date", "code", "shares", "value"]


def _log(msg):
    print(msg, flush=True)


TICKER_PARAM_MAX = 200               # Direct API: the `ticker` filter is capped at 200 chars
TICKER_BATCH_MAX = 30                # ... and at 30 tickers per request


def _chunks(items, n=TICKER_BATCH_MAX, max_chars=TICKER_PARAM_MAX):
    """Batches of at most min(n, 30) tickers whose comma-joined form fits the
    API's `ticker` parameter."""
    n = min(n, TICKER_BATCH_MAX)
    out, cur = [], []
    for t in items:
        if cur and (len(cur) >= n or len(",".join(cur + [t])) > max_chars):
            out.append(cur)
            cur = []
        cur.append(t)
    if cur:
        out.append(cur)
    return out


def _concat(frames, columns) -> pd.DataFrame:
    """concat that skips empty frames (pandas warns on them) and keeps `columns`."""
    parts = [f for f in frames if len(f)]
    if not parts:
        return pd.DataFrame(columns=list(columns))
    return pd.concat(parts, ignore_index=True)[list(columns)]


# ---- pure transforms (unit-tested) ----
def sp500_universe(sp500: pd.DataFrame) -> list[str]:
    """Every ticker that ever appears in the sp500 table, including the
    counterpart of each add/remove — the research universe definition."""
    vals = set(sp500["ticker"].dropna())
    if "contraticker" in sp500.columns:
        vals |= set(sp500["contraticker"].dropna())
    return sorted(v for v in vals if isinstance(v, str) and v and v != "N/A")


def membership_from_sp500(sp500: pd.DataFrame, sectors: dict) -> pd.DataFrame:
    """Membership stints (ticker, start_date, end_date, sector) from the first
    `historical` snapshot plus the add/remove event stream."""
    sp = sp500.copy()
    sp["date"] = pd.to_datetime(sp["date"])
    hist = sp[sp["action"] == "historical"]
    first_snap_date = hist["date"].min()
    first_snap = set(hist[hist["date"] == first_snap_date]["ticker"])
    events = sp[sp["action"].isin(["added", "removed"])].sort_values("date", kind="stable")
    intervals = {t: [[first_snap_date, pd.NaT]] for t in first_snap}
    for r in events.itertuples():
        spans = intervals.setdefault(r.ticker, [])
        open_span = spans[-1] if spans and pd.isna(spans[-1][1]) else None
        if r.action == "added":
            if open_span is None:
                spans.append([r.date, pd.NaT])
            else:  # already a member (snapshot stint): the add dates its start
                open_span[0] = min(open_span[0], r.date)
        elif open_span is not None:
            open_span[1] = r.date
    rows = [(t, s, e, sectors.get(t)) for t, spans in intervals.items() for s, e in spans]
    mem = pd.DataFrame(rows, columns=["ticker", "start_date", "end_date", "sector"])
    return mem.sort_values(["ticker", "start_date"]).reset_index(drop=True)


def prices_from_sep(raw: pd.DataFrame) -> pd.DataFrame:
    """SEP/SFP rows -> the project's price schema on the total-return basis:
    close = closeadj, open scaled by the same factor, volume as is."""
    factor = raw["closeadj"] / raw["close"]
    out = pd.DataFrame({"date": pd.to_datetime(raw["date"]), "ticker": raw["ticker"],
                        "open": raw["open"] * factor, "close": raw["closeadj"],
                        "volume": raw["volume"]})
    # Sharadar occasionally serves a row twice (249 exact duplicates in the
    # research SEP pull); the research prices table has none
    return (out.dropna(subset=["close"]).drop_duplicates(["ticker", "date"], keep="last")
            .sort_values(["ticker", "date"]).reset_index(drop=True))


def upsert(old: pd.DataFrame | None, new: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """Rows of `new` replace rows of `old` with the same key."""
    if old is None or old.empty:
        return new.reset_index(drop=True)
    if new.empty:
        return old.reset_index(drop=True)
    idx_old = pd.MultiIndex.from_frame(old[keys])
    idx_new = pd.MultiIndex.from_frame(new[keys])
    keep = old[~idx_old.isin(idx_new)]
    return (pd.concat([keep, new[old.columns]], ignore_index=True)
            .sort_values(keys).reset_index(drop=True))


def fundamentals_from_sf1(raw: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    """SF1 rows -> the world's fundamentals slice (FUND_COLS semantics)."""
    if raw.empty:
        return pd.DataFrame(columns=FUND_COLS)
    df = raw[raw["ticker"].isin(universe) & raw["dimension"].isin(["ARQ", "ART"])]
    df = df[[c for c in FUND_COLS if c in df.columns]].copy()
    for c in ("calendardate", "date", "reportperiod"):
        df[c] = pd.to_datetime(df[c])
    num = [c for c in df.columns
           if c not in ("ticker", "dimension", "calendardate", "date", "reportperiod")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    return (df.drop_duplicates(["ticker", "dimension", "calendardate", "date"])
            .sort_values(["ticker", "dimension", "reportperiod", "date"])
            .reset_index(drop=True))


def _sf2_clean(raw: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    df = raw[raw["ticker"].isin(universe) & raw["transactioncode"].isin(["P", "S"])].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["transactiondate"] = pd.to_datetime(df["transactiondate"], errors="coerce")
    for c in ("transactionshares", "transactionvalue", "transactionpricepershare"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["date"])


def insiders_from_sf2(raw: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    """SF2 rows -> the world's insiders slice (INSIDER_COLS semantics)."""
    cols = INSIDER_COLS + ["signed_value"]
    if raw.empty:
        return pd.DataFrame(columns=cols)
    df = _sf2_clean(raw, universe)
    df["signed_value"] = (np.sign(df["transactionshares"].fillna(0))
                          * df["transactionvalue"].abs())
    return df[cols].sort_values(["ticker", "date"]).reset_index(drop=True)


def form4_from_sf2(raw: pd.DataFrame, universe: set[str]) -> pd.DataFrame:
    """SF2 non-derivative open-market rows in the SEC Form 4 schema the panel
    consumes (features/insiders.py). Verified equal to the SEC quarterly data
    for 2026Q1 on ten tickers (SF2 slightly richer on two)."""
    if raw.empty:
        return pd.DataFrame(columns=FORM4_COLS)
    df = _sf2_clean(raw, universe)
    df = df[df["securityadcode"].astype(str).str.startswith("N")]
    shares = df["transactionshares"].abs()
    out = pd.DataFrame({"ticker": df["ticker"], "filed": df["date"],
                        "trans_date": df["transactiondate"],
                        "code": df["transactioncode"], "shares": shares,
                        "value": (shares * df["transactionpricepershare"]).abs()})
    return (out[FORM4_COLS].sort_values(["ticker", "filed", "trans_date"])
            .reset_index(drop=True))


def membership_diff(old: pd.DataFrame, new: pd.DataFrame) -> pd.DataFrame:
    """Stints present in exactly one of the two membership tables."""
    cols = ["ticker", "start_date", "end_date"]
    a = old[cols].assign(side="old")
    b = new[cols].assign(side="new")
    both = pd.concat([a, b], ignore_index=True)
    return both.drop_duplicates(cols, keep=False).sort_values(["ticker", "start_date"])


RENAMEABLE = ("prices", "sharadar_prices", "membership", "fundamentals", "insiders",
              "edgar", "sec8k", "shortint", "form4_sec", "form4", "form4_bridge")


def detect_renames(old_tk: pd.DataFrame, new_tk: pd.DataFrame, known: set[str]) -> dict[str, str]:
    """Symbols Sharadar rewrote since the stored tickers table: the same
    permaticker now carries a different ticker, and every table's history
    moved with it (EQR -> VMRK, 2026-09-01). Only unambiguous cases: one old
    symbol we hold, one new symbol nobody holds, the old one gone."""
    def mapping(tk, keep):
        t = tk[tk["ticker"].isin(keep)] if keep is not None else tk
        t = t.dropna(subset=["permaticker"])[["permaticker", "ticker"]].drop_duplicates()
        counts = t.groupby("permaticker")["ticker"].nunique()
        return dict(t[t["permaticker"].isin(counts[counts == 1].index)].values)
    old, new = mapping(old_tk, known), mapping(new_tk, None)
    new_symbols = set(new_tk["ticker"])
    return {o: new[p] for p, o in old.items()
            if p in new and new[p] != o and new[p] not in known and o not in new_symbols}


def apply_renames(store: DataStore, renames: dict[str, str], log=_log) -> dict[str, int]:
    """Rewrite old symbols in every stored table that has a ticker column."""
    touched = {}
    for name in RENAMEABLE:
        if not store.exists(name):
            continue
        df = store.read(name)
        if "ticker" not in df.columns:
            continue
        m = df["ticker"].isin(renames)
        if m.any():
            df.loc[m, "ticker"] = df.loc[m, "ticker"].map(renames)
            store.write(name, df)
            touched[name] = int(m.sum())
    log(f"renamed {renames} in {touched}")
    return touched


def related_symbols(tk: pd.DataFrame) -> dict[str, list[str]]:
    """ticker -> Sharadar's `relatedtickers` (previous symbols, other classes)."""
    out: dict[str, list[str]] = {}
    if "relatedtickers" not in tk.columns:
        return out
    for r in tk.dropna(subset=["relatedtickers"]).itertuples():
        out.setdefault(r.ticker, []).extend(str(r.relatedtickers).replace(",", " ").split())
    return out


# ---- store-level steps ----
def bootstrap_live_store(live_dir, research_dir="data/sharadar_world2000",
                         data_dir="data", log=_log) -> DataStore:
    """One-time copy of the frozen research world into the live directory."""
    live, research, data = Path(live_dir), Path(research_dir), Path(data_dir)
    store = DataStore(live)
    if store.exists("prices"):
        return store
    log(f"bootstrap: copying the research world {research} -> {live}")
    for name in RESEARCH_TABLES:
        shutil.copy2(research / f"{name}.parquet", live / f"{name}.parquet")
    shutil.copy2(research / "form4.parquet", live / "form4_sec.parquet")
    shutil.copy2(research / "form4.parquet", live / "form4.parquet")
    for name in ("sharadar_prices", "sharadar_sp500", "sharadar_tickers"):
        shutil.copy2(data / f"{name}.parquet", live / f"{name}.parquet")
    research_manifest = DataStore(research).manifest
    for k in ("corrupt_tickers", "feature_coverage", "edgar", "sec8k", "shortint", "fred"):
        if k in research_manifest:
            store.set_manifest(k, research_manifest[k])
    raw = store.read("sharadar_prices")
    form4_sec = store.read("form4_sec")
    store.set_manifest("sharadar", {
        "since": str(raw["lastupdated"].max().date()),
        "form4_sec_through": str(form4_sec["filed"].max().date()),
        "bootstrapped_from": str(research)})
    return store


def _fetch(table, key, fetch_fn, **filters):
    df = fetch_table(table, key, fetch_fn=fetch_fn, **filters)
    if fetch_fn is None:
        time.sleep(REQUEST_PAUSE_S)
    return df


def refresh_sharadar(store: DataStore, key: str, fetch_fn=None, log=_log,
                     today=None) -> dict:
    """Bring membership, prices, fundamentals, insiders and the Form 4 bridge
    up to date from Sharadar Direct. Returns a freshness report."""
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    meta = dict(store.manifest.get("sharadar", {}))
    since = meta["since"]
    report = {}

    # universe + symbol renames (applied to every stored table first)
    sp500 = _fetch("sp500", key, fetch_fn, **{"date.gte": UNIVERSE_START})
    universe = sp500_universe(sp500)
    tk = pd.concat([_fetch("tickers", key, fetch_fn, ticker=",".join(b))
                    for b in _chunks(universe, 100)], ignore_index=True)
    raw = store.read("sharadar_prices")
    renames = {}
    if store.exists("sharadar_tickers"):
        renames = detect_renames(store.read("sharadar_tickers"), tk, set(raw["ticker"]))
    if renames:
        apply_renames(store, renames, log)
        raw = store.read("sharadar_prices")
    store.write("sharadar_sp500", sp500)
    store.write("sharadar_tickers", tk)
    report["renames"] = renames

    # membership: sp500 stints + sicsector for the universe
    sectors = dict(tk.dropna(subset=["sicsector"]).drop_duplicates("ticker")
                   [["ticker", "sicsector"]].values)
    old_mem = store.read("membership")
    old_sectors = dict(old_mem.dropna(subset=["sector"]).drop_duplicates("ticker")
                       [["ticker", "sector"]].values)
    # keep the research sector for tickers that already had one (stability);
    # `tickers` only supplies sectors for newcomers
    mem = membership_from_sp500(sp500, {**sectors, **old_sectors})
    diff = membership_diff(old_mem, mem)
    store.write("membership", mem)
    current = sorted(mem[mem["end_date"].isna()]["ticker"])
    report["membership"] = {"stints": int(len(mem)), "current": len(current),
                            "events_through": str(sp500["date"].max().date()),
                            "changed_stints": diff.astype(str).to_dict("records")}
    log(f"membership: {len(mem)} stints, {len(current)} current; "
        f"{len(diff)} stint changes vs stored")

    # prices: SEP for the universe, SFP for the funds, incremental by lastupdated
    known = set(raw["ticker"])
    frames = []
    for b in _chunks([t for t in universe if t in known], 40):
        frames.append(_fetch("stocks", key, fetch_fn, ticker=",".join(b),
                             **{"from": UNIVERSE_START, "lastupdated.gte": since}))
    new_tickers = [t for t in universe if t not in known]
    for b in _chunks(new_tickers, 40):
        frames.append(_fetch("stocks", key, fetch_fn, ticker=",".join(b),
                             **{"from": UNIVERSE_START}))
    fund_filters = {"from": UNIVERSE_START}
    if all(f in known for f in FUND_TICKERS):
        fund_filters["lastupdated.gte"] = since
    frames.append(_fetch("funds", key, fetch_fn, ticker=",".join(FUND_TICKERS),
                         **fund_filters))
    new = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame(columns=SEP_COLS)
    if not new.empty:
        new = new[SEP_COLS]
        for c in ("open", "high", "low", "close", "volume", "closeadj", "closeunadj"):
            new[c] = pd.to_numeric(new[c], errors="coerce")
        new = new.drop_duplicates(["ticker", "date"], keep="last")
    raw = upsert(raw[SEP_COLS], new, ["ticker", "date"]).drop_duplicates(["ticker", "date"], keep="last")
    keep = set(universe) | set(FUND_TICKERS)
    orphans = sorted(set(raw["ticker"]) - keep)
    if orphans:
        log(f"prices: dropping {len(orphans)} tickers no longer in the universe: {orphans}")
        raw = raw[raw["ticker"].isin(keep)].reset_index(drop=True)
    store.write("sharadar_prices", raw)
    prices = prices_from_sep(raw)
    store.write("prices", prices)
    if not new.empty:
        since = max(since, str(new["lastupdated"].max().date()))
    report["prices"] = {"rows_updated": int(len(new)),
                        "tickers_updated": int(new["ticker"].nunique()) if len(new) else 0,
                        "new_tickers": [t for t in new_tickers if t in set(new["ticker"])],
                        # pre-snapshot casualties Sharadar lists but has no
                        # prices for (CBB1); refetched harmlessly each week
                        "no_history": [t for t in new_tickers if t not in set(new["ticker"])],
                        "through": str(prices["date"].max().date()),
                        "spy_through": str(prices[prices["ticker"] == "SPY"]["date"].max().date()),
                        "n_tickers": int(prices["ticker"].nunique())}
    log(f"prices: {len(new)} rows updated ({report['prices']['tickers_updated']} tickers), "
        f"through {report['prices']['through']}")

    # fundamentals: full refetch of ARQ + ART for the universe
    frames = []
    for b in _chunks(universe, 100):
        for dim in ("ARQ", "ART"):
            frames.append(_fetch("fundamentals", key, fetch_fn, ticker=",".join(b),
                                 dimension=dim, **{"from": FUNDAMENTALS_START}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)    # all-NA columns in some batches
        raw_sf1 = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
            if any(not f.empty for f in frames) else pd.DataFrame(columns=FUND_COLS)
    fund = fundamentals_from_sf1(raw_sf1, set(universe))
    old_fund = store.read("fundamentals")
    if len(fund) < 0.98 * len(old_fund):
        raise RuntimeError(f"fundamentals refetch returned {len(fund)} rows vs "
                           f"{len(old_fund)} stored — refusing to shrink the table")
    store.write("fundamentals", fund)
    report["fundamentals"] = {"rows": int(len(fund)), "rows_before": int(len(old_fund)),
                              "filed_through": str(fund["date"].max().date())}
    log(f"fundamentals: {len(fund)} rows (was {len(old_fund)}), filed through "
        f"{report['fundamentals']['filed_through']}")

    # insiders + Form 4 bridge: window-replace by filing date
    old_ins = store.read("insiders")
    sec_through = pd.Timestamp(meta["form4_sec_through"])
    bridge_old = store.read("form4_bridge") if store.exists("form4_bridge") else \
        pd.DataFrame(columns=FORM4_COLS)
    lo = min(old_ins["date"].max(),
             bridge_old["filed"].max() if len(bridge_old) else sec_through + pd.Timedelta(days=1))
    lo = (lo - pd.Timedelta(days=14)).normalize()
    raw_sf2 = _fetch("insiders", key, fetch_fn,
                     **{"from": lo.date().isoformat(), "to": today.date().isoformat()})
    ins_new = insiders_from_sf2(raw_sf2, set(universe))
    ins = _concat([old_ins[old_ins["date"] < lo], ins_new], old_ins.columns)
    ins = ins.sort_values(["ticker", "date"]).reset_index(drop=True)
    store.write("insiders", ins)
    bridge_new = form4_from_sf2(raw_sf2, set(universe))
    bridge = _concat([bridge_old[bridge_old["filed"] < lo], bridge_new], FORM4_COLS)
    bridge = bridge[bridge["filed"] > sec_through]
    bridge = bridge.sort_values(["ticker", "filed", "trans_date"]).reset_index(drop=True)
    store.write("form4_bridge", bridge)
    form4 = _concat([store.read("form4_sec"), bridge], FORM4_COLS)
    store.write("form4", form4.sort_values(["ticker", "filed", "trans_date"])
                .reset_index(drop=True))
    report["insiders"] = {"rows": int(len(ins)), "filed_through": str(ins["date"].max().date()),
                          "window_from": str(lo.date()), "window_rows": int(len(ins_new))}
    report["form4"] = {"sec_through": str(sec_through.date()), "bridge_rows": int(len(bridge)),
                       "filed_through": str(form4["filed"].max().date())}
    log(f"insiders: {len(ins_new)} rows in window from {lo.date()}, filed through "
        f"{report['insiders']['filed_through']}; form4 bridge {len(bridge)} rows")

    meta.update({"since": since, "refreshed_at": str(pd.Timestamp.now()),
                 "universe": len(universe)})
    store.set_manifest("sharadar", meta)
    return report


def sharadar_cik_map(tickers, user_agent, related=None, cik_map=None) -> dict:
    """SEC CIK lookup keyed by Sharadar tickers (BRK.B), via the normalized
    form the SEC map uses (BRK-B); falls back to a ticker's previous symbols
    (`related`) while the SEC map lags a rename."""
    from stocks_ml.data.edgar import load_cik_map
    from stocks_ml.data.membership import normalize_symbol
    ciks = cik_map if cik_map is not None else load_cik_map(user_agent)
    out = {}
    for t in tickers:
        for cand in [t] + list((related or {}).get(t, [])):
            if normalize_symbol(cand) in ciks:
                out[t] = ciks[normalize_symbol(cand)]
                break
    return out


def refresh_sec(store: DataStore, cfg, current: list[str], log=_log) -> dict:
    """EDGAR companyfacts (every current member, replaced), 8-K metadata,
    FINRA short interest and FRED — the free-data ingesters."""
    from stocks_ml.data.edgar import ingest_edgar
    from stocks_ml.data.fred import ingest_fred
    from stocks_ml.data.sec8k import ingest_sec8k
    from stocks_ml.data.shortint import ingest_shortint
    report = {}
    related = related_symbols(store.read("sharadar_tickers")) if store.exists("sharadar_tickers") else {}
    ciks = sharadar_cik_map(current, cfg.user_agent, related=related)
    missing = sorted(set(current) - set(ciks))
    if missing:
        log(f"no SEC CIK for {missing}")
    s = ingest_edgar(store, current, cfg.edgar_concepts, cfg.user_agent,
                     cik_map=ciks, refresh_days=0)
    edgar = store.read("edgar")
    report["edgar"] = {**s, "filed_through": str(edgar["filed"].max().date())}
    log(f"edgar: {s['n_ok']} tickers, {len(s['failed_tickers'])} failed, filed through "
        f"{report['edgar']['filed_through']}")
    s = ingest_sec8k(store, current, cfg.user_agent, cik_map=ciks)
    sec8k = store.read("sec8k")
    report["sec8k"] = {**s, "filed_through": str(sec8k["filed"].max().date())}
    log(f"sec8k: {s['n_filings']} filings, {len(s['failed_tickers'])} failed, filed through "
        f"{report['sec8k']['filed_through']}")
    report["shortint"] = ingest_shortint(store, cfg.user_agent)
    log(f"shortint: {report['shortint']}")
    report["fred"] = ingest_fred(store, cfg.fred_series, cfg.user_agent)
    log(f"fred: {report['fred']}")
    return report


def refresh_world(live_dir, cfg, data_dir="data", fetch_fn=None, log=_log,
                  sec: bool = True) -> dict:
    from stocks_ml.data.sharadar import api_key
    store = bootstrap_live_store(live_dir, data_dir=data_dir, log=log)
    report = {"sharadar": refresh_sharadar(store, api_key(data_dir), fetch_fn=fetch_fn, log=log)}
    if sec:
        mem = store.read("membership")
        current = sorted(mem[mem["end_date"].isna()]["ticker"])
        report["sec"] = refresh_sec(store, cfg, current, log=log)
    return report


class _PanelStore(DataStore):
    """The world store as build_panel sees it: no ballast fund in prices."""

    def read(self, name):
        df = super().read(name)
        if name == "prices":
            df = df[~df["ticker"].isin(PANEL_EXCLUDED)].reset_index(drop=True)
        return df


def build_world_panel(live_dir, cfg, log=_log) -> pd.DataFrame:
    """panel.parquet + panel_sf.parquet for the world — the research recipe."""
    import copy

    from stocks_ml.features.panel import build_panel
    from stocks_ml.features.ranking import rank_normalize
    from stocks_ml.features.sharadar_fundamentals import (
        SF_RAW_COLS, SFI_RAW_COLS, _asof, sharadar_fundamental_features,
        sharadar_insider_features)

    cfg2 = copy.copy(cfg)
    for k, v in (("data_dir", str(live_dir)), ("backtest_start", BACKTEST_START)):
        object.__setattr__(cfg2, k, v)
    store = _PanelStore(live_dir)
    t0 = time.time()
    panel = build_panel(store, cfg2)
    store.write("panel", panel)
    log(f"panel: {len(panel):,} rows through {panel['date'].max().date()} "
        f"({time.time() - t0:.0f}s)")

    prices = store.read("prices")
    fund = store.read("fundamentals")
    ins = store.read("insiders")
    cw = prices.pivot(index="date", columns="ticker", values="close").sort_index().ffill()
    wk = cw.reindex(pd.Index(sorted(panel["date"].unique())), method="ffill")
    close = pd.Series(wk.stack().reindex(
        pd.MultiIndex.from_frame(panel[["date", "ticker"]])).values, index=panel.index)
    ff = sharadar_fundamental_features(fund, panel, close)
    shares = _asof(panel, fund[fund["dimension"] == "ARQ"], ["sharesbas"])["sharesbas"]
    fi = sharadar_insider_features(ins, panel, mktcap=close * shares)
    panel_sf = rank_normalize(pd.concat([panel, ff, fi], axis=1), SF_RAW_COLS + SFI_RAW_COLS)
    panel_sf.to_parquet(Path(live_dir) / "panel_sf.parquet", index=False)
    log(f"panel_sf: {panel_sf.shape[0]:,} x {panel_sf.shape[1]} ({time.time() - t0:.0f}s)")
    return panel_sf
