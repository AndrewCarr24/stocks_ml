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

import os
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
    # 2026-09-16, the feature search: raw dollar totals only — never the
    # vendor's per-share or ratio fields (bvps/eps/dps are the exception,
    # kept as they were; the panel cancels their split restatement against
    # close_split). Ratios are computed here, point-in-time, from these.
    "receivables", "inventory", "payables", "deferredrev", "depamor", "intangibles",
    "ppnenet", "investments", "retearn", "cor", "opex", "opinc", "sgna", "rnd", "sbcomp",
    "taxexp", "intexp", "ebt", "ncf", "ncfi", "ncff", "ncfdiv", "ncfcommon", "ncfdebt",
    "ncfbus", "ncfinv", "netinccmn", "workingcapital", "tangibles", "invcap", "roa", "roic",
    "assetturnover", "payoutratio", "assetsc", "liabilitiesc", "debtc", "debtnc",
    "fiscalperiod", "lastupdated",
]
HOLDINGS_COLS = ["ticker", "date", "shrholders", "shrunits", "shrvalue", "totalvalue", "percentoftotal",
                 "cllholders", "putholders", "cllunits", "putunits"]
PRICES_HL_COLS = ["ticker", "date", "high", "low"]
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


# ---- the top-N universe (2026-09-18: the owner's direction after the S&P 500 line) ----
SNAPSHOT_START = "1998-01-01"        # quarter-end market-cap snapshots from here
SNAPSHOT_FALLBACK_DAYS = 6           # a quarter end on a weekend/holiday: the last session before it
EQUITY_CATEGORIES = ("Domestic Common Stock", "Domestic Common Stock Primary Class")


def quarter_ends(start, end) -> list[pd.Timestamp]:
    """Calendar quarter ends in [start, end]."""
    lo, hi = pd.Timestamp(start), pd.Timestamp(end)
    q = pd.date_range(lo - pd.offsets.QuarterEnd(1), hi, freq="QE")
    return [d for d in q if lo <= d <= hi]


def universe_equities(tk: pd.DataFrame) -> set[str]:
    """The tickers a top-N universe may hold: domestic common stock (one class
    per company: primary, never a secondary class). The venue is NOT a
    criterion: the tickers table carries today's exchange, and excluding
    today's OTC names removed Fannie Mae and Freddie Mac from 2006-2008 —
    a whole-history exclusion decided by where a name trades in 2026
    (2026-09-21). The market-cap snapshots decide who is in at each date."""
    cat = tk["category"].fillna("")
    return set(tk.loc[cat.isin(EQUITY_CATEGORIES), "ticker"].dropna())


def membership_from_top(snapshots: pd.DataFrame, eligible: set, sectors: dict, n: int) -> pd.DataFrame:
    """Membership stints (ticker, start_date, end_date, sector) of the top-n
    eligible names by market cap at each snapshot date: a name enters at the
    first snapshot that ranks it, stays until the first snapshot that does
    not (the stint ends that day), open-ended if ranked at the last one.
    Point in time: nothing after a snapshot date informs its membership."""
    s = snapshots.dropna(subset=["marketcap"]).copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s[s["ticker"].isin(eligible)]
    dates = sorted(s["date"].unique())
    top = {}
    for d, g in s.groupby("date"):
        top[d] = set(g.sort_values("marketcap", ascending=False).drop_duplicates("ticker")["ticker"].head(n))
    rows, open_since = [], {}
    for d in dates:
        for t in list(open_since):
            if t not in top[d]:
                rows.append((t, open_since.pop(t), d))
        for t in top[d]:
            open_since.setdefault(t, d)
    rows += [(t, d0, pd.NaT) for t, d0 in open_since.items()]
    mem = pd.DataFrame(rows, columns=["ticker", "start_date", "end_date"])
    mem["sector"] = mem["ticker"].map(sectors)
    return mem.sort_values(["ticker", "start_date"]).reset_index(drop=True)


def membership_topped_up(base: pd.DataFrame, snapshots: pd.DataFrame, eligible: set, sectors: dict, n: int) -> pd.DataFrame:
    """A membership (e.g. the S&P 500 stints, point in time) topped up at
    each snapshot with the largest eligible non-members by market cap until
    the universe holds n names: the index plus the next names by size —
    "sp800" for n=800. The base stints are kept as they are; the top-up
    names get quarterly stints (membership_from_top's rule)."""
    from stocks_ml.data.membership import members_asof
    s = snapshots.dropna(subset=["marketcap"]).copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s[s["ticker"].isin(eligible)]
    rows, open_since = [], {}
    for d in sorted(s["date"].unique()):
        inside = set(members_asof(base, d))
        g = s[(s["date"] == d) & ~s["ticker"].isin(inside)].sort_values("marketcap", ascending=False).drop_duplicates("ticker")
        top = set(g["ticker"].head(max(0, n - len(inside))))
        for t in list(open_since):
            if t not in top:
                rows.append((t, open_since.pop(t), d))
        for t in top:
            open_since.setdefault(t, d)
    rows += [(t, d0, pd.NaT) for t, d0 in open_since.items()]
    up = pd.DataFrame(rows, columns=["ticker", "start_date", "end_date"])
    up["sector"] = up["ticker"].map(sectors)
    out = pd.concat([base[["ticker", "start_date", "end_date", "sector"]], up], ignore_index=True)
    return out.sort_values(["ticker", "start_date"]).reset_index(drop=True)


def membership_sized(base: pd.DataFrame, snapshots: pd.DataFrame, eligible: set, sectors: dict, n: int) -> pd.DataFrame:
    """A universe of n names around an index (the tuner's dial, 2026-09-20):
    at each snapshot the index's members as of that date ranked by market
    cap — the largest n of them when the index is bigger than n, topped up
    with the largest eligible non-members when it is smaller. Quarterly
    stints for everyone (membership_from_top's rule), so every n is built
    the same way; n = the index's size is the index at quarterly resolution."""
    from stocks_ml.data.membership import members_asof
    s = snapshots.dropna(subset=["marketcap"]).copy()
    s["date"] = pd.to_datetime(s["date"])
    s = s[s["ticker"].isin(eligible)]
    rows, open_since = [], {}
    for d in sorted(s["date"].unique()):
        g = s[s["date"] == d].sort_values("marketcap", ascending=False).drop_duplicates("ticker")
        inside = set(members_asof(base, d))
        members = list(g[g["ticker"].isin(inside)]["ticker"])[:n]
        if len(members) < n:
            members += list(g[~g["ticker"].isin(inside)]["ticker"].head(n - len(members)))
        top = set(members)
        for t in list(open_since):
            if t not in top:
                rows.append((t, open_since.pop(t), d))
        for t in top:
            open_since.setdefault(t, d)
    rows += [(t, d0, pd.NaT) for t, d0 in open_since.items()]
    mem = pd.DataFrame(rows, columns=["ticker", "start_date", "end_date"])
    mem["sector"] = mem["ticker"].map(sectors)
    return mem.sort_values(["ticker", "start_date"]).reset_index(drop=True)


def fetch_snapshots(key, fetch_fn, dates, log=_log) -> pd.DataFrame:
    """DAILY market caps at each date (the last session on or before it)."""
    frames = []
    for d in dates:
        for back in range(SNAPSHOT_FALLBACK_DAYS + 1):
            day = (pd.Timestamp(d) - pd.Timedelta(days=back)).date().isoformat()
            df = _fetch("daily", key, fetch_fn, date=day, fields="ticker,date,marketcap")
            if len(df):
                df["marketcap"] = pd.to_numeric(df["marketcap"], errors="coerce")
                frames.append(df[["ticker", "date", "marketcap"]].assign(snapshot=pd.Timestamp(d)))
                break
        else:
            log(f"snapshot {pd.Timestamp(d).date()}: no DAILY rows within {SNAPSHOT_FALLBACK_DAYS} days")
    out = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=["ticker", "date", "marketcap", "snapshot"])
    out["date"] = out["snapshot"]          # membership keys on the calendar quarter end
    return out.drop(columns=["snapshot"])


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
    """SEP/SFP rows -> the project's price schema. open/close are the
    total-return basis (close = closeadj, open scaled by the same factor):
    returns and the ledger's units live there. closeunadj (nominal — what the
    tape actually showed that day) and close_split (SEP's split-adjusted
    close) ride along for LEVEL features under price_basis='nominal':
    closeadj levels encode future splits (the 2026-09 leak finding)."""
    factor = raw["closeadj"] / raw["close"]
    out = pd.DataFrame({"date": pd.to_datetime(raw["date"]), "ticker": raw["ticker"],
                        "open": raw["open"] * factor, "close": raw["closeadj"],
                        "volume": raw["volume"],
                        "closeunadj": raw["closeunadj"], "close_split": raw["close"]})
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
    if len(raw_sf2) == 0 and ((old_ins["date"] >= lo).any()
                              or (len(bridge_old) and (bridge_old["filed"] >= lo).any())):
        # an empty SF2 page (outage, bad key) must not erase the stored
        # window: the replace below would delete every row in [lo, today]
        raise RuntimeError(f"SF2 returned no rows for {lo.date()} -> {today.date()} but the "
                           "store has rows in that window; refusing to erase them")
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


def refresh_extras(store: DataStore, key: str, fetch_fn=None, log=_log,
                   full_fundamentals: bool = True) -> dict:
    """The extra Sharadar tables of the 2026-09-16 feature search, for any
    store: `sharadar_tickers` (metadata: Sharadar sector/industry beside the
    SIC division the membership table carries), `holdings` (SF3A —
    institutional holdings by security, quarterly; the vendor's series
    starts 2013, so it cannot drive a model selected on 2006-2015),
    `prices_hl` (SEP high/low, the nominal tape, for range/gap features) and
    the fundamentals table refetched with FUND_COLS' wider field list.
    Every table is filtered to the store's membership universe."""
    mem = store.read("membership")
    universe = sorted(set(mem["ticker"]))
    report = {}
    tk = pd.concat([_fetch("tickers", key, fetch_fn, ticker=",".join(b))
                    for b in _chunks(universe, 100)], ignore_index=True)
    tk = tk.drop_duplicates("ticker", keep="last")
    store.write("sharadar_tickers", tk)
    report["tickers"] = {"rows": int(len(tk)),
                         "sectors": int(tk["sector"].nunique()) if "sector" in tk.columns else 0,
                         "industries": int(tk["industry"].nunique()) if "industry" in tk.columns else 0}
    log(f"tickers: {len(tk)} rows, {report['tickers']['sectors']} sectors, "
        f"{report['tickers']['industries']} industries")
    frames = [_fetch("holdings_ticker", key, fetch_fn, ticker=",".join(b)) for b in _chunks(universe, 100)]
    hold = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    hold = hold[[c for c in HOLDINGS_COLS if c in hold.columns]].copy()
    hold["date"] = pd.to_datetime(hold["date"])
    for c in hold.columns:
        if c not in ("ticker", "date"):
            hold[c] = pd.to_numeric(hold[c], errors="coerce")     # the API serves some counts as strings
    hold = hold.drop_duplicates(["ticker", "date"], keep="last").sort_values(["ticker", "date"]).reset_index(drop=True)
    store.write("holdings", hold)
    report["holdings"] = {"rows": int(len(hold)), "from": str(hold["date"].min().date()),
                          "through": str(hold["date"].max().date()), "tickers": int(hold["ticker"].nunique())}
    log(f"holdings: {len(hold)} rows, {report['holdings']['from']} -> {report['holdings']['through']}")
    frames = []
    for b in _chunks(universe, 40):
        frames.append(_fetch("stocks", key, fetch_fn, ticker=",".join(b), fields="ticker,date,high,low",
                             **{"from": UNIVERSE_START}))
    hl = pd.concat([f for f in frames if not f.empty], ignore_index=True)
    hl = hl[PRICES_HL_COLS].copy(); hl["date"] = pd.to_datetime(hl["date"])
    for c in ("high", "low"):
        hl[c] = pd.to_numeric(hl[c], errors="coerce")
    hl = hl.drop_duplicates(["ticker", "date"], keep="last").sort_values(["ticker", "date"]).reset_index(drop=True)
    store.write("prices_hl", hl)
    report["prices_hl"] = {"rows": int(len(hl)), "through": str(hl["date"].max().date())}
    log(f"prices_hl: {len(hl)} rows through {report['prices_hl']['through']}")
    if full_fundamentals:
        frames = []
        for b in _chunks(universe, 100):
            for dim in ("ARQ", "ART"):
                frames.append(_fetch("fundamentals", key, fetch_fn, ticker=",".join(b),
                                     dimension=dim, **{"from": FUNDAMENTALS_START}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            raw_sf1 = pd.concat([f for f in frames if not f.empty], ignore_index=True)
        fund = fundamentals_from_sf1(raw_sf1, set(universe))
        old_fund = store.read("fundamentals")
        if len(fund) < 0.98 * len(old_fund):
            raise RuntimeError(f"fundamentals refetch returned {len(fund)} rows vs {len(old_fund)} stored")
        store.write("fundamentals", fund)
        report["fundamentals"] = {"rows": int(len(fund)), "columns": int(fund.shape[1]),
                                  "filed_through": str(fund["date"].max().date())}
        log(f"fundamentals: {len(fund)} rows x {fund.shape[1]} columns (was {old_fund.shape[1]})")
    store.set_manifest("extras", {"at": str(pd.Timestamp.today().normalize().date()), **report})
    return report


def cik_from_tickers(tk: pd.DataFrame) -> dict[str, int]:
    """ticker -> SEC CIK from Sharadar's tickers table (`secfilings` is the
    EDGAR browse URL, CIK=NNNNNNNNNN), delisted names included: the SEC's own
    ticker list names only today's registrants, so a map built from it makes
    "has EDGAR data" mean "still listed in 2026" — the survivorship look-ahead
    of 2026-09-21."""
    if tk is None or "secfilings" not in tk.columns:
        return {}
    cik = tk["secfilings"].fillna("").astype(str).str.extract(r"CIK=(\d+)")[0]
    rows = tk.assign(_cik=cik).dropna(subset=["_cik"])
    if "isdelisted" in rows.columns:                       # a reused symbol: the listed holder wins
        rows = rows.sort_values("isdelisted", key=lambda c: c.astype(str).str.upper().eq("Y"))
    rows = rows.drop_duplicates("ticker", keep="first")
    out = {t: int(c) for t, c in zip(rows["ticker"], rows["_cik"])}
    # a renamed company lists its old symbols in relatedtickers (VMRK -> EQR): the old symbol,
    # which a frozen membership still uses, takes the row's CIK unless it has a row of its own
    if "relatedtickers" in rows.columns:
        for t, c in zip(rows["relatedtickers"], rows["_cik"]):
            for old in str(t or "").replace(",", " ").split():
                out.setdefault(old, int(c))
    return out


def sharadar_cik_map(tickers, user_agent, related=None, cik_map=None, tickers_table=None) -> dict:
    """SEC CIK lookup keyed by Sharadar tickers (BRK.B): Sharadar's own CIK
    first (cik_from_tickers, every name ever priced), then the SEC map via
    the normalized form it uses (BRK-B), then a ticker's previous symbols
    (`related`) while the SEC map lags a rename."""
    from stocks_ml.data.edgar import load_cik_map
    from stocks_ml.data.membership import normalize_symbol
    own = cik_from_tickers(tickers_table)
    out = {t: own[t] for t in tickers if t in own}
    rest = [t for t in tickers if t not in out]
    if not rest:
        return out
    ciks = cik_map if cik_map is not None else load_cik_map(user_agent)     # only fetched when needed
    for t in rest:
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
    tk = store.read("sharadar_tickers") if store.exists("sharadar_tickers") else None
    ciks = sharadar_cik_map(current, cfg.user_agent, related=related_symbols(tk) if tk is not None else {},
                            tickers_table=tk)
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


def refetch_sec_universe(root, cfg, log=_log, fetch_facts_fn=None, fetch_submissions_fn=None,
                         fetch_file_fn=None, cik_map=None, data_dir="data") -> dict:
    """EDGAR companyfacts and 8-K metadata for EVERY name that was ever a
    member of the world at `root`, by Sharadar's CIK (cik_from_tickers), so a
    departed name has the same SEC history as a current one. The store's
    survivor-only tables are kept beside the new ones as
    edgar.survivors_<date>.parquet / sec8k.survivors_<date>.parquet; the
    panel is untouched (append_sec_columns adds the corrected features)."""
    from stocks_ml.data.edgar import ingest_edgar
    from stocks_ml.data.sec8k import ingest_sec8k
    root = Path(root)
    store = DataStore(root)
    mem = store.read("membership")
    names = sorted(mem["ticker"].dropna().unique())
    # the store's tickers rows plus the project's full pull (every Sharadar name, renames included:
    # a symbol the frozen membership still uses may only survive as another row's relatedtickers)
    tks = [store.read("sharadar_tickers")] if store.exists("sharadar_tickers") else []
    if (Path(data_dir) / "sharadar_tickers.parquet").exists():
        tks.append(pd.read_parquet(Path(data_dir) / "sharadar_tickers.parquet"))
    tk = pd.concat(tks, ignore_index=True) if tks else None
    ciks = sharadar_cik_map(names, cfg.user_agent, related=related_symbols(tk) if tk is not None else {},
                            tickers_table=tk, cik_map=cik_map)
    missing = sorted(set(names) - set(ciks))
    before = {}
    stamp = pd.Timestamp.today().date().isoformat()
    for name in ("edgar", "sec8k"):
        if store.exists(name):
            before[name] = int(store.read(name)["ticker"].nunique())
            backup = root / f"{name}.survivors_{stamp}.parquet"
            if not backup.exists():
                shutil.copy2(root / f"{name}.parquet", backup)
    log(f"refetch: {len(names)} names ever in the membership, {len(ciks)} with a CIK"
        + (f", none for {missing}" if missing else ""))
    report = {"names": len(names), "with_cik": len(ciks), "no_cik": missing, "before": before}
    if store.exists("edgar"):                            # start clean: refetched rows REPLACE a name's rows
        store.write("edgar", store.read("edgar").iloc[0:0])
    s = ingest_edgar(store, names, cfg.edgar_concepts, cfg.user_agent, fetch_facts_fn=fetch_facts_fn,
                     cik_map=ciks, refresh_days=0)
    report["edgar"] = s
    log(f"edgar: {s['n_ok']} names with facts (was {before.get('edgar', 0)}), {len(s['failed_tickers'])} without")
    if store.exists("sec8k"):
        store.write("sec8k", store.read("sec8k").iloc[0:0])
    s = ingest_sec8k(store, names, cfg.user_agent, fetch_submissions_fn=fetch_submissions_fn,
                     fetch_file_fn=fetch_file_fn, cik_map=ciks)
    report["sec8k"] = s
    log(f"sec8k: {s['n_tickers']} names, {s['n_filings']} filings (was {before.get('sec8k', 0)} names)")
    store.set_manifest("sec_universe", {"at": stamp, "names": len(names), "with_cik": len(ciks),
                                        "edgar_names": report["edgar"]["n_ok"], "sec8k_names": report["sec8k"]["n_tickers"]})
    return report


SEC_COLS = ["x_evt_filed_5d", "x_days_since_filing", "x_pead",
            "x_evt_8k_7d", "x_evt_earnings_8k_7d", "x_days_since_earnings_8k"]


def sec_columns(store, panel: pd.DataFrame) -> pd.DataFrame:
    """The filing-window / PEAD and 8-K features (features.events) recomputed
    from the store's edgar and sec8k tables on the panel's rows, under their
    x_ names, unranked."""
    from stocks_ml.features.events import filing_features, sec8k_features
    prices = store.read("prices")
    prices = prices[prices["ticker"].isin(set(panel["ticker"]))]
    dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    edgar = store.read("edgar")
    sec8k = store.read("sec8k")
    ff = filing_features(edgar, prices, dates)
    e8 = sec8k_features(sec8k, sorted(prices["ticker"].unique()), dates)
    both = ff.merge(e8, on=["date", "ticker"], how="outer")
    both = both.rename(columns={c: "x_" + c[2:] for c in both.columns if c.startswith("f_")})
    key = pd.MultiIndex.from_frame(panel[["date", "ticker"]])
    out = both.set_index(["date", "ticker"]).reindex(key)[SEC_COLS]
    return pd.DataFrame(out.to_numpy(), columns=SEC_COLS, index=panel.index)


def append_sec_columns(root, log=_log) -> list[str]:
    """Append the filing-window / PEAD and 8-K features recomputed from the
    store's (refetched, every-name) edgar and sec8k tables to a frozen panel
    as x_ columns, ranked within the week and neutral-filled like the f_
    originals: the survivorship-free versions, opt-in by recipe
    (features=x_pead,...; drop=f_pead,...). Existing columns untouched."""
    from stocks_ml.features.ranking import rank_normalize
    root = Path(root)
    path = root / "panel_sf.parquet"
    panel = pd.read_parquet(path)
    have = [c for c in SEC_COLS if c in panel.columns]
    if have:
        log(f"{path}: {have} present; nothing to append")
        return []
    panel["date"] = pd.to_datetime(panel["date"])
    x = sec_columns(_PanelStore(root), panel)
    ranked = rank_normalize(pd.concat([panel[["date", "ticker"]], x], axis=1), SEC_COLS)
    for c in SEC_COLS:
        panel[c] = ranked[c].to_numpy()
    tmp = path.with_suffix(".tmp.parquet")          # atomic: a walk reading the panel sees the old file or the new
    panel.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    log(f"{path}: appended {SEC_COLS} (filing/PEAD and 8-K features from the every-name SEC tables)")
    return SEC_COLS


def append_short_dtc(root, log=_log, col: str = "x_short_dtc") -> list[str]:
    """Append days-to-cover on the RAW volume basis (features/panel.raw_volume)
    to a frozen panel as `col`, ranked within the week and neutral-filled like
    f_short_dtc, whose split-adjusted volume read the future split factor
    (+0.35 on 2016-2024; found 2026-09-21). Opt-in by recipe
    (features=x_short_dtc, drop=f_short_dtc). Existing columns untouched."""
    from stocks_ml.features.insiders import short_features
    from stocks_ml.features.panel import _wide, raw_volume
    from stocks_ml.features.ranking import rank_normalize
    root = Path(root)
    path = root / "panel_sf.parquet"
    panel = pd.read_parquet(path)
    if col in panel.columns:
        log(f"{path}: {col} present; nothing to append")
        return []
    panel["date"] = pd.to_datetime(panel["date"])
    store = _PanelStore(root)
    prices = store.read("prices")
    prices = prices[prices["ticker"].isin(set(panel["ticker"]))]
    shortint = store.read("shortint") if store.exists("shortint") else pd.DataFrame(
        columns=["ticker", "settlement_date", "publication_date", "short_interest"])
    grid = panel[["date", "ticker"]].drop_duplicates().assign(shares=np.nan)
    sf = short_features(shortint, grid, raw_volume(prices, _wide(prices, "volume")))
    raw = sf.set_index(["date", "ticker"])["f_short_dtc"].reindex(pd.MultiIndex.from_frame(panel[["date", "ticker"]]))
    ranked = rank_normalize(pd.DataFrame({"date": panel["date"], "ticker": panel["ticker"], col: raw.to_numpy()}), [col])[col]
    panel[col] = ranked.to_numpy()
    tmp = path.with_suffix(".tmp.parquet")
    panel.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    log(f"{path}: appended {col} (days-to-cover on raw volume)")
    return [col]


def refetch_form4_from_sf2(root, key, fetch_fn=None, log=_log) -> dict:
    """Replace a store's SEC Form 4 table with one derived from Sharadar SF2
    for EVERY name ever in its membership (form4_from_sf2, the research
    builder's own rule). The S&P store's Form 4 table was pulled through the
    SEC's current-ticker map and covered 25-68% of the names that later left
    the index against 90-94% of those that stayed (leak_audit.coverage_by_survival,
    2026-09-21). The old table is kept as form4.survivors_<date>.parquet;
    the frozen panel is untouched."""
    root = Path(root)
    store = DataStore(root)
    names = sorted(store.read("membership")["ticker"].dropna().unique())
    stamp = pd.Timestamp.today().date().isoformat()
    before = int(store.read("form4")["ticker"].nunique()) if store.exists("form4") else 0
    if store.exists("form4") and not (root / f"form4.survivors_{stamp}.parquet").exists():
        shutil.copy2(root / "form4.parquet", root / f"form4.survivors_{stamp}.parquet")
    frames = [_fetch("insiders", key, fetch_fn, ticker=",".join(b), **{"from": UNIVERSE_START}) for b in _chunks(names, 30)]
    raw = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(not f.empty for f in frames) \
        else pd.DataFrame(columns=INSIDER_COLS + ["securityadcode", "transactionpricepershare"])
    form4 = form4_from_sf2(raw, set(names))
    store.write("form4", form4)
    if store.exists("form4_sec"):
        store.write("form4_sec", form4)
    report = {"names": len(names), "before": before, "after": int(form4["ticker"].nunique()), "rows": int(len(form4))}
    log(f"form4 from SF2: {report['after']} names, {report['rows']:,} rows (was {before} names)")
    store.set_manifest("form4_universe", {"at": stamp, **report})
    return report


def append_market_dispersion(root, log=_log, col: str = "x_mkt_dispersion") -> list[str]:
    """Append the cross-sectional return dispersion computed over THAT DATE'S
    INDEX MEMBERS to a frozen panel as `col`. The stored f_mkt_dispersion took
    the std across every column of the price file — the ever-member set — so
    31% of the 2006 cross-section were names that only joined the index later
    (2026-09-23). Rank-exempt like its parent, so the raw level is kept.
    Opt-in by recipe (features=x_mkt_dispersion, drop=f_mkt_dispersion)."""
    from stocks_ml.features.panel import _wide
    root = Path(root)
    path = root / "panel_sf.parquet"
    panel = pd.read_parquet(path)
    if col in panel.columns:
        log(f"{path}: {col} present; nothing to append")
        return []
    panel["date"] = pd.to_datetime(panel["date"])
    store = _PanelStore(root)
    prices = store.read("prices")
    mem = store.read("membership")
    r5 = _wide(prices, "close").pct_change(5, fill_method=None)
    inside = pd.DataFrame(False, index=r5.index, columns=r5.columns)
    for st in mem.itertuples():
        if st.ticker not in inside.columns or pd.isna(st.start_date):
            continue
        m = r5.index >= pd.Timestamp(st.start_date)
        if pd.notna(st.end_date):
            m &= r5.index <= pd.Timestamp(st.end_date)
        inside.loc[m, st.ticker] = True
    disp = r5.where(inside).std(axis=1, ddof=1)
    panel[col] = panel["date"].map(disp.reindex(pd.DatetimeIndex(sorted(panel["date"].unique()))).to_dict()).astype(float)
    tmp = path.with_suffix(".tmp.parquet")
    panel.to_parquet(tmp, index=False)
    os.replace(tmp, path)
    log(f"{path}: appended {col} (dispersion across the date's index members)")
    return [col]


def _stream_batches(path: Path, frames_iter) -> int:
    """Write DataFrames to one parquet file batch by batch (never all in
    memory: a top-2000 SEP history is ~25M rows). Returns the row count."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    writer, n = None, 0
    try:
        for df in frames_iter:
            if df.empty:
                continue
            tab = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(path, tab.schema)
            writer.write_table(tab.cast(writer.schema))
            n += len(df)
    finally:
        if writer is not None:
            writer.close()
    return n


DERIVED_TABLES = ("sharadar_tickers", "daily_snapshots", "sharadar_prices", "prices", "fundamentals", "insiders",
                  "form4", "form4_sec", "form4_bridge", "edgar", "sec8k", "shortint", "fred")


def derive_research_store(root, parent, cfg, n: int, log=_log, panel: bool = True,
                          membership_from=None, top_up: bool = False, sized: bool = False) -> dict:
    """A top-n world carved from a built top-m world (n <= m): every pulled
    table is shared by symlink, the membership is the parent's snapshots cut
    at n (the same point-in-time rule), and the panel is rebuilt so every
    feature is ranked within THIS universe. Nothing is fetched. Refuses an
    existing world."""
    root, parent = Path(root), Path(parent)
    if (root / "panel_sf.parquet").exists() or (root / "membership.parquet").exists():
        raise SystemExit(f"{root} already holds a research world; it is never rebuilt")
    src = DataStore(parent)
    if not src.exists("daily_snapshots"):
        raise SystemExit(f"{parent} has no daily_snapshots: derive from a top-N world")
    m = int(src.manifest.get("universe", {}).get("n", 0))
    if m and n > m:
        raise SystemExit(f"top-{n} cannot be derived from a top-{m} world")
    store = DataStore(root)
    for name in DERIVED_TABLES:
        if src.exists(name):
            (root / f"{name}.parquet").symlink_to((parent / f"{name}.parquet").resolve())
    tk = src.read("sharadar_tickers")
    snaps = src.read("daily_snapshots")
    sectors = dict(tk.dropna(subset=["sicsector"]).drop_duplicates("ticker")[["ticker", "sicsector"]].values)
    if membership_from:
        # a CONTROL world: another store's membership (e.g. the S&P 500 stints) on this
        # store's tables and build — same names as the champion's world, new plumbing
        mem = DataStore(membership_from).read("membership")
        have = set(src.read("prices")["ticker"]) if src.exists("prices") else set(mem["ticker"])
        missing = sorted(set(mem["ticker"]) - have)
        mem = mem[mem["ticker"].isin(have)].reset_index(drop=True)
        log(f"membership from {membership_from}: {len(mem)} stints; {len(missing)} names without prices here dropped: {missing[:8]}")
        if sized:
            mem = membership_sized(mem, snaps, universe_equities(tk), sectors, n)
            log(f"sized to {n} names per snapshot (the index's largest {n}, or the index topped up): {len(mem)} stints")
        elif top_up:
            mem = membership_topped_up(mem, snaps, universe_equities(tk), sectors, n)
            log(f"topped up to {n} names per snapshot with the largest non-members by market cap: {len(mem)} stints")
    else:
        mem = membership_from_top(snaps, universe_equities(tk), sectors, n)
    store.write("membership", mem)
    universe = sorted(set(mem["ticker"]))
    report = {"universe": f"top{n}" if not membership_from else f"membership of {membership_from}" + (f" sized to {n}" if sized else f" topped up to {n}" if top_up else ""), "n": n,
              "derived_from": str(parent),
              "membership": {"snapshots": int(snaps["date"].nunique()), "stints": int(len(mem)),
                             "names": len(universe), "current": int(mem["end_date"].isna().sum())}}
    for k in ("sharadar", "edgar", "sec8k", "shortint", "fred"):
        if k in src.manifest:
            store.set_manifest(k, src.manifest[k])
    rule = (f"{n} names at each quarter end: the members of {membership_from} ranked by market cap — the largest {n}, "
            f"or all of them topped up with the largest non-members — on {parent}'s tables and build" if membership_from and sized else
            f"the membership of {membership_from} topped up to {n} names at each quarter end with the largest "
            f"non-members by market cap, on {parent}'s tables and build" if membership_from and top_up else
            f"the membership of {membership_from} (a control: the same names on {parent}'s tables and build)"
            if membership_from else f"top{n} by market cap at each calendar quarter end, derived from "
                                    f"{parent} (its snapshots and tables, shared by symlink)")
    store.set_manifest("universe", {"rule": rule, "n": n, "derived_from": str(parent),
                                    "membership_from": membership_from, **report["membership"]})
    log(f"universe top{n} from {parent}: {len(mem)} stints over {len(universe)} names, "
        f"{report['membership']['current']} current; tables shared")
    if panel:
        build_world_panel(root, cfg, log=log)
    return report


def build_research_store(root, key, cfg, n: int = 2000, fetch_fn=None, log=_log, sec: bool = True,
                         panel: bool = True, start: str = SNAPSHOT_START, today=None) -> dict:
    """A research world for the top-n universe, from nothing: quarter-end
    market-cap snapshots -> membership stints; then for every name that was
    ever a member, SEP prices (streamed to disk), SF1 ARQ/ART, SF2 insiders
    (the Form 4 file is SF2-derived here), the free SEC/FINRA/FRED ingesters,
    and the panel. Refuses a directory that already holds a panel: a research
    world is built once (the frozen-panel rule)."""
    root = Path(root)
    if (root / "panel_sf.parquet").exists() or (root / "membership.parquet").exists():
        raise SystemExit(f"{root} already holds a research world; it is never rebuilt")
    store = DataStore(root)
    today = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    report = {"universe": f"top{n}", "n": n}

    tk = fetch_table("tickers", key, fetch_fn=fetch_fn, **{"table": "stocks"})   # every equity Sharadar prices
    store.write("sharadar_tickers", tk)
    eligible = universe_equities(tk)
    dates = quarter_ends(start, today)
    snaps = fetch_snapshots(key, fetch_fn, dates, log=log)
    store.write("daily_snapshots", snaps)
    sectors = dict(tk.dropna(subset=["sicsector"]).drop_duplicates("ticker")[["ticker", "sicsector"]].values)
    mem = membership_from_top(snaps, eligible, sectors, n)
    store.write("membership", mem)
    universe = sorted(set(mem["ticker"]))
    report["membership"] = {"snapshots": int(snaps["date"].nunique()), "stints": int(len(mem)),
                            "names": len(universe), "current": int(mem["end_date"].isna().sum())}
    log(f"universe top{n}: {report['membership']['snapshots']} snapshots {pd.Timestamp(start).date()} -> "
        f"{today.date()}, {len(mem)} stints over {len(universe)} names, {report['membership']['current']} current")

    # prices: SEP for the universe, SFP for the funds — streamed batch by batch
    def sep_batches():
        batches = _chunks(universe, 30)
        for i, b in enumerate(batches, 1):
            df = _fetch("stocks", key, fetch_fn, ticker=",".join(b), **{"from": UNIVERSE_START})
            if i % 20 == 0 or i == len(batches):
                log(f"  prices: batch {i}/{len(batches)}")
            yield df
        yield _fetch("funds", key, fetch_fn, ticker=",".join(FUND_TICKERS), **{"from": UNIVERSE_START})
    def as_sep(df):
        if df.empty:
            return df
        df = df[SEP_COLS].copy()
        for c in ("open", "high", "low", "close", "volume", "closeadj", "closeunadj"):
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df["date"] = pd.to_datetime(df["date"])
        df["lastupdated"] = pd.to_datetime(df["lastupdated"], errors="coerce")
        return df.drop_duplicates(["ticker", "date"], keep="last")
    raw_path, px_path = store._path("sharadar_prices"), store._path("prices")
    def both():
        for df in sep_batches():
            df = as_sep(df)
            if df.empty:
                continue
            import pyarrow as pa, pyarrow.parquet as pq
            yield df, prices_from_sep(df)
    import pyarrow as pa
    import pyarrow.parquet as pq
    wr = wp = None
    n_rows = 0
    try:
        for raw, px in both():
            t1, t2 = pa.Table.from_pandas(raw, preserve_index=False), pa.Table.from_pandas(px, preserve_index=False)
            if wr is None:
                wr, wp = pq.ParquetWriter(raw_path, t1.schema), pq.ParquetWriter(px_path, t2.schema)
            wr.write_table(t1.cast(wr.schema)); wp.write_table(t2.cast(wp.schema))
            n_rows += len(px)
    finally:
        for w in (wr, wp):
            if w is not None:
                w.close()
    prices_meta = pq.read_metadata(px_path) if px_path.exists() else None
    report["prices"] = {"rows": n_rows}
    log(f"prices: {n_rows:,} rows for {len(universe)} names + {FUND_TICKERS}")

    # fundamentals: SF1 ARQ + ART, full history
    frames = []
    for b in _chunks(universe, 30):
        for dim in ("ARQ", "ART"):
            frames.append(_fetch("fundamentals", key, fetch_fn, ticker=",".join(b), dimension=dim,
                                 **{"from": FUNDAMENTALS_START}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FutureWarning)
        raw_sf1 = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
            if any(not f.empty for f in frames) else pd.DataFrame(columns=FUND_COLS)
    fund = fundamentals_from_sf1(raw_sf1, set(universe))
    store.write("fundamentals", fund)
    report["fundamentals"] = {"rows": int(len(fund)), "names": int(fund["ticker"].nunique())}
    log(f"fundamentals: {len(fund):,} rows, {report['fundamentals']['names']} names")
    del frames, raw_sf1

    # insiders: SF2 full history; the Form 4 file is SF2-derived (form4_from_sf2)
    frames = [_fetch("insiders", key, fetch_fn, ticker=",".join(b), **{"from": UNIVERSE_START})
              for b in _chunks(universe, 30)]
    raw_sf2 = pd.concat([f for f in frames if not f.empty], ignore_index=True) \
        if any(not f.empty for f in frames) else pd.DataFrame(columns=INSIDER_COLS + ["securityadcode", "transactionpricepershare"])
    ins = insiders_from_sf2(raw_sf2, set(universe))
    form4 = form4_from_sf2(raw_sf2, set(universe))
    store.write("insiders", ins)
    store.write("form4", form4)
    store.write("form4_sec", form4)                   # the frozen file a live copy would extend
    store.write("form4_bridge", pd.DataFrame(columns=FORM4_COLS))
    report["insiders"] = {"rows": int(len(ins)), "form4_rows": int(len(form4))}
    log(f"insiders: {len(ins):,} rows; form4 {len(form4):,} rows")
    del frames, raw_sf2
    store.set_manifest("sharadar", {"since": str(today.date()), "form4_sec_through": str(today.date()),
                                    "built_at": str(pd.Timestamp.now()), "universe": f"top{n}"})
    store.set_manifest("universe", {"rule": f"top{n} domestic common stock by market cap at each calendar "
                                            f"quarter end since {pd.Timestamp(start).date()}; a name is a member "
                                            f"from the first snapshot that ranks it until the first that does not",
                                    "n": n, **report["membership"]})
    if sec:
        report["sec"] = refresh_sec(store, cfg, universe, log=log)
    if panel:
        build_world_panel(root, cfg, log=log)
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


def sharadar_frame(store, cfg, panel: pd.DataFrame, fundamentals_file=None) -> pd.DataFrame:
    """The Sharadar fundamental features for `panel` (build_world_panel's
    step, factored out): ratios on the split-adjusted close under the
    nominal basis, ranked within the week."""
    from stocks_ml.features.ranking import rank_normalize
    from stocks_ml.features.sharadar_fundamentals import SF_RAW_COLS, sharadar_fundamental_features
    prices = store.read("prices")
    fund = pd.read_parquet(fundamentals_file) if fundamentals_file else store.read("fundamentals")
    nominal = getattr(cfg, "price_basis", "closeadj") == "nominal"
    px_field = "close_split" if nominal else "close"
    cw = prices.pivot(index="date", columns="ticker", values=px_field).sort_index().ffill()
    wk = cw.reindex(pd.Index(sorted(panel["date"].unique())), method="ffill")
    close = pd.Series(wk.stack().reindex(
        pd.MultiIndex.from_frame(panel[["date", "ticker"]])).values, index=panel.index)
    ff = sharadar_fundamental_features(fund, panel, close)
    return rank_normalize(pd.concat([panel[["date", "ticker"]], ff], axis=1), SF_RAW_COLS)[SF_RAW_COLS]


def append_sector_relative(root, log=_log, cols=None) -> list[str]:
    """Append the sector-relative ranks (features/panel.sector_relative_ranks)
    of the admitted features to a panel as x_sr_* columns — derived from the
    panel's own ranked columns and the membership's sector map; existing
    columns untouched. Columns already present are left alone."""
    from stocks_ml.features.panel import SR_PREFIX, feature_cols, sector_relative_ranks
    root = Path(root)
    path = root / "panel_sf.parquet"
    panel = pd.read_parquet(path)
    cols = [c for c in (cols or feature_cols(panel)) if SR_PREFIX + c not in panel.columns]
    if not cols:
        log(f"{path}: every sector-relative column present; nothing to append")
        return []
    mem = DataStore(root).read("membership")
    smap = dict(mem.dropna(subset=["sector"]).drop_duplicates("ticker")[["ticker", "sector"]].values)
    sr = sector_relative_ranks(panel, panel["ticker"].map(smap), cols)
    for c in sr.columns:
        panel[c] = sr[c].to_numpy()
    panel.to_parquet(path, index=False)
    log(f"{path}: appended {len(sr.columns)} sector-relative columns ({SR_PREFIX}*)")
    return list(sr.columns)


def append_clean_dollar_volume(root, log=_log, col: str = "x_dollar_vol") -> list[str]:
    """Append the split-consistent dollar volume (split-adjusted close x
    split-adjusted volume, 20-day mean, log, ranked within the week) to a
    frozen panel as `col`: the fixed f_dollar_vol, opt-in by recipe
    (features=x_dollar_vol,drop=f_dollar_vol). Existing columns untouched."""
    from stocks_ml.features.ranking import rank_normalize
    root = Path(root)
    path = root / "panel_sf.parquet"
    panel = pd.read_parquet(path)
    if col in panel.columns:
        log(f"{path}: {col} present; nothing to append")
        return []
    panel["date"] = pd.to_datetime(panel["date"])
    prices = _PanelStore(root).read("prices")
    px = "close_split" if "close_split" in prices.columns else "close"
    c = prices.pivot(index="date", columns="ticker", values=px).sort_index()
    v = prices.pivot(index="date", columns="ticker", values="volume").sort_index()
    dv = np.log((c * v).rolling(20).mean().where(lambda d: d > 0))
    wk = dv.reindex(pd.Index(sorted(panel["date"].unique())), method="ffill")
    raw = pd.Series(wk.stack().reindex(pd.MultiIndex.from_frame(panel[["date", "ticker"]])).values, index=panel.index)
    ranked = rank_normalize(pd.DataFrame({"date": panel["date"], "ticker": panel["ticker"], col: raw}), [col])[col]
    panel[col] = ranked.to_numpy()
    panel.to_parquet(path, index=False)
    log(f"{path}: appended {col} (split-consistent dollar volume)")
    return [col]


def append_sf_columns(root, cfg, log=_log) -> list[str]:
    """Append the Sharadar columns a frozen panel lacks (a newer SF_RAW_COLS),
    after verifying that EVERY Sharadar column it already has recomputes
    exactly from the store's tables (the fundamentals vintage the panel was
    built from: leak_audit.fundamentals_file). Existing columns are never
    touched; a mismatch refuses the append (the 2026-09-16 rule)."""
    from stocks_ml.features.sharadar_fundamentals import SF_RAW_COLS
    from stocks_ml.leak_audit import fundamentals_file
    root = Path(root)
    store = _PanelStore(root)
    path = root / "panel_sf.parquet"
    panel = pd.read_parquet(path)
    panel["date"] = pd.to_datetime(panel["date"])
    have = [c for c in SF_RAW_COLS if c in panel.columns]
    new = [c for c in SF_RAW_COLS if c not in panel.columns]
    if not new:
        log(f"{path}: every Sharadar column present; nothing to append")
        return []
    sf = sharadar_frame(store, cfg, panel[["date", "ticker"]].reset_index(drop=True), fundamentals_file(root))
    sf.index = panel.index
    worst = max(float(np.nanmax(np.abs(panel[c].to_numpy(float) - sf[c].to_numpy(float)))) for c in have)
    if not worst <= 1e-9:
        raise SystemExit(f"{path}: the existing Sharadar columns do not recompute from the store's tables "
                         f"(max |diff| {worst:.3g}); nothing appended")
    for c in new:
        panel[c] = sf[c].to_numpy()
    panel.to_parquet(path, index=False)
    log(f"{path}: verified {len(have)} Sharadar columns exact; appended {new}")
    return new


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
    # f_sf ratios divide split-restated per-share values by this close; the
    # SPLIT-adjusted close (close_split) cancels the restatement exactly and
    # equals the live computation, where the future factor is 1 for every
    # name. The closeadj basis (status quo) keeps the deployed behavior.
    nominal = getattr(cfg, "price_basis", "closeadj") == "nominal"
    if nominal and "close_split" not in prices.columns:
        raise RuntimeError("price_basis='nominal' needs close_split in the prices table: "
                           "regenerate it from sharadar_prices (world.prices_from_sep)")
    px_field = "close_split" if nominal else "close"
    cw = prices.pivot(index="date", columns="ticker", values=px_field).sort_index().ffill()
    wk = cw.reindex(pd.Index(sorted(panel["date"].unique())), method="ffill")
    close = pd.Series(wk.stack().reindex(
        pd.MultiIndex.from_frame(panel[["date", "ticker"]])).values, index=panel.index)
    ff = sharadar_fundamental_features(fund, panel, close)
    shares = _asof(panel, fund[fund["dimension"] == "ARQ"], ["sharesbas"])["sharesbas"]
    fi = sharadar_insider_features(ins, panel, mktcap=close * shares)
    panel_sf = rank_normalize(pd.concat([panel, ff, fi], axis=1), SF_RAW_COLS + SFI_RAW_COLS)
    # the range family (features/panel.high_low_features) from SEP high/low:
    # prices_hl (world --extras) or the raw SEP rows the live refresh keeps
    hl = None
    if store.exists("prices_hl"):
        hl = store.read("prices_hl")
    elif store.exists("sharadar_prices"):
        raw_hl = store.read("sharadar_prices")
        if {"high", "low"} <= set(raw_hl.columns):
            hl = raw_hl[["ticker", "date", "high", "low"]].copy()
            hl["date"] = pd.to_datetime(hl["date"])
    if hl is not None:
        from stocks_ml.features.panel import HL_COLS, high_low_features
        x = high_low_features(hl, prices, panel_sf)
        panel_sf = rank_normalize(pd.concat([panel_sf, x], axis=1), HL_COLS)
        log(f"panel_sf: + {len(HL_COLS)} high/low columns (x_hl_*)")
    # sector-relative volatility (features/panel.sector_relative_volatility) on the
    # membership's SIC divisions and, when the tickers table is stored, Sharadar's 11 sectors
    from stocks_ml.features.panel import sector_relative_volatility
    mem = store.read("membership")
    maps = {"sic": dict(mem.dropna(subset=["sector"]).drop_duplicates("ticker")[["ticker", "sector"]].values)}
    if store.exists("sharadar_tickers"):
        tk = store.read("sharadar_tickers")
        if "sector" in tk.columns:
            maps["s11"] = dict(tk.dropna(subset=["sector"]).drop_duplicates("ticker")[["ticker", "sector"]].values)
    sv = pd.concat([sector_relative_volatility(panel_sf, panel_sf["ticker"].map(m), tag) for tag, m in maps.items()], axis=1)
    panel_sf = rank_normalize(pd.concat([panel_sf, sv], axis=1), list(sv.columns))
    log(f"panel_sf: + {sv.shape[1]} sector-relative volatility columns (x_sv_*: {', '.join(maps)})")
    from stocks_ml.features.panel import VX_COLS, volatility_size_interactions
    panel_sf = rank_normalize(pd.concat([panel_sf, volatility_size_interactions(panel_sf)], axis=1), VX_COLS)
    log(f"panel_sf: + {len(VX_COLS)} volatility x size columns (x_vx_*)")
    from stocks_ml.features.panel import PEER_COLS, comovement_peer_features
    peers = comovement_peer_features(prices, panel_sf)
    panel_sf = rank_normalize(pd.concat([panel_sf, peers], axis=1), PEER_COLS)
    log(f"panel_sf: + {len(PEER_COLS)} co-movement peer columns (x_cm_*)")
    # x_dollar_vol: the split-consistent dollar volume under its research name. A panel built by this
    # code already has it as f_dollar_vol; the frozen research panel keeps its leaky f_dollar_vol and
    # carries the clean column appended as x_dollar_vol (append_clean_dollar_volume), so the recipe
    # `features=x_dollar_vol,drop=f_dollar_vol` is the same 64-column matrix everywhere.
    panel_sf["x_dollar_vol"] = panel_sf["f_dollar_vol"]
    # x_pead & co: the filing/PEAD and 8-K features under their research names. A store whose SEC tables
    # cover every name ever (refetch_sec_universe; a research world built since 2026-09-21) has them as
    # the f_ columns; the frozen S&P panel keeps its survivor-only f_ columns and carries the corrected
    # ones appended as x_ (append_sec_columns).
    for c in SEC_COLS:
        panel_sf[c] = panel_sf["f_" + c[2:]]
    panel_sf["x_short_dtc"] = panel_sf["f_short_dtc"]      # raw-volume days-to-cover since 2026-09-21 (append_short_dtc)
    panel_sf["x_mkt_dispersion"] = panel_sf["f_mkt_dispersion"]    # members-only cross-section since 2026-09-23
    panel_sf.to_parquet(Path(live_dir) / "panel_sf.parquet", index=False)
    log(f"panel_sf: {panel_sf.shape[0]:,} x {panel_sf.shape[1]} ({time.time() - t0:.0f}s)")
    return panel_sf
