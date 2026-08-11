from __future__ import annotations

from io import StringIO

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"


def normalize_symbol(s: str) -> str:
    return str(s).strip().upper().replace(".", "-")


def build_membership(current: pd.DataFrame, changes: pd.DataFrame, floor_date: pd.Timestamp) -> pd.DataFrame:
    """Walk the changes table backwards from today to reconstruct stints.

    Invariant while walking (dates descending): `open_stints` maps ticker -> the
    stint whose start date we have not yet discovered. An 'added' row closes that
    discovery (sets start). A 'removed' row opens a new earlier stint ending then.

    Within a single change date, all 'added' events are applied before any
    'removed' events so that a same-date add+remove of one ticker (in separate
    rows) yields the same two stints regardless of row order.
    """
    tickers = current["ticker"].map(normalize_symbol)
    sectors = dict(zip(tickers, current["sector"]))
    added_dates = (dict(zip(tickers, pd.to_datetime(current["date_added"], errors="coerce")))
                   if "date_added" in current else {})
    open_stints: dict[str, dict] = {
        t: {"ticker": t, "start_date": added_dates.get(t, pd.NaT), "end_date": pd.NaT}
        for t in tickers
    }
    done: list[dict] = []

    ordered = changes.sort_values("date", ascending=False, kind="stable")
    for date, group in ordered.groupby("date", sort=False):
        for _, row in group.iterrows():
            if pd.notna(row["added"]):
                added = normalize_symbol(row["added"])
                if added in open_stints:
                    stint = open_stints.pop(added)
                    if pd.isna(stint["start_date"]):
                        stint["start_date"] = date
                    done.append(stint)
        for _, row in group.iterrows():
            if pd.notna(row["removed"]):
                removed = normalize_symbol(row["removed"])
                # ticker was a member before this date; start unknown so far
                open_stints[removed] = {"ticker": removed, "start_date": pd.NaT, "end_date": date}

    for stint in open_stints.values():
        if pd.isna(stint["start_date"]):
            stint["start_date"] = floor_date
        done.append(stint)

    mem = pd.DataFrame(done)
    mem["sector"] = mem["ticker"].map(sectors)
    return mem.sort_values(["ticker", "start_date"]).reset_index(drop=True)


def members_asof(membership: pd.DataFrame, date) -> list[str]:
    d = pd.Timestamp(date)
    live = membership[
        (membership["start_date"] <= d)
        & (membership["end_date"].isna() | (membership["end_date"] > d))
    ]
    return sorted(live["ticker"].unique())


def _clean_wiki_tables(current_raw: pd.DataFrame, changes_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    current_cols = {str(c).lower(): c for c in current_raw.columns}
    added_key = next((original for lowered, original in current_cols.items()
                      if "date" in lowered and "added" in lowered), None)
    current = pd.DataFrame({
        "ticker": current_raw["Symbol"].map(normalize_symbol),
        "sector": current_raw["GICS Sector"].astype(str),
        "date_added": (pd.to_datetime(current_raw[added_key], errors="coerce")
                       if added_key is not None else pd.NaT),
    })
    cols = changes_raw.columns
    if isinstance(cols, pd.MultiIndex):
        flat = ["_".join(str(c) for c in col).lower() for col in cols]
    else:
        flat = [str(c).lower() for c in cols]
    changes_raw = changes_raw.set_axis(flat, axis=1)
    date_col = next(c for c in flat if "date" in c)
    added_col = next(c for c in flat if "added" in c and "ticker" in c)
    removed_col = next(c for c in flat if "removed" in c and "ticker" in c)
    reason_col = next(c for c in flat if "reason" in c)
    changes = pd.DataFrame({
        "date": pd.to_datetime(changes_raw[date_col], errors="coerce"),
        "added": changes_raw[added_col],
        "removed": changes_raw[removed_col],
        "reason": changes_raw[reason_col],
    }).dropna(subset=["date"])
    return current, changes


def fetch_sp500_tables(user_agent: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Network: fetch and clean Wikipedia's current-constituents and changes tables."""
    resp = requests.get(WIKI_URL, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    return _clean_wiki_tables(tables[0], tables[1])


def ingest_membership(store, cfg, fetch_fn=None) -> pd.DataFrame:
    fetch = fetch_fn or (lambda: fetch_sp500_tables(cfg.user_agent))
    current, changes = fetch()
    mem = build_membership(current, changes, cfg.membership_floor)
    store.write("membership", mem)
    store.set_manifest("membership", {"n_tickers": int(mem["ticker"].nunique()),
                                      "n_stints": int(len(mem))})

    removed = changes[changes["removed"].notna()]
    removals = pd.DataFrame({
        "ticker": removed["removed"].map(normalize_symbol),
        "date": removed["date"],
        "reason": removed["reason"],
    }).reset_index(drop=True)
    store.write("removals", removals)

    return mem
