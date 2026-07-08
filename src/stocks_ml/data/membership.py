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
    """
    sectors = dict(zip(current["ticker"], current["sector"]))
    open_stints: dict[str, dict] = {
        t: {"ticker": t, "start_date": pd.NaT, "end_date": pd.NaT} for t in current["ticker"]
    }
    done: list[dict] = []

    for _, row in changes.sort_values("date", ascending=False).iterrows():
        added = normalize_symbol(row["added"]) if pd.notna(row["added"]) else None
        removed = normalize_symbol(row["removed"]) if pd.notna(row["removed"]) else None
        if added and added in open_stints:
            stint = open_stints.pop(added)
            stint["start_date"] = row["date"]
            done.append(stint)
        if removed:
            # ticker was a member before this date; start unknown so far
            open_stints[removed] = {"ticker": removed, "start_date": pd.NaT, "end_date": row["date"]}

    for stint in open_stints.values():
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


def fetch_sp500_tables(user_agent: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Network: fetch and clean Wikipedia's current-constituents and changes tables."""
    resp = requests.get(WIKI_URL, headers={"User-Agent": user_agent}, timeout=30)
    resp.raise_for_status()
    tables = pd.read_html(StringIO(resp.text))
    current_raw, changes_raw = tables[0], tables[1]

    current = pd.DataFrame({
        "ticker": current_raw["Symbol"].map(normalize_symbol),
        "sector": current_raw["GICS Sector"].astype(str),
    })

    changes_raw.columns = ["_".join(str(c) for c in col).lower() for col in changes_raw.columns]
    date_col = next(c for c in changes_raw.columns if c.startswith("date"))
    added_col = next(c for c in changes_raw.columns if "added" in c and "ticker" in c)
    removed_col = next(c for c in changes_raw.columns if "removed" in c and "ticker" in c)
    changes = pd.DataFrame({
        "date": pd.to_datetime(changes_raw[date_col], errors="coerce"),
        "added": changes_raw[added_col],
        "removed": changes_raw[removed_col],
    }).dropna(subset=["date"])
    return current, changes


def ingest_membership(store, cfg, fetch_fn=None) -> pd.DataFrame:
    fetch = fetch_fn or (lambda: fetch_sp500_tables(cfg.user_agent))
    current, changes = fetch()
    mem = build_membership(current, changes, cfg.membership_floor)
    store.write("membership", mem)
    store.set_manifest("membership", {"n_tickers": int(mem["ticker"].nunique()),
                                      "n_stints": int(len(mem))})
    return mem
