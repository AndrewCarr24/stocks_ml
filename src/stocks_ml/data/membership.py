from __future__ import annotations

import re
from io import StringIO

import pandas as pd
import requests

WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
# 2026-08-11: Wikipedia moved the "Selected changes" table off the main article
# (revision 1368903137, "move to Historical components of the S&P 500").
WIKI_CHANGES_URL = "https://en.wikipedia.org/wiki/Historical_components_of_the_S%26P_500"


def normalize_symbol(s: str) -> str:
    # Strip footnote/formatting junk Wikipedia cells sometimes carry (the
    # 2026-08 "Historical components" page renders some tickers as "ITT |").
    cleaned = re.sub(r"[^A-Za-z0-9.\-]", "", str(s).strip().split()[0] if str(s).strip() else "")
    return cleaned.upper().replace(".", "-")


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


def _flat_cols(table: pd.DataFrame) -> list[str]:
    cols = table.columns
    if isinstance(cols, pd.MultiIndex):
        return ["_".join(str(c) for c in col).lower() for col in cols]
    return [str(c).lower() for c in cols]


def _locate_table(tables: list, required_fragments: list[str], page: str) -> pd.DataFrame:
    """The table whose (flattened) columns contain every required fragment.

    Wikipedia reshuffles pages without notice (the 2026-08 move of the changes
    table broke positional indexing); locating by column content survives
    reordering, added navboxes, and section moves within a page."""
    for t in tables:
        flat = _flat_cols(t)
        if all(any(frag in c for c in flat) for frag in required_fragments):
            return t
    raise ValueError(
        f"no table with columns matching {required_fragments} on {page}; "
        "Wikipedia's structure changed again — inspect the page and update "
        "membership.py (see the WIKI_CHANGES_URL move note)")


def fetch_sp500_tables(user_agent: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Network: fetch and clean the current-constituents table (main article)
    and the membership-changes table (moved to its own article 2026-08)."""
    headers = {"User-Agent": user_agent}
    resp = requests.get(WIKI_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    current_raw = _locate_table(pd.read_html(StringIO(resp.text)),
                                ["symbol", "gics sector"], WIKI_URL)
    resp = requests.get(WIKI_CHANGES_URL, headers=headers, timeout=30)
    resp.raise_for_status()
    changes_raw = _locate_table(pd.read_html(StringIO(resp.text)),
                                ["date", "added", "removed", "reason"],
                                WIKI_CHANGES_URL)
    return _clean_wiki_tables(current_raw, changes_raw)


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
