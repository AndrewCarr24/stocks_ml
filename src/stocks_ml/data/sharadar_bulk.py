"""Ingest Sharadar Direct bulk-CSV tables into world-store parquets.

Bulk zips come from GET /data/{table}?years=full (see data/sharadar.py and
sharadar.com/llms.txt). This module filters them to a world's ticker universe
and keeps only the point-in-time-safe slices:
  * fundamentals: AR* dimensions only (ARQ instant/quarterly, ART trailing-12m).
    MR* rows are restated backward in time -> lookahead; never ingest them.
    `date` is the SEC filing date (the point-in-time key downstream).
  * insiders: open-market Form 3/4/5 rows; `date` is the filing date.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FUND_COLS = [
    "ticker", "dimension", "calendardate", "date", "reportperiod",
    "revenue", "netinc", "gp", "assets", "equity", "debt", "ebitda", "ebit",
    "fcf", "ncfo", "capex", "currentratio", "de", "sharesbas", "shareswa",
    "bvps", "eps", "epsusd", "marketcap", "liabilities", "cashneq",
    "divyield", "dps", "grossmargin", "ebitdamargin", "netmargin", "roe",
]
INSIDER_COLS = ["ticker", "date", "transactiondate", "transactioncode",
                "transactionshares", "transactionvalue", "ownername"]


def ingest_fundamentals(csv_path, store_dir, universe: set[str],
                        chunksize: int = 500_000) -> int:
    keep = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize,
                             usecols=lambda c: c in FUND_COLS,
                             low_memory=False):
        m = chunk["ticker"].isin(universe) & chunk["dimension"].isin(["ARQ", "ART"])
        keep.append(chunk[m])
    df = pd.concat(keep, ignore_index=True)
    for c in ("calendardate", "date", "reportperiod"):
        df[c] = pd.to_datetime(df[c])
    num = [c for c in df.columns
           if c not in ("ticker", "dimension", "calendardate", "date", "reportperiod")]
    df[num] = df[num].apply(pd.to_numeric, errors="coerce")
    df = df.sort_values(["ticker", "dimension", "reportperiod", "date"])
    df.to_parquet(Path(store_dir) / "fundamentals.parquet", index=False)
    return len(df)


def ingest_insiders(csv_path, store_dir, universe: set[str],
                    chunksize: int = 1_000_000) -> int:
    keep = []
    for chunk in pd.read_csv(csv_path, chunksize=chunksize,
                             usecols=lambda c: c in INSIDER_COLS,
                             low_memory=False):
        m = chunk["ticker"].isin(universe) & chunk["transactioncode"].isin(["P", "S"])
        keep.append(chunk[m])
    df = pd.concat(keep, ignore_index=True)
    for c in ("date", "transactiondate"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    for c in ("transactionshares", "transactionvalue"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date"])
    # signed dollar flow: buys positive, sales negative
    df["signed_value"] = np.sign(df["transactionshares"].fillna(0)) * df[
        "transactionvalue"].abs()
    df = df.sort_values(["ticker", "date"])
    df.to_parquet(Path(store_dir) / "insiders.parquet", index=False)
    return len(df)
