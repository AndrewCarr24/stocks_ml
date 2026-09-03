"""Membership helpers shared by the world store and the panel builder.

Membership itself comes from Sharadar's `sp500` events (data/world.py:
membership_from_sp500); the frame has one row per stint with `ticker`,
`start_date`, `end_date` (NaT while current) and `sector`.
"""
from __future__ import annotations

import re

import pandas as pd


def normalize_symbol(s: str) -> str:
    """SEC/Wikipedia-style symbol (BRK.B, 'ITT |') -> the project's BRK-B form."""
    cleaned = re.sub(r"[^A-Za-z0-9.\-]", "", str(s).strip().split()[0] if str(s).strip() else "")
    return cleaned.upper().replace(".", "-")


def members_asof(membership: pd.DataFrame, date) -> list[str]:
    d = pd.Timestamp(date)
    live = membership[
        (membership["start_date"] <= d)
        & (membership["end_date"].isna() | (membership["end_date"] > d))
    ]
    return sorted(live["ticker"].unique())
