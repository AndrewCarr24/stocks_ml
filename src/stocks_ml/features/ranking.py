from __future__ import annotations

import pandas as pd

RANK_EXEMPT_PREFIXES = ("f_mkt_", "f_macro_", "f_month", "f_woq", "f_sec_", "f_evt_")


def rank_normalize(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    grouped = out.groupby("date")[cols]
    out[cols] = grouped.rank(pct=True).mul(2).sub(1.0).where(out[cols].notna())
    return out
