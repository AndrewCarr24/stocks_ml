from __future__ import annotations

import pandas as pd

RANK_EXEMPT_PREFIXES = ("f_mkt_", "f_macro_", "f_month", "f_woq", "f_sec_", "f_evt_")


def rank_normalize(df: pd.DataFrame, cols: list[str], neutral_fill: bool = True) -> pd.DataFrame:
    """Rank features cross-sectionally, then neutral-fill missing ranked values.

    Filling after ranking preserves observed cross-sectional ranks while keeping
    optional features from deleting stocks or weeks. Pass ``neutral_fill=False``
    when raw post-rank missingness must be inspected.
    """
    out = df.copy()
    grouped = out.groupby("date")[cols]
    out[cols] = grouped.rank(pct=True).mul(2).sub(1.0).where(out[cols].notna())
    if neutral_fill:
        out[cols] = out[cols].fillna(0.0)
    return out
