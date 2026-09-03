from __future__ import annotations

import pandas as pd

CORRUPT_RATIO = 1.9          # adjusted data should contain no split-sized jumps
CORRUPT_MIN_EVENTS = 2       # one genuine mega-move is possible; repeats are corruption


def drop_corrupt_series(prices: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Drop tickers whose ADJUSTED closes jump by a split-sized ratio (>1.9x or
    <1/1.9x day-over-day) two or more times. Properly adjusted series contain no
    split jumps, so repeated occurrences signal a corrupted adjustment history.
    A single genuine crash/bounce (e.g., AIG March 2009) survives the threshold.
    Part of the panel recipe (features/panel.py), so it stays even though the
    Sharadar world has never tripped it."""
    ratios = (prices.sort_values(["ticker", "date"])
                    .groupby("ticker")["close"].pct_change().add(1.0))
    extreme = (ratios > CORRUPT_RATIO) | (ratios < 1.0 / CORRUPT_RATIO)
    events = extreme.groupby(prices["ticker"]).sum()
    corrupt = sorted(events[events >= CORRUPT_MIN_EVENTS].index)
    return prices[~prices["ticker"].isin(corrupt)].reset_index(drop=True), corrupt
