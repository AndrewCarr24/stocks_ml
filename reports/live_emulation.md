# Champion as deployed

Same 2006-2024 picks (5y window, 4w label, K=4) graded by `selection.simulate`,
which runs the live job's ledger (`stocks_ml.ledger`: target weights filled at
the next session's open, 5 bp a side on every trade, 0.5%-of-NAV dust threshold,
NAV marked at Friday's close) — the official record — and once more with fills
priced at the decision date's close. The two differ only in fill timing.

## 2006-01 -> 2024-06 (pre-holdout)

| series | grade |
|---|---|
| as deployed (fills at the next open) | $1,553 (+15.9%/yr, SR 0.72, DD 62%) |
| fills at the decision close | $1,580 (+16.1%/yr, SR 0.72, DD 62%) |
| sp500 (close-to-close) | $623 (+10.4%/yr, SR 0.64, DD 55%) |

## 2006 -> 2012

| series | grade |
|---|---|
| as deployed (fills at the next open) | $141 (+5.0%/yr, SR 0.31, DD 62%) |
| fills at the decision close | $145 (+5.4%/yr, SR 0.33, DD 62%) |
| sp500 (close-to-close) | $130 (+3.8%/yr, SR 0.28, DD 55%) |

## 2013 -> 2024-06

| series | grade |
|---|---|
| as deployed (fills at the next open) | $1,103 (+23.2%/yr, SR 1.01, DD 42%) |
| fills at the decision close | $1,092 (+23.1%/yr, SR 1.00, DD 44%) |
| sp500 (close-to-close) | $479 (+14.6%/yr, SR 0.93, DD 32%) |

## 2021 -> 2024-06

| series | grade |
|---|---|
| as deployed (fills at the next open) | $168 (+16.1%/yr, SR 0.77, DD 30%) |
| fills at the decision close | $173 (+16.9%/yr, SR 0.81, DD 28%) |
| sp500 (close-to-close) | $155 (+13.4%/yr, SR 0.86, DD 24%) |

## Fill timing, 2006-01 -> 2024-06

- decision close -> next open: -0.11%/yr ($1,580 -> $1,553), +0.00 on Sharpe

Weekend gap at the 963 rotations (mean open-after-signal / close-at-signal - 1): incoming names +3.1 bp, outgoing names +5.7 bp, difference -2.7 bp per rotation. A fill at the decision close credits the incoming gap to the book; the deployed book earns the outgoing one. One sleeve of four rotates per week at 70% book weight, so the difference is worth about -0.24%/yr before compounding.

Trades as deployed: 9727 fills over 964 weeks; fees $164.66 on a $100 start.

## Notes

- Training labels are open-to-open (first session after the rank date to 20 sessions later): the model learns the returns a Monday-open fill earns, and the engine now grades on the same basis.
- Before 2026-09 the engine bought at the rank date's close, rebalanced the whole book for free every week and charged 10 bp on the rotated sleeve only; it graded these picks $1,644 (+16.3%/yr, SR 0.72; ledger champion_r5_7030_cap2_regraded before this change). The move to the live rules is the champion's headline change from that number.
- Holdings weeks with no cached prediction are held through; a sleeve that missed its rotation catches up (STALE_WEEKS), as in the live job.
