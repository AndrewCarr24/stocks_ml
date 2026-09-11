# 95% confidence intervals for the clean line

Computed 2026-09-11 on the delisting-honest world's clean nominal line
(K=16, champion settings, windows ending 2024-07-19 exclusive; holdout
untouched). Scripts: scratchpad `ci_seed.py` (seed draws) and `ci_summary.py`
(bootstrap + nesting); artifacts `ci_seed_paths.parquet`, `ci_summary_block8.csv`.

Two uncertainties, kept separate and then nested:

- **Seed noise** — same history, different model fit: 200 K=16 ensembles drawn
  by resampling the 16 saved copies with replacement, each re-simulated under
  the live ledger rules.
- **History noise** — same model, different history: circular block bootstrap
  of the weeks (8-week blocks; 13-week blocks read the same), strategy and
  SPY resampled on the SAME weeks, 4,000 resamples of the point path.
- **Combined** — 20 history resamples of each of the 200 seed paths.

| window | metric | point | seed-only 95% | history-only 95% | combined 95% |
|---|---|---|---|---|---|
| 2016-2024 | excess CAGR vs SPY | +4.2%/yr | +3.4 … +8.5 | −5.1 … +14.2 | −3.5 … +16.3 |
| 2016-2024 | CAGR | +19.2%/yr | +18.2 … +24.1 | +1.5 … +39.6 | +2.9 … +41.8 |
| 2016-2024 | terminal $100 | $449 | $419 … $635 | $114 … $1,726 | $128 … $1,980 |
| pre-holdout (2006-2024) | excess CAGR vs SPY | +2.8%/yr | +1.9 … +5.8 | −3.4 … +9.0 | −2.4 … +10.6 |
| pre-holdout | CAGR | +13.4%/yr | +12.4 … +16.7 | +1.3 … +26.4 | +2.5 … +27.8 |
| pre-holdout | terminal $100 | $1,028 | $877 … $1,751 | $127 … $7,705 | $158 … $9,440 |
| 2006-2015 | excess CAGR vs SPY | +1.5%/yr | −1.5 … +4.7 | −6.3 … +10.3 | −6.4 … +11.6 |
| 2006-2015 | terminal $100 | $229 | $169 … $309 | $48 … $995 | $50 … $1,078 |
| sp500 | 2006-2015 / 2016-2024 / pre-holdout | $197 (+7.0%/yr) / $316 (+14.4%/yr) / $621 (+10.4%/yr) | | | |

Paired weekly log-excess t-statistic: 0.4 (2006-2015), 0.8 (2016-2024),
0.8 (pre-holdout). Share of nested resamples with positive excess: 0.7 / 0.9 /
0.9.

Reading:

- History noise is about five times wider than seed noise. The point K=16
  path sits at the low end of its own seed distribution (most re-seeded
  ensembles land above $449 on 2016-2024).
- The edge over SPY is positive at the point and in ~90% of resampled
  histories, but its 95% interval includes zero on 8.5 years and on 18 years.
  At this tracking error (~15%/yr against SPY for a 6-name book) a t of 2 at
  the measured edge would need on the order of 50 years of record. This is a
  property of any concentrated book measured against a 500-name index, not
  of this model.
- Selection stays mechanical (t reported, not gated — owner's rule); this is
  the honest description of what a dollar is being bet on: an expected few
  %/yr over SPY, a band that comfortably includes no edge, and 40-60%
  drawdowns along the way.
