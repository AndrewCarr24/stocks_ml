# Cohen, Malloy & Pomorski (2012) — "Decoding Inside Information"

PDF: `papers/cohen-malloy-pomorski-2012-decoding-inside-information.pdf` (NBER WP 16454)

## What the paper shows

Split Form 4 insider trades into **routine** (the insider traded in the same
calendar month in each of the three preceding years — think scheduled
diversification / 10b5-1-style behavior) and **opportunistic** (everything else,
for insiders with ≥3 years of history). Portfolios following *opportunistic*
trades earn large abnormal returns (~82 bps/month value-weighted long-short;
~180 bps equal-weighted) at a **monthly** horizon; routine trades predict nothing.
Opportunistic trades also predict news and are stronger where information
asymmetry is higher — i.e. the effect **attenuates in large, heavily-analyzed
firms** like ours, but the value-weighted result means it does not vanish there.

## What it implies for stocks_ml

- **It explains our null, and prescribes the retry.** Project history #5 records
  "insider data at weekly horizon" as a null. CMP say two things were stacked
  against that test: (a) *undifferentiated* Form 4 flow is mostly routine noise —
  their entire result is that the aggregate signal lives in a classifiable
  subset; (b) the drift accrues over months, not five days. The principled retry
  is opportunistic-only net buying at the `label_4w` horizon in the monthly
  pipeline — not more weekly insider features.
- **Feasibility with our data: possible, but needs one ingestion change.** The
  classification requires *per-insider* trade histories. The SEC quarterly bulk
  zips we already download contain the REPORTINGOWNER table (owner CIK) — but
  `data/insiders.py::extract_transactions` currently joins only SUBMISSION ↔
  NONDERIV_TRANS and keeps `[ticker, filed, trans_date, code, shares, value]`,
  discarding the owner's identity. Change: join REPORTINGOWNER on
  ACCESSION_NUMBER and carry `owner_cik` through `FORM4_COLS` (a re-ingest of the
  2006+ zips, which the idempotent `ingest` already knows how to do).
- **The classification is point-in-time by construction** — "routine" at time t
  depends only on that insider's trades in years t−3…t−1, so it composes with our
  next-calendar-day filing convention. With Form 4 data from 2006, insiders are
  classifiable from ~2009, comfortably before the 2015-03 evaluation start; the
  eligible-insider count in the early folds should still be reported per
  missing-data policy #2, since index joiners' insiders arrive with thin history.
- **Feature shape, if built:** `f_insider_opp_net_13w` — the existing
  `f_insider_net_13w` machinery (windowed net dollar value scaled by dollar
  volume) restricted to opportunistic insiders; optionally
  `f_evt_insider_opp_buy_2w` mirroring the existing event flag. Reuses
  `_windowed_asof_sum` unchanged.
- **Expectation setting for S&P 500:** their information-asymmetry gradient means
  we operate in the *weakest* segment of their effect, and HXZ-style
  value-weighted discipline shrinks it further. This is a moderate-effort,
  moderate-prior candidate — behind the momentum and SUE bundles, ahead of any
  new model family. Budget one family-level ablation (t ≥ 3 per the
  Harvey–Liu–Zhu rule) and let it die quickly if it doesn't clear.

## Concrete candidates

1. Ingestion: carry `owner_cik` from REPORTINGOWNER into the Form 4 parquet.
2. `f_insider_opp_net_13w` (+ event-flag variant), ablated on `label_4w` in the
   monthly pipeline; weekly label secondary.
