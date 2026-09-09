# The nominal-basis rebuild — registration

Written 2026-09-09, before any number, on the owner's go ("proceed with what
you are proposing", 2026-09-09) after the split-leak finding: the store's
`close` is Sharadar's `closeadj`, restated through the download date, so
every cross-sectional price LEVEL at week t encodes the stock's future
splits, and SF1 per-share fields carry the same restatement. The pure
future-adjustment factor ranked members with NW t = -7.0 on 2006-2015; the
nominal price ranked them at t = 0.5; the champion's K=16 scores correlate
-0.29 with the factor and their IC falls from 0.016 (t 2.0) to 0.0075
(t 0.9) when the factor is removed. The bundle was screened on the leak
(r_sf_bvps in 20 of 40 formulas, r_close in 10, f_log_mktcap in 9). The
live job scores at factor 1 for every name, so the leak cannot pay live in
either direction: the record and the deployed strategy are different
strategies. This program measures the honest one.

## The basis recipe

Returns stay on the total-return basis; levels move to what the tape showed
that day. Nominal is exactly the live job's arithmetic, where the future
factor is 1 for every name (emulate-live mandate).

- The store's `prices` table carries two new columns straight from raw SEP:
  `closeunadj` (nominal) and `close_split` (SEP's split-adjusted close).
  `open`/`close` (closeadj basis) are unchanged: labels, momentum,
  volatility, hi/lo, beta, ballast, fills, and the ledger's units do not
  move.
- `r_close`, `x_price_level`: `closeunadj`.
- `r_sf_<field>` per-share LEVELS (bvps, dps, eps, epsdil, epsusd, sps,
  fcfps, ncfps, tbvps, sharesbas, shareswa, shareswadil — those present):
  stored value x split_factor(t), split_factor = closeunadj/close_split at
  the panel date (the cumulative splits between t and the download, the
  exact restatement Sharadar applied). `_yoy` fields are same-basis ratios:
  unchanged.
- `f_sf_earnings_yield` = epsusd / close_split; `f_sf_book_to_market` =
  bvps / close_split; `f_sf_fcf_yield` and `f_sf_sales_to_price` = dollar
  total / (close_split x sharesbas). Dividing the split-restated per-share
  by the split-adjusted close cancels the factor exactly and equals the
  live computation.
- Free-EDGAR block: mktcap = as-reported shares x `closeunadj` (feeds
  f_log_mktcap, f_earnings_yield, f_book_to_market, f_sales_to_price,
  f_cf_to_price).
- `f_dollar_vol`, `f_amihud_4w/12w`: `closeunadj` x raw volume (the current
  closeadj x raw-volume product mixes bases).
- `f_short_ratio`/`f_short_dtc`: already nominal-consistent (as-reported SI
  / raw volume / as-reported shares); verified in the gate, unchanged.
- Gate flag `Config.price_basis`: "closeadj" (default; byte-identical
  status quo, the deployed champion and the live job stay on it) vs
  "nominal" (the above). Adoption of a nominal-basis champion is the
  owner's decision after the board.

## Data fixes riding in the same rebuild (they change the panel)

1. EDGAR extraction takes the UNION of a concept's tags instead of the
   first tag that ever yielded data (MSFT revenues froze in 2011; 152/478
   members' revenues stale > 400 days), and the as-of join gets a 400-day
   tolerance like the f_sf block. Requires a companyfacts refetch: free
   SEC API, ~1,200 requests.
2. An empty companyfacts payload or an empty SF2 page keeps the stored
   rows instead of erasing them.
3. 8-K dedupe by (ticker, accession) instead of accession alone (dual-class
   listings: GOOGL had 0 rows to GOOG's 114); targeted refetch of CIKs
   shared by two tickers.
4. Fundamentals as-of: reportperiod must be monotone (an older comparative
   filed later may not supersede a newer value; ~0.7% of rows) and ARQ
   duplicate (ticker, reportperiod) rows dedupe to the last filed before
   the shift(4) YoY (5.7% of rows lost YoY to misalignment).
5. Short-interest symbol normalization verified against the panel
   (BRK.B/BF.B).

NOT included: the delisting-within-horizon universe gap (slice_row requires
a finite 4-week forward fill; live uses a 7-day-close rule). That is a
selection-rule change the owner has not decided; this program keeps
slice_row as it stands so the clean board is comparable layer for layer.

## The gate (pass before any performance number is read)

On the rebuilt nominal panel:

- Identity: r_close equals raw SEP closeunadj (|rel err| < 1e-6) on every
  member-week of 120 sampled weeks, 2006-2015 (selection.sample_weeks).
- Spot restatements: NVDA r_sf_bvps mid-2015 reads ~7.6 (stored: 0.19);
  AAPL f_log_mktcap mid-2013 implies ~$350-450B (panel today: $11B).
- Diagnostics reported, not gated (true nominal levels correlate with the
  factor through genuine momentum): Spearman of each rebuilt input with the
  future-adjustment factor, before/after.

Fail any check: stop, fix, re-run the gate. No walk starts before the gate
passes.

## The program (all K=16, replication.py seeding, new experiment dirs with
recipe stamps, every grade window ending at selection.HOLDOUT_START
exclusive, sp500 row beside every number, holdout untouched)

1. `nominal_clean_2006_2015` — base features only (ctx.extra = []), every
   rank week 2006-01-06 -> 2015-12-31, preds saved per copy.
2. The generated-bundle screen (ops/openfe_arm.py procedure, unchanged
   parameters) fit strictly inside 2006-2015 on nominal inputs -> up to 40
   g_ formulas. Same bar for challenger and incumbent: the bundle is
   admitted by the layer's own argmax metric (cost-adjusted compounded %/yr
   at the chosen book on 2006-2015, calendar chains), t stats reported not
   gated. The generated bundle is the clean replacement (no asterisk); the
   leaky ideas bundle is never in the arm.
3. `nominal_bundle_2006_2015` — the same sixteen copies with the screened
   bundle.
4. The 2006-2015 argmax line (clean or bundle) extends once to
   2016-01 -> 2024-07-12 (`nominal_*_2016_2024`); the other line may extend
   for the board but its 2016-2024 number decides nothing.
5. The board: clean line and bundle line at the champion's standing
   settings AND at the cascade re-decided on 2006-2015 (book by compounded
   %/yr, floor/stop/cap by Sharpe at the picks), SPY beside each. The old
   record ($4,256 / $900 / $473 vs SPY $608/$310/$197) is quoted as
   contaminated-basis history only.
6. Nothing reads 2024-07-19+. Deployment changes (champion spec, live
   basis, funding) are the owner's decisions after the board.

## Cost

EDGAR refetch minutes-to-hours (free, rate-limited); prices-table and panel
rebuild tens of minutes; one screen; three-to-four 16-copy walk segments
(about 1.5-2x last week's K=16 program) in the background on this Mac.
