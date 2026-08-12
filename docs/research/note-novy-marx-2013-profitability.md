# Novy-Marx (2013) — "The Other Side of Value: The Gross Profitability Premium"

PDF: `papers/novy-marx-2013-gross-profitability.pdf` (NBER WP 15940)

## What the paper shows

Gross profitability — (revenue − COGS) / **total assets** — predicts returns about
as strongly as book-to-market, with three properties that matter for us: it works
**within large caps** (most anomalies don't), it is a *growth-side* signal nearly
uncorrelated with (slightly negatively correlated to) value so the two combine
almost frictionlessly, and it is extremely **slow-moving** — portfolios formed on
years-old gross-profit data still earn the premium, turnover is tiny, and the
spread accrues over months-to-years, not days.

## What it implies for stocks_ml

- **We already hold the feature, and the construction checks out.** The premium
  is specifically about scaling by *total assets*, not equity — scaling by book
  equity mixes value back in and weakens it (paper, §1). Verified:
  `features/fundamentals.py:66` computes `f_gross_profitability =
  gross_profit / assets` from the XBRL `gross_profit` (duration, 10-K) and
  `assets` (instant) concepts — exactly GP/AT. Both inputs are in the mapped
  concept lists (`DURATION_CONCEPTS` / `INSTANT_CONCEPTS`), filing-dated with the
  next-day availability shift. One residual caveat: firms that don't tag
  `GrossProfit` directly (some report only revenue and COGS) come through as
  missing — coverage stats per policy #2 would show whether a
  revenues-minus-COGS fallback is worth adding.
- **The horizon mismatch is the real content.** A signal whose spread accrues over
  quarters is nearly invisible in a forward *5-trading-day* median-relative label —
  it contributes at most a tiny weekly tilt, which is presumably why fundamentals
  rank below price features for our champion. This is the strongest paper-backed
  argument for the monthly-cadence pipeline (`label_4w`, already in wave 1 of the
  league): profitability and the other EDGAR features are exactly the family that
  should differentially benefit from the 4-week horizon. Concretely testable: an
  ablation of the fundamentals family on the weekly vs the 4-week label — the
  paper predicts a bigger marginal IC on `label_4w`.
- **Value + profitability is the endorsed combination.** Novy-Marx's headline
  portfolio ranks on the *sum* of the two ranks. In our setup, both enter the
  model jointly, which subsumes the fixed 50/50 combination — but it also means a
  degenerate refit (the tie-guard scenario) loses both tilts at once. If the
  monthly pipeline ever needs a non-ML fallback (champion rule already falls back
  to momentum), rank(`f_gross_profitability`) + rank(`f_book_to_market`) is the
  natural fundamentals-side baseline, and this paper is the citation for it.
- **Large-cap validity is unusual and valuable.** Most cross-sectional results die
  in the S&P 500 (see Green–Hand–Zhang note); Novy-Marx documents the profitability
  spread among the largest 500 explicitly. Among our fundamentals, this one has
  the best prior for surviving in our universe — a reason to protect its data
  quality (COGS availability in EDGAR pre-2010 is sparse; coverage stats per
  missing-data policy #2 matter here).

## Concrete candidates

1. Verify `f_gross_profitability` = (revenue − COGS)/total assets; fix if the
   denominator drifted.
2. Fundamentals-family ablation on `label_4w` vs weekly label (identical folds);
   adopt in the monthly pipeline if the 4-week marginal IC confirms the horizon story.
3. Optional: rank-sum value+profitability baseline strategy for the pipeline league.
