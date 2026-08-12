# Green, Hand & Zhang (2017) — "The Characteristics that Provide Independent Information about Average U.S. Monthly Stock Returns"

PDF: `papers/green-hand-zhang-2017-characteristics.pdf` — note this is the free
2014 working-paper draft ("The Remarkable Multidimensionality in the Cross-Section
of Expected U.S. Stock Returns", Wharton Jacobs Levy copy). The published RFS 2017
version (SSRN 2262374 is paywalled to scripts) tightens the microcap and
multiple-testing corrections; headline numbers below are from the published version.

## What the paper shows

94 characteristics thrown *simultaneously* into Fama–MacBeth regressions, with
microcaps down-weighted and a Harvey–Liu–Zhu-style t≥3 hurdle. Results: only ~12
characteristics carry independent information in non-microcap stocks 1980–2014,
and predictability collapses around 2003 — after which roughly two survive.
Survivors in non-microcaps include: book-to-market, share turnover volatility,
return volatility, 1-month momentum/reversal, 12-month momentum, **net share
issuance (chcsho)**, **R&D-to-market-cap**, **earnings-announcement return (ear)**,
**number of consecutive earnings increases (nincr)**, cash holdings, and zero-trade
days.

## What it implies for stocks_ml

- **The joint-regression logic is our daily reality.** A gradient-boosted model on
  rank features *is* a joint conditional test — univariate anomaly literature
  over-counts, and GHZ explain why most of the 94 add nothing marginal. This
  supports our ablation-on-identical-folds policy (missing-data policy #4) and
  argues against adopting features because they "are a known anomaly."
- **We hold most survivors already** (`f_book_to_market`, `f_vol_*`,
  `f_idio_vol_60d`, `f_mom_4w`, `f_mom_12w`, `f_abn_volume`, `f_dollar_vol`,
  `f_pead` ≈ their *ear*). The gaps that are computable from data we already
  ingest (EDGAR) are precisely the interesting ones:
  **net share issuance** (change in split-adjusted shares outstanding — shares
  outstanding is in EDGAR company facts), **R&D/market cap**, and **nincr**
  (consecutive quarterly EPS increases). All three survived their non-microcap,
  post-multiple-testing filter; none is in `panel.py` today.
- **The post-2003 decay result sets our prior.** Our entire evaluation window
  (2015-03 onward) sits in the regime where GHZ find almost nothing survives in
  non-microcaps. That is the strongest published argument that (a) IC ≈ 0.02 is a
  respectable ceiling for this universe/era, and (b) fold-stability (missing-data
  policy #6) should be weighted over mean IC, since anything that "worked" mostly
  pre-2003 can't help us anyway.
- **Microcap warning doesn't bite here** — S&P 500 constituents only — but the same
  mechanism (a few extreme names driving an equal-weighted result) reappears as
  our concentrated top-8 timing-luck problem documented in the tie-guard note.
  Their fix (down-weighting the extremes' influence when *judging* a signal) is an
  argument for continuing to judge candidates by rank IC across all ~500 names,
  never by top-8 backtest dollars.

## Concrete candidates

1. EDGAR feature bundle: `f_net_issuance` (YoY % change in shares outstanding),
   `f_rd_to_mktcap`, `f_nincr` — filing-dated, neutral-filled per policy #3.
2. Keep `f_pead` (their *ear*) — externally validated; consider the SUE complement
   from the Bernard–Thomas note as its fundamental-side twin.
