# Research notes

One-page notes on what each paper implies for stocks_ml specifically (features,
labels, CV, evaluation). PDFs live in `papers/`. Ranked action list:
`recommendations-2026-08.md`.

Written in August 2026 against the weekly pipeline retired on 2026-09-01, so
the code they cite (`backtest/metrics.py`, `pipelines.py`, `models/candidates.py`,
`reports/backtest.md`, `ledger.json`) resolves at tag `legacy-final`, not on
main. What came of them still stands: recommendation #3 is the trials ledger
(`models/trials_ledger.json`); the #1 and #2 feature families are computed but
sit in `PENDING_ABLATION_FEATURES` (`features/panel.py`), never admitted.

| Note | Paper | Main implication for us |
|---|---|---|
| [note-gu-kelly-xiu-2020.md](note-gu-kelly-xiu-2020.md) | Gu, Kelly & Xiu 2020 | IC ~0.02 is the right ceiling for large caps; price-trend features dominate; test longer training windows |
| [note-kelly-malamud-zhou-complexity.md](note-kelly-malamud-zhou-complexity.md) | Kelly, Malamud & Zhou | One principled new family: random-Fourier-feature ridge with heavy shrinkage |
| [note-bailey-lopez-de-prado-2014-dsr.md](note-bailey-lopez-de-prado-2014-dsr.md) | Bailey & López de Prado 2014 | Our DSR uses the wrong variance (single-path, not cross-trial); need a trials ledger |
| [note-green-hand-zhang-2017.md](note-green-hand-zhang-2017.md) | Green, Hand & Zhang 2017 | Missing survivors we can compute from EDGAR: net issuance, R&D/mktcap, nincr |
| [note-novy-marx-2013-profitability.md](note-novy-marx-2013-profitability.md) | Novy-Marx 2013 | `f_gross_profitability` is right, but its horizon fits `label_4w`, not weekly |
| [note-novy-marx-2012-intermediate-momentum.md](note-novy-marx-2012-intermediate-momentum.md) | Novy-Marx 2012 | Biggest feature gap: no momentum beyond 12 *weeks*; add 26w/52w/intermediate |
| [note-jegadeesh-titman-1993.md](note-jegadeesh-titman-1993.md) | Jegadeesh & Titman 1993 | New momentum features should skip the most recent week (reversal contamination) |
| [note-bernard-thomas-1989-pead.md](note-bernard-thomas-1989-pead.md) | Bernard & Thomas 1989 | `f_pead` is the price shadow; true SUE from EDGAR earnings is buildable |
| [note-harvey-liu-zhu-2016.md](note-harvey-liu-zhu-2016.md) | Harvey, Liu & Zhu 2016 | Adopt features at paired ΔIC t ≥ 3, family-level, logged as trials |
| [note-timmermann-2006-forecast-combinations.md](note-timmermann-2006-forecast-combinations.md) | Timmermann 2006 | Equal-weight rank blends of *different* families (XGB+ENet); trim staggered ensemble |
| [note-hou-xue-zhang-2020-replicating-anomalies.md](note-hou-xue-zhang-2020-replicating-anomalies.md) | Hou, Xue & Zhang 2020 | The veto list: momentum/investment/profitability survive microcap screens; liquidity anomalies don't |
| [note-cohen-malloy-pomorski-2012-insiders.md](note-cohen-malloy-pomorski-2012-insiders.md) | Cohen, Malloy & Pomorski 2012 | Routine-vs-opportunistic split needs `owner_cik` (one ingestion change); retry insiders at `label_4w` |
| [note-short-interest-cross-section.md](note-short-interest-cross-section.md) | Boehmer, Jones & Zhang 2008 + SI literature | We have the right pair (ratio + days-to-cover); the missing transform is *changes* |

The S&P 500-only caveat runs through every note: most published effect sizes are
small/microcap-driven, and each note flags whether the result survives
value-weighting / large-cap screens (Hou–Xue–Zhang is the systematic check).
