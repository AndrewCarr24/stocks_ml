# Backtest report

Champion model: **lgbm_tuned** · strategies × candidates tried: **325** (used to deflate Sharpe)

## Headline ($100 invested at start)

| strategy | $100 → | CAGR | Sharpe | Deflated Sharpe | Sortino | max DD | worst week | underwater (d) | costs $ | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| equal_topk | $738 | 9.8% | 0.45 | 0.69 | 0.75 | 64.9% | -30.5% | 1922 | 252.45 | 1117 |
| vol_scaled | $139 | 1.6% | 0.19 | 0.24 | 0.29 | 25.4% | -11.5% | 2926 | 50.26 | 1117 |
| kelly | $694 | 9.5% | 0.57 | 0.85 | 0.89 | 53.0% | -18.3% | 825 | 105.12 | 1117 |
| kelly_spy | $851 | 10.5% | 0.60 | 0.87 | 0.93 | 53.0% | -18.3% | 825 | 120.92 | 1117 |
| topk_spy | $781 | 10.1% | 0.46 | 0.69 | 0.75 | 67.7% | -30.5% | 1816 | 267.18 | 1117 |
| spy_hold | $960 | 11.1% | 0.65 | – | 1.02 | 55.2% | -19.8% | 1773 | – | – |
| cash | $145 | 1.7% | 14.66 | – | 28,423.70 | 0.0% | -0.0% | 16 | – | – |

## Regime-sliced performance (ann. Sharpe / hit rate)

| strategy | bull | bear | high_vol | low_vol |
|---|---|---|---|---|
| equal_topk | 1.00 / 52.2% | -0.22 / 46.2% | -0.32 / 46.5% | 2.02 / 54.6% |
| vol_scaled | 0.77 / 42.0% | -1.43 / 39.2% | -0.93 / 38.2% | 1.42 / 44.1% |
| kelly | 1.57 / 55.0% | -0.72 / 46.3% | -0.53 / 48.3% | 3.09 / 57.3% |
| kelly_spy | 1.59 / 56.0% | -0.61 / 48.9% | -0.49 / 49.8% | 3.18 / 58.4% |
| topk_spy | 1.07 / 54.4% | -0.30 / 49.1% | -0.37 / 49.2% | 2.17 / 56.7% |
| spy_hold | 1.84 / 57.1% | -0.94 / 47.6% | -0.69 / 48.6% | 3.87 / 60.4% |
| cash | 14.49 / 100.0% | 16.14 / 98.3% | 12.80 / 99.3% | 16.50 / 99.9% |

## Stress windows (total return)

| window | equal_topk | vol_scaled | kelly | kelly_spy | topk_spy | spy_hold | cash |
|---|---|---|---|---|---|---|---|
| GFC | -13.9% | -2.4% | -23.9% | -23.9% | -27.0% | -36.6% | 0.2% |
| Q4 2018 | -21.8% | -13.3% | -14.5% | -16.4% | -21.8% | -13.8% | 0.6% |
| COVID crash | -20.0% | -18.3% | -26.1% | -25.1% | -24.2% | -17.0% | 0.1% |
| 2022 bear | -37.8% | -14.3% | -14.2% | -14.2% | -40.6% | -18.6% | 2.0% |

## Holdout period (≥ 2024-08-30, never used for model selection — the primary strategy comparison)

| strategy | $100 → | total return | ann. Sharpe | max DD | worst week |
|---|---|---|---|---|---|
| equal_topk | $122 | 22.2% | 0.46 | 36.4% | -13.6% |
| vol_scaled | $100 | 0.0% | – | 0.0% | 0.0% |
| kelly | $111 | 11.0% | 0.42 | 18.8% | -8.4% |
| kelly_spy | $111 | 11.0% | 0.42 | 18.8% | -8.4% |
| topk_spy | $126 | 25.9% | 0.50 | 36.5% | -13.6% |
| spy_hold | $139 | 39.3% | 1.09 | 18.8% | -9.1% |
| cash | $108 | 8.1% | 190.17 | 0.0% | 0.0% |

![holdout equity curves](equity_holdout.png)

## Recent five years ($100 at 2021-08-31; overlaps model-selection window — see holdout for the clean test)

| strategy | $100 → | CAGR | Sharpe | max DD |
|---|---|---|---|---|
| equal_topk | $89 | -2.3% | 0.07 | 51.6% |
| vol_scaled | $85 | -3.1% | -0.73 | 17.1% |
| kelly | $130 | 5.3% | 0.41 | 23.2% |
| kelly_spy | $141 | 7.1% | 0.50 | 23.2% |
| topk_spy | $91 | -1.9% | 0.09 | 51.1% |
| spy_hold | $182 | 12.7% | 0.78 | 24.5% |
| cash | $120 | 3.7% | 36.27 | 0.0% |

## Honesty notes

- MinTRL: with N=325 trials, the expected-max weekly SR of a zero-skill strategy is 0.049; a live strategy matching the holdout live-strategy weekly SR (0.072) needs **≥ 5012 weeks** of shadow ledger to certify SR>0 at 95%.
- Sharpe deflation uses N=325 trials and the ledger's empirical cross-trial SR variance (0.0144).
- Regime flags (SPY 200d SMA, VIX median) use full-sample statistics; they are reporting lenses, not tradable signals.
- Fundamentals are sparse before ~2009 (EDGAR XBRL phase-in).
- Delisted tickers missing from the free price source are absent from the panel; residual survivorship bias is reported in the ingestion manifest.
- Positions whose ticker stops trading are liquidated at the last available (forward-filled) price — optimistic for bankruptcies.

![equity curves](equity.png)