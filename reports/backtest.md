# Backtest report

Champion model: **xgb_tuned** · strategies × candidates tried: **301** (used to deflate Sharpe)

## Headline ($100 invested at start)

| strategy | $100 → | CAGR | Sharpe | Deflated Sharpe | Sortino | max DD | worst week | underwater (d) | costs $ | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| equal_topk | $1,558 | 13.8% | 0.58 | 0.82 | 0.94 | 68.1% | -26.4% | 1328 | 193.04 | 1111 |
| vol_scaled | $113 | 0.6% | 0.13 | 0.12 | 0.19 | 25.6% | -5.9% | 6794 | 9.75 | 1111 |
| kelly | $366 | 6.3% | 0.44 | 0.61 | 0.66 | 50.2% | -17.1% | 2881 | 30.05 | 1111 |
| kelly_spy | $845 | 10.6% | 0.63 | 0.87 | 0.97 | 50.2% | -17.1% | 1123 | 47.59 | 1111 |
| topk_spy | $3,406 | 18.0% | 0.68 | 0.92 | 1.11 | 67.6% | -26.4% | 921 | 320.85 | 1111 |
| spy_hold | $930 | 11.1% | 0.65 | – | 1.01 | 55.2% | -19.8% | 1773 | – | – |
| cash | $144 | 1.7% | 14.57 | – | 28,157.27 | 0.0% | -0.0% | 16 | – | – |

## Regime-sliced performance (ann. Sharpe / hit rate)

| strategy | bull | bear | high_vol | low_vol |
|---|---|---|---|---|
| equal_topk | 1.25 / 47.4% | -0.20 / 45.1% | -0.14 / 44.3% | 2.11 / 49.1% |
| vol_scaled | 0.71 / 7.3% | -1.16 / 16.4% | -0.81 / 7.9% | 0.91 / 10.3% |
| kelly | 1.52 / 49.7% | -0.96 / 46.7% | -0.67 / 45.7% | 2.99 / 51.9% |
| kelly_spy | 1.73 / 56.4% | -0.84 / 48.9% | -0.61 / 49.3% | 3.46 / 59.4% |
| topk_spy | 1.44 / 55.1% | -0.22 / 48.8% | -0.17 / 48.9% | 2.51 / 57.8% |
| spy_hold | 1.84 / 57.1% | -0.94 / 47.6% | -0.68 / 48.6% | 3.85 / 60.5% |
| cash | 14.38 / 100.0% | 16.14 / 98.3% | 12.74 / 99.3% | 16.37 / 99.9% |

## Stress windows (total return)

| window | equal_topk | vol_scaled | kelly | kelly_spy | topk_spy | spy_hold | cash |
|---|---|---|---|---|---|---|---|
| GFC | -36.4% | -13.8% | -33.4% | -33.4% | -37.6% | -36.6% | 0.2% |
| Q4 2018 | -20.2% | 0.0% | -11.3% | -19.4% | -23.0% | -13.8% | 0.6% |
| COVID crash | -18.3% | 0.0% | -28.7% | -27.5% | -29.1% | -17.0% | 0.1% |
| 2022 bear | -12.2% | 0.0% | -8.1% | -8.1% | -14.3% | -18.6% | 2.0% |

## Holdout period (≥ 2024-07-19, never used for model selection — the primary strategy comparison)

| strategy | $100 → | total return | ann. Sharpe | max DD | worst week |
|---|---|---|---|---|---|
| equal_topk | $200 | 100.1% | 1.14 | 27.0% | -14.4% |
| vol_scaled | $100 | 0.0% | – | 0.0% | 0.0% |
| kelly | $113 | 12.6% | 0.50 | 15.8% | -7.6% |
| kelly_spy | $115 | 15.2% | 0.57 | 15.8% | -7.6% |
| topk_spy | $206 | 105.6% | 1.17 | 27.0% | -14.4% |
| spy_hold | $139 | 38.6% | 1.05 | 18.8% | -9.1% |
| cash | $108 | 8.2% | 157.31 | 0.0% | 0.1% |

![holdout equity curves](equity_holdout.png)

## Recent five years ($100 at 2021-07-17; overlaps model-selection window — see holdout for the clean test)

| strategy | $100 → | CAGR | Sharpe | max DD |
|---|---|---|---|---|
| equal_topk | $183 | 12.9% | 0.57 | 32.6% |
| vol_scaled | $100 | 0.0% | – | 0.0% |
| kelly | $114 | 2.7% | 0.26 | 19.7% |
| kelly_spy | $164 | 10.5% | 0.73 | 19.7% |
| topk_spy | $260 | 21.0% | 0.79 | 33.5% |
| spy_hold | $187 | 13.4% | 0.82 | 24.5% |
| cash | $119 | 3.6% | 33.40 | 0.0% |

## Honesty notes

- MinTRL: with N=301 trials, the expected-max weekly SR of a zero-skill strategy is 0.052; a live strategy matching the holdout live-strategy weekly SR (0.181) needs **≥ 177 weeks** of shadow ledger to certify SR>0 at 95%.
- Sharpe deflation uses N=301 trials and the ledger's empirical cross-trial SR variance (0.0171).
- Regime flags (SPY 200d SMA, VIX median) use full-sample statistics; they are reporting lenses, not tradable signals.
- Fundamentals are sparse before ~2009 (EDGAR XBRL phase-in).
- Delisted tickers missing from the free price source are absent from the panel; residual survivorship bias is reported in the ingestion manifest.
- Positions whose ticker stops trading are liquidated at the last available (forward-filled) price — optimistic for bankruptcies.

![equity curves](equity.png)