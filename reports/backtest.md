# Backtest report

Champion model: **xgb** · strategies × candidates tried: **12** (used to deflate Sharpe)

## Headline ($100 invested at start)

| strategy | $100 → | CAGR | Sharpe | Deflated Sharpe | Sortino | max DD | worst week | underwater (d) | costs $ | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| equal_topk | $12,170 | 25.1% | 0.76 | 0.97 | 1.28 | 66.7% | -33.7% | 805 | 2,068.53 | 278 |
| vol_scaled | $328 | 5.7% | 0.53 | 0.78 | 0.81 | 33.3% | -13.9% | 2338 | 84.25 | 278 |
| kelly | $949 | 11.1% | 0.60 | 0.86 | 0.93 | 50.5% | -20.0% | 806 | 179.45 | 278 |
| spy_hold | $959 | 11.1% | 0.65 | – | 1.02 | 55.2% | -19.8% | 1773 | – | – |
| cash | $145 | 1.7% | 14.68 | – | 28,391.73 | 0.0% | -0.0% | 16 | – | – |

## Regime-sliced performance (ann. Sharpe / hit rate)

| strategy | bull | bear | high_vol | low_vol |
|---|---|---|---|---|
| equal_topk | 1.29 / 54.2% | 0.12 / 48.9% | 0.00 / 48.7% | 2.34 / 56.6% |
| vol_scaled | 1.24 / 38.4% | -1.03 / 34.5% | -0.80 / 29.4% | 2.06 / 44.2% |
| kelly | 1.57 / 56.4% | -0.58 / 48.6% | -0.47 / 49.4% | 3.21 / 59.1% |
| spy_hold | 1.86 / 57.1% | -0.94 / 47.9% | -0.68 / 48.6% | 3.81 / 60.5% |
| cash | 14.45 / 100.0% | 16.48 / 98.3% | 12.74 / 99.3% | 16.56 / 99.9% |

## Stress windows (total return)

| window | equal_topk | vol_scaled | kelly | spy_hold | cash |
|---|---|---|---|---|---|
| GFC | -8.4% | -5.3% | -30.5% | -36.6% | 0.2% |
| Q4 2018 | -20.5% | -10.7% | -14.4% | -13.8% | 0.6% |
| COVID crash | -29.3% | -33.1% | -27.1% | -17.0% | 0.1% |
| 2022 bear | -12.0% | 0.0% | -13.5% | -18.6% | 2.0% |

## Holdout period (≥ 2024-07-19, never used for model selection)

| strategy | window return |
|---|---|
| equal_topk | 48.6% |
| vol_scaled | 0.0% |
| kelly | 16.8% |
| spy_hold | 40.7% |
| cash | 8.1% |

## Honesty notes

- Sharpe deflation assumes 12 strategy/model trials.
- Regime flags (SPY 200d SMA, VIX median) use full-sample statistics; they are reporting lenses, not tradable signals.
- Fundamentals are sparse before ~2009 (EDGAR XBRL phase-in).
- Delisted tickers missing from the free price source are absent from the panel; residual survivorship bias is reported in the ingestion manifest.
- Positions whose ticker stops trading are liquidated at the last available (forward-filled) price — optimistic for bankruptcies.

![equity curves](equity.png)