# Backtest report

Champion model: **xgb_tuned** · strategies × candidates tried: **27** (used to deflate Sharpe)

## Headline ($100 invested at start)

| strategy | $100 → | CAGR | Sharpe | Deflated Sharpe | Sortino | max DD | worst week | underwater (d) | costs $ | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| equal_topk | $949 | 11.0% | 0.47 | 0.56 | 0.77 | 76.4% | -32.2% | 4403 | 153.62 | 278 |
| vol_scaled | $138 | 1.5% | 0.25 | 0.19 | 0.37 | 25.9% | -6.7% | 6636 | 15.78 | 278 |
| kelly | $1,216 | 12.3% | 0.66 | 0.84 | 1.03 | 52.7% | -19.2% | 988 | 137.35 | 278 |
| spy_hold | $945 | 11.0% | 0.65 | – | 1.01 | 55.2% | -19.8% | 1773 | – | – |
| cash | $145 | 1.7% | 14.69 | – | 28,434.11 | 0.0% | -0.0% | 16 | – | – |

## Regime-sliced performance (ann. Sharpe / hit rate)

| strategy | bull | bear | high_vol | low_vol |
|---|---|---|---|---|
| equal_topk | 1.09 / 53.4% | -0.52 / 48.5% | -0.40 / 47.6% | 2.07 / 56.2% |
| vol_scaled | 0.90 / 7.9% | -0.95 / 16.8% | -0.67 / 7.6% | 1.07 / 11.6% |
| kelly | 1.70 / 56.6% | -0.64 / 49.4% | -0.38 / 49.7% | 3.34 / 59.5% |
| spy_hold | 1.85 / 57.1% | -0.94 / 47.9% | -0.68 / 48.6% | 3.79 / 60.5% |
| cash | 14.47 / 100.0% | 16.48 / 98.3% | 12.74 / 99.3% | 16.58 / 99.9% |

## Stress windows (total return)

| window | equal_topk | vol_scaled | kelly | spy_hold | cash |
|---|---|---|---|---|---|
| GFC | -72.6% | -19.0% | -38.3% | -36.6% | 0.2% |
| Q4 2018 | -18.1% | 0.0% | -9.9% | -13.8% | 0.6% |
| COVID crash | -23.4% | 0.0% | -26.2% | -17.0% | 0.1% |
| 2022 bear | -16.8% | 0.0% | -4.1% | -18.6% | 2.0% |

## Holdout period (≥ 2024-07-26, never used for model selection)

| strategy | window return |
|---|---|
| equal_topk | 90.1% |
| vol_scaled | 0.0% |
| kelly | 32.7% |
| spy_hold | 39.8% |
| cash | 8.1% |

## Honesty notes

- Sharpe deflation assumes 27 strategy/model trials.
- Regime flags (SPY 200d SMA, VIX median) use full-sample statistics; they are reporting lenses, not tradable signals.
- Fundamentals are sparse before ~2009 (EDGAR XBRL phase-in).
- Delisted tickers missing from the free price source are absent from the panel; residual survivorship bias is reported in the ingestion manifest.
- Positions whose ticker stops trading are liquidated at the last available (forward-filled) price — optimistic for bankruptcies.

![equity curves](equity.png)