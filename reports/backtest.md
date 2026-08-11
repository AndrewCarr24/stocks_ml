# Backtest report

Champion model: **xgb_tuned** · strategies × candidates tried: **45** (used to deflate Sharpe)

## Headline ($100 invested at start)

| strategy | $100 → | CAGR | Sharpe | Deflated Sharpe | Sortino | max DD | worst week | underwater (d) | costs $ | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| equal_topk | $2,644 | 16.6% | 0.65 | 0.78 | 1.02 | 69.7% | -30.5% | 947 | 268.92 | 275 |
| vol_scaled | $679 | 9.4% | 0.85 | 0.95 | 1.31 | 23.3% | -10.5% | 2001 | 65.74 | 275 |
| kelly | $554 | 8.4% | 0.53 | 0.58 | 0.82 | 48.8% | -21.1% | 1123 | 36.05 | 275 |
| kelly_spy | $1,274 | 12.7% | 0.71 | 0.84 | 1.10 | 48.8% | -21.1% | 1123 | 63.00 | 275 |
| topk_spy | $5,838 | 21.1% | 0.73 | 0.88 | 1.17 | 71.9% | -30.5% | 855 | 515.41 | 275 |
| spy_hold | $930 | 11.1% | 0.65 | – | 1.01 | 55.2% | -19.8% | 1773 | – | – |
| cash | $144 | 1.7% | 14.57 | – | 28,157.27 | 0.0% | -0.0% | 16 | – | – |

## Regime-sliced performance (ann. Sharpe / hit rate)

| strategy | bull | bear | high_vol | low_vol |
|---|---|---|---|---|
| equal_topk | 1.28 / 38.8% | -0.09 / 39.4% | 0.02 / 35.3% | 1.91 / 42.0% |
| vol_scaled | 1.47 / 39.0% | -0.71 / 40.0% | -0.46 / 35.7% | 2.23 / 42.2% |
| kelly | 1.57 / 46.9% | -0.58 / 47.9% | -0.33 / 43.8% | 2.85 / 49.8% |
| kelly_spy | 1.84 / 56.6% | -0.69 / 48.6% | -0.41 / 49.5% | 3.51 / 59.4% |
| topk_spy | 1.61 / 57.1% | -0.37 / 48.9% | -0.20 / 50.6% | 2.63 / 59.3% |
| spy_hold | 1.84 / 57.1% | -0.94 / 47.6% | -0.68 / 48.6% | 3.85 / 60.5% |
| cash | 14.38 / 100.0% | 16.14 / 98.3% | 12.74 / 99.3% | 16.37 / 99.9% |

## Stress windows (total return)

| window | equal_topk | vol_scaled | kelly | kelly_spy | topk_spy | spy_hold | cash |
|---|---|---|---|---|---|---|---|
| GFC | -39.4% | -9.9% | -34.0% | -34.0% | -49.3% | -36.6% | 0.2% |
| Q4 2018 | -5.5% | -2.6% | -3.9% | -12.8% | -12.5% | -13.8% | 0.6% |
| COVID crash | 6.0% | -12.8% | 1.1% | -17.3% | -13.3% | -17.0% | 0.1% |
| 2022 bear | -34.5% | -8.3% | -10.3% | -10.3% | -37.9% | -18.6% | 2.0% |

## Holdout period (≥ 2024-07-19, never used for model selection — the primary strategy comparison)

| strategy | $100 → | total return | ann. Sharpe | max DD | worst week |
|---|---|---|---|---|---|
| equal_topk | $127 | 26.8% | 0.52 | 32.3% | -15.8% |
| vol_scaled | $113 | 13.0% | 0.53 | 14.4% | -4.4% |
| kelly | $119 | 18.5% | 0.73 | 15.7% | -7.5% |
| kelly_spy | $122 | 22.3% | 0.80 | 15.7% | -7.5% |
| topk_spy | $139 | 38.9% | 0.63 | 32.3% | -15.8% |
| spy_hold | $139 | 38.6% | 1.05 | 18.8% | -9.1% |
| cash | $108 | 8.2% | 157.31 | 0.0% | 0.1% |

![holdout equity curves](equity_holdout.png)

## Recent five years ($100 at 2021-07-17; overlaps model-selection window — see holdout for the clean test)

| strategy | $100 → | CAGR | Sharpe | max DD |
|---|---|---|---|---|
| equal_topk | $129 | 5.2% | 0.32 | 39.2% |
| vol_scaled | $124 | 4.5% | 0.50 | 14.4% |
| kelly | $130 | 5.4% | 0.46 | 20.3% |
| kelly_spy | $176 | 12.0% | 0.84 | 20.3% |
| topk_spy | $181 | 12.6% | 0.54 | 44.7% |
| spy_hold | $187 | 13.4% | 0.82 | 24.5% |
| cash | $119 | 3.6% | 33.40 | 0.0% |

## Honesty notes

- Sharpe deflation assumes 45 strategy/model trials.
- Regime flags (SPY 200d SMA, VIX median) use full-sample statistics; they are reporting lenses, not tradable signals.
- Fundamentals are sparse before ~2009 (EDGAR XBRL phase-in).
- Delisted tickers missing from the free price source are absent from the panel; residual survivorship bias is reported in the ingestion manifest.
- Positions whose ticker stops trading are liquidated at the last available (forward-filled) price — optimistic for bankruptcies.

![equity curves](equity.png)