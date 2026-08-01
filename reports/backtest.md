# Backtest report

Champion model: **xgb_tuned** · strategies × candidates tried: **27** (used to deflate Sharpe)

## Headline ($100 invested at start)

| strategy | $100 → | CAGR | Sharpe | Deflated Sharpe | Sortino | max DD | worst week | underwater (d) | costs $ | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| equal_topk | $7,078 | 21.9% | 0.72 | 0.91 | 1.22 | 65.9% | -28.6% | 960 | 692.07 | 278 |
| vol_scaled | $219 | 3.7% | 0.39 | 0.40 | 0.59 | 27.6% | -9.5% | 2390 | 53.45 | 278 |
| kelly | $1,114 | 11.9% | 0.65 | 0.83 | 1.02 | 50.4% | -20.0% | 1157 | 139.38 | 278 |
| spy_hold | $950 | 11.0% | 0.65 | – | 1.01 | 55.2% | -19.8% | 1773 | – | – |
| cash | $145 | 1.7% | 14.72 | – | 28,520.23 | 0.0% | -0.0% | 16 | – | – |

## Regime-sliced performance (ann. Sharpe / hit rate)

| strategy | bull | bear | high_vol | low_vol |
|---|---|---|---|---|
| equal_topk | 1.34 / 54.2% | -0.18 / 48.6% | -0.10 / 48.3% | 2.33 / 56.9% |
| vol_scaled | 1.02 / 38.4% | -1.09 / 35.4% | -1.26 / 28.9% | 1.92 / 44.9% |
| kelly | 1.68 / 56.5% | -0.67 / 48.7% | -0.41 / 49.1% | 3.35 / 59.5% |
| spy_hold | 1.85 / 57.1% | -0.94 / 47.9% | -0.70 / 48.6% | 3.82 / 60.5% |
| cash | 14.51 / 100.0% | 16.48 / 98.3% | 12.79 / 99.3% | 16.60 / 99.9% |

## Stress windows (total return)

| window | equal_topk | vol_scaled | kelly | spy_hold | cash |
|---|---|---|---|---|---|
| GFC | -32.6% | -9.4% | -36.8% | -36.6% | 0.2% |
| Q4 2018 | -15.8% | -8.3% | -10.3% | -13.8% | 0.6% |
| COVID crash | -22.8% | -22.8% | -25.1% | -17.0% | 0.1% |
| 2022 bear | 8.1% | 0.0% | -2.9% | -18.6% | 2.0% |

## Holdout period (≥ 2024-08-09, never used for model selection)

| strategy | window return |
|---|---|
| equal_topk | 128.1% |
| vol_scaled | 0.0% |
| kelly | 16.8% |
| spy_hold | 43.5% |
| cash | 8.1% |

## Honesty notes

- Sharpe deflation assumes 27 strategy/model trials.
- Regime flags (SPY 200d SMA, VIX median) use full-sample statistics; they are reporting lenses, not tradable signals.
- Fundamentals are sparse before ~2009 (EDGAR XBRL phase-in).
- Delisted tickers missing from the free price source are absent from the panel; residual survivorship bias is reported in the ingestion manifest.
- Positions whose ticker stops trading are liquidated at the last available (forward-filled) price — optimistic for bankruptcies.

![equity curves](equity.png)