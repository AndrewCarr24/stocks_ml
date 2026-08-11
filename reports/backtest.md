# Backtest report

Champion model: **xgb_tuned** · strategies × candidates tried: **27** (used to deflate Sharpe)

## Headline ($100 invested at start)

| strategy | $100 → | CAGR | Sharpe | Deflated Sharpe | Sortino | max DD | worst week | underwater (d) | costs $ | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| equal_topk | $1,142 | 12.1% | 0.52 | 0.65 | 0.85 | 71.7% | -30.8% | 1232 | 241.15 | 275 |
| vol_scaled | $114 | 0.6% | 0.13 | 0.08 | 0.19 | 25.7% | -5.8% | 6942 | 9.91 | 275 |
| kelly | $506 | 7.9% | 0.54 | 0.68 | 0.85 | 50.0% | -18.8% | 1472 | 35.06 | 275 |
| spy_hold | $935 | 11.1% | 0.65 | – | 1.01 | 55.2% | -19.8% | 1773 | – | – |
| cash | $144 | 1.7% | 14.57 | – | 28,174.40 | 0.0% | -0.0% | 16 | – | – |

## Regime-sliced performance (ann. Sharpe / hit rate)

| strategy | bull | bear | high_vol | low_vol |
|---|---|---|---|---|
| equal_topk | 1.07 / 47.8% | -0.34 / 45.4% | -0.24 / 44.2% | 1.92 / 49.9% |
| vol_scaled | 0.69 / 7.9% | -1.29 / 13.4% | -0.84 / 6.6% | 0.92 / 11.1% |
| kelly | 1.44 / 48.6% | -0.69 / 46.2% | -0.51 / 44.4% | 2.98 / 51.1% |
| spy_hold | 1.84 / 57.1% | -0.94 / 47.6% | -0.69 / 48.6% | 3.86 / 60.5% |
| cash | 14.39 / 100.0% | 16.14 / 98.3% | 12.76 / 99.3% | 16.37 / 99.9% |

## Stress windows (total return)

| window | equal_topk | vol_scaled | kelly | spy_hold | cash |
|---|---|---|---|---|---|
| GFC | -49.7% | -9.9% | -32.4% | -36.6% | 0.2% |
| Q4 2018 | -21.3% | 0.0% | -9.6% | -13.8% | 0.6% |
| COVID crash | 0.0% | 0.0% | 0.0% | -17.0% | 0.1% |
| 2022 bear | -0.9% | 0.0% | -6.4% | -18.6% | 2.0% |

## Holdout period (≥ 2024-07-19, never used for model selection)

| strategy | window return |
|---|---|
| equal_topk | -5.8% |
| vol_scaled | 0.0% |
| kelly | 14.3% |
| spy_hold | 39.4% |
| cash | 8.3% |

## Honesty notes

- Sharpe deflation assumes 27 strategy/model trials.
- Regime flags (SPY 200d SMA, VIX median) use full-sample statistics; they are reporting lenses, not tradable signals.
- Fundamentals are sparse before ~2009 (EDGAR XBRL phase-in).
- Delisted tickers missing from the free price source are absent from the panel; residual survivorship bias is reported in the ingestion manifest.
- Positions whose ticker stops trading are liquidated at the last available (forward-filled) price — optimistic for bankruptcies.

![equity curves](equity.png)