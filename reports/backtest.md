# Backtest report

Champion model: **xgb_tuned** · strategies × candidates tried: **45** (used to deflate Sharpe)

## Headline ($100 invested at start)

| strategy | $100 → | CAGR | Sharpe | Deflated Sharpe | Sortino | max DD | worst week | underwater (d) | costs $ | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| equal_topk | $2,722 | 16.8% | 0.60 | 0.70 | 0.94 | 76.1% | -31.4% | 2639 | 376.71 | 1111 |
| vol_scaled | $440 | 7.2% | 0.69 | 0.82 | 1.07 | 31.6% | -8.4% | 2850 | 67.37 | 1111 |
| kelly | $387 | 6.6% | 0.43 | 0.40 | 0.65 | 53.2% | -21.3% | 1640 | 30.46 | 1111 |
| kelly_spy | $1,086 | 11.9% | 0.66 | 0.78 | 1.01 | 53.2% | -21.3% | 1168 | 55.64 | 1111 |
| topk_spy | $5,532 | 20.8% | 0.68 | 0.81 | 1.08 | 76.0% | -31.4% | 884 | 634.35 | 1111 |
| spy_hold | $930 | 11.1% | 0.65 | – | 1.01 | 55.2% | -19.8% | 1773 | – | – |
| cash | $144 | 1.7% | 14.57 | – | 28,157.27 | 0.0% | -0.0% | 16 | – | – |

## Regime-sliced performance (ann. Sharpe / hit rate)

| strategy | bull | bear | high_vol | low_vol |
|---|---|---|---|---|
| equal_topk | 1.22 / 45.6% | -0.06 / 49.0% | -0.04 / 43.6% | 1.97 / 48.5% |
| vol_scaled | 1.26 / 33.7% | -0.75 / 34.7% | -0.32 / 28.0% | 1.75 / 38.8% |
| kelly | 1.54 / 49.2% | -0.70 / 48.4% | -0.43 / 45.6% | 2.79 / 51.9% |
| kelly_spy | 1.82 / 56.6% | -0.71 / 48.8% | -0.47 / 49.8% | 3.49 / 59.3% |
| topk_spy | 1.38 / 54.7% | -0.10 / 50.4% | -0.07 / 50.1% | 2.29 / 56.8% |
| spy_hold | 1.84 / 57.1% | -0.94 / 47.6% | -0.68 / 48.6% | 3.85 / 60.5% |
| cash | 14.38 / 100.0% | 16.14 / 98.3% | 12.74 / 99.3% | 16.37 / 99.9% |

## Stress windows (total return)

| window | equal_topk | vol_scaled | kelly | kelly_spy | topk_spy | spy_hold | cash |
|---|---|---|---|---|---|---|---|
| GFC | -61.5% | -9.3% | -35.7% | -35.7% | -60.2% | -36.6% | 0.2% |
| Q4 2018 | -10.6% | -9.6% | -2.3% | -14.4% | -18.2% | -13.8% | 0.6% |
| COVID crash | -38.5% | -23.9% | -23.0% | -22.8% | -38.5% | -17.0% | 0.1% |
| 2022 bear | -19.9% | 0.0% | -8.0% | -8.0% | -22.5% | -18.6% | 2.0% |

## Holdout period (≥ 2024-07-19, never used for model selection — the primary strategy comparison)

| strategy | $100 → | total return | ann. Sharpe | max DD | worst week |
|---|---|---|---|---|---|
| equal_topk | $178 | 77.7% | 0.89 | 32.2% | -15.6% |
| vol_scaled | $100 | 0.0% | – | 0.0% | 0.0% |
| kelly | $114 | 14.4% | 0.59 | 15.5% | -7.5% |
| kelly_spy | $121 | 20.8% | 0.75 | 15.5% | -7.5% |
| topk_spy | $178 | 77.6% | 0.88 | 32.2% | -15.6% |
| spy_hold | $139 | 38.6% | 1.05 | 18.8% | -9.1% |
| cash | $108 | 8.2% | 157.31 | 0.0% | 0.1% |

![holdout equity curves](equity_holdout.png)

## Recent five years ($100 at 2021-07-17; overlaps model-selection window — see holdout for the clean test)

| strategy | $100 → | CAGR | Sharpe | max DD |
|---|---|---|---|---|
| equal_topk | $162 | 10.1% | 0.45 | 44.5% |
| vol_scaled | $100 | 0.0% | – | 0.0% |
| kelly | $113 | 2.4% | 0.25 | 22.5% |
| kelly_spy | $168 | 11.0% | 0.78 | 18.7% |
| topk_spy | $196 | 14.5% | 0.56 | 46.2% |
| spy_hold | $187 | 13.4% | 0.82 | 24.5% |
| cash | $119 | 3.6% | 33.40 | 0.0% |

## Honesty notes

- Sharpe deflation assumes 45 strategy/model trials.
- Regime flags (SPY 200d SMA, VIX median) use full-sample statistics; they are reporting lenses, not tradable signals.
- Fundamentals are sparse before ~2009 (EDGAR XBRL phase-in).
- Delisted tickers missing from the free price source are absent from the panel; residual survivorship bias is reported in the ingestion manifest.
- Positions whose ticker stops trading are liquidated at the last available (forward-filled) price — optimistic for bankruptcies.

![equity curves](equity.png)