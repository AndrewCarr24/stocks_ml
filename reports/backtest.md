# Backtest report

Champion model: **xgb_tuned** · strategies × candidates tried: **45** (used to deflate Sharpe)

## Headline ($100 invested at start)

| strategy | $100 → | CAGR | Sharpe | Deflated Sharpe | Sortino | max DD | worst week | underwater (d) | costs $ | fits |
|---|---|---|---|---|---|---|---|---|---|---|
| equal_topk | $840 | 10.5% | 0.46 | 0.46 | 0.74 | 71.0% | -27.6% | 2730 | 155.00 | 1111 |
| vol_scaled | $103 | 0.1% | 0.05 | 0.02 | 0.08 | 25.9% | -7.0% | 6794 | 8.25 | 1111 |
| kelly | $366 | 6.3% | 0.44 | 0.41 | 0.66 | 50.2% | -17.1% | 2881 | 30.05 | 1111 |
| kelly_spy | $845 | 10.6% | 0.63 | 0.74 | 0.97 | 50.2% | -17.1% | 1123 | 47.59 | 1111 |
| topk_spy | $1,764 | 14.4% | 0.55 | 0.62 | 0.90 | 71.1% | -27.6% | 1210 | 255.79 | 1111 |
| spy_hold | $930 | 11.1% | 0.65 | – | 1.01 | 55.2% | -19.8% | 1773 | – | – |
| cash | $144 | 1.7% | 14.57 | – | 28,157.27 | 0.0% | -0.0% | 16 | – | – |

## Regime-sliced performance (ann. Sharpe / hit rate)

| strategy | bull | bear | high_vol | low_vol |
|---|---|---|---|---|
| equal_topk | 0.99 / 46.8% | -0.19 / 44.6% | -0.15 / 43.8% | 1.73 / 48.4% |
| vol_scaled | 0.57 / 7.0% | -1.02 / 16.0% | -0.67 / 7.6% | 0.73 / 9.9% |
| kelly | 1.52 / 49.7% | -0.96 / 46.7% | -0.67 / 45.7% | 2.99 / 51.9% |
| kelly_spy | 1.73 / 56.4% | -0.84 / 48.9% | -0.61 / 49.3% | 3.46 / 59.4% |
| topk_spy | 1.15 / 54.4% | -0.19 / 49.1% | -0.16 / 48.8% | 2.06 / 57.0% |
| spy_hold | 1.84 / 57.1% | -0.94 / 47.6% | -0.68 / 48.6% | 3.85 / 60.5% |
| cash | 14.38 / 100.0% | 16.14 / 98.3% | 12.74 / 99.3% | 16.37 / 99.9% |

## Stress windows (total return)

| window | equal_topk | vol_scaled | kelly | kelly_spy | topk_spy | spy_hold | cash |
|---|---|---|---|---|---|---|---|
| GFC | -48.3% | -20.7% | -33.4% | -33.4% | -48.5% | -36.6% | 0.2% |
| Q4 2018 | -23.4% | 0.0% | -11.3% | -19.4% | -24.0% | -13.8% | 0.6% |
| COVID crash | -41.6% | 0.0% | -28.7% | -27.5% | -44.3% | -17.0% | 0.1% |
| 2022 bear | 2.4% | 0.0% | -8.1% | -8.1% | -13.1% | -18.6% | 2.0% |

## Holdout period (≥ 2024-07-19, never used for model selection — the primary strategy comparison)

| strategy | $100 → | total return | ann. Sharpe | max DD | worst week |
|---|---|---|---|---|---|
| equal_topk | $167 | 67.2% | 0.82 | 30.7% | -15.9% |
| vol_scaled | $100 | 0.0% | – | 0.0% | 0.0% |
| kelly | $113 | 12.6% | 0.50 | 15.8% | -7.6% |
| kelly_spy | $115 | 15.2% | 0.57 | 15.8% | -7.6% |
| topk_spy | $169 | 69.5% | 0.84 | 30.7% | -15.9% |
| spy_hold | $139 | 38.6% | 1.05 | 18.8% | -9.1% |
| cash | $108 | 8.2% | 157.31 | 0.0% | 0.1% |

![holdout equity curves](equity_holdout.png)

## Recent five years ($100 at 2021-07-17; overlaps model-selection window — see holdout for the clean test)

| strategy | $100 → | CAGR | Sharpe | max DD |
|---|---|---|---|---|
| equal_topk | $190 | 13.7% | 0.56 | 30.7% |
| vol_scaled | $100 | 0.0% | – | 0.0% |
| kelly | $114 | 2.7% | 0.26 | 19.7% |
| kelly_spy | $164 | 10.5% | 0.73 | 19.7% |
| topk_spy | $231 | 18.2% | 0.67 | 35.7% |
| spy_hold | $187 | 13.4% | 0.82 | 24.5% |
| cash | $119 | 3.6% | 33.40 | 0.0% |

## Honesty notes

- Sharpe deflation assumes 45 strategy/model trials.
- Regime flags (SPY 200d SMA, VIX median) use full-sample statistics; they are reporting lenses, not tradable signals.
- Fundamentals are sparse before ~2009 (EDGAR XBRL phase-in).
- Delisted tickers missing from the free price source are absent from the panel; residual survivorship bias is reported in the ingestion manifest.
- Positions whose ticker stops trading are liquidated at the last available (forward-filled) price — optimistic for bankruptcies.

![equity curves](equity.png)