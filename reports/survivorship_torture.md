# Survivorship torture test

Champion model: **xgb** (same fitted estimator as the committed baseline). Empirically measured removal-haircut stress test: how much of the backtest's headline performance survives when every simulated index-removal liquidation is penalized using empirically measured post-removal returns, without buying delisted-price data.

## Measured post-removal outcomes (the empirical evidence)

| reason class | n events | n truncated | median post_ret | q25 post_ret | haircut | fallback |
|---|---|---|---|---|---|---|
| acquisition | 18 | 1 | 0.0% | -16.1% | 0.0% | – |
| decline | 92 | 4 | 5.8% | -11.2% | 0.0% | – |
| restructuring | 14 | 0 | 4.9% | -18.0% | 0.0% | – |
| unknown | 3 | 0 | 34.9% | 3.2% | 0.0% | – |

## Headline: committed baseline (reports/backtest.md) vs. tortured

| strategy | baseline $100→ | tortured $100→ | baseline CAGR | tortured CAGR | baseline max DD | tortured max DD |
|---|---|---|---|---|---|---|
| equal_topk | $12,170 | $11,677 | 25.1% | 24.8% | 66.7% | 66.7% |
| vol_scaled | $328 | $323 | 5.7% | 5.6% | 33.3% | 33.3% |
| kelly | $949 | $946 | 11.1% | 11.0% | 50.5% | 50.5% |

## GFC stress window (total return): baseline vs. tortured

| strategy | baseline | tortured |
|---|---|---|
| equal_topk | -8.4% | -8.4% |
| vol_scaled | -5.3% | -5.3% |
| kelly | -30.5% | -30.5% |

## Interpretation

- Haircuts fire only when the simulated strategy is still holding a ticker at the moment it is actually removed from the S&P 500 (within a 35-day window of the real removal date); they cannot simulate holding through the tickers that are fully missing from the free price source and so never entered the panel at all -- those casualties are not represented here at all.
- The haircut is applied as a proceeds reduction at the liquidation exec day, including cases where the base (non-tortured) simulator would otherwise mark the position at a stale forward-filled price.
- Truncated removal events (price history ends within 60 trading days of removal, i.e. the ticker's data disappears near the event -- typically the worst outcomes) are assigned their class's empirical 25th-percentile loss rather than their own unmeasurable return.
- Classes whose empirical median post-removal return is non-negative (acquisitions, restructurings) get a 0.0 haircut -- the measured outcome, not an assumption.