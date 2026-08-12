# Pipeline league — one exam, many recipes

All pipelines: $100 walked forward from 2005-01-03 through the same cost-aware simulator (5bps one-way), staggered-refit ensembles, tie guard. They differ in training objective, strategy, and cadence:

- **incumbent_weekly**: champion regression (RMSE-trained), weekly, topk_spy — the baseline
- **ltr_weekly**: learning-to-rank (rank:ndcg, week = query group), weekly, topk_spy — CV-tuned
- **monthly_reg**: CV-tuned 4-week-label regression, monthly cadence, topk_spy
- **quintile_prob_weekly**: P(top-quintile) classifier, weekly; only >50% confidence fills a slot, rest in SPY

Trials to date across the project (feeds Sharpe-deflation honesty): each league row adds one.

## Holdout (≥ 2024-07-19) — the primary comparison

Never used for tuning or selection of any pipeline.

| pipeline | $100 → | total return | ann. Sharpe | max DD | worst week |
|---|---|---|---|---|---|
| incumbent_weekly | $169 | +69.5% | 0.84 | 30.7% | -15.9% |
| ltr_weekly | $213 | +113.2% | 1.03 | 36.4% | -18.0% |
| monthly_reg | $145 | +45.2% | 0.68 | 27.2% | -14.3% |
| quintile_prob_weekly | $118 | +17.5% | 0.41 | 28.8% | -12.2% |
| spy_hold (benchmark) | $139 | +38.6% | 1.05 | 18.8% | -9.1% |
| cash (benchmark) | $108 | +8.2% | 157.31 | 0.0% | 0.1% |

## Since 2005-01-03 (full history)

Same exam as the strategy zoo. Pre-holdout years overlap the tuned pipelines' CV/tuning windows — context, not verdict; the holdout section above is the clean test.

| pipeline | $100 → | total return | ann. Sharpe | max DD | worst week |
|---|---|---|---|---|---|
| incumbent_weekly | $1,764 | +1663.8% | 0.55 | 71.1% | -27.6% |
| ltr_weekly | $2,293 | +2192.8% | 0.57 | 80.5% | -29.0% |
| monthly_reg | $10,876 | +10776.1% | 0.82 | 65.4% | -22.0% |
| quintile_prob_weekly | $734 | +634.1% | 0.47 | 59.3% | -27.4% |
| spy_hold (benchmark) | $930 | +829.9% | 0.65 | 55.2% | -19.8% |
| cash (benchmark) | $144 | +43.9% | 14.57 | 0.0% | -0.0% |

## Cost and fit accounting

| pipeline | rebalances/cadence | model fits | costs $ |
|---|---|---|---|
| incumbent_weekly | weekly | 1111 | 255.79 |
| ltr_weekly | weekly | 1111 | 312.95 |
| monthly_reg | every 4 wk | 276 | 339.65 |
| quintile_prob_weekly | weekly | 1111 | 207.62 |

Honesty notes:
- Wave-1 challengers are deliberately untuned (conventional defaults); the incumbent had an Optuna budget. A challenger that competes while untuned is the interesting signal.
- Every pipeline graded here spends holdout novelty; the shadow ledger makes the final call.

![pipeline holdout equity](equity_pipelines.png)