# Pre-registered strategy blend (v1) — backtest

Owner-requested framework (2026-08-13): allocate across sleeve strategies under
the four practitioner principles, with the meta-layer capacity deliberately
tiny (stacking's out-of-sample discipline, Timmermann's near-equal weights).
Rules and constants in `backtest/blend.py` were fixed before this backtest ran
and were not tuned against it; this run is one ledger trial.

Sleeves: champion_topk16, ltr_topk12, monthly_topk16, kelly_spy, vol_scaled,
spy_sleeve. Quarterly reviews; equal weights with trailing-3y-Sharpe rank tilt
bounded to [0.75, 1.25]; max 25% of the gap moved per review; 52-week probation
at half share; suspension at >40% drawdown with reinstatement below 20%;
dead-money halving after 3 negative trailing years.

## Results ($100 at start)

| | since 2005 | Sharpe | max DD | holdout | holdout Sharpe | holdout DD |
|---|---|---|---|---|---|---|
| blend | $1,348 | 0.68 | 56% | $150 | 1.05 | 19% |
| champion_topk16 (best sleeve) | $3,406 | 0.68 | 68% | $206 | 1.17 | 27% |
| spy_hold | $930 | 0.65 | 55% | $139 | 1.05 | 19% |

Blend costs over 21 years: $74.53 (sleeve trades net against each other).

## Reading

- The blend achieves the champion's full-history Sharpe (0.68) at **SPY-level
  risk** (max DD 56% vs the champion's 68%; holdout DD 19% vs 27%) while
  beating SPY's dollars by ~45%. It is the "hold-through-anything"
  configuration: at no point in 21 years does its path diverge from SPY's risk
  profile enough to break an investor, which was the design goal — the answer
  to "how do I not capitulate in year 7?" is "own the thing whose pain is
  bounded."
- The price is equally visible: roughly a third of capital sits in sleeves
  that earned little (kelly_spy, vol_scaled), and the bounded tilt is too
  weak to fire them — by design. Diversification drag IS the premium paid for
  never having to make a panic decision. The concentrated champion made 2.5x
  more but demanded sitting through drawdowns half again as deep.
- The meta-weight path (reports/equity_blend.png, lower panel) shows the rules
  working mechanically: GFC drawdowns suspended three sleeves in 2009 and
  hysteresis reinstated them gradually; LTR was suspended through 2020-2023
  after its 77% crash and earned its way back; no weight ever moved more than
  the change budget allows.

## Caveats

- Sleeve roster chosen with hindsight (these six exist because they survived
  this project's research); a 2005 deployment could not have held them. The
  meta-RULES are pre-registered; the roster is not.
- Sleeve NAVs inherit every data caveat of their pipelines (survivorship,
  adjusted prices, 5bps costs), monthly_topk16 most of all.
- Suspension thresholds are round a-priori numbers. Tuning them against this
  backtest is forbidden (each variant would be a new ledger trial and the
  gains would be winner's curse).
