# Bailey & López de Prado (2014) — "The Deflated Sharpe Ratio"

PDF: `papers/bailey-lopez-de-prado-2014-deflated-sharpe.pdf` (author copy, davidhbailey.com)

## What the paper shows

The observed maximum Sharpe over N tried strategies grows like
`E[max SR] ≈ sqrt(V[SR across trials]) · ((1−γ)·Φ⁻¹(1−1/N) + γ·Φ⁻¹(1−1/(N·e)))`
even when every strategy's true Sharpe is zero. The Deflated Sharpe Ratio is the
Probabilistic Sharpe Ratio evaluated against that expected-max benchmark instead
of 0, with non-normality (skew, kurtosis) folded into the test statistic. Two
inputs are non-negotiable: an honest **N** (all trials, including abandoned ones)
and the **variance of Sharpe estimates across those trials** — not the sampling
variance of the single winning strategy.

## What it implies for stocks_ml

This is the one paper on the list we have already partially implemented, so the
note is mostly an audit of `backtest/metrics.py::deflated_sharpe`.

- **What's right:** the expected-max formula, the Euler–Mascheroni weighting, the
  skew/kurt-adjusted PSR denominator, and the fact that `pipelines.py` reports a
  project-wide trial count ("Trials to date … feeds Sharpe-deflation").
- **What deviates from the paper:** `metrics.py:71` uses
  `var_sr = (1 − skew·sr + (kurt−1)/4·sr²)/(T−1)` — the *estimator* variance of the
  single reported SR — where the paper calls for the *cross-trial* variance
  `V[{SR_n}]` of the Sharpes actually produced by the N trials. With our history
  (two invalidated backtests, an Optuna run that gamed CV, hundreds of tuning
  trials), cross-trial dispersion is real and observable; proxying it with the
  single-path estimator variance under-deflates when trials were diverse and
  over-deflates when they were near-clones. Fix: persist every trial's holdout-free
  backtest Sharpe (tuning runs, pipeline league entrants, strategy variants) in a
  small trials ledger, and compute `sr0` from the ledger's empirical variance.
- **N is currently a judgment call.** The league "grows the Sharpe-deflation trial
  count" by hand. The ledger above makes N an artifact, not an estimate — every
  `tune`/`train`/`pipelines` invocation appends. Bailey–LdP explicitly warn that
  selective forgetting of trials is how DSR gets gamed.
- **Refit-anchor sensitivity is a multiple-testing problem.** The tie-guard
  investigation showed holdout topk_spy ranging $139→$252 across two refit
  anchors. Each anchor tried is a trial; reporting the better path without
  counting the other is exactly the selection bias this paper prices. The official
  report already leads with the anchored run — good — but both anchors belong in N.
- **Weekly, not daily, returns for the DSR inputs.** Our strategy rebalances
  weekly; skew/kurtosis of daily NAV changes mixes microstructure into the
  non-normality correction. `summarize()` currently feeds daily returns; a weekly
  variant would match the decision frequency.

## Concrete candidates

1. Trials ledger (JSON, git-tracked like `ledger.json`) auto-appended by tune /
   train / pipelines; DSR computed from its empirical cross-trial variance and count.
2. Report DSR alongside Sharpe in the *holdout* section of `reports/backtest.md`
   (iron rule #0 says holdout leads; DSR is the honest version of that headline).
