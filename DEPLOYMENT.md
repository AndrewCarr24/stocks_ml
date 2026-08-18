# Deployment contract (v1 — pre-registered 2026-08-18)

The rules below are decided BEFORE real money moves, so that decisions during
drawdowns are executions of policy, not judgments made in pain. Constants are
round numbers fixed a priori; changing any of them requires a calm-quarter
review (rule 4) and a trials-ledger entry. This file is the answer to "what
would we do if 2015-2024 happened again": mostly, follow the table.

## 0. Current standing configuration

- Live (paper): champion xgb_tuned + topk_spy, k=16 (`config.yaml`).
- Challenger (paper): LTR ranker + topk_spy, k=12 (`challenger_top_k`).
- Real money deployed: none. MinTRL at N≈300 trials says ~177 weeks of live
  record are needed to certify skill outright; the champion-vs-challenger
  relative verdict needs far less (rule 3).

## 1. Sizing (when real money starts)

- The strategy sleeve is capped at a fraction of investable savings the owner
  sets ON A CALM DAY and records here before funding: ____%.
- Default posture until then: SPY core + strategy sleeve, blend-don't-switch.
- The sleeve must be sized so a repeat of its OWN backtest worst cases —
  a −68% sleeve drawdown (2008-class) and nine consecutive years of trailing
  SPY (2015-2024) — would be tolerable without intervention.

## 2. Kill / de-risk criteria (mechanical, layered)

Monitor three layers separately; each trigger maps to ONE action.

| layer | metric (weekly, from the ledgers/reports) | trigger | action |
|---|---|---|---|
| model | trailing 104-week mean weekly IC of the live model | ≤ 0 | retire the model (strategy may keep running on challenger/SPY) |
| pipeline | live sleeve drawdown from its all-time high | > 40% | suspend sleeve to SPY; reinstate only below 20% (hysteresis, mirrors blend rules) |
| regime | equal-weight-500 minus SPY trailing 156w | any | NO action — known structural tilt; report only |
| execution | realized costs vs simulator assumption | > 3× modeled | halt until explained |

The regime row exists to forbid the classic error: firing the model because
the equal-weight tilt is out of favor.

## 3. Challenger promotion (shadow race)

- Pre-registered rule: after ≥ 52 race weeks, the challenger is promoted to
  live IF (a) its ledger NAV leads the champion's AND (b) its trailing-52-week
  Sharpe exceeds the champion's by ≥ 0.30. Otherwise it keeps racing.
- Promotion means BLEND, not switch: challenger enters at 25% of the sleeve,
  +25 percentage points per subsequent quarter it still satisfies (a).
- A new challenger may replace a failed one at any time (probation applies);
  at most one live promotion decision per quarter.

## 4. Change budget (anti-panic clause)

- Configuration changes (models, k, strategies, features) are considered only
  at scheduled month-end reviews.
- Any change proposed while the sleeve is in a >15% drawdown must ALSO pass
  the question, answered in writing in the PR: "would we adopt this in a flat
  quarter?" — with the trials-ledger entry linking the evidence.
- Backtest-derived changes follow the standing selection protocol: chosen on
  pre-holdout evidence, holdout read once, every variant a ledger trial.

## 5. Standing honesty machinery (already enforced in code)

- Trials ledger (models/trials_ledger.json) records every attempt; deflated
  Sharpe uses its cross-trial variance; MinTRL is printed in backtest.md.
- The holdout is spent; further holdout grading requires strong cause.
- The shadow race status is appended to every Saturday signal (cli.py), so
  the race verdict accumulates in public with no manual bookkeeping.
