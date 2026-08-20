# stocks_ml — agent context

ML system that forecasts S&P 500 stock returns weekly, backtests investment
strategies, and runs a live paper-trading account. Built entirely on free data.
The owner's goal: turn $100 into more, without ever blowing up the account.

## Current state (2026-07-21)

- **Champion model:** Optuna-tuned XGBoost (`models/champion.json`, mean weekly
  rank IC 0.0198 across 4 CV folds, all positive, full 488-week coverage).
  Optuna-tuned ElasticNet is second at 0.0190, also positive in all folds.
- **Evaluation design (owner-specified, do not change casually):** 4 walk-forward
  CV folds testing 2015-03 → 2024-07 (`eval_start: 2015-03-01`, `n_cv_folds: 4`
  in config). Each fold fits one frozen model on the immediately preceding
  2-calendar-year window (`cv_train_years: 2`) with a 10-day purge; exact dates
  are in `reports/rolling_cv.md`. The last 2 years
  (2024-07 → now) are a **holdout** never used for tuning or selection.
- **Live:** GitHub Actions runs the weekly cycle every Saturday 13:00 UTC
  (signals + paper ledger, committed back to the repo) and a monthly retrain on
  the 1st. Live strategy is `topk_spy` (config `live_strategy`): the model's
  top-k equal-weighted, unfilled slots held in SPY. k-sweep 2026-08-12
  (pre-holdout-selected, protocol declared before results): champion k=16
  (pre-holdout Sharpe 0.63 vs 0.52 at the old k=8; its now-admissible holdout
  row: \$206 / Sharpe 1.17 / DD 27% — first champion config to beat SPY's 1.05
  holdout Sharpe). LTR challenger runs its own `challenger_top_k: 12`.
  Magnitude-proportional weighting and score-cutoff dynamic-k both LOST to
  equal weights pre-holdout (LTR scores are not scale-comparable across
  refits — never use absolute score cutoffs across weeks).
- **Shadow race (2026-08-11):** the Saturday cycle also runs an LTR challenger
  (WeekGroupedXGBRanker, models/ltr_optuna.json — selected by NDCG@8 on CV
  folds, never mean IC: mean-IC selection demonstrably degrades a top-k
  ranker) on its own paper ledger `ledger_ltr.json` with the same topk_spy
  strategy and the same $100 start. Files: signals/<date>-ltr.md,
  <date>-ltr-trades.json. The race verdict — champion vs challenger on truly
  unseen weeks — is the ONLY evidence that may change `live_strategy` or the
  champion; further holdout grading of league variants is frozen (winner's
  curse: ~7 configs have already been graded on it).
- **Tie guard (2026-08-11):** `select_top_k` in strategies.py fills the top-k
  with whole equal-value prediction groups only; a group larger than the
  remaining slots is refused (unfilled slots → cash, or SPY under SpyFloor).
  Degenerate refits — early stopping keeping ~no trees when its recent
  validation tail shows no signal — previously collapsed `nlargest` into
  buying the first 8 tickers alphabetically. Never reintroduce order-dependent
  tie-breaking. Guarded holdout results are strongly refit-anchor sensitive:
  the official 2005-anchored run (reports/backtest.md) gives topk_spy $139 ≈
  spy_hold $139, equal_topk $127, while a 2024-07-anchored rerun gave topk_spy
  $252, equal_topk $222. The spread is timing luck of concentrated top-8 bets
  (e.g. a healthy June-2026 refit bet semiconductors into a sector crash, week
  −15.8%), not a bug — treat single-path holdout numbers with humility.

## Commands

```bash
uv sync                      # install (Python 3.12; automl_tool needs <3.13)
uv run pytest                # 255 tests; MUST stay green with 0 warnings
uv run stocks-ml ingest      # fetch all data + rebuild panel (idempotent)
uv run stocks-ml tune --family xgb|lgbm|catboost|enet [--optuna]
uv run stocks-ml train       # champion tournament -> models/selection.md
uv run stocks-ml backtest    # -> reports/backtest.md
uv run stocks-ml pipelines   # multi-pipeline league -> reports/pipelines.md
uv run stocks-ml signals     # weekly live signal
uv run stocks-ml ledger init|apply|mark|show
uv run stocks-ml torture     # survivorship stress test
```

**Pipeline league (2026-08-11, owner's design):** pipelines may differ in
training objective, strategy, and cadence, but are all judged by one $100
cost-aware walk-forward exam (backtest/pipelines.py). Wave 1: incumbent
regression; weekly learning-to-rank (WeekGroupedXGBRanker, rank:ndcg with
week query groups, per-week quintile relevance grades); monthly-cadence
regression on the panel's `label_4w` (4-week horizon; purge 42 days —
consumers must purge past the label span); P(top-quintile) classifier
(TopQuintileClassifier, extremes-only training) traded via ConfidenceTopK
(floor 0.5) + SpyFloor. Challengers are deliberately untuned in wave 1; the
league grows the Sharpe-deflation trial count; the shadow ledger stays the
final judge.

CI commits artifacts (signals/, ledger.json, models/, reports/) on its own
schedule, so the repository may need to be synchronized before local work.
However, do not run `git` commands unless the owner explicitly requests them;
when synchronization cannot be confirmed, work from the current checkout and
state that limitation in the final summary.

## Architecture

```
src/stocks_ml/
  data/       store.py (parquet DataStore + manifest), prices.py (yfinance,
              corrupt-series filter), membership.py (point-in-time S&P 500 from
              Wikipedia), fred.py (macro; only audited T10Y2Y/FEDFUNDS admitted), edgar.py
              (fundamentals, filing-dated), insiders.py (Form 4), shortint.py (FINRA)
  features/   panel.py (build_panel = the one place features/labels are made),
              fundamentals.py, events.py, insiders.py, ranking.py
  models/     cv.py (purged walk-forward CV, weekly rank IC), candidates.py
              (model zoo + wrappers), tuning.py (random search),
              optuna_tuning.py (TPE, CV-selected), champion.py (tournament)
  backtest/   simulator.py (no-lookahead walk-forward), strategies.py
              (equal_topk / vol_scaled / kelly + SpyFloor variants kelly_spy /
              topk_spy; select_top_k tie guard), metrics.py, report.py,
              survivorship.py
  live/       signals.py, ledger.py
  cli.py
```

Label: forward 5-trading-day return, open-to-open, minus the cross-sectional
median that week (so the task is *ranking* stocks, not predicting the market).
Features are rank-normalized to (-1,1] per week; prefixes: `f_` = model feature,
`aux_` = raw helper (never ranked), `f_evt_`/`f_mkt_`/`f_macro_`/`f_sec_` =
rank-exempt (time-only or binary). Only ALFRED-audited `T10Y2Y` and `FEDFUNDS`
macro features are admitted; revision-prone macro series and all sector-derived
features are excluded because Wikipedia sectors are not effective-dated. See
`reports/source_point_in_time_audit.md`. Metric: mean weekly Spearman rank
IC ("IC"). For scale: 0.01 is real, 0.02 is good, 0.05+ means suspect a bug.

## Iron rules (breaking these silently corrupts everything)

0. **Strategy evaluation leads with the HOLDOUT section/chart** (backtest.md's
   holdout table + reports/equity_holdout.png), never the since-2005 headline —
   the full-history window overlaps model selection. Owner-mandated.

1. **No lookahead.** Every feature at week t uses only data knowable at t's
  close: date-only EDGAR/Form 4 records become available the next calendar day;
  revision-prone FRED and non-effective-dated sector features are model-excluded;
   labels start the next trading day, CV has a 10-day purge gap, early stopping
   uses a time-ordered tail (never a random split). `tests/test_no_lookahead.py`
   proves this by corrupting future data and asserting past outputs unchanged —
   if it fails, fix the code, NEVER the test.
2. **The holdout (last 2 years) is untouchable.** No tuning or selection may see
  it. Optuna and random search select exclusively on the pre-holdout purged CV
  folds. Holdout results may be reported only after all choices are frozen.
3. **Champion eligibility:** a candidate needs a valid (non-NaN) IC in *every*
  folds, and the tournament falls back to the momentum baseline if no ML model
  beats the baselines. Watch the "test weeks" column in `models/selection.md`:
  healthy means every week in the dynamically derived evaluation calendar
  (488 in the current artifact). Less means the model produced constant
  (unrankable) predictions in some weeks — a degeneracy, not a virtue (see
  history #4).
4. **Tests green, zero warnings**, no network in tests (fetchers are injectable;
   fixtures only). Silence third-party noise via their own APIs, not warning filters.
5. Money math in the simulator is guarded: weights ≥ 0, sum ≤ 1, cost-netted
   buys, no leverage. `run_backtest` raises if a strategy violates this.

## Missing-data and feature-admission policy

The current 4-fold, 2015-03+ evaluation design is fixed. New features must fit
the benchmark; a feature's late start or sparse history is not a reason to move
fold boundaries or shorten the primary evaluation window.

1. Every candidate must produce rankable predictions for every week in the same
  evaluation calendar. The count is derived from the current panel, fixed
  `eval_start`, and rolling holdout rather than hard-coded, so it grows as new
  data arrives (the current artifact has 488 weeks). Never drop rows or weeks
  because an optional feature or prediction is missing or constant; incomplete
  coverage makes a candidate ineligible.
2. Distinguish structural pre-source absence, cross-sectional gaps, incidental
  retrieval failures, and economically meaningful absence. Track and report
  weekly and cross-sectional coverage by feature family.
3. Optional ranked features are neutral-filled after ranking (`0` in rank
  space) rather than causing observations to be dropped. Add point-in-time
  missingness indicators only where absence may carry information. Any fitted
  imputation parameters must use training data only.
4. Keep a stable, full-history core feature set. Evaluate additions by ablation
  on identical folds, weeks, and eligible stocks, with and without missingness
  indicators where appropriate.
5. Late-starting features must be tested both in the official full benchmark
  (neutral before availability) and, when useful, in a separately labeled,
  predeclared common-window ablation. The secondary window never replaces or
  mixes with the primary leaderboard.
6. Prefer improvements that are stable across folds, not gains concentrated in
  one period. Before production adoption, stress-test realistic random gaps
  and complete outages of the feature family.

Model-native missing-value support does not override this policy: it may handle
scattered company-level gaps, but it cannot justify missing weeks, unequal
scoring calendars, or all-missing folds.

## Hard-won history (why things are the way they are)

1. **automl_tool** (owner's package) selects by MAE, and the label's median is 0
   by construction — so it picks constant-predictors. Structurally excluded by
   the eligibility gate. Fix would be adding a scoring param upstream.
2. **Corrupt price data:** free sources garble delisted tickers (CPWR showed
   +300% "weeks" from broken split adjustments). `drop_corrupt_series` removes
   series with repeated split-sized jumps. Known false positive: GME (real
   squeeze) — accepted. ~200 delisted tickers (incl. Lehman) have NO free price
   data at all → residual survivorship bias, quantified in `data/manifest.json`
   and stress-tested in `stocks-ml torture` (verdict: removal-exit channel
   clean; missing-bankruptcies channel unquantifiable with free data).
3. **First backtests were invalidated twice** (corrupt data → fake 43% CAGR;
   then a constant-predictor champion). Treat any CAGR > ~25% or IC > ~0.05 as
   a bug until proven otherwise. `docs/run-notes-2026-07-11.md` documents both.
4. **The Optuna gaming incident:** short-interest data only exists from 2017-12.
   In earlier 5-fold CV, the 2012-15 fold had all-NaN short columns; some model
   fits collapsed to constant predictions there, those weeks were silently
   dropped from scoring, and Optuna found a config that "won" (fake IC 0.0281
   on 369/611 weeks; honest value 0.0143). Resolution (owner's design): drop
   that fold — 4 folds from 2015-03 — and keep short features (a controlled A/B
   showed they add ~+0.002 IC consistently). CatBoost still partially collapses
   (visible as 371 test weeks in selection.md) but loses, so it's tolerated.
5. **What moved the needle vs what didn't:** hyperparameter tuning (+42% rel.)
   and new features (+32% rel.) worked; insider data at weekly horizon, model
   zoo (LightGBM/CatBoost), and prediction-ensembles were nulls. Lesson: new
   *information* helps; new *algorithms* mostly don't. ElasticNet nearly ties
   XGBoost — the signal is largely linear in rank space.
6. **Old IC numbers are not comparable across evaluation designs.** The metric
   window/folds changed (5-fold-2012+ → 4-fold-2015+). Compare only within
   the current design.
7. **The survivorship reckoning (2026-08-19, Sharadar re-grade).** On the
   survivorship-clean world (official 500-member menus 1998+, 100% price
   coverage incl. delisted via suffixed tickers, dividend-adjusted opens),
   NO config beats SPY on pre-holdout Sharpe: best 0.57 (seccap4) vs SPY
   0.60; the old world claimed 0.73. SPY reproduces exactly across worlds
   (665 vs 666) — the build is validated; the historical edge was data
   artifact. What survives: the family ORDERING (seccap 0.57 > eased 0.54 >
   banded 0.45 ≥ k16 0.44 — the strategy-layer mechanics are real), and the
   holdout-era edge (1.20-1.27 vs SPY 1.11; near-complete data in both
   worlds). The honest thesis is now recent-era-only. Clean-world store:
   data/sharadar_world/; walk: data/walks/sharadar_champion.parquet. All
   pre-Sharadar absolute levels vs SPY are deprecated for decisions.
8. **The clean-world retrain (2026-08-19, post-reckoning).** Exam-objective
   ASHA on the Sharadar world (24 configs incl. incumbent, rungs incl. real
   GFC): incumbent params scored 0.391 on true 2005-2015 (not even top-8);
   winner = config-4 region (depth 2, lr 0.003, 500 trees, reg_alpha 1.34 —
   MORE conservative than the incumbent). Final = 7-member neighborhood-
   averaged book: pre-holdout SR 0.640 vs SPY 0.60, +2.7%/yr paired t=1.56,
   DSR 0.864 (N=554) — the first measured real historical edge; suggestive,
   not proven. Holdout 1.05 vs old-params-on-clean 1.25: era tension — old
   params look better recently, new params are real historically. Both
   should shadow-race on the clean pipeline. Final book:
   scratch world_final_book.parquet; neighbor walks data/walks/world_*. All
   six pending feature families re-rejected on clean data (t -1.4 to -2.3).
9. **The untuned-knobs campaign (2026-08-20, owner-directed).** Multi-objective
   Optuna (pre-tax SR + post-tax earnings, both from walked exams — no proxies)
   over the never-tuned XGB dimensions found a new champion region: depth-2 +
   lossguide + dart dropout + num_parallel_tree 4 + gamma pruning — every
   winning knob is another form of internal ensembling/skepticism. t12
   neighborhood-avg (7 books): pre-holdout SR 0.653 ($1,098 vs SPY $674),
   holdout 1.11 at SPY-equal 19% DD, +3.3%/yr paired t=1.79, DSR 0.882
   (N=562) — supersedes the config-4 region (0.640/t=1.56) as research
   champion. First config to beat SPY post-tax ($847 vs $536, t12 solo).
   Params: scratch optuna_world_finalists.json[0]; neighbors t12_neighbors
   .json; book t12_final_book.parquet; walks data/walks/world_t12*.
10. **Screens do not predict the exam (2026-08 model campaign).** Three screen
   designs — mean CV IC, top-8 realized excess on CV folds, and a tail-objective
   Optuna search — each produced winners (2-3x claimed improvements) that LOST
   the walked $100 exam by 0.08-0.21 Sharpe. Ten model challengers (lgbm,
   catboost, enet, RFF, MLP, 4y window, label variants, retuned params), zero
   wins; every book blend landed at 0.68-0.72 vs the champion's 0.73. Standing
   rule: walks are cheap (~40 min) — walk everything, screens are junk-filters
   only. The one live blend path: a neural sleeve at SR >= ~0.67 with rank corr
   ~0.1 (MLP hit 0.63/0.11, blend 0.72 — lost by 0.01); TabM is the candidate.
8. **The vol_scaled cash-lock (found 2026-08-18):** drawdown is measured vs the
   all-time peak, so a strategy at zero exposure freezes its NAV and its
   drawdown can never improve — the full stop was an absorbing state. It fired
   in COVID (2020-03-16, dd 32%) and vol_scaled sat in cash for the remaining
   6.4 years of every zoo backtest since. Fixed with a 13-week cool-off, then
   half exposure (`VolScaledTopK.REENTRY_WEEKS`); zoo reports generated before
   the fix understate vol_scaled. Lesson: any risk rule that can zero exposure
   must have a non-drawdown-based way back in.

## Data source quirks

- **yfinance:** unofficial; batches of 100, retries, per-ticker failures
  recorded in manifest (never fatal). Stooq fallback is DEAD (JS anti-bot wall).
- **Wikipedia** membership: changes-table columns are a MultiIndex with
  "Effective Date"; parser matches by containment.
- **EDGAR:** 10 req/s, User-Agent required. Fundamentals sparse pre-2010.
  Insider Form 4 via quarterly bulk zips (2006+). 90-day staleness refresh.
- **FINRA short interest:** free unauthenticated Query API, history ~2018+,
  publication lag = settlement + 14 days.
- Tiingo probed as backfill for missing delisted tickers: covers 114/202 but
  none of the 2008 casualties. Not integrated (decided against the dependency).

## Environment gotchas

- **iCloud syncs ~/Documents and repeatedly sets a hidden flag on `.venv`
  files; Python 3.12 then skips `.pth` files and `import stocks_ml` breaks.**
  Mitigations in place: `pythonpath = ["src"]` in pyproject (pytest immune) and
  a sitecustomize shim inside `.venv` (lost if venv is recreated). If imports
  break mysteriously, this is why. Durable fix: keep `.venv` out of iCloud.
- **iCloud also EVICTS large `data/*.parquet` contents under disk pressure**
  (2026-08-19: prices.parquet at 0 blocks, panel.parquet half-evicted).
  Symptom: `Parquet magic bytes not found in footer` on files that read fine
  hours earlier. Check `stat -f blocks:%b` before assuming corruption; fix
  with `brctl download` and verify blocks ≈ size/512. Never rebuild the store
  over an eviction — the local "corruption" is not data loss.
- macOS: no `timeout` command; GNU-isms differ.
- `models/`, `reports/`, `signals/`, `ledger.json` are git-TRACKED (audit
  trail); `data/` is not (rebuilt by ingest; CI caches it).

## Open items / likely next steps

- **Better data for survivorship (owner-requested 2026-08-12):** the monthly
  pipeline's outsized crisis-recovery returns (Apr-2009 picks: AIG/C/FITB/HBAN
  at $1.86-$23) lean on the distressed-rebound trade, exactly where the ~200
  missing delisted tickers flatter results most. A delisted-inclusive price
  source (Tiingo key exists — rotate it, was exposed in chat — or similar)
  is the fix. PROBED 2026-08-12 (reports/monthly_probes.md): a $5 price
  floor on picks collapses monthly_reg to SPY ($913 vs $930, Sharpe 0.65 =
  SPY) — the entire excess came from sub-$5 distressed survivors. Treat the
  monthly pipeline as unproven pending delisted-inclusive data.
- **Research-backed roadmap in docs/research/ (owner-supplied papers, 2026-08):**
  ranked actions in recommendations-2026-08.md — (1) momentum block past 12
  weeks, skip-adjusted (f_mom_26w, f_mom_52w_skip4w, f_mom_interm); (2) EDGAR
  bundle: f_sue, f_net_issuance, f_nincr (+R&D/mktcap), ablate on label_4w
  where fundamentals should shine; (3) trials ledger + paired-ΔIC t≥3
  adoption hurdle + DSR with cross-trial variance (metrics.py currently uses
  single-path variance — known deviation from Bailey-López de Prado). Also:
  IC≈0.02 is the published large-cap ceiling (GKX/GHZ/HXZ) — calibrate
  expectations; don't expand liquidity/idio-vol as alpha (HXZ veto);
  RankBlend(xgb,enet) and trimmed staggered ensemble are the two principled
  combination candidates (Timmermann).
- **Reviewer triage (2026-08-12, second-agent review of docs/research):**
  built now — trials ledger + cross-trial DSR + MinTRL (+ ENet shrinkage audit:
  the selected config is already max-viable-shrinkage — no action).
  **Ablations RUN 2026-08-12, all three families REJECTED at t>=3**
  (reports/ablation_*.md): momentum skip/interm ΔIC −0.005 (t=−0.94), EDGAR
  sue/nincr/issuance weekly −0.008 (t=−1.47), EDGAR on label_4w +0.0008
  (t=0.18 — Novy-Marx's horizon prediction shows the right SIGN but is noise).
  The six candidates stay in PENDING_ABLATION_FEATURES permanently unless
  re-tested with new evidence; plain 26w/52w momentum already in the panel
  likely spans the skip-adjusted variants. Still queued: rank-blend xgb+enet
  precheck; decile-spread diagnostic. Deferred pending owner design call:
  train-window-length candidates; selection-stage IC deflation column.
- **Banded top-k (2026-08-18, pre-registered, cached-walk study):** BandedTopK
  in strategies.py (enter top-k, hold until rank decays past exit band —
  hysteresis against noise churn). banded(16,32) beats plain k16 on EVERY
  column, same panel vintage: pre-holdout $1,848/SR 0.64/DD 64% vs
  $1,194/0.57/66%; holdout $233/SR 1.28/DD 33% vs $198/1.05/36%; costs
  $222 vs $228. Recommended live_strategy change AT THE NEXT SCHEDULED
  REVIEW (DEPLOYMENT.md rule 4 — not mid-race, owner decision); both
  configs are ledger trials.
- **DEPLOYMENT.md (2026-08-18):** pre-registered sizing/kill/promotion/
  change-budget contract; weekly signals now append a shadow-race
  scoreboard (race_status in ledger.py). Membership ingest hardened:
  falls back to stored data on fetch failure or implausible swings
  (>15 tickers/week), hard-fails after 3 consecutive fallback weeks.
- Let the shadow ledger accumulate (Saturday cycle) before real money — this
  was always the plan; backtest numbers lean on friendly fill assumptions.
- Possible research: honest Optuna re-run on the current 4-fold design;
  multi-horizon models (monthly would suit fundamentals/insider data);
  EDGAR filing-text features ("Lazy Prices"); VIX-scaled transaction costs.
- CatBoost fold-collapse could be fixed or the family dropped.

## Where to look

- `docs/superpowers/specs/…-design.md` — original system spec
- `docs/superpowers/plans/…` — the 19-task build plan
- `docs/run-notes-2026-07-11.md` — first real-data run + invalidated results
- `models/selection.md`, `models/tuning_*.md` — current leaderboards
- `reports/backtest.md` — dollar-terms results; `reports/survivorship_torture.md`
- `.superpowers/sdd/progress.md` — full build/decision ledger (gitignored, local)
