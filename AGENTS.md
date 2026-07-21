# stocks_ml — agent context

ML system that forecasts S&P 500 stock returns weekly, backtests investment
strategies, and runs a live paper-trading account. Built entirely on free data.
The owner's goal: turn $100 into more, without ever blowing up the account.

## Current state (2026-07-20)

- **Champion model:** tuned XGBoost (`models/champion.json`, mean weekly rank
  IC 0.0241 across 4 CV folds, all positive, full 488-week coverage).
- **Evaluation design (owner-specified, do not change casually):** 4 walk-forward
  CV folds testing 2015-03 → 2024-07 (`eval_start: 2015-03-01`, `n_cv_folds: 4`
  in config). Training always reaches back to 2005. The last 2 years
  (2024-07 → now) are a **holdout** never used for tuning or selection.
- **Live:** GitHub Actions runs the weekly cycle every Saturday 13:00 UTC
  (signals + paper ledger, committed back to the repo) and a monthly retrain on
  the 1st. Live strategy is `vol_scaled` (config `live_strategy`); there is an
  open, evidence-backed recommendation to switch to `kelly` — see
  `reports/backtest.md` (kelly: $100→$1,216 since 2005, best deflated Sharpe).
- Latest backtest: equal_topk ≈ market with 76% max drawdown; kelly beats
  market modestly with half that pain; vol_scaled barely earns but rarely loses.

## Commands

```bash
uv sync                      # install (Python 3.12; automl_tool needs <3.13)
uv run pytest                # 201 tests; MUST stay green with 0 warnings
uv run stocks-ml ingest      # fetch all data + rebuild panel (idempotent)
uv run stocks-ml tune --family xgb|lgbm|catboost|enet [--optuna]
uv run stocks-ml train       # champion tournament -> models/selection.md
uv run stocks-ml backtest    # -> reports/backtest.md
uv run stocks-ml signals     # weekly live signal
uv run stocks-ml ledger init|apply|mark|show
uv run stocks-ml torture     # survivorship stress test
```

Always `git pull` before local work — CI commits artifacts (signals/, ledger.json,
models/, reports/) on its own schedule.

## Architecture

```
src/stocks_ml/
  data/       store.py (parquet DataStore + manifest), prices.py (yfinance,
              corrupt-series filter), membership.py (point-in-time S&P 500 from
              Wikipedia), fred.py (macro, publication-lagged), edgar.py
              (fundamentals, filing-dated), insiders.py (Form 4), shortint.py (FINRA)
  features/   panel.py (build_panel = the one place features/labels are made),
              fundamentals.py, events.py, insiders.py, ranking.py
  models/     cv.py (purged walk-forward CV, weekly rank IC), candidates.py
              (model zoo + wrappers), tuning.py (random search),
              optuna_tuning.py (TPE, holdout-judged), champion.py (tournament)
  backtest/   simulator.py (no-lookahead walk-forward), strategies.py
              (equal_topk / vol_scaled / kelly), metrics.py, report.py,
              survivorship.py
  live/       signals.py, ledger.py
  cli.py
```

Label: forward 5-trading-day return, open-to-open, minus the cross-sectional
median that week (so the task is *ranking* stocks, not predicting the market).
Features are rank-normalized to (-1,1] per week; prefixes: `f_` = model feature,
`aux_` = raw helper (never ranked), `f_evt_`/`f_mkt_`/`f_macro_`/`f_sec_` =
rank-exempt (time-only or binary). Metric everywhere: mean weekly Spearman rank
IC ("IC"). For scale: 0.01 is real, 0.02 is good, 0.05+ means suspect a bug.

## Iron rules (breaking these silently corrupts everything)

1. **No lookahead.** Every feature at week t uses only data knowable at t's
   close: EDGAR facts join on *filing* date, FRED series are publication-lagged,
   labels start the next trading day, CV has a 10-day purge gap, early stopping
   uses a time-ordered tail (never a random split). `tests/test_no_lookahead.py`
   proves this by corrupting future data and asserting past outputs unchanged —
   if it fails, fix the code, NEVER the test.
2. **The holdout (last 2 years) is untouchable.** No tuning or selection may see
   it. Optuna adoption is judged on it (`optuna_tuning.py`) — that is its only use.
3. **Champion eligibility:** a candidate needs a valid (non-NaN) IC in *every*
   fold, and the tournament falls back to the momentum baseline if no ML model
   beats the baselines. Watch the "test weeks" column in `models/selection.md`:
   healthy = 488. Less means the model produced constant (unrankable)
   predictions in some weeks — a degeneracy, not a virtue (see history #4).
4. **Tests green, zero warnings**, no network in tests (fetchers are injectable;
   fixtures only). Silence third-party noise via their own APIs, not warning filters.
5. Money math in the simulator is guarded: weights ≥ 0, sum ≤ 1, cost-netted
   buys, no leverage. `run_backtest` raises if a strategy violates this.

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
- macOS: no `timeout` command; GNU-isms differ.
- `models/`, `reports/`, `signals/`, `ledger.json` are git-TRACKED (audit
  trail); `data/` is not (rebuilt by ingest; CI caches it).

## Open items / likely next steps

- Decide `live_strategy`: stay `vol_scaled` (safe, near-zero return) or switch
  to `kelly` (evidence-backed recommendation). One line in config/config.yaml.
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
