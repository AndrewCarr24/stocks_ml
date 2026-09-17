# stocks_ml — agent context

ML system that ranks S&P 500 stocks on a one-month horizon, wraps the ranking
in a rules-based long-only book with an index/Treasury ballast, and runs that
system every week as a $100 paper portfolio. Point-in-time Sharadar data; a
mechanical selection procedure written as code; a sealed holdout from
2024-07-19. The owner's goal: turn $100 into more without ever blowing up
the account (a Roth IRA — capital preservation is the binding constraint).

**Simplified 2026-09-13.** The tree holds only the essential processes:
training, backtesting, evaluation, the interactive explorer, and the live
weekly job. Every research driver, campaign report and explorer variant
that produced the champion was deleted, not rewritten; all of it is at the
annotated tag `pre-simplify` (`git show pre-simplify:<path>`,
`git worktree add ../pre-simplify pre-simplify`). The legacy one-week
pipeline is at `legacy-final`. [docs/simplification_log.md](docs/simplification_log.md)
maps what was removed to where it went.

## Current state (2026-09-13)

- **Champion since 2026-09-14: the rank-label model (`label_4w_sector_rank`).**
  Depth-3 XGBoost, fixed parameters, on the panel's 64 `f_` columns
  (nominal price basis, last-print labels, no extra features), trained on
  the stock's 4-week return minus its sector's same-week median **replaced
  by its within-week rank as a normal score**, over a **trailing 8-year
  window**, 16 copies averaged. Strategy layers decided by `stocks-ml
  procedure` on its K=16 walk of 2006-2015: **top-3 per sleeve / halfgate /
  no stop / cap 2**. The walk: `data/experiments/labels/rank_k16/{select,extend}`.
  Chosen by the challenger flow (README "Challenging the champion") from
  three tempered labels: rank beat the incumbent on the selection metric
  (three-book mean +9.8 vs +8.7 at K=4; paired t +0.24 reported).
- **Record** (`stocks-ml eval` 2026-09-14, reports/champion_eval.md; each
  model at its own settings; SPY on the same weeks): 2006-2015 $585 vs $197
  | 2016-2024 $844 (+28.2%/yr, SR 0.92, DD 38%) vs $316 | 2006-2024 $4,940
  (+23.4%/yr, SR 0.82, DD 49%) vs $621; paired weekly t vs SPY +2.53
  (2006-2024) / +1.71 (2016-2024); nested 95% CI on the 2016-2024 excess
  −4.8..+33.1 around +12.2%/yr, P(excess>0) 0.92; falsification test vs
  the previous champion (ls_w8) t +0.76, not rejected; leak audit PASS.
  Ledger rows `label_test_2006_2015_k4_*`, `procedure_2006-01-01_2015-12-31_
  labels_rank_k16_select`, `eval_champion_rank_k16_k16`.
- **Before it** (2026-09-12 → 14): ls_w8 — the same model on the raw
  sector-relative label, top-10 / halfgate / no cap: $302 / $660 / $1,994
  (SR 0.53 / 0.88 / 0.70; DD 63 / 33 / 63%). Its walk stays at
  `data/experiments/clean_program/stage_e/ls_w8` (with a 2002-2005 `pre`
  segment for the rolling experiment).
- **The rolling procedure (2026-09-13, `backtest --rolling` /
  `procedure --lookback`, src/stocks_ml/rolling.py):** the strategy layers
  re-decided every week on the trailing k years of the walk's own
  out-of-sample predictions (the walk gained a 2002-2005 `pre` segment for
  the burn-in), k chosen on 2010-2015 among {3, 5, 8, expanding}. Result
  (ledger `rolling_lookback_2010-2015_trailing_5_c1`): the trailing rules
  flip every 5-7 weeks and sit on the spec's configuration 0-6% of the
  time; the expanding rule is stable (19 changes) but converges to
  6 / halfgate / stop / cap 2, never the spec's; on 2010-2015 every rule
  and the fixed settings (+7.8%/yr, in-sample) trailed SPY (+13.0%).
  The chosen rule, trailing 5, on 2016-2024: $610 (+23.5%/yr, SR 0.89,
  DD 41%) vs the fixed settings' $660 (+24.6%, 0.88, 33%), paired weekly
  t −0.35. The layers are a coin flip on any window; the model carries the
  edge. The spec is unchanged. Outputs under `<walk>/rolling/` (data/,
  git-ignored).
- **Before it** (2026-09-11 → 12): the same clean model with the
  week-centred label on a 5-year window, top-6 / 60-40 / cap 2: $229 /
  $449 / $1,028. Before 2026-09-11: a split-leak-contaminated bundle model
  ($4,256 on 2006-2024) — history for scale only, at the tag.
- **The paper ledger** (ledger_r5.json, signals_r5/) has run since
  2026-09-01; the first live signal on the rank-label champion will be 2026-09-19.
  Real money waits for the ledger to accumulate.
- **The holdout (2024-07-19 →)** has never been read. It is graded once, on
  the owner's go, with the champion frozen; the result never feeds
  re-selection.

## The tree

```
src/stocks_ml/
  cli.py            world | train | backtest | procedure | procedure-card | eval | app | r5-weekly
  train.py          stocks-ml train: the walk (weekly refit, K copies) -> <walk>/<seg>/preds.parquet + spec.json
  backtest.py       stocks-ml backtest: a walk through the strategy -> the owner's table vs sp500
  procedure.py      stocks-ml procedure: decide_strategy on the walk's 2006-2015 segment -> the spec
  procedure_card.py stocks-ml procedure-card: PROCEDURE.md from the spec
  eval.py           stocks-ml eval: table, 95% intervals, falsification vs an incumbent, leak audit, charts
  leak_audit.py     the audit eval runs: identity gate + report-only factor/delisting statistics
  selection.py      the engine both research and live share: Ctx (a world in memory), MODEL_PARAMS,
                    ensemble_preds (the model call), ensemble_holdings + simulate (the backtest),
                    decide_strategy (the layers), metrics, HOLDOUT_START, K_COPIES
  ledger.py         the live rules and paper ledger: sleeves, ballast (FLOORS), fills at 5 bp, NAV
  app/build.py      stocks-ml app: replays simulate with its trace hook into app.html
  live/r5.py        stocks-ml r5-weekly: refresh, rank (ensemble_preds), rotate, ledger, report
  data/             store.py (parquet DataStore), world.py (refresh + panel rebuild), sharadar.py,
                    membership.py, prices.py, edgar.py, sec8k.py, shortint.py, fred.py
  features/         panel.py (build_panel: the one place features and labels are made),
                    fundamentals.py, events.py, insiders.py, sharadar_fundamentals.py, ranking.py
  models/           xgb.py (TimeTailEarlyStopXGB), walk.py (walk_forward_predictions),
                    replication.py (WeekBootstrapEstimator, the K-copy protocol), trials.py (the ledger)
config/config.yaml  the panel recipe (price basis, delisting rule, purge, FRED lags)
models/             champion_spec.json (the deployed spec; decision fields written only by
                    stocks-ml procedure), trials_ledger.json (every evaluated configuration)
PROCEDURE.md        generated from the spec (stocks-ml procedure-card)
reports/            champion_eval.md + the two champion-vs-SPY charts (stocks-ml eval);
                    clean_improvement.md (the program that chose the label and window)
.github/workflows/champion.yml   the Saturday job (the only workflow)
ops/                r5_weekly.sh (the cycle by hand on the Mac), r5_seed.sh (seed the Actions cache)
signals_r5/, ledger_r5.json      the paper record, written by the job
tests/              pytest; tests/e2e (Playwright, the explorer page; skipped without playwright)
data/               git-ignored: the research world data/sharadar_world2000_nominal_dl, the live
                    world data/r5_live, walks under data/experiments/, the Sharadar key
docs/simplification_log.md       what the 2026-09-13 simplification removed and where it is
```

## Commands

```bash
uv sync                                  # Python 3.12; `uv sync --group eval` adds matplotlib
uv run pytest                            # must stay green with zero warnings; no network
uv run stocks-ml world [--dir data/r5_live] [--no-refresh] [--no-sec]
uv run stocks-ml train --out <walk>/select --start 2006-01-01 --end 2015-12-31 [--label L] [--train-years N] [--k 16] [--every 1] [--features a,b] [--params name=v,...] [--workers 4]
uv run stocks-ml train --check <walk>/select/preds.parquet          # refit the first weeks, compare
uv run stocks-ml train --out <walk>/twin --copies 17-32 ...         # the champion's seed twin (challenge-fast builds it)
uv run stocks-ml backtest --preds <walk>/select/preds.parquet <walk>/extend/preds.parquet [--book/--floor/--stop/--cap/--k]
uv run stocks-ml procedure --preds <walk>/select/preds.parquet [--check]   # writes the spec + PROCEDURE.md
uv run stocks-ml eval [--walk W] [--incumbent I] [--ci-draws 200] [--no-charts]
uv run stocks-ml app                                                 # reports/champion_explorer.html
uv run stocks-ml challenge --out <dir> --candidate label=L --candidate train_years=N [--candidate "features=a+b"] [--candidate "params=name:v"] [--k16]
uv run stocks-ml challenge-fast --out <dir> --candidate ... [--per-year 26] [--seed 0]   # the prototype: stratified sample, K=16, seed-twin null, ranks only
uv run stocks-ml r5-weekly [--as-of F] [--no-refresh] [--no-sec] [--dry-run] [--commit]
/opt/homebrew/Caskroom/miniconda/base/bin/python -m pytest tests/e2e  # the Playwright page test
```

`train`, `backtest`, `procedure`, `eval`, `app`, `challenge` and `challenge-fast` read the research world
`data/sharadar_world2000_nominal_dl` (nominal basis, last-print labels;
built once, never refreshed). `r5-weekly` reads `data/r5_live` and needs
the Sharadar key (`data/.sharadar_key` or `SHARADAR_API_KEY`). The Mac has
no `gh`; push with `GIT_TERMINAL_PROMPT=0 git push origin main` (the token
comes from the keychain through `git credential fill`; never print it).

## Iron rules (breaking these silently corrupts everything)

1. **No lookahead.** Every feature at week t uses only data knowable at t's
   close: date-only EDGAR/Form 4 records become available the next calendar
   day; FRED series carry publication lags; labels start at the next open;
   training is purged past the label's span (35 days for the 4-week label);
   early stopping uses a time-ordered tail. Level features use the nominal
   close and split-only adjusted per-share ratios, so a future split never
   reaches the model (the 2026-09 split leak). `tests/test_no_lookahead.py`
   corrupts future data and asserts past outputs unchanged — if it fails,
   fix the code, never the test.
2. **The holdout (2024-07-19 onward) is untouchable.** `train`, `backtest`,
   `eval`, `leak_audit` and `app` refuse it in code. It is read once, on the
   owner's go, after every choice is frozen.
3. **Selection is mechanical.** A candidate model is a recipe (label,
   window, features, params): prototyped by `stocks-ml challenge-fast`
   (stratified sample, 26/yr, K=16, gap vs the luckier of the incumbent's two seed sets, flagged above the 90th percentile of the centred seed-twin null; ranks only), then `stocks-ml challenge` (every week, K=16): the
   argmax of the model score (the mean over the top-3/6/10 books of the
   cost-adjusted compounded %/yr; owner's rule 2026-09-14) on 2006-2015
   alone, incumbent included; the strategy layers are
   `selection.decide_strategy` (book by cost-adjusted compounded %/yr;
   floor, stop, cap by Sharpe, stop and cap adopted only if higher). t
   statistics are reported, not gated. A doubt about a winner becomes a
   pre-registered falsification test, never a discretionary override.
   2016-2024 is read once per candidate (`stocks-ml eval`).
4. **The spec is written by code.** `models/champion_spec.json`'s decision
   and model fields, and `PROCEDURE.md`, come only from `stocks-ml
   procedure`; tests fail on a hand edit of the spec or of the live job's
   settings. Prose fields in the spec may be edited.
5. **One engine.** `selection.simulate` and the live job share
   `ledger.py`: weights ≥ 0, sum ≤ 1, sells before buys, 5 bp a side, never
   overdrawn, no leverage. A change to the live rules is a change to the
   backtest.
6. **Every week must be rankable.** The live job refuses to signal on fewer
   than 100 rankable names or an empty ensemble; a walk skips a week with
   fewer than 20 distinct scores.
7. **Tests green, zero warnings, no network in tests.** Silence third-party
   noise through their own APIs, not warning filters.
8. **Leak audit before believing.** A strong result triggers
   `stocks-ml eval`'s audit; adoption needs PASS.

## The owner's standing instructions

- **Charts show 2016-2024 (out of sample) only.** Never a curve that
  includes 2006-2015: those years chose the settings and the picture
  overstates the model (owner, 2026-09-14). `stocks-ml eval` writes one
  chart, the app replays 2016 -> holdout.
- **Sample first.** Exploration runs on samples (`train --every N --k 4`);
  full-population runs only on an explicit go, and a go covers only the run
  the owner clearly saw — restate what runs, on which years, at what cost.
  No frozen decision may read a sample (the procedure refuses them).
- **Propose before implementing.** Say what will be tried and why; wait.
- **Same-basis comparisons.** Never compare compounded with arithmetic
  returns, or different windows, frequencies or cost rules; every
  performance table carries the sp500 row on the same weeks.
- **Judge by spreads, not one number.** A 6-10 name book's terminal wealth
  carries a ~±30% path band; read the K-copy spread and the paired weekly
  t, never a single terminal figure.
- **Emulate live.** Backtests run the live ledger's rules; measure any gap
  and move the backtest toward live.
- **Commit only when asked.** Never print the Sharadar key or the GitHub
  token; nothing under `data/` and no built explorer page is ever staged.
- **Writing.** Answer first, short sentences, say what the numbers mean in
  words, no insider shorthand.
- Do not scrap the champion; do not re-tune on a calendar (nested
  experiments showed calendar re-selection adds drawdown without reliable
  return). Re-selection happens on a structural trigger only: a new data
  source passing its gate, a pre-registered kill criterion, or the owner.

## Data source quirks

- **Sharadar Direct** (api.sharadar.com/v1.0/data/{table}, `x-api-key`
  header; never data.nasdaq.com — the key is foreign there): `ticker`
  filter ≤ 30 tickers per call; `lastupdated.gte` on SEP/SFP returns only
  re-adjusted or new rows (a dividend re-adjustment bumps every row of a
  ticker, so the upsert is exact); SF1 `lastupdated` is bumped for a
  ticker's whole history (refetch in full); SF2 filters on the filing date;
  a renamed company keeps its `permaticker`. Point-in-time join for
  fundamentals: filing `date` ≤ the decision Friday, ARQ/ART dimensions
  only (MR* rows are restated backward — lookahead). SF1 per-share fields
  are split-restated: divide by the split-only adjusted close.
- **EDGAR:** 10 req/s, User-Agent required; fundamentals sparse before
  2010; SEC ticker format BRK-B vs Sharadar BRK.B (`normalize_symbol`).
- **FINRA short interest:** history ~2018+, publication lag = settlement
  + 14 days. **FRED:** only ALFRED-audited `T10Y2Y` and `FEDFUNDS` are
  features; the rest are revision-prone.
- **IEF** (the ballast fund) must not be in the panel's price frame
  (`f_mkt_dispersion` is a cross-section over every series).

## Environment gotchas

- The repo lives at `~/Documents/projects/stocks_ml.nosync` (iCloud never
  syncs a `.nosync` path); `~/Documents/projects/stocks_ml` is a symlink to
  it. Keep both. iCloud used to evict `.venv` files and `data/*.parquet`
  contents (0%-CPU hangs, "Parquet magic bytes not found"); the cure is a
  per-file `brctl download`, or a sequential `cp` to local tmp and a swap.
  Never delete a `*.parquet.evicted` placeholder.
- macOS: no `timeout` command, no `gh`. uv is at
  `/opt/homebrew/Caskroom/miniconda/base/bin/uv`; the e2e test runs under
  the miniconda Python (playwright is installed there).
- `models/`, `reports/`, `signals_r5/`, `ledger_r5.json` are tracked;
  `data/` and `reports/champion_explorer.html` are not (licensed data).
- **Seed luck is ~2.5 points on the model score at K=16.** The champion's
  seeds 1-16 score +13.3 on the full 2006-2015 walk, seeds 17-32 (its
  twin, `<walk>/twin/`) +15.8. No comparison at K=16 can call a gap inside
  that; `challenge` and `challenge-fast` count both seed sets as the
  incumbent (2026-09-16).
- `train --workers N` fits copies in N spawned processes, each with
  cpu_count // N XGBoost threads (a depth-3 hist fit cannot use 14 cores);
  the fits are identical, only the wall time changes (bench 2026-09-14: 32 s/week
  serial, 14 s at 4 workers, 12 s at 7, bit-identical). Default 7 on the CLI,
  1 in-process for tests.
- A DataFrame column built from a scalar `Timestamp` is datetime64[s] in
  pandas 2.3; `merge_asof` against [ns] fails — build it as a list.

## Open items

- Let the paper ledger accumulate before real money; write the r5
  sizing/kill/promotion contract before any real order.
- The holdout exam (2024-07-19 →): once, on the owner's go.
- Not started, owner's call: the `ls_w8_x` arm (the package with 16
  engineered features — a K=16 walk would be a second read of 2016-2024);
  a taxable post-tax table (the account is a Roth, so post-tax = pre-tax
  today); the character study (the edge sits in high-volatility,
  beaten-down names, mostly after 2016).
- The incumbent walks `data/experiments/nominal_clean_2006_2015_dl` /
  `_2016_2024_dl` predate `stocks-ml train` and carry no `recipe` in their
  records; `eval --incumbent` on them needs one added (label_4w, 5 years).

## Where to look

- `PROCEDURE.md` — the champion's procedure card (generated).
- `models/trials_ledger.json` — every evaluated configuration, in order.
- `reports/clean_improvement.md` — the program that chose the champion's
  label and window; `reports/champion_eval.md` — the current grade.
- `docs/simplification_log.md` — the 2026-09-13 simplification.
- Tag `pre-simplify` — every research driver (`ops/*.py`), report,
  explorer variant and paper note that preceded the simplification, with
  the full history in its AGENTS.md. Tag `legacy-final` — the legacy
  one-week pipeline.
