# Simplification log (2026-09-13)

The owner's directive: keep only the essential processes — training,
backtesting, eval, the interactive app — as code, navigable and
reproducible; everything else goes. The tree before this work is at the git
tag `pre-simplify` (commit a1d59c5); anything removed below can be
recovered from it (`git show pre-simplify:<path>`).

One commit per step. Each entry says what moved where, so a reader of the
old reports or ledger rows can find the code that produced them.

## Step 1 — the processes as commands (new code)

| new | what it is | replaces |
|---|---|---|
| `src/stocks_ml/train.py` — `stocks-ml train` | the walk: the champion model refit every week, K copies, `<out>/preds.parquet` + `<out>/spec.json` record, resumable under a guard; `--check` refits a saved walk's first weeks and compares | `ops/clean_program.py` (`walk`, `context`, the Stage E record), `selection.stage_wsweep`/`stage_holdings` |
| `src/stocks_ml/backtest.py` — `stocks-ml backtest` | a walk through the strategy: the owner's table ($100, %/yr, SR, DD on 2006-2015 / 2016-2024 / 2006-2024, sp500 row, paired weekly t); settings from the spec's decision unless overridden | `ops/clean_program.py` (`_load`, `_sim`, `_row_vs_spy`, `table_md`) |
| `src/stocks_ml/leak_audit.py` | the identity gate (f_sf ratio recomputed at a segment's middle week) plus the report-only factor and delisting statistics, per segment | `ops/leak_audit.py` (its shifted-feature retention gate was tied to the retired generated bundle's inputs; the identity check is the champion's own mechanism) |
| `src/stocks_ml/eval.py` — `stocks-ml eval` | the one look: table, incumbent + falsification, leak audit, 95% intervals (seed / history / nested), charts, `<walk>/eval.json`, `reports/<label>_eval.md`, ledger row | `ops/clean_program.py` (`stage_e`, `confidence`, `stage_e_md`, `champion_chart`) |
| `src/stocks_ml/app/build.py` + `app.html` — `stocks-ml app` | the champion's explorer page from the spec's walk; the notes are read off the spec and `<walk>/eval.json`, never typed | `app/oos/build.py` + `app/oos/app.html` (four variants: oos, oos_x, select, champion) |
| `src/stocks_ml/cli.py` | eight commands: world, train, backtest, procedure, procedure-card, eval, app, r5-weekly | the old cli (`select`, `procedure`, `procedure-card`, `r5-weekly`) |
| `tests/test_train.py`, `test_backtest.py`, `test_eval.py`, `test_leak_audit.py`, `test_app.py`, `tests/e2e/test_app_e2e.py` | tests of the new modules | `tests/test_oos_app.py`, `tests/e2e/test_oos_app.py` |

Walk layout: `<walk>/select/preds.parquet` (2006-2015) + `<walk>/extend/preds.parquet`
(2016 → 2024-07-18), each with `spec.json`; `<walk>/procedure.json` caches
the procedure's decision on the select segment; `<walk>/eval.json`,
`<walk>/eval_weekly.parquet`, `<walk>/holdings_k16.parquet` are written by
eval and app. The champion's walk is `data/experiments/clean_program/stage_e/ls_w8`
(named by the spec's `procedure.preds.path`).

Verified before the commit: `stocks-ml backtest` on the champion's two
segments prints the spec's record exactly ($1,994 / $660 / $302 vs $621 /
$316 / $197; t +1.90 / +1.52); `stocks-ml train --check` refits copy 1 of the
first two weeks and matches to 1e-6; `stocks-ml eval` reproduces the table,
the leak verdict and the intervals; `stocks-ml app` builds the page.
`pyproject.toml` gained a non-default `eval` group (matplotlib) for the
charts.

## Step 2 — the removals (9,583 lines deleted, 119 added)

Everything below is at tag `pre-simplify`; the ledger rows and reports
that cite these tools name them as they were.

| removed | what it was | where the process lives now |
|---|---|---|
| `ops/clean_program.py`, `nominal_program.py`, `k16_program.py` | the research programs (stage B–E, the nominal rebuild, the K=16 regrade) | `stocks-ml train` / `backtest` / `eval` / `procedure` |
| `ops/leak_audit.py` | the leak audit with the shifted-feature retention gate | `leak_audit.py` inside `stocks-ml eval` (identity gate; factor and delisting report-only) |
| `ops/openfe_arm*.py`, `tails_exam*.py`, `tails_features_probe.py`, `decline_features_probe.py`, `champion_bundle_regrade.py`, `fill_basis_regrade.py`, `regrade_campaign.py`, `k16_seed_spread.py`, `leverage_exam.py`, `live_emulation.py` | one-off exams and regrades (each has a report at the tag and a ledger row) | a challenger is now a walk: `train` → `backtest` (selection metric) → `eval` |
| `ops/com.stocks-ml.r5-weekly.plist` | the Mac launchd schedule for the weekly job | GitHub Actions (`.github/workflows/champion.yml`); `ops/r5_weekly.sh` stays for a manual Mac run |
| `app/oos/` | the four explorer variants | `stocks-ml app` (`src/stocks_ml/app/`) |
| `src/stocks_ml/feature_screen.py`, `features/candidates.py`, `features/derived.py`, `features/generated.py`, `features/bundle.py` | the feature screen, the `x_` candidate ideas, the generated (`g_`) bundle and its formulas | none: the model's inputs are the panel's `f_` columns; the spec's `features` list (empty) names any extra panel column, read by `live/r5.py` and `models/walk.py`; a new feature set is a challenger model, admitted by the selection metric |
| `selection.py`: `stage_grid`, `stage_wsweep`, `stage_holdings`, `stage_screen`, `run_cascade`, `run_select`, `decide_engine`, `decide_horizon`, `decide_window`, `load_windows`, `sample_weeks`, `holdings_name`, `_stage_*`, `_load`, `WINDOWS`, `REF_WINDOW` | the cascade's search stages (horizon and window menus; the 2026-09-01 procedure) | the model (label, window) is chosen by the argmax of the selection metric across candidate walks; the strategy layers by `decide_strategy` (unchanged) |
| tests: `test_bundle`, `test_candidates`, `test_derived_features`, `test_feature_screen`, `test_fill_basis_regrade`, `test_generated`, `test_live_emulation`, `test_regrade_campaign`; eight cascade tests in `test_selection.py`; one in `test_procedure.py` | tests of the removed code | — |

Edits to kept code: `data/world.py` no longer adds `x_`/`g_` columns to the
live `panel_sf` (the model reads `feature_cols` only); `live/r5.py` reads
`features` from the spec instead of `features/bundle.py`; `backtest.run`
prints the **selection metric** (cost-adjusted compounded %/yr per book on
2006-2015, `selection.decide_book` — the number the procedure's book layer
reads, so a challenger model's admission is code); `procedure_card` and
its test lost the bundle wording; `tests/e2e/test_app.py` renamed
`test_app_e2e.py` (pytest basename clash with `tests/test_app.py`).

Verified after the removals: 245 tests pass; `stocks-ml procedure --check`
matches the spec; `stocks-ml backtest` on the champion prints the record and
the selection metric {3: +2.14, 6: +7.86, 10: +8.11} equal to the
procedure's book evidence; `stocks-ml r5-weekly --no-refresh --dry-run
--as-of 2026-08-28` ranks 502 names in 22 s with `features: []` from the
spec (the Mac's live store ends 2026-09-01; the Saturday job refreshes).
`selection.py` went from 700 to 412 lines.
