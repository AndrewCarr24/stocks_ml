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
