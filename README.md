# stocks_ml

`stocks_ml` ranks S&P 500 stocks on a one-month horizon with a small gradient-
boosted model, wraps the ranking in a rules-based long-only book with an
index/Treasury ballast, and runs that system every week as a $100 paper
portfolio. Its selection procedure is mechanical and pre-registered, its data
is point-in-time, and the last two years are a sealed holdout.

There is no broker integration. The weekly job proposes a book; a human
places the orders.

## The champion

The production system is `r5` — declared 2026-09-01 after a month-horizon
rebuild, specified in [models/champion_spec.json](models/champion_spec.json)
and summarized in [PROCEDURE.md](PROCEDURE.md).

| Component | Spec |
|---|---|
| Model | depth-3 XGBoost, untuned by design (hyperparameter search measured as noise) |
| Target | the stock's 4-week open-to-open return minus that week's median member's; 35-day purge |
| Training | weekly refit on the trailing 5 years; early stopping on a purged, time-ordered tail's weekly rank correlation |
| Ensemble | K=4 copies (seed + whole-week bootstrap), predictions averaged |
| Book | top-6, equal weight, four staggered sleeves — one rotates each week, so every name is held four weeks; max 2 per sector |
| Ballast | 70% book / 30% ballast; ballast in SPY, shifted to IEF one third per breached SPY trailing mean (30/40/52 weeks) |
| Costs | 5 bp one-way, fills at Monday's open |

**Record, 2006 → 2024-06 (selection window, costs included):** $100 →
**$1,586**, +16.8%/yr, Sharpe 0.74, max drawdown 58% — vs SPY $563,
+10.2%/yr, Sharpe 0.62, drawdown 54%. Read this with the spec's own caveat:
the edge is era-concentrated (strong 2013–2020, index-like in whipsaw and
mega-cap regimes), every number from the selection window carries
design-iteration shine (measured at about +3.8%/yr on dollars, ~0 on Sharpe),
and sizing should assume SPY-like outcomes in adverse regimes. The holdout
(2024-07-19 onward) has not been graded; it is a single-use exam.
[reports/r5_package_2006_2024.png](reports/r5_package_2006_2024.png) shows two
of the package's ballast variants against SPY over the same window.

### How it was chosen

The procedure is a fixed cascade run on the selection window only, one
decision per layer, each by a metric declared in advance:

| step | menu | decided by |
|---|---|---|
| horizon | 1w vs 4w | cost-adjusted compounded return of the top-6 book |
| training window | 1–5 years | top-6 edge vs a random basket on identical weeks |
| book size | top-3 / 6 / 10 | cost-adjusted compounded return |
| stagger | always on | mechanism, not searched |
| ballast | none / half-gate / 80-20 / 70-30 / 60-40 | Sharpe |
| stop-loss | off / −25% | Sharpe (adopt only if higher) — off |
| sector cap | off / 2-of-book | Sharpe (adopt only if higher) — on |

`stocks-ml select` runs it end to end. The cascade was validated by a nested
test ([reports/nested_selection_protocol.md](reports/nested_selection_protocol.md)):
run on 2006–2015 alone it chose a near-champion configuration, whose frozen
grade on 2016 → 2024-07 was $497 (+21.4%/yr, Sharpe 1.00, drawdown 36%) vs
SPY $295 (+14.0%/yr, 0.85, 32%) — the same span selected in-window gives
$643, which is where the +3.8%/yr inflation estimate comes from.
Every configuration ever evaluated is in
[models/trials_ledger.json](models/trials_ledger.json). The champion is never
re-tuned on a calendar: nested experiments showed calendar re-selection at any
cadence adds drawdown without reliable return. Re-selection happens only on a
structural trigger (a new data source passes its gate, a pre-registered kill
criterion fires, or the owner directs it).

## The weekly job

[`.github/workflows/champion.yml`](.github/workflows/champion.yml) is the only
workflow. Every Saturday 13:00 UTC it runs `stocks-ml r5-weekly --commit`
([src/stocks_ml/live/r5.py](src/stocks_ml/live/r5.py)):

1. **Refresh the live world** ([data/world.py](src/stocks_ml/data/world.py)):
   Sharadar S&P 500 membership, prices (incremental by `lastupdated`, upserted
   by ticker and date), fundamentals (full refetch, refuses to shrink), insider
   filings; then SEC EDGAR company facts and 8-Ks, FINRA short interest and
   FRED. Symbol renames are detected by permanent ticker id and rewritten in
   every stored table so history carries over.
2. **Rebuild the panel** with the research recipe, then fit the K=4 ensemble
   on the trailing five years and rank every current member.
3. **Rotate the due sleeve** (the schedule is anchored so a rerun on the same
   Friday is a no-op), set the ballast state from SPY's trailing means, and
   write target weights.
4. **Keep the paper ledger** ([ledger_r5.json](ledger_r5.json)): fill last
   week's targets at Monday's open, sells before buys, 5 bp each way, never
   overdrawn, rebalances under 0.5% of NAV skipped; then mark NAV at Friday's
   close against SPY buy-and-hold from the same $100.
5. **Commit** `signals_r5/<friday>.md` (the book with per-name deltas, the
   sleeves, the top-15 candidates, data freshness) and the ledger, as
   `r5: signal <friday>`. The report is also on the run's summary page.

A run fails loudly — the panel must end on the most recent Friday — rather
than signal on stale data. Manual runs (`workflow_dispatch`) accept `as_of`
and `dry_run`.

**Licensed data never enters this public repository.** The Sharadar key is the
`SHARADAR_API_KEY` repository secret. The live data store (about 0.5 GB)
travels between runs as an AES-256 tarball in the Actions cache, encrypted
with the `R5_STORE_KEY` secret; a snapshot is saved only after a successful
run, and a Wednesday keepalive job stops GitHub's 7-day idle eviction. When
the cache is empty, the run seeds itself from a draft release uploaded by
[ops/r5_seed.sh](ops/r5_seed.sh) (drafts are invisible to the public) and
deletes it afterwards. [ops/r5_weekly.sh](ops/r5_weekly.sh) runs the same
cycle on a Mac by hand.

## Data

- **Sharadar** (Nasdaq Data Link, licensed; key required): S&P 500
  constituents with join/leave dates from 1998, daily prices for every
  historical member including delisted ones (dividend-adjusted opens and
  closes), quarterly fundamentals dated by filing, and insider transactions.
  This is what makes the universe survivorship-clean; the earlier free-data
  world was missing about 200 delisted constituents, and the edge it showed
  turned out to be a data artifact.
- **SEC EDGAR** company facts and 8-K filings, **SEC Form 4** insider filings
  (the Sharadar insider table bridges the gap after the last quarterly SEC
  file), **FINRA** short interest, **FRED** macro series (only the ALFRED-
  audited `T10Y2Y` and `FEDFUNDS` families are admitted as features).

Point-in-time rules: membership is effective-dated; filings become usable the
next calendar day; FINRA and macro observations carry their publication lags;
features are rank-normalized within each week; the label starts at the next
open and records when it becomes observable; training, early stopping and
validation are separated by purge gaps sized to the label horizon. The
no-lookahead tests corrupt future inputs and require past outputs to stay
unchanged — if they fail, fix the implementation, not the tests.

## Installation

Python 3.12 (`automl_tool`, a legacy benchmark candidate, needs < 3.13).

```bash
uv sync
uv run pytest        # 341 tests; must stay green with zero warnings
```

Running the champion locally needs `data/.sharadar_key` (or
`SHARADAR_API_KEY`) and a live store under `data/r5_live/`; `data/` is
git-ignored in full.

```bash
uv run stocks-ml r5-weekly [--as-of FRIDAY] [--no-refresh] [--no-sec] [--dry-run] [--commit]
uv run stocks-ml select --sel-start A --sel-end B [--eval-start C --eval-end D]
uv run stocks-ml procedure-card      # regenerate PROCEDURE.md from the champion spec
uv run stocks-ml leaderboard         # render the earnings-ranked board of evaluated books
```

All commands read [config/config.yaml](config/config.yaml).

## The legacy pipeline

The repository began as a one-week-horizon system on free data (yfinance
prices, Wikipedia membership) with its own tuning tournament, backtester,
strategy zoo and paper ledger. It is retired but still runs, and its history
is why the champion looks the way it does:

- Week-ahead ranking skill existed around 2001–2004 and has been
  indistinguishable from noise since; month-scale structure is real and
  concentrated in a handful of features, and Sharadar fundamentals add to it
  at that horizon (all pre-registered tests).
- No in-sample screen — CV rank IC, realized top-k excess, Optuna objectives
  — predicted out-of-sample dollars; walked exams are the only evidence
  accepted.
- Re-tuning on a calendar was destructive at every cadence tested.

```bash
uv run stocks-ml ingest [--full]          # free-data world + panel
uv run stocks-ml tune --family xgb|lgbm|catboost|enet [--optuna]
uv run stocks-ml train                    # champion tournament -> models/selection.md
uv run stocks-ml backtest                 # -> reports/backtest.md
uv run stocks-ml pipelines                # multi-pipeline league -> reports/pipelines.md
uv run stocks-ml signals                  # legacy weekly signal
uv run stocks-ml ledger init|apply|mark|show
uv run stocks-ml torture                  # survivorship stress test
```

Its artifacts (`models/champion.joblib`, `signals/`, `ledger.json`,
`ledger_ltr.json`, `reports/backtest.md`) are frozen history. The free-data
limitations documented in [AGENTS.md](AGENTS.md) — missing delisted tickers,
unofficial price sources, non-effective-dated sectors — apply to that world,
not to the Sharadar world the champion runs on.

## Where to look

- [AGENTS.md](AGENTS.md) — the full project context: every campaign, verdict
  and rule, in order.
- [PROCEDURE.md](PROCEDURE.md) — the champion's procedure card (generated
  from the spec).
- [models/trials_ledger.json](models/trials_ledger.json) — every evaluated
  configuration.
- [reports/](reports/) — backtests, ablations, the nested-selection protocol
  and the survivorship torture tests.
- [docs/research/](docs/research/) — notes on the papers the design leans on.
