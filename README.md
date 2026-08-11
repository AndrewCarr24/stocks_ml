# stocks_ml

`stocks_ml` is an availability-dated historical-reconstruction and
machine-learning system that ranks S&P 500 stocks by expected weekly excess
return, evaluates long-only strategies, and runs a $100 live paper portfolio.
It uses only free data and has no broker integration.

The model predicts each stock's next five-trading-day open-to-open return minus
that week's cross-sectional median return. This is a stock-ranking problem, not
a market-direction forecast. The primary metric is mean weekly Spearman rank
information coefficient (IC).

## Current checked-in state

- **Champion:** Optuna-tuned XGBoost, mean weekly rank IC **0.0198**.
- **Runner-up:** Optuna-tuned ElasticNet, mean IC **0.0190**.
- Both models have positive IC in all four validation folds and rankable
  predictions for all **488/488** evaluation weeks.
- The most recent two calendar years are an untouched holdout: tuning and
  champion selection cannot inspect them.
- The production panel currently contains 48 features. Optional feature gaps
  are retained and handled conservatively rather than deleting stocks or weeks.
- The checked-in live strategy is `vol_scaled`.
- The test suite contains 231 tests and is expected to pass with zero warnings.

For exact model results see [models/selection.md](models/selection.md). For the
latest strategy results see [reports/backtest.md](reports/backtest.md). The
source-timing audit is in
[reports/source_point_in_time_audit.md](reports/source_point_in_time_audit.md).

## Installation

Python 3.12 is recommended. `automl_tool`, one of the benchmark candidates,
requires Python earlier than 3.13.

```bash
uv sync
uv run pytest
```

All commands read [config/config.yaml](config/config.yaml). Important settings
include the forecast horizon, purge gap, CV dates, transaction costs, strategy
risk limits, FRED availability lags, and `live_strategy`.

## Full research pipeline

Run the current end-to-end pipeline with:

```bash
uv run stocks-ml ingest
uv run stocks-ml train
uv run stocks-ml backtest
```

These are deliberately separate commands. Data can be refreshed without model
selection, and a frozen champion recipe can be backtested without retuning it.

### 1. `stocks-ml ingest`

Ingestion updates the source datasets and then rebuilds the complete weekly
modeling panel. The stages run in this order:

1. **S&P 500 membership** — downloads Wikipedia's current constituent table and
   effective-dated additions/removals. Historical membership is reconstructed
   backward from the change log, while the current table's `Date added` anchors
   active membership stints.
2. **Adjusted prices** — downloads adjusted daily open, high, low, close, and
   volume from yfinance for every historical member and SPY. Requests are
   batched and retried; individual ticker failures are recorded rather than
   aborting the entire run. Histories with repeated split-sized jumps are
   rejected as likely corrupt.
3. **Macro data** — downloads configured FRED series and applies explicit
   calendar-day publication lags. Only the ALFRED-audited `T10Y2Y` and
   `FEDFUNDS` families enter the production model. Revision-prone series remain
   stored for diagnostics or reporting but are excluded from model features.
4. **Fundamentals** — downloads SEC EDGAR Company Facts for current members and
   maps configured XBRL concepts such as revenue, income, assets, equity, cash
   flow, and shares.
5. **Corporate events** — downloads effective-dated SEC 8-K submission data for
   the historical universe.
6. **Insider activity** — ingests SEC Form 4 filings.
7. **Short interest** — ingests FINRA short-interest history with its
   publication delay.
8. **Panel construction** — creates the point-in-time weekly member grid,
   features, labels, coverage statistics, and the persisted model panel.

Parquet datasets are written under `data/`; source status, failures, and feature
coverage are summarized in [data/manifest.json](data/manifest.json). Incremental
price ingestion updates known histories and fetches full history for new
tickers. It does not repair arbitrary old gaps within an existing ticker. Use
`uv run stocks-ml ingest --full` to delete all persisted Parquet datasets and
refetch complete price histories. This option retains `data/manifest.json`;
manifest-driven completion markers, notably older Form 4 quarters, are not
reset by `--full`.

#### Panel, features, and labels

The panel is sampled at the last trading day on or before each Friday. Its base
grid is the point-in-time index membership universe, and optional data is
left-joined so a missing fundamental, filing, short-interest observation, or
price-derived feature cannot silently remove a member.

Feature families include:

- momentum, reversal, volatility, beta, and residual-return signals;
- liquidity, volume, overnight, and intraday measures;
- filing-dated fundamentals and 8-K event indicators;
- Form 4 insider and FINRA short-interest features;
- audited market and macro features.

Ordinary stock-level features are ranked cross-sectionally to $(-1, 1]$ each
week. Optional missing values are neutral-filled with zero only after ranking.
Time-only and binary event/market/macro features are rank-exempt. Current GICS
sectors from Wikipedia are not effective-dated, so sector-derived features are
excluded from production matrices.

The label starts at the next trading-day open and ends five trading days later.
It is centered using only that week's point-in-time members, and
`label_end_date` records when it becomes observable. Price calculations do not
implicitly fill missing observations into zero returns.

### Optional: tune model families

Hyperparameter tuning is not part of every `train` run. Existing tuned recipes
under `models/` are reusable inputs to the champion tournament. To refresh one
family with Optuna:

```bash
uv run stocks-ml tune --family xgb --optuna
uv run stocks-ml tune --family enet --optuna
```

Supported families are `xgb`, `lgbm`, `catboost`, and `enet`; omit `--optuna`
to use the random-search tuner. Both methods optimize only pre-holdout purged
walk-forward CV rank IC. Optuna writes the selected recipe, a Markdown report,
and complete trial diagnostics under `models/`.

Optuna-tuned XGBoost uses a 5,000-tree safety ceiling with early stopping on a
separate, time-ordered, purged tail of each training set. Its stopping metric is
mean weekly Spearman IC—not row-level RMSE. The untuned and random-search XGBoost
candidates use lower tree ceilings.

### 2. `stocks-ml train`

Training runs the champion tournament. It loads the panel and available tuned
recipes, then compares model candidates, an ensemble when applicable, and zero
and momentum baselines on the same calendar:

- four contiguous validation folds beginning in March 2015;
- one frozen model per fold;
- the immediately preceding two calendar years for each fold's training data;
- a ten-calendar-day purge between training and validation;
- no labels whose return horizon reaches the protected holdout;
- no holdout covariates in tuning, selection, or rankability checks.

A candidate is eligible only if every fold has finite IC and every expected
week has finite, non-constant, rankable predictions. Missing weeks cannot be
silently dropped to improve a score. An ML candidate must also beat the
baselines; otherwise selection falls back to momentum.

The command writes:

- `models/selection.md` — candidate scores, fold ICs, and coverage;
- `models/champion.json` — champion identity and selection metadata;
- `models/champion.joblib` — the frozen estimator **recipe**.

The serialized champion is intentionally a recipe, not permanently fitted
weights. Backtests and live signals clone and refit it using only the training
history available at each decision date.

### 3. `stocks-ml backtest`

The backtest walks chronologically through weekly signal dates. At each
four-week retraining point it clones the champion recipe and fits it on the
preceding two calendar years, ending ten calendar days before the signal. It
predicts the current reconstructed universe, asks the strategy for target
weights, and trades at the first following trading-day open.

The simulator enforces long-only weights, total exposure no greater than 100%,
cost-netted purchases, and no leverage. It charges the configured one-way
transaction cost (currently 5 bps) and marks NAV daily at closes. The first
capital observation is retained so initial costs and the first execution-day
return are not erased by normalization.

All three strategies use only positive model forecasts:

- **`equal_topk`** — equal slots for up to the top eight names;
- **`vol_scaled`** — inverse-volatility top-eight weights, a 15% annualized
  portfolio-volatility cap, and drawdown hysteresis that halves exposure at 15%
  drawdown and moves to cash at 25%;
- **`kelly`** — capped quarter-Kelly sizing with no short positions or leverage.

The report adds SPY buy-and-hold and Treasury-bill cash benchmarks, regime and
stress-period slices, and a separately labeled view of the untouched holdout.
It writes [reports/backtest.md](reports/backtest.md) and `reports/equity.png`.
The holdout may be reported here only after all model and strategy choices are
frozen; it is never fed back into tuning or champion selection.

## Weekly live paper cycle

Initialize the ledger once:

```bash
uv run stocks-ml ledger init --cash 100
```

After the Friday close, the manual shadow cycle is:

```bash
uv run stocks-ml ingest
uv run stocks-ml signals
uv run stocks-ml ledger apply
uv run stocks-ml ledger mark
```

What happens during that cycle:

1. **Refresh data and rebuild the panel.** The same ingestion and feature code
   used by research runs, preventing a separate live feature implementation
   from drifting away from training.
2. **Refit the frozen champion recipe.** `signals` clones the selected recipe
   and fits it on labeled observations from the trailing two calendar years,
   stopping ten calendar days before the latest panel date.
3. **Score the latest universe.** The model predicts every eligible member
   represented on the latest persisted panel date. The configured
   `live_strategy` converts positive forecasts and trailing volatility into
   long-only target weights.
4. **Restore risk state.** The volatility-scaled strategy replays ledger NAV
   history to restore its drawdown-guard hysteresis before sizing positions.
5. **Create an order proposal.** Current paper NAV and the latest available
   closes are used to convert target weights to fractional target shares and
   share deltas. The command writes `signals/YYYY-MM-DD.md` and
   `signals/YYYY-MM-DD-trades.json`.
6. **Apply the paper trades.** `ledger apply` sells before buying, charges the
   configured one-way costs, prevents a cash overdraft, and skips an already
   recorded trade file by default. `--force` deliberately overrides that
   duplicate-file protection.
7. **Mark the account.** `ledger mark` values cash and positions at the latest
   available closes and appends the result to NAV history.

`ledger.json` stores cash, fractional positions, trades, applied signal files,
and NAV marks. Use `uv run stocks-ml ledger show` to inspect it.

### Important execution limitation

Signal reports state that orders should execute at the next market open, but
the automated paper workflow currently writes and applies each trade using the
latest-close reference price embedded in the generated JSON. It does not wait
for or retrieve actual next-open fills. This makes the ledger a shadow signal
tracker, not a realistic execution simulator. For manual paper accounting,
replace the JSON prices with actual fills before applying the file.

No broker credentials or order-routing code exist in this repository.

## GitHub Actions

Two scheduled workflows share one concurrency group so they cannot modify
tracked artifacts simultaneously:

- **Weekly:** `.github/workflows/weekly.yml` runs Saturdays at 13:00 UTC. It
  restores cached data, runs `ingest → signals → ledger apply → ledger mark`,
  saves the refreshed cache, commits `signals/` and `ledger.json`, and opens a
  tracking issue containing the signal report.
- **Monthly:** `.github/workflows/retrain.yml` runs at 06:00 UTC on the first of
  each month. It restores cached data, runs `ingest → train → backtest`, saves
  the cache, and commits `models/` and `reports/`. It reuses existing tuned
  hyperparameters; it does **not** launch Optuna.

Both workflows support manual `workflow_dispatch` runs. Because CI commits the
audit artifacts, synchronize the repository before a local weekly cycle or
retrain to avoid diverging histories.

Partial yfinance failures are logged and summarized in the manifest. A failed
ticker does not necessarily fail the whole job; workflow-level failures appear
in the Actions tab.

## Point-in-time safeguards

- Historical index membership is reconstructed into effective-dated stints.
- Date-only EDGAR and Form 4 records become usable on the next calendar day.
- FINRA and macro observations receive explicit publication lags.
- Only ALFRED-audited macro families enter production features.
- Revision-prone macro and non-effective-dated sector features are excluded.
- Labels begin after the signal date and carry their own observability date.
- Outer CV and model-internal validation use time-ordered purge gaps.
- Optional feature gaps do not delete rows or weeks.
- Every candidate must cover the identical dynamically derived evaluation
  calendar.
- Strategy weights are validated as long-only and unlevered.

The no-lookahead tests deliberately corrupt future inputs and require past
outputs to remain unchanged. If those tests fail, fix the implementation—not
the tests.

## Known limitations

- yfinance is an unofficial source and can be rate-limited or return malformed
  adjusted histories.
- Membership, adjusted price histories, and FRED data are reconstructed from
   current source snapshots rather than a complete archive of every historical
   source vintage. Unknown membership starts fall back to the configured 1996
   floor. The admitted macro series were compared with available ALFRED
   vintages, but `T10Y2Y` vintage coverage begins in 2014. These limitations are
   more relevant to the 2005+ backtest than to the March 2015+ selection window.
- Free data is unavailable for roughly 200 historical/delisted constituents,
  including important bankruptcies. The resulting residual survivorship bias
  cannot be eliminated with the current sources. Run
  `uv run stocks-ml torture` and see
  [reports/survivorship_torture.md](reports/survivorship_torture.md) for the
  quantified stress tests.
- The backtest forward-fills its daily open/close matrices for valuation and
  execution. Stale terminal prices for delisted securities are therefore a
  remaining realism risk; removal-haircut torture tests provide a conservative
  sensitivity analysis.
- Wikipedia's current sectors are not historically effective-dated, so sector
  features are intentionally unavailable.
- Fundamentals are sparse in early history, short interest begins around 2018,
  and data-source outages remain possible. Neutral filling protects calendar
  comparability but cannot create missing information.
- Backtest fills and the automated live paper fills are simplified. Results do
  not establish that the same returns are achievable with real orders.
- `data/manifest.json` writes are not atomic. If it is interrupted and becomes
  invalid, remove it and rerun ingestion. Ledger writes are atomic.

## Command reference

```bash
uv run stocks-ml ingest
uv run stocks-ml ingest --full
uv run stocks-ml tune --family xgb --optuna  # family: xgb, lgbm, catboost, or enet
uv run stocks-ml tune --family xgb           # random search when --optuna is omitted
uv run stocks-ml train
uv run stocks-ml backtest
uv run stocks-ml signals
uv run stocks-ml ledger init --cash 100
uv run stocks-ml ledger apply
uv run stocks-ml ledger apply --file path/to/trades.json
uv run stocks-ml ledger apply --file path/to/trades.json --force
uv run stocks-ml ledger mark
uv run stocks-ml ledger show
uv run stocks-ml torture
uv run pytest
```

The original design is documented in
[docs/superpowers/specs/2026-07-07-stocks-ml-design.md](docs/superpowers/specs/2026-07-07-stocks-ml-design.md),
with implementation history in
[docs/run-notes-2026-07-11.md](docs/run-notes-2026-07-11.md).
