# stocks_ml — ML Stock Forecasting & Investment System

**Date:** 2026-07-07
**Status:** Approved design, pre-implementation

## Goal

A system that forecasts S&P 500 stock returns, backtests investment strategies
built on those forecasts against real historical data, and produces weekly
trade signals for a real (initially $100) portfolio. Guiding principle: expected
profit matters, but capital preservation is the binding constraint — if the
account goes to zero, the game is over.

## Decisions made during design

| Decision | Choice |
|---|---|
| Forecast target | Forward 5-trading-day (weekly) return **relative to universe median**; horizon is a config value so daily is a config change |
| Universe | S&P 500 constituents, point-in-time membership (survivorship-bias-free) |
| Execution | Signals + paper ledger; manual execution at broker. No broker API in v1 |
| Position sizing | Three pluggable strategies backtested head-to-head; results pick the default |
| Architecture | Custom modular Python package with purpose-built walk-forward backtester (Approach A) |
| Constraints | Long-only, no leverage, no shorting, cash is the fallback asset |
| Data | Free/open sources only; daily bars only (no hourly, no paywalls) |

## Architecture

```
                ┌─────────────────────────────────────────────┐
                │                data store (parquet)          │
                │  prices/  membership/  fred/  manifest.json  │
                └───────▲──────────────────────────┬───────────┘
   ingest (network) ────┘                          │ read-only
                                                   ▼
        ┌──────────┐    ┌───────────┐    ┌──────────────────┐
        │ features │───▶│  models   │───▶│     backtest     │──▶ scenario report
        │  (panel) │    │ (champion)│    │ (walk-forward    │    (md/html)
        └──────────┘    └───────────┘    │  sim+strategies) │
                              │          └──────────────────┘
                              ▼
                        ┌──────────┐
                        │   live   │──▶ weekly signals + paper ledger
                        └──────────┘
```

Package layout:

```
src/stocks_ml/
  data/        # ingestion: yfinance prices, Wikipedia membership, FRED econ
  features/    # panel construction, feature + label engineering
  models/      # candidate models, walk-forward CV, champion selection
  backtest/    # simulator, Strategy interface, strategies, metrics, report
  live/        # signal generation, paper ledger
  cli.py       # ingest | train | backtest | signals | ledger
config/        # YAML: universe, horizon, costs, strategy params, FRED series
data/          # local parquet store (gitignored)
docs/superpowers/specs/
tests/
```

Tooling: Python 3.11+, `uv` for env management, `pytest`, `automl_tool`
installed from GitHub source (`AndrewCarr24/automl_tool`).

## Component 1: Data ingestion (`stocks_ml.data`)

All network access lives here. Everything downstream reads local parquet only.
`stocks-ml ingest` is idempotent and incremental (fetches only dates newer than
the local store; `--full` forces a rebuild).

**Sources:**

1. **Prices** — daily adjusted OHLCV for all tickers ever in the S&P 500 over
   the backtest window, via `yfinance` (free, no key). Batched downloads with
   retry/backoff; Stooq (via `pandas-datareader`) as documented fallback.
   Delisted/renamed tickers that yfinance cannot serve are recorded in the
   manifest and excluded, with their count reported (partial survivorship bias
   remains for unfetchable names; see Risks).
2. **Universe membership** — Wikipedia's S&P 500 page: current constituents
   table (with GICS sector) plus the historical changes table. Reconstructed
   into a point-in-time membership parquet: `(ticker, start_date, end_date)`.
   Backtests only trade names in the index as of each rebalance date.
3. **Economic indicators** — ~10 FRED series fetched keylessly
   (e.g., VIXCLS, T10Y2Y, ICSA, CPIAUCSL, FEDFUNDS, UNRATE, UMCSENT, DTWEXBGS).
   Each series carries a per-series publication lag (config) and is shifted by
   that lag before use, so the model never sees a value before its real-world
   release date.

**Storage:** parquet files partitioned by source; `manifest.json` records last
fetch dates, ticker failures, and row counts. No database server.

**Error handling:** per-ticker failures don't abort the run — they're logged,
recorded in the manifest, and summarized at the end. A hard failure of the
whole source (e.g., Wikipedia layout change) aborts with a clear message rather
than writing partial garbage.

## Component 2: Features and labels (`stocks_ml.features`)

Builds one cross-sectional panel: one row per (ticker, rebalance date). A
single model learns across all stocks — ~500× the training data of per-ticker
models.

**Features (all computed strictly from data available at prediction time):**
- Momentum: trailing 1, 4, 12, 26, 52-week returns (skip-week variants for
  short horizons to avoid reversal contamination)
- Risk: realized volatility over 4 and 12 weeks; downside deviation
- Volume: dollar-volume trend, abnormal volume
- Price position: distance from 52-week high/low
- Sector: GICS sector (categorical)
- Market context: SPY trailing returns and realized vol
- Macro: lagged FRED series levels and changes
- Calendar: month, week-of-quarter

**Label:** forward `horizon` (default 5) trading-day return minus the
cross-sectional median return that week. Relative labels make the task
"rank stocks," which is what the strategies consume. Raw forward return is
also stored for the backtester's accounting.

**Leakage discipline:** features end at the rebalance date's close; labels
start the next trading day. Econ series are publication-lagged. The panel
builder is the single place labels are created, and it is unit-tested for
overlap.

## Component 3: Champion model selection (`stocks_ml.models`)

**Candidates:**
1. `automl_tool` — `AutoML().fit_pipeline()` (GridSearchCV over its model zoo)
2. Hand-tuned XGBoost regressor (challenger)
3. Zero-forecast baseline (predicts 0 for everything)
4. Momentum-rank baseline (predicted return = trailing 12-week return rank)

Baselines are permanent members of every evaluation report: if ML doesn't beat
naive momentum, we want to see it, not hide it.

**Evaluation:** expanding-window walk-forward CV over the panel. Each fold
trains on all data up to time t, skips a purge gap of `horizon` days (so no
train label overlaps a test feature window), and tests on the following block.
**Primary metric: mean Spearman rank IC** per week on test folds (how well the
model orders the cross-section); secondary: RMSE, IC t-stat, decile spread.

**Champion selection:** best mean rank IC across folds wins, subject to beating
both baselines. The champion's *recipe* (model family + hyperparameters) is
what's selected; the backtest and live systems refit that recipe on trailing
data at each retrain — the selection folds are never reused as final evidence
of performance (that's the backtest's job, on periods after model selection).

**Retraining cadence:** every 4 weeks (config) in both backtest and live,
expanding window.

## Component 4: Backtester (`stocks_ml.backtest`)

**Simulator:** steps weekly through history. At each rebalance date t:
1. Slice all data as-of t (prices ≤ t, publication-lagged econ, membership as
   of t). The model/strategy physically cannot see beyond t.
2. Refit champion recipe if a retrain is due (expanding window ending at
   t − purge gap).
3. Predict the cross-section; hand predictions + current portfolio + risk
   state to the strategy, which returns target weights.
4. Execute: trade from current to target weights at next open, charging
   transaction costs (default 5 bps one-way slippage/spread; commissions $0;
   fractional shares assumed). Track positions, cash, NAV daily.

**Strategy interface:** `propose_weights(predictions, prices, portfolio_state,
risk_state) -> weights` — pure function, pluggable, independently testable.

**Strategies:**
1. **EqualWeightTopK** — equal dollars in top k (default 8) predictions;
   cash filter: if fewer than k predictions are positive, unfilled slots stay
   in cash.
2. **VolScaledTopK + drawdown guard** — top k weighted ∝ 1/realized-vol,
   scaled so forecast portfolio vol ≤ target (default 15% annualized);
   hard guard: if portfolio drawdown from high-water mark exceeds threshold
   (default 15%), de-risk to 50% cash; > 25%, fully to cash until drawdown
   recovers to half the threshold (hysteresis so it doesn't flap).
3. **FractionalKelly** — weight ∝ predicted excess return / predicted variance,
   scaled to quarter-Kelly, per-position cap (default 20%), renormalized;
   negative-edge names get zero.
4. **Benchmarks:** SPY buy-and-hold, equal-weight universe, 100% cash (T-bill
   rate from FRED).

**Scenario report** (markdown + HTML with equity curves), per strategy:
- "$100 → $X" terminal wealth over the full test window and per-year
- CAGR, Sharpe, Sortino, max drawdown, worst week, longest underwater spell
- Time-in-cash %, annual turnover, total cost drag
- Stress windows called out separately: 2018 Q4, Feb–Mar 2020, 2022
- Champion vs. baselines rank-IC table alongside, so model quality and
  strategy quality are visible independently

**Integrity tests (first-class):** a test that corrupts all future data after
t and asserts signals at t are unchanged; a test that labels and features
never overlap; a test that membership filtering excludes not-yet-added and
already-removed tickers.

## Component 5: Live operation (`stocks_ml.live`)

Weekly cycle (manual, ~5 minutes):

```
stocks-ml ingest            # refresh data
stocks-ml signals           # emits target portfolio + trade list vs ledger
# user executes trades at broker manually
stocks-ml ledger mark       # record fills / update NAV with latest prices
```

- `signals` loads the current champion recipe, refits on trailing data if the
  retrain cadence says so, predicts, runs the chosen strategy, and writes
  `signals/YYYY-MM-DD.md`: target weights, concrete dollar/share amounts for
  the ledger's current value, and the diff (buy/sell list) from current
  holdings.
- **Paper ledger:** JSON state file — positions, cash, NAV history, trade log.
  `ledger mark` fetches latest closes and appends NAV. Over time this is the
  live track record to compare against backtest expectations before real money
  scales beyond $100.
- Strategy for live defaults to the backtest winner (config-pinned, human-
  changeable).

## Configuration

Single `config/config.yaml`: horizon (5), rebalance cadence (weekly), retrain
cadence (4 weeks), top-k, vol target, drawdown thresholds, Kelly fraction,
cost bps, FRED series list + lags, backtest start date (2005), CV fold spec.
Switching to daily = change horizon + cadence; code paths are identical.

## Testing

- Unit tests per module (pytest); deterministic seeds.
- Golden no-lookahead integrity tests (described above) run in CI-style via
  `pytest` before any backtest result is trusted.
- Strategy unit tests with hand-constructed prediction fixtures (e.g., DD
  guard triggers at exactly the threshold, Kelly caps respected, weights sum
  ≤ 1, never negative).
- Ingestion tests against small cached fixtures (no network in tests).

## Risks & mitigations

- **yfinance instability** (unofficial API): retries/backoff, incremental
  caching, Stooq fallback; ingestion failures are visible, not silent.
- **Residual survivorship bias**: point-in-time membership fixes selection of
  *names*, but some delisted tickers' price history may be unfetchable from
  free sources. The manifest quantifies coverage; the report states it. If
  coverage is poor pre-2010, shorten the backtest window rather than pretend.
- **Overfitting to the backtest**: champion selection uses only CV folds;
  strategy comparison uses the full walk-forward; final sanity check reserves
  the most recent ~2 years as a period never used for any selection decision.
- **Weak edge overall**: entirely possible the ML beats nothing. The baselines
  and benchmarks make this visible, and the honest outcome may be "vol-scaled
  momentum with a drawdown guard" — the system still functions as a platform.
- **Small-account frictions**: $100 with fractional shares makes weekly top-8
  rebalancing feasible at commission-free brokers; cost model still charges
  spread. The report includes cost drag so we see if the account is too small
  for the strategy's turnover.

## Out of scope (v1)

- Broker API integration (design keeps `signals` output structured so an
  Alpaca adapter can consume it later)
- Intraday/hourly data, options, shorting, leverage, non-US assets
- Automated scheduling (user runs the weekly cycle manually)
- Deep-learning models
