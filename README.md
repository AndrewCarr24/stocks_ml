# stocks_ml

ML stock forecasting, backtesting, and paper trading on free data
(yfinance, Wikipedia, FRED, SEC EDGAR). See
`docs/superpowers/specs/2026-07-07-stocks-ml-design.md` for the design.

## Quickstart

    uv sync
    uv run stocks-ml ingest      # ~10-20 min first run (500+ tickers, EDGAR)
    uv run stocks-ml train       # champion selection -> models/selection.md
    uv run stocks-ml backtest    # -> reports/backtest.md + equity.png

## Weekly live cycle (after Friday close)

    uv run stocks-ml ingest
    uv run stocks-ml signals             # -> signals/YYYY-MM-DD.md + -trades.json
    # execute the trade list at your broker (fractional shares)
    uv run stocks-ml ledger apply        # record the executed trades
    uv run stocks-ml ledger mark         # record NAV

Initialize the paper ledger once: `uv run stocks-ml ledger init --cash 100`

## Tests

    uv run pytest

## Automation (GitHub Actions)

Two scheduled workflows run the system unattended and commit the resulting
artifacts back to this repo, so git history is the shadow-deployment audit
trail:

- **`.github/workflows/weekly.yml`** — Saturdays 13:00 UTC (after Friday
  close): `ingest` → `signals` → `ledger apply` → `ledger mark`, then commits
  `signals/` and `ledger.json` and opens a tracking issue with that week's
  signal.
- **`.github/workflows/retrain.yml`** — 06:00 UTC on the 1st of each month:
  `ingest` → `train` → `backtest`, then commits `models/` and `reports/`.

Both workflows also accept `workflow_dispatch:` for a manual run: go to the
**Actions** tab, pick the workflow, and click **Run workflow**.

Because CI commits `signals/`, `ledger.json`, `models/`, and `reports/`, run
`git pull` before any local session that touches the weekly cycle or
retrains, to avoid diverging from the CI-tracked history.

**Risks and where failures surface:** `ingest` calls yfinance from a
GitHub-hosted runner IP, which can occasionally be rate-limited; per-ticker
ingest failures are summarized in `data/manifest.json` and printed in the
job log, so a partial ingest doesn't necessarily fail the run. Full job
failures show up as red runs in the Actions tab and trigger the default
GitHub Actions failure email. No broker credentials exist anywhere in this
repo or its automation — all trades recorded by `ledger apply` are
paper-only.

## Operational notes

- `manifest.json` writes are not atomic. If it becomes corrupted (e.g. the
  process was killed mid-write), delete it and re-run `ingest`.
- `ingest --full` re-fetches prices from scratch but keeps `manifest.json`.
- `ledger apply` records fills at signal-time prices. If your actual fills
  differed, edit the trades JSON prices before applying.
- An empty, un-initialized ledger makes `signals` assume a $100 portfolio.
  Run `ledger init` first.
