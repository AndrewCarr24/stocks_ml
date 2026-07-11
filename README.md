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

## Operational notes

- `manifest.json` writes are not atomic. If it becomes corrupted (e.g. the
  process was killed mid-write), delete it and re-run `ingest`.
- `ingest --full` re-fetches prices from scratch but keeps `manifest.json`.
- `ledger apply` records fills at signal-time prices. If your actual fills
  differed, edit the trades JSON prices before applying.
- An empty, un-initialized ledger makes `signals` assume a $100 portfolio.
  Run `ledger init` first.
