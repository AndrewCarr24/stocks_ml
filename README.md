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
