from __future__ import annotations

import pandas as pd
from sklearn.base import clone

from stocks_ml.backtest.strategies import RiskState, make_strategies
from stocks_ml.features.panel import feature_cols
from stocks_ml.live.ledger import latest_closes
from stocks_ml.models.champion import load_champion
from stocks_ml.models.candidates import dated_features


def replay_guard_state(nav_history, dd_derisk: float, dd_full: float) -> bool:
    """Replay the drawdown-guard hysteresis over the ledger's NAV history."""
    guarded, hwm = False, 0.0
    for _, nav in nav_history:
        hwm = max(hwm, nav)
        dd = 1.0 - nav / hwm if hwm > 0 else 0.0
        if dd >= dd_derisk:
            guarded = True
        elif dd < dd_derisk / 2:
            guarded = False
    return guarded


def generate_signals(store, cfg, ledger, models_dir="models") -> tuple[str, list]:
    panel = store.read("panel")
    prices = store.read("prices")
    fcols = feature_cols(panel)
    champ_name, estimator = load_champion(models_dir)

    labeled = panel[panel["label"].notna()]
    latest = panel["date"].max()
    train_end = latest - pd.Timedelta(days=cfg.purge_days)
    train_start = train_end - pd.DateOffset(years=cfg.cv_train_years)
    labeled = labeled[labeled["date"].between(train_start, train_end)]
    model = clone(estimator).fit(dated_features(labeled, fcols), labeled["label"])

    rows = panel[panel["date"] == latest]
    preds = pd.Series(model.predict(rows[fcols]), index=rows["ticker"].values)
    vols = pd.Series(rows["aux_vol"].values, index=rows["ticker"].values)

    closes = latest_closes(prices)
    nav = ledger.nav(closes) if (ledger.positions or ledger.cash) else 100.0

    hwm = max((n for _, n in ledger.nav_history), default=nav)
    dd = max(0.0, 1.0 - nav / hwm) if hwm > 0 else 0.0
    strategy = make_strategies(cfg)[cfg.live_strategy]
    if hasattr(strategy, "restore_guard"):
        strategy.restore_guard(replay_guard_state(ledger.nav_history, cfg.dd_derisk, cfg.dd_full))
    weights = strategy.propose_weights(preds, vols, RiskState(drawdown=dd))

    lines = [f"# Signals for {pd.Timestamp(latest).date()}",
             f"Champion: **{champ_name}** · strategy: **{cfg.live_strategy}** · "
             f"portfolio value: ${nav:,.2f} · drawdown: {dd:.1%}", "",
             "## Target portfolio", "",
             "| ticker | weight | target $ | target shares | current shares | Δ shares |",
             "|---|---|---|---|---|---|"]
    trades = []
    tickers = sorted(set(weights.index) | set(ledger.positions))
    for t in tickers:
        w = float(weights.get(t, 0.0))
        price = float(closes.get(t, 0.0))
        cur = float(ledger.positions.get(t, 0.0))
        target_shares = (w * nav / price) if price > 0 else 0.0
        delta = target_shares - cur
        if abs(delta) > 1e-6 and price > 0:
            trades.append((t, delta, price))
        lines.append(f"| {t} | {w:.1%} | ${w * nav:,.2f} | {target_shares:.4f} "
                     f"| {cur:.4f} | {delta:+.4f} |")
    cash_w = 1.0 - float(weights.sum())
    lines += ["", f"Cash target: {cash_w:.1%} (${cash_w * nav:,.2f})", "",
              "Execute at next market open; then run `stocks-ml ledger apply` "
              "and `stocks-ml ledger mark`."]
    return "\n".join(lines), trades
