from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.base import clone

from stocks_ml.backtest.strategies import RiskState
from stocks_ml.features.panel import feature_cols

MIN_TRAIN_ROWS = 50  # low floor: one real week has ~500 rows; small value keeps synthetic tests tradeable


@dataclass
class BacktestResult:
    nav: pd.Series
    weights: pd.DataFrame
    total_costs: float
    n_fits: int


def run_backtest(panel, prices, strategy, estimator, cfg, start=None, end=None) -> BacktestResult:
    fcols = feature_cols(panel)
    open_w = prices.pivot(index="date", columns="ticker", values="open").sort_index().ffill()
    close_w = prices.pivot(index="date", columns="ticker", values="close").sort_index().ffill()
    cal = close_w.index

    rdates = pd.DatetimeIndex(sorted(panel["date"].unique()))
    if start:
        rdates = rdates[rdates >= pd.Timestamp(start)]
    if end:
        rdates = rdates[rdates <= pd.Timestamp(end)]

    labeled = panel[panel["label"].notna()]
    model, last_fit, n_fits = None, None, 0
    cash, shares = 100.0, {}
    navs, weight_rows = {}, {}
    hwm, total_costs = 100.0, 0.0

    def mark(day) -> float:
        px = close_w.loc[day]
        return cash + sum(s * px.get(t, 0.0) for t, s in shares.items())

    for i, t in enumerate(rdates):
        need_fit = model is None or (t - last_fit).days >= cfg.retrain_weeks * 7
        if need_fit:
            train = labeled[labeled["date"] <= t - pd.Timedelta(days=cfg.purge_days)]
            if cfg.train_sample_rows:
                train = train.sort_values("date").tail(cfg.train_sample_rows)
            if len(train) >= MIN_TRAIN_ROWS:
                model = clone(estimator).fit(train[fcols], train["label"])
                last_fit, n_fits = t, n_fits + 1
        if model is None:
            continue

        rows = panel[panel["date"] == t]
        preds = pd.Series(model.predict(rows[fcols]), index=rows["ticker"].values)
        preds = preds.dropna()
        vols = pd.Series(rows["aux_vol"].values, index=rows["ticker"].values)

        # drawdown as of t: NAV marks so far end at t (never beyond — see marking
        # below); hwm is maintained from every daily mark, so intra-week peaks count
        past_navs = pd.Series(navs).sort_index()
        nav_t = past_navs.iloc[-1] if not past_navs.empty else 100.0
        dd = 1.0 - nav_t / hwm

        weights = strategy.propose_weights(preds, vols, RiskState(drawdown=dd))
        weight_rows[t] = weights

        ei = cal.searchsorted(t, side="right")
        if ei >= len(cal):
            break
        exec_day = cal[ei]
        opens = open_w.loc[exec_day]
        port_val = cash + sum(s * opens.get(tk, 0.0) for tk, s in shares.items())

        # first pass sizes the trade to estimate cost, then invest NET of cost so
        # cash can never go negative (no implicit leverage)
        tradable = {tk: w for tk, w in weights.items() if opens.get(tk, 0) > 0}
        current_dollars = {tk: s * opens.get(tk, 0.0) for tk, s in shares.items()}
        est_traded = sum(abs(tradable.get(tk, 0.0) * port_val - current_dollars.get(tk, 0.0))
                         for tk in set(tradable) | set(current_dollars))
        cost = est_traded * cfg.cost_bps / 1e4
        investable = port_val - cost
        target_dollars = {tk: w * investable for tk, w in tradable.items()}
        total_costs += cost
        shares = {tk: d / opens[tk] for tk, d in target_dollars.items()}
        cash = investable - sum(target_dollars.values())

        # mark daily NAV from execution up to (and including) the NEXT rebalance
        # date, but never past it — the next signal's drawdown must not see the
        # day after its own decision point
        span_end = rdates[i + 1] if i + 1 < len(rdates) else cal[-1]
        for day in cal[(cal >= exec_day) & (cal <= span_end)]:
            navs[day] = mark(day)
            hwm = max(hwm, navs[day])

    nav = pd.Series(navs).sort_index()
    weights_df = pd.DataFrame(weight_rows).T.fillna(0.0)
    return BacktestResult(nav=nav, weights=weights_df, total_costs=total_costs, n_fits=n_fits)
