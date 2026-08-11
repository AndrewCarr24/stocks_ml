from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


def latest_closes(prices: pd.DataFrame) -> pd.Series:
    """Last available close per ticker (forward-fill convention, matching the simulator)."""
    return prices.sort_values("date").groupby("ticker")["close"].last()


@dataclass
class Ledger:
    cash: float = 0.0
    positions: dict = field(default_factory=dict)
    nav_history: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    applied_files: list = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        return cls(cash=raw["cash"], positions=raw["positions"],
                   nav_history=raw["nav_history"], trades=raw["trades"],
                   applied_files=raw.get("applied_files", []))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        payload = json.dumps(
            {"cash": self.cash, "positions": self.positions,
             "nav_history": self.nav_history, "trades": self.trades,
             "applied_files": self.applied_files}, indent=2, allow_nan=False)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        tmp.replace(path)

    def nav(self, closes: pd.Series) -> float:
        return self.cash + sum(s * float(closes.get(t, 0.0))
                               for t, s in self.positions.items())

    def mark(self, closes: pd.Series, date) -> float:
        nav = self.nav(closes)
        self.nav_history.append([str(pd.Timestamp(date).date()), nav])
        return nav

    def apply_trades(self, trades: list, date, cost_bps: float = 0.0) -> None:
        if cost_bps < 0:
            raise ValueError("cost_bps must be nonnegative")
        for _, delta, price in trades:
            if not (math.isfinite(delta) and math.isfinite(price) and price > 0):
                raise ValueError("trade delta and price must be finite, with price > 0")
        # Process sells before buys so sale proceeds can fund purchases.
        ordered = ([t for t in trades if t[1] < 0] +
                   [t for t in trades if t[1] >= 0])
        for ticker, delta, price in ordered:
            held = self.positions.get(ticker, 0.0)
            delta = max(delta, -held)  # cannot sell more than held
            if delta == 0:
                continue
            notional = delta * price
            fee = abs(notional) * cost_bps / 1e4
            cash_change = notional + fee
            if delta > 0 and self.cash - cash_change < -1e-9:
                raise ValueError(
                    f"trade would overdraw cash: buy {delta:.4f} {ticker} @ "
                    f"${price:,.2f} costs ${cash_change:,.2f}, cash ${self.cash:,.2f}")
            self.cash -= cash_change
            new = held + delta
            if abs(new) < 1e-9:
                self.positions.pop(ticker, None)
            else:
                self.positions[ticker] = new
            self.trades.append([str(pd.Timestamp(date).date()), ticker, delta, price, fee])
