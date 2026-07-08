from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd


@dataclass
class Ledger:
    cash: float = 0.0
    positions: dict = field(default_factory=dict)
    nav_history: list = field(default_factory=list)
    trades: list = field(default_factory=list)

    @classmethod
    def load(cls, path: str | Path) -> "Ledger":
        p = Path(path)
        if not p.exists():
            return cls()
        raw = json.loads(p.read_text())
        return cls(cash=raw["cash"], positions=raw["positions"],
                   nav_history=raw["nav_history"], trades=raw["trades"])

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(
            {"cash": self.cash, "positions": self.positions,
             "nav_history": self.nav_history, "trades": self.trades}, indent=2))

    def nav(self, closes: pd.Series) -> float:
        return self.cash + sum(s * float(closes.get(t, 0.0))
                               for t, s in self.positions.items())

    def mark(self, closes: pd.Series, date) -> float:
        nav = self.nav(closes)
        self.nav_history.append([str(pd.Timestamp(date).date()), nav])
        return nav

    def apply_trades(self, trades: list, date) -> None:
        for ticker, delta, price in trades:
            held = self.positions.get(ticker, 0.0)
            delta = max(delta, -held)  # cannot sell more than held
            if delta == 0:
                continue
            self.cash -= delta * price
            new = held + delta
            if abs(new) < 1e-9:
                self.positions.pop(ticker, None)
            else:
                self.positions[ticker] = new
            self.trades.append([str(pd.Timestamp(date).date()), ticker, delta, price])
