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


CHALLENGER_TAG = "ltr"   # shadow-race challenger; its files carry this suffix


def find_latest_trades(signals_dir="signals", tag: str | None = None):
    """Newest trades file for the champion (tag=None) or a tagged challenger.

    Champion trade files are `<date>-trades.json`; challenger files are
    `<date>-<tag>-trades.json`. The champion glob must exclude tagged files or
    `ledger apply` would silently apply the challenger's trades to the real
    ledger."""
    from pathlib import Path as _P

    files = sorted(_P(signals_dir).glob("*-trades.json"))
    if tag:
        files = [f for f in files if f.name.endswith(f"-{tag}-trades.json")]
    else:
        files = [f for f in files if not f.name.endswith(f"-{CHALLENGER_TAG}-trades.json")]
    return files[-1] if files else None


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


def race_status(champion: "Ledger", challenger: "Ledger") -> str:
    """One-paragraph shadow-race scoreboard for the weekly signal markdown.

    Reads only the two ledgers' NAV histories; DEPLOYMENT.md rule 3 is decided
    from exactly these numbers, so printing them weekly makes the race verdict
    accumulate in public with no manual bookkeeping."""
    def last_nav(led):
        return led.nav_history[-1][1] if led.nav_history else float("nan")

    weeks = min(len(champion.nav_history), len(challenger.nav_history))
    if weeks == 0:
        return ("## Shadow race\n\nNo marked weeks yet — the race starts with "
                "the first `ledger mark` on both ledgers.")
    lead = "challenger" if last_nav(challenger) > last_nav(champion) else "champion"
    return (f"## Shadow race (week {weeks})\n\n"
            f"champion ${last_nav(champion):,.2f} vs challenger "
            f"${last_nav(challenger):,.2f} — {lead} leads. Promotion review at "
            f"week 52 per DEPLOYMENT.md rule 3.")
