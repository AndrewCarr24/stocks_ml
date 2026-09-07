"""The paper book's rules and accounting, shared by the backtest engine
(selection.simulate) and the live weekly job (live/r5.py).

A signal dated t (the week's last trading day) rotates one of the K sleeves
to the week's ranked picks, reads the trend ballast, and stores target
weights. The ledger fills those weights at the first open after t, paying
COST_BPS a side on every trade and skipping rebalances under MIN_TRADE_FRAC
of NAV, then marks NAV at the last close on or before each week's label.
The backtest and the live job differ only in where the picks come from.

Units are on Sharadar's total-return price basis (closeadj): dividends and
splits arrive as retroactive rescalings of the whole history, so every mark
stores a reference close per position and `rebase` rescales units by
old/new reference close before the next run touches the book.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

ANCHOR = pd.Timestamp("2001-01-05")        # week 0 of the sleeve schedule
BALLAST_WINDOWS = (30, 40, 52)             # weeks; each third: IEF when SPY < its MA
COST_BPS = 5.0                             # per side (procedure card)
STALE_WEEKS = 5                            # a sleeve this old missed a rotation
MIN_TRADE_FRAC = 0.005                     # skip rebalances under 0.5% of NAV (full exits always run)
FUNDS = ("SPY", "IEF")
TOP_N = 15                                 # names a signal ranks; sleeves pick from these


# ---- rules ----
def pick_capped(names, cap, k, smap):
    """The first k names in rank order with at most `cap` per sector (no cap:
    the first k); a short capped list is topped up from the rest in rank order."""
    if cap is None:
        return names[:k]
    out, cnt = [], {}
    for n in names:
        s = smap.get(n, "UNK")
        if cnt.get(s, 0) < cap:
            out.append(n)
            cnt[s] = cnt.get(s, 0) + 1
        if len(out) == k:
            break
    return out if len(out) == k else (out + [n for n in names if n not in out])[:k]


def friday_of(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t + pd.Timedelta(days=(4 - t.weekday()) % 7)


def week_index(t) -> int:
    """Weeks since the anchor, counted on the week's Friday: a signal dated
    Thursday (holiday Friday) belongs to its own week, as in
    selection.week_slot, and rotates a sleeve like any other week."""
    return int((friday_of(t) - ANCHOR).days // 7)


def due_sleeve(t, n_sleeves: int) -> int:
    return week_index(t) % n_sleeves


def rotate_sleeves(sleeves: dict, t, ranked: list[str], smap: dict, n_sleeves: int,
                   book: int, cap: int | None, top_n: int = TOP_N) -> tuple[dict, list[int]]:
    """One sleeve rotates per week (week_index mod n_sleeves); empty sleeves
    fill immediately (the first week), stale ones catch up. Returns (new
    sleeves, rotated sleeve ids)."""
    t = pd.Timestamp(t)
    out, rotated = {}, []
    for k in range(n_sleeves):
        s = sleeves.get(str(k), {"names": [], "since": None})
        age = (t - pd.Timestamp(s["since"])).days // 7 if s.get("since") else None
        if age is not None and age <= 0 and s["names"]:
            # already rotated on this signal date: a rerun of the same
            # Friday must not rotate it again (once per week)
            out[str(k)] = {"names": list(s["names"]), "since": s["since"]}
        elif k == due_sleeve(t, n_sleeves) or not s["names"] \
                or (age is not None and age >= STALE_WEEKS):
            names = pick_capped(list(ranked[:top_n]), cap, book, smap)
            out[str(k)] = {"names": names, "since": str(t.date())}
            rotated.append(k)
        else:
            out[str(k)] = {"names": list(s["names"]), "since": s["since"]}
    return out, rotated


def ballast_state(spy_weekly: pd.Series, t) -> dict[str, str]:
    """Per moving-average window: 'IEF' when SPY's weekly close is below its
    trailing W-week mean, else 'SPY'. Uses closes through t's week."""
    hist = spy_weekly[spy_weekly.index <= friday_of(t)].dropna()
    out = {}
    for w in BALLAST_WINDOWS:
        below = len(hist) >= w and float(hist.iloc[-1]) < float(hist.iloc[-w:].mean())
        out[str(w)] = "IEF" if below else "SPY"
    return out


def target_weights(sleeves: dict, ballast: dict, floor: float) -> dict[str, float]:
    """Sleeves equal-weighted at `floor` of NAV, names equal within a sleeve;
    the rest split evenly across the ballast's thirds."""
    w: dict[str, float] = {}
    for s in sleeves.values():
        for n in s["names"]:
            w[n] = w.get(n, 0.0) + floor / (len(sleeves) * len(s["names"]))
    for fund in ballast.values():
        w[fund] = w.get(fund, 0.0) + (1.0 - floor) / len(ballast)
    return dict(sorted(w.items(), key=lambda kv: (-kv[1], kv[0])))


def sleeve_counts(sleeves: dict) -> dict[str, int]:
    out: dict[str, int] = {}
    for s in sleeves.values():
        for n in s["names"]:
            out[n] = out.get(n, 0) + 1
    return out


# ---- prices ----
def close_asof(closes: pd.DataFrame, ticker: str, date) -> tuple[float, pd.Timestamp | None]:
    if ticker not in closes.columns:
        return np.nan, None
    s = closes[ticker].loc[:pd.Timestamp(date)].dropna()
    return (float(s.iloc[-1]), s.index[-1]) if len(s) else (np.nan, None)


def fill_price(closes: pd.DataFrame, opens: pd.DataFrame, ticker: str,
               fill_date) -> tuple[float, pd.Timestamp | None]:
    """Open on the fill date, else the first open within five sessions, else
    the last close before it (a name that stopped trading is closed out at
    its final print)."""
    if ticker in opens.columns:
        s = opens[ticker].loc[pd.Timestamp(fill_date):].iloc[:5].dropna()
        if len(s):
            return float(s.iloc[0]), s.index[0]
    return close_asof(closes, ticker, fill_date)


# ---- the ledger ----
@dataclass
class Ledger:
    cash: float = 0.0
    positions: dict = field(default_factory=dict)    # ticker -> units (closeadj basis)
    refs: dict = field(default_factory=dict)         # ticker -> [date, close] at last mark
    sleeves: dict = field(default_factory=dict)      # "0".."K-1" -> {names, since}
    pending: dict | None = None                       # {decision_date, weights}
    nav_history: list = field(default_factory=list)  # [date, nav, spy_nav]
    trades: list = field(default_factory=list)       # [fill_date, ticker, units, price, fee]
    bench: dict = field(default_factory=dict)        # SPY buy-and-hold: cash, units, ref
    started: str | None = None

    @classmethod
    def new(cls, capital: float, t) -> "Ledger":
        return cls(cash=float(capital), bench={"cash": float(capital), "units": 0.0, "ref": None},
                   started=str(pd.Timestamp(t).date()))

    @classmethod
    def load(cls, path) -> "Ledger | None":
        p = Path(path)
        if not p.exists():
            return None
        return cls(**json.loads(p.read_text()))

    def save(self, path) -> None:
        path = Path(path)
        tmp = path.with_suffix(path.suffix + ".tmp")
        with tmp.open("w") as fh:
            fh.write(json.dumps(asdict(self), indent=2, allow_nan=False))
            fh.flush()
            os.fsync(fh.fileno())
        tmp.replace(path)

    def rebase(self, closes: pd.DataFrame) -> dict[str, float]:
        """Rescale units where the vendor re-adjusted a held name's history
        since the last mark (value-preserving). Returns the factors applied."""
        factors = {}
        for tk, (d, c_old) in list(self.refs.items()):
            if tk not in self.positions or not c_old:
                continue
            c_new, _ = close_asof(closes, tk, d)
            if np.isfinite(c_new) and c_new > 0 and abs(c_new / c_old - 1.0) > 1e-9:
                factors[tk] = c_old / c_new
                self.positions[tk] *= factors[tk]
                self.refs[tk] = [d, c_new]
        ref = self.bench.get("ref")
        if ref and self.bench.get("units"):
            c_new, _ = close_asof(closes, "SPY", ref[0])
            if np.isfinite(c_new) and c_new > 0 and abs(c_new / ref[1] - 1.0) > 1e-9:
                factors["SPY(bench)"] = ref[1] / c_new
                self.bench["units"] *= factors["SPY(bench)"]
                self.bench["ref"] = [ref[0], c_new]
        return factors

    def fill_pending(self, closes: pd.DataFrame, opens: pd.DataFrame, t,
                     cost_bps: float = COST_BPS) -> list:
        """Execute the stored target weights at the first open after their
        decision date: sells first, buys scaled to the cash left after fees
        (never overdrawn); rebalances under MIN_TRADE_FRAC of NAV are skipped
        (dust trades are not worth a $100 book's spread). Orders decided on
        `t` itself wait for the next run."""
        if not self.pending:
            return []
        d = pd.Timestamp(self.pending["decision_date"])
        t = pd.Timestamp(t)
        after = opens.index[opens.index > d]
        if d >= t or not len(after):
            return []
        fd, fee = after[0], cost_bps / 1e4
        weights = {k: float(v) for k, v in self.pending["weights"].items()}
        names = sorted(set(weights) | set(self.positions))
        px = {tk: fill_price(closes, opens, tk, fd) for tk in names}
        nav = self.cash + sum(u * px[tk][0] for tk, u in self.positions.items()
                              if np.isfinite(px[tk][0]))
        delta = {tk: weights.get(tk, 0.0) * nav - self.positions.get(tk, 0.0) * px[tk][0]
                 for tk in names if np.isfinite(px[tk][0]) and px[tk][0] > 0}
        floor = MIN_TRADE_FRAC * nav
        fills = []
        for tk in sorted(delta, key=delta.get):            # sells first (most negative)
            if delta[tk] >= -1e-9:
                break
            exit_all = weights.get(tk, 0.0) <= 0.0
            if not exit_all and -delta[tk] < floor:
                continue
            p, when = px[tk]
            held = self.positions.get(tk, 0.0)
            units = held if exit_all else min(held, -delta[tk] / p)
            if units <= 0:
                continue
            f = units * p * fee
            self.cash += units * p - f
            self._add_units(tk, -units)
            fills.append([str(when.date()), tk, -units, p, f])
        buys = {tk: v for tk, v in delta.items() if v >= floor}
        total = sum(buys.values())
        scale = min(1.0, self.cash / (total * (1 + fee))) if total > 0 else 0.0
        for tk in sorted(buys):
            p, when = px[tk]
            dollars = buys[tk] * scale
            if dollars <= 1e-9:
                continue
            self.cash -= dollars * (1 + fee)
            self._add_units(tk, dollars / p)
            fills.append([str(when.date()), tk, dollars / p, p, dollars * fee])
        if self.bench.get("cash", 0.0) > 0 and not self.bench.get("units"):
            p, when = px.get("SPY") or fill_price(closes, opens, "SPY", fd)
            self.bench["units"] = self.bench["cash"] / (p * (1 + fee))
            self.bench["cash"] = 0.0
        self.trades.extend(fills)
        self.pending = None
        return fills

    def _add_units(self, tk: str, units: float) -> None:
        new = self.positions.get(tk, 0.0) + units
        if abs(new) < 1e-12:
            self.positions.pop(tk, None)
        else:
            self.positions[tk] = new

    def mark(self, closes: pd.DataFrame, t) -> tuple[float, float]:
        """NAV at the last close on or before t; refresh the reference closes."""
        t = pd.Timestamp(t)
        nav, refs = self.cash, {}
        for tk, u in self.positions.items():
            c, when = close_asof(closes, tk, t)
            if np.isfinite(c):
                nav += u * c
                refs[tk] = [str(when.date()), c]
        self.refs = refs
        spy, when = close_asof(closes, "SPY", t)
        bench = self.bench.get("cash", 0.0) + self.bench.get("units", 0.0) * spy
        if self.bench.get("units"):
            self.bench["ref"] = [str(when.date()), spy]
        row = [str(t.date()), nav, bench]
        self.nav_history = [r for r in self.nav_history if r[0] != row[0]] + [row]
        return nav, bench

    def value_of(self, closes: pd.DataFrame, t) -> dict[str, float]:
        return {tk: u * close_asof(closes, tk, t)[0] for tk, u in self.positions.items()}
