"""The paper book's rules and accounting, shared by the backtest engine
(selection.simulate) and the live weekly job (live/r5.py).

A signal dated t (the week's last trading day) rotates one of the K sleeves
to the week's ranked picks, reads the trend ballast, and stores target
weights. The ledger fills those weights at the first open after t, paying
COST_BPS a side on every trade and skipping rebalances under MIN_TRADE_FRAC
of NAV, then marks NAV at the last close on or before each week's label.
The backtest and the live job differ only in where the picks come from.

Units are on Sharadar's total-return price basis (closeadj): dividends and
splits arrive as retroactive rescalings of the whole history, so each
position stores two reference closes anchored at its fill date and `rebase`
rescales units by old/new reference close before the next run touches the
book — the factor must agree at both reference dates, so a transient bad
print is skipped (and logged) rather than booked as a split.
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
# The ballast floor menu the procedure chooses from (selection.decide_strategy)
# and the live job executes (live/r5.py), one rule for both: floor_split.
FLOORS = ("none", "halfgate", "80/20", "70/30", "60/40")
FLOOR_FRACTION = {"80/20": 0.8, "70/30": 0.7, "60/40": 0.6}   # the fixed book / ballast splits
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
        age = week_index(t) - week_index(s["since"]) if s.get("since") else None
        if age is not None and age <= 0 and s["names"]:
            # already rotated in this signal's week: a rerun of the same week
            # must not rotate it again (once per week). Ages count rank weeks
            # (week_index), not calendar days, so a holiday Thursday is a
            # week older than the previous Friday, never the same age.
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


def floor_split(floor: str, gates: dict) -> tuple[float, dict]:
    """The floor menu as (book fraction of NAV, ballast thirds) for
    target_weights. `gates` is ballast_state's per-window SPY/IEF reading.
      none      the whole of NAV in the book
      halfgate  book at 1 - g/2, g the share of gates down (100% with none
                down, 50% with all three); the rest in IEF only
      80/20 ..  a fixed book fraction; the rest one third per gate, SPY or IEF
    """
    if floor == "none":
        return 1.0, {}
    if floor == "halfgate":
        below = {w: f for w, f in gates.items() if f == "IEF"}
        return 1.0 - 0.5 * len(below) / len(gates), below
    return FLOOR_FRACTION[floor], dict(gates)


def target_weights(sleeves: dict, ballast: dict, floor: float) -> dict[str, float]:
    """Sleeves equal-weighted at `floor` of NAV (the book fraction floor_split
    returns), names equal within a sleeve; the rest split evenly across the
    ballast's thirds."""
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
               fill_date, for_buy: bool = False) -> tuple[float, pd.Timestamp | None]:
    """Open on the fill date, else the first open within five sessions, else
    the last close before it (a name that stopped trading is closed out at
    its final print). Buys need a live open: `for_buy` disables the close
    fallback so a dead name is never "bought" at its last print."""
    if ticker in opens.columns:
        s = opens[ticker].loc[pd.Timestamp(fill_date):].iloc[:5].dropna()
        if len(s):
            return float(s.iloc[0]), s.index[0]
    if for_buy:
        return np.nan, None
    return close_asof(closes, ticker, fill_date)


def _ref_points(closes: pd.DataFrame, ticker: str, date) -> list:
    """Two reference closes on or before `date` (the fill date), newest
    first: [d1, c1, d2, c2]. `rebase` requires the vendor's implied
    adjustment factor to agree at both dates before rescaling units."""
    if ticker not in closes.columns:
        return []
    s = closes[ticker].loc[:pd.Timestamp(date)].dropna().iloc[-2:]
    if not len(s):
        return []
    d1, c1 = s.index[-1], float(s.iloc[-1])
    d2, c2 = (s.index[0], float(s.iloc[0])) if len(s) == 2 else (d1, c1)
    return [str(d1.date()), c1, str(d2.date()), c2]


# ---- the ledger ----
@dataclass
class Ledger:
    cash: float = 0.0
    positions: dict = field(default_factory=dict)    # ticker -> units (closeadj basis)
    refs: dict = field(default_factory=dict)         # ticker -> [d1, c1, d2, c2] at fill (legacy: [date, close])
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

    def rename(self, mapping: dict[str, str]) -> dict[str, str]:
        """Apply the vendor's symbol renames (world.detect_renames) to the
        book: positions, references, sleeve names and pending weights move
        to the new symbol before any price lookup happens. Without this a
        renamed holding has no price column: unsellable and worth $0 in NAV.
        Returns the renames that touched the ledger."""
        hit = {}
        for old, new in mapping.items():
            if old in self.positions:
                self.positions[new] = self.positions.pop(old) + self.positions.get(new, 0.0)
                hit[old] = new
            if old in self.refs:
                self.refs[new] = self.refs.pop(old)
                hit[old] = new
            for s in self.sleeves.values():
                if old in s.get("names", []):
                    s["names"] = [new if n == old else n for n in s["names"]]
                    hit[old] = new
            w = (self.pending or {}).get("weights") or {}
            if old in w:
                w[new] = w.pop(old) + w.get(new, 0.0)
                hit[old] = new
        return hit

    def rebase(self, closes: pd.DataFrame, log=None) -> dict[str, float]:
        """Rescale units where the vendor re-adjusted a held name's history
        (value-preserving). References anchor at the position's fill date and
        stay there, so an adjustment published any number of weeks after its
        ex-date is still caught. The implied factor must agree at both
        reference dates or the rescale is skipped (and logged): a transient
        bad print is not a split. Returns the factors applied."""
        missing = [tk for tk in self.positions if tk not in closes.columns]
        if missing:
            raise RuntimeError(
                f"held names have no price column (vendor rename or dropped series): {missing}; "
                "apply the world's symbol renames to the ledger before rebasing")

        def factor_of(ref, tk):
            d1, c1 = ref[0], ref[1]
            d2, c2 = (ref[2], ref[3]) if len(ref) == 4 else (ref[0], ref[1])
            n1, w1 = close_asof(closes, tk, d1)
            n2, w2 = close_asof(closes, tk, d2)
            if (w1, w2) != (pd.Timestamp(d1), pd.Timestamp(d2)):
                # a stale or reshaped store has no print on the reference
                # date itself: comparing another day's close would book price
                # drift as an adjustment
                if log:
                    log(f"rebase: {tk} has no print at its reference dates {d1}/{d2} "
                        f"(store ends earlier?); units not rescaled")
                return None, ref
            if not (np.isfinite(n1) and n1 > 0 and np.isfinite(n2) and n2 > 0) or not c1 or not c2:
                return None, ref
            f1, f2 = c1 / n1, c2 / n2
            if abs(f1 - 1.0) <= 1e-9:
                return None, [d1, n1, d2, n2]
            if abs(f1 / f2 - 1.0) > 1e-6:
                if log:
                    log(f"rebase: {tk} adjustment disagrees at {d1} ({f1:.6f}) vs {d2} "
                        f"({f2:.6f}); units not rescaled")
                return None, ref
            return f1, [d1, n1, d2, n2]

        factors = {}
        for tk in sorted(self.positions):
            ref = self.refs.get(tk)
            if not ref:
                continue
            f, newref = factor_of(ref, tk)
            self.refs[tk] = newref
            if f is not None:
                factors[tk] = f
                self.positions[tk] *= f
        ref = self.bench.get("ref")
        if ref and self.bench.get("units"):
            f, newref = factor_of(ref, "SPY")
            self.bench["ref"] = newref
            if f is not None:
                factors["SPY(bench)"] = f
                self.bench["units"] *= f
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
        buy_px = {tk: fill_price(closes, opens, tk, fd, for_buy=True)
                  for tk, v in delta.items() if v >= floor}
        buys = {tk: v for tk, v in delta.items()
                if v >= floor and np.isfinite(buy_px[tk][0]) and buy_px[tk][0] > 0}
        total = sum(buys.values())
        scale = min(1.0, self.cash / (total * (1 + fee))) if total > 0 else 0.0
        for tk in sorted(buys):
            p, when = buy_px[tk]
            dollars = buys[tk] * scale
            if dollars <= 1e-9:
                continue
            self.cash -= dollars * (1 + fee)
            self._add_units(tk, dollars / p)
            fills.append([str(when.date()), tk, dollars / p, p, dollars * fee])
        for tk in self.positions:
            if len(self.refs.get(tk) or []) != 4:
                pts = _ref_points(closes, tk, fd)
                if pts:
                    self.refs[tk] = pts               # anchor at fill; rebase keeps it there
        if self.bench.get("cash", 0.0) > 0 and not self.bench.get("units"):
            p, when = px.get("SPY") or fill_price(closes, opens, "SPY", fd)
            self.bench["units"] = self.bench["cash"] / (p * (1 + fee))
            self.bench["cash"] = 0.0
            self.bench["ref"] = _ref_points(closes, "SPY", when or fd) or None
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
        """NAV at the last close on or before t. References stay anchored at
        the fill date (see rebase); mark only prunes refs of exited names and
        backfills one that is missing, legacy, or degenerate."""
        t = pd.Timestamp(t)
        nav, missing = self.cash, []
        for tk, u in self.positions.items():
            c, when = close_asof(closes, tk, t)
            if not np.isfinite(c):
                missing.append(tk)
                continue
            nav += u * c
            ref = self.refs.get(tk) or []
            if len(ref) != 4 or ref[0] == ref[2]:
                self.refs[tk] = _ref_points(closes, tk, when)
        if missing:
            raise RuntimeError(
                f"held names have no price on or before {t.date()}: {missing} "
                "(vendor rename or dropped series; NAV would silently shrink)")
        self.refs = {tk: v for tk, v in self.refs.items() if tk in self.positions}
        spy, when = close_asof(closes, "SPY", t)
        bench = self.bench.get("cash", 0.0) + self.bench.get("units", 0.0) * spy
        ref = self.bench.get("ref") or []
        if self.bench.get("units") and (len(ref) != 4 or ref[0] == ref[2]):
            self.bench["ref"] = _ref_points(closes, "SPY", when)
        row = [str(t.date()), nav, bench]
        self.nav_history = sorted([r for r in self.nav_history if r[0] != row[0]] + [row])
        return nav, bench

    def value_of(self, closes: pd.DataFrame, t) -> dict[str, float]:
        return {tk: u * close_asof(closes, tk, t)[0] for tk, u in self.positions.items()}
