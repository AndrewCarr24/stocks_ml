"""The r5 champion's weekly signal job (`stocks-ml r5-weekly`).

Runs on the owner's Mac (launchd, Saturday morning): it needs the Sharadar
key and a fresh world store, neither of which belongs in Actions. Steps:

  1. data/world.py refreshes the live world and rebuilds panel_sf.parquet
  2. selection.ensemble_preds ranks this Friday's members exactly as the
     research pipeline did (K=4 week-bootstrap copies, 4w label, 5y window)
  3. the sleeve schedule rotates one of four 6-name sleeves (sector cap 2)
  4. the 70/30 trend ballast decides SPY vs IEF per moving-average third
  5. a paper ledger fills LAST week's orders at Monday's open, marks NAV at
     Friday's close and stores this week's target weights as pending orders

Everything the job decides is written to signals_r5/<friday>.md (+ .json)
and ledger_r5.json. The rules mirror selection.simulate — the function the
champion was scored with — with two live-only additions: a name must have
traded within the last five sessions to be rankable (simulate required a
forward return, which implies the same), and a sleeve that missed its
rotation because the job skipped a week is rotated at the next run.

Units in the ledger are on Sharadar's total-return price basis (closeadj):
dividends and splits arrive as retroactive rescalings of the whole history,
so every mark stores a reference close per position and the next run
rescales units by old/new reference close before doing anything else.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from stocks_ml.selection import HORIZONS, Ctx, ensemble_preds, pick_capped

SPEC = {"horizon": "4w", "train_years": 5, "book": 6, "cap": 2, "floor": 0.7,
        "top_n": 15}                       # models/champion_spec.json
N_SLEEVES = HORIZONS[SPEC["horizon"]]["kweeks"]
ANCHOR = pd.Timestamp("2001-01-05")        # week 0 of the sleeve schedule
BALLAST_WINDOWS = (30, 40, 52)             # weeks; each third: IEF when SPY < its MA
COST_BPS = 5.0                             # per side (procedure card)
STALE_WEEKS = 5                            # a sleeve this old missed a rotation
MIN_UNIVERSE = 100                         # rankable names needed for a signal
MIN_TRADE_FRAC = 0.005                     # skip rebalances under 0.5% of NAV (full exits always run)
TRADABLE_DAYS = 7                          # a name must have a close this recent
FUNDS = ("SPY", "IEF")


def _log(msg):
    print(msg, flush=True)


# ---- pure rules (unit-tested) ----
def week_index(t) -> int:
    return int((pd.Timestamp(t) - ANCHOR).days // 7)


def due_sleeve(t) -> int:
    return week_index(t) % N_SLEEVES


def friday_of(t) -> pd.Timestamp:
    t = pd.Timestamp(t)
    return t + pd.Timedelta(days=(4 - t.weekday()) % 7)


def rotate_sleeves(sleeves: dict, t, ranked: list[str], smap: dict) -> tuple[dict, list[int]]:
    """One sleeve rotates per week (week_index mod 4, as simulate's
    `i % period == c`); empty sleeves fill immediately (simulate's first
    week), stale ones catch up. Returns (new sleeves, rotated sleeve ids)."""
    t = pd.Timestamp(t)
    out, rotated = {}, []
    for k in range(N_SLEEVES):
        s = sleeves.get(str(k), {"names": [], "since": None})
        age = (t - pd.Timestamp(s["since"])).days // 7 if s.get("since") else None
        if age is not None and age <= 0 and s["names"]:
            # already rotated on this signal date: a rerun of the same
            # Friday must not rotate it again (simulate: once per week)
            out[str(k)] = {"names": list(s["names"]), "since": s["since"]}
        elif k == due_sleeve(t) or not s["names"] or (age is not None and age >= STALE_WEEKS):
            names = pick_capped(list(ranked[:SPEC["top_n"]]), SPEC["cap"], SPEC["book"], smap)
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


def target_weights(sleeves: dict, ballast: dict, floor: float = SPEC["floor"]) -> dict[str, float]:
    """simulate's book: sleeves equal-weighted, names equal within a sleeve;
    the (1 - floor) ballast split evenly across the MA thirds."""
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


# ---- paper ledger ----
def _close_asof(cw: pd.DataFrame, ticker: str, date) -> tuple[float, pd.Timestamp | None]:
    if ticker not in cw.columns:
        return np.nan, None
    s = cw[ticker].loc[:pd.Timestamp(date)].dropna()
    return (float(s.iloc[-1]), s.index[-1]) if len(s) else (np.nan, None)


def _fill_price(cw: pd.DataFrame, ow: pd.DataFrame, ticker: str, fill_date) -> tuple[float, pd.Timestamp | None]:
    """Open on the fill date, else the first open within five sessions, else
    the last close before it (a name that stopped trading is closed out at
    its final print)."""
    if ticker in ow.columns:
        s = ow[ticker].loc[pd.Timestamp(fill_date):].dropna().iloc[:5]
        if len(s):
            return float(s.iloc[0]), s.index[0]
    return _close_asof(cw, ticker, fill_date)


@dataclass
class R5Ledger:
    cash: float = 0.0
    positions: dict = field(default_factory=dict)    # ticker -> units (closeadj basis)
    refs: dict = field(default_factory=dict)         # ticker -> [date, close] at last mark
    sleeves: dict = field(default_factory=dict)      # "0".."3" -> {names, since}
    pending: dict | None = None                       # {decision_date, weights}
    nav_history: list = field(default_factory=list)  # [date, nav, spy_nav]
    trades: list = field(default_factory=list)       # [fill_date, ticker, units, price, fee]
    bench: dict = field(default_factory=dict)        # SPY buy-and-hold: cash, units, ref
    started: str | None = None

    @classmethod
    def new(cls, capital: float, t) -> "R5Ledger":
        return cls(cash=float(capital), bench={"cash": float(capital), "units": 0.0, "ref": None},
                   started=str(pd.Timestamp(t).date()))

    @classmethod
    def load(cls, path) -> "R5Ledger | None":
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

    def rebase(self, cw: pd.DataFrame) -> dict[str, float]:
        """Rescale units where the vendor re-adjusted a held name's history
        since the last mark (value-preserving). Returns the factors applied."""
        factors = {}
        for tk, (d, c_old) in list(self.refs.items()):
            if tk not in self.positions or not c_old:
                continue
            c_new, _ = _close_asof(cw, tk, d)
            if np.isfinite(c_new) and c_new > 0 and abs(c_new / c_old - 1.0) > 1e-9:
                factors[tk] = c_old / c_new
                self.positions[tk] *= factors[tk]
                self.refs[tk] = [d, c_new]
        ref = self.bench.get("ref")
        if ref and self.bench.get("units"):
            c_new, _ = _close_asof(cw, "SPY", ref[0])
            if np.isfinite(c_new) and c_new > 0 and abs(c_new / ref[1] - 1.0) > 1e-9:
                factors["SPY(bench)"] = ref[1] / c_new
                self.bench["units"] *= factors["SPY(bench)"]
                self.bench["ref"] = [ref[0], c_new]
        return factors

    def fill_pending(self, cw: pd.DataFrame, ow: pd.DataFrame, t, cost_bps: float = COST_BPS) -> list:
        """Execute the stored target weights at the first open after their
        decision date: sells first, buys scaled to the cash left after fees
        (never overdrawn); rebalances under MIN_TRADE_FRAC of NAV are skipped
        (simulate charges only rotations, and dust trades are not worth a
        $100 book's spread). Orders decided on `t` itself wait for next run."""
        if not self.pending:
            return []
        d = pd.Timestamp(self.pending["decision_date"])
        t = pd.Timestamp(t)
        after = ow.index[ow.index > d]
        if d >= t or not len(after):
            return []
        fd, fee = after[0], cost_bps / 1e4
        weights = {k: float(v) for k, v in self.pending["weights"].items()}
        names = sorted(set(weights) | set(self.positions))
        px = {tk: _fill_price(cw, ow, tk, fd) for tk in names}
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
            p, when = px.get("SPY") or _fill_price(cw, ow, "SPY", fd)
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

    def mark(self, cw: pd.DataFrame, t) -> tuple[float, float]:
        """NAV at the last close on or before t; refresh the reference closes."""
        t = pd.Timestamp(t)
        nav, refs = self.cash, {}
        for tk, u in self.positions.items():
            c, when = _close_asof(cw, tk, t)
            if np.isfinite(c):
                nav += u * c
                refs[tk] = [str(when.date()), c]
        self.refs = refs
        spy, when = _close_asof(cw, "SPY", t)
        bench = self.bench.get("cash", 0.0) + self.bench.get("units", 0.0) * spy
        if self.bench.get("units"):
            self.bench["ref"] = [str(when.date()), spy]
        row = [str(t.date()), nav, bench]
        self.nav_history = [r for r in self.nav_history if r[0] != row[0]] + [row]
        return nav, bench

    def value_of(self, cw: pd.DataFrame, t) -> dict[str, float]:
        return {tk: u * _close_asof(cw, tk, t)[0] for tk, u in self.positions.items()}


# ---- the weekly run ----
def rank_members(preds: pd.Series, prices: pd.DataFrame, t) -> pd.Series:
    """Predictions for names that traded within the last few sessions,
    highest first."""
    t = pd.Timestamp(t)
    recent = prices[(prices["date"] > t - pd.Timedelta(days=TRADABLE_DAYS))
                    & (prices["date"] <= t)]
    tradable = set(recent.dropna(subset=["close"])["ticker"])
    p = preds[preds.index.isin(tradable)]
    if len(p) < MIN_UNIVERSE:
        raise RuntimeError(f"only {len(p)} rankable names at {t.date()} (need {MIN_UNIVERSE})")
    return p.sort_values(ascending=False)


def last_friday(today=None) -> pd.Timestamp:
    d = pd.Timestamp(today or pd.Timestamp.today()).normalize()
    return d - pd.Timedelta(days=(d.weekday() - 4) % 7)


def run_weekly(live_dir, cfg, as_of=None, refresh=True, sec=True, dry_run=False,
               capital=100.0, out_dir="signals_r5", ledger_path="ledger_r5.json",
               log=_log) -> dict:
    from stocks_ml.data.world import build_world_panel, refresh_world
    t0 = time.time()
    report = {"run_at": str(pd.Timestamp.now()), "spec": SPEC, "dry_run": dry_run}
    if refresh:
        report["refresh"] = refresh_world(live_dir, cfg, sec=sec, log=log)
    if refresh or not (Path(live_dir) / "panel_sf.parquet").exists():
        build_world_panel(live_dir, cfg, log=log)

    ctx = Ctx(str(live_dir))
    t = pd.Timestamp(as_of) if as_of else ctx.weeks[-1]
    if t not in ctx.members:
        raise RuntimeError(f"{t.date()} is not a panel date; latest is {ctx.weeks[-1].date()}")
    if as_of is None and friday_of(t) < last_friday():       # holiday Fridays: Thursday is fine
        raise RuntimeError(f"panel ends {t.date()} but the last Friday was "
                           f"{last_friday().date()}: prices are not refreshed yet")
    log(f"signal date {t.date()} (sleeve {due_sleeve(t)} due); fitting {SPEC}")
    t1 = time.time()
    preds = ensemble_preds(ctx, t, SPEC["horizon"], SPEC["train_years"])
    if preds is None:
        raise RuntimeError(f"no ensemble prediction for {t.date()}")
    ranked = rank_members(preds, ctx.prices, t)
    log(f"ranked {len(ranked)} names in {time.time() - t1:.0f}s; "
        f"top-{SPEC['top_n']}: {', '.join(ranked.index[:SPEC['top_n']])}")

    ledger = R5Ledger.load(ledger_path) or R5Ledger.new(capital, t)
    daily = ctx.prices.sort_values("date")
    cw = daily.pivot(index="date", columns="ticker", values="close").sort_index().ffill()
    ow = daily.pivot(index="date", columns="ticker", values="open").sort_index()
    factors = ledger.rebase(cw)
    fills = ledger.fill_pending(cw, ow, t)
    nav, bench = ledger.mark(cw, t)
    sleeves, rotated = rotate_sleeves(ledger.sleeves, t, list(ranked.index), ctx.smap)
    ballast = ballast_state(ctx.spy_w, t)
    weights = target_weights(sleeves, ballast)
    ledger.sleeves = sleeves
    ledger.pending = {"decision_date": str(t.date()), "weights": weights}

    held = ledger.value_of(cw, t)
    signal = {
        "date": str(t.date()), "sleeve_due": due_sleeve(t), "rotated": rotated,
        "sleeves": sleeves, "ballast": ballast, "weights": weights,
        "nav": nav, "spy_nav": bench, "cash": ledger.cash,
        "held_value": held, "fills": fills, "rebase_factors": factors,
        "top": [(tk, float(v)) for tk, v in ranked.iloc[:SPEC["top_n"]].items()],
        "n_ranked": int(len(ranked)), "positions": ledger.positions,
        "freshness": _freshness(ctx, report.get("refresh")),
        "elapsed_s": round(time.time() - t0),
    }
    report["signal"] = signal
    md = render_markdown(signal, ctx.smap)
    if dry_run:
        log("dry run: ledger and signal files not written")
    else:
        out = Path(out_dir)
        out.mkdir(exist_ok=True)
        (out / f"{t.date()}.md").write_text(md)
        (out / f"{t.date()}.json").write_text(json.dumps(signal, indent=2, default=str))
        ledger.save(ledger_path)
        log(f"wrote {out / f'{t.date()}.md'} and {ledger_path}")
    log(md)
    return report


def _freshness(ctx: Ctx, refresh: dict | None) -> dict:
    out = {"panel": str(ctx.weeks[-1].date()),
           "prices": str(ctx.prices["date"].max().date())}
    if refresh:
        for src, rep in refresh.get("sharadar", {}).items():
            for k in ("through", "filed_through", "events_through"):
                if k in rep:
                    out[src] = rep[k]
        for src, rep in refresh.get("sec", {}).items():
            for k in ("filed_through", "coverage_end", "last_date"):
                if k in rep:
                    out[src] = rep[k]
    return out


def render_markdown(sig: dict, smap: dict) -> str:
    nav, bench = sig["nav"], sig["spy_nav"]
    counts = sleeve_counts(sig["sleeves"])
    lines = [f"# r5 signal — {sig['date']}", "",
             "Champion r5 (PROCEDURE.md): 70/30 trend ballast, top-6 four-sleeve "
             "stagger, sector cap 2, 4-week label, 5-year window, K=4.", "",
             f"Paper NAV **${nav:,.2f}** · SPY buy-and-hold ${bench:,.2f} · "
             f"cash ${sig['cash']:,.2f}", "",
             "## This week", "",
             f"Sleeve {sig['sleeve_due']} due; rotated {sig['rotated']}.",
             "Ballast: " + ", ".join(f"{w}w→{f}" for w, f in sig["ballast"].items()), "",
             "## Target book (execute at Monday's open)", "",
             "| ticker | sector | sleeves | weight | target $ | held $ | Δ $ |",
             "|---|---|---|---|---|---|---|"]
    held = sig["held_value"]
    for tk in sorted(set(sig["weights"]) | set(held),
                     key=lambda x: -sig["weights"].get(x, 0.0)):
        w = sig["weights"].get(tk, 0.0)
        cur = held.get(tk, 0.0)
        sector = "ballast" if tk in FUNDS else smap.get(tk, "?")
        lines.append(f"| {tk} | {sector} | {counts.get(tk, '')} | {w:.2%} | "
                     f"${w * nav:,.2f} | ${cur:,.2f} | {w * nav - cur:+,.2f} |")
    lines += ["", "## Sleeves", ""]
    for k, s in sig["sleeves"].items():
        lines.append(f"- sleeve {k} (since {s['since']}): {', '.join(s['names'])}")
    lines += ["", f"## Top-{len(sig['top'])} of {sig['n_ranked']} ranked", "",
              "| rank | ticker | sector | score |", "|---|---|---|---|"]
    for i, (tk, v) in enumerate(sig["top"], 1):
        lines.append(f"| {i} | {tk} | {smap.get(tk, '?')} | {v:+.4f} |")
    if sig["fills"]:
        lines += ["", "## Fills since the last signal", "",
                  "| date | ticker | units | price | fee |", "|---|---|---|---|---|"]
        for d, tk, u, p, f in sig["fills"]:
            lines.append(f"| {d} | {tk} | {u:+.4f} | ${p:,.2f} | ${f:.4f} |")
    if sig["rebase_factors"]:
        lines += ["", "Re-adjusted histories (units rescaled): " +
                  ", ".join(f"{k} ×{v:.6f}" for k, v in sig["rebase_factors"].items())]
    lines += ["", "## Data", "",
              ", ".join(f"{k} {v}" for k, v in sig["freshness"].items()),
              "", f"Run time {sig['elapsed_s']}s."]
    return "\n".join(lines)
