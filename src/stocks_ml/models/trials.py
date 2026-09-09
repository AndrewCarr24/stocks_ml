"""Trials ledger: the project's complete, append-only record of what was tried.

Every configuration the selection procedure evaluates lands here (one row per
trial, upserted by name), so the count N of everything ever tried — the
input a deflated Sharpe or a t >= 3 adoption hurdle needs — is a census, not
a guess. Rows record pre-holdout evidence only; holdout results never enter
the ledger. Rows written by tooling removed on 2026-09-02 (tag legacy-final)
stay in the file: they count toward N like any other trial.
"""
from __future__ import annotations

import fcntl
import json
import math
from datetime import date
from pathlib import Path

import numpy as np

LEDGER_PATH = "models/trials_ledger.json"


def _sanitize(v):
    """JSON-safe values at any depth: numpy scalars unwrapped, non-finite
    floats -> None (json.dumps would otherwise emit a bare NaN that no strict
    parser reads back), containers recursed."""
    if isinstance(v, np.bool_):
        return bool(v)
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (float, np.floating)):
        return float(v) if math.isfinite(v) else None
    if isinstance(v, np.ndarray):
        return [_sanitize(x) for x in v.tolist()]
    if isinstance(v, dict):
        return {str(k): _sanitize(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_sanitize(x) for x in v]
    return v


def load_ledger(path: str | Path = LEDGER_PATH) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return json.loads(p.read_text())


def record_trials(entries: list[dict], path: str | Path = LEDGER_PATH) -> int:
    """Upsert entries (each: kind, name, and optional cv_metric /
    pre_holdout_sharpe / notes). Returns the new total count.

    Re-running an identical config is not a new trial: rows are keyed by
    (kind, name) and updated in place, so reruns keep the census honest
    instead of inflating N. The read-modify-write cycle holds an exclusive
    flock so parallel runs (walks recording as they finish) cannot drop each
    other's rows."""
    p = Path(path)
    lock = p.with_suffix(p.suffix + ".lock")
    with lock.open("w") as lk:
        fcntl.flock(lk.fileno(), fcntl.LOCK_EX)
        ledger = load_ledger(p)
        index = {(r.get("kind"), r.get("name")): i for i, r in enumerate(ledger)}
        today = str(date.today())
        for e in entries:
            row = _sanitize({"date": today, **e})
            key = (row.get("kind"), row.get("name"))
            if key in index:
                ledger[index[key]] = row
            else:
                index[key] = len(ledger)
                ledger.append(row)
        tmp = p.with_suffix(p.suffix + ".tmp")
        tmp.write_text(json.dumps(ledger, indent=1, allow_nan=False))
        tmp.replace(p)
    return len(ledger)
