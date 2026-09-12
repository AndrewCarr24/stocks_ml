"""Render PROCEDURE.md (the procedure card) from models/champion_spec.json.

The spec file is the single durable source of truth for the champion's
settings. Its strategy layers (book, floor, stop, cap) are written by
`stocks-ml procedure` (stocks_ml.procedure), never by hand; the prose fields
are edited in the spec. Rerun `stocks-ml procedure-card` after either; never
edit PROCEDURE.md by hand.
"""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

from stocks_ml.selection import LABELS_4W

SPEC_PATH = Path("models/champion_spec.json")
CARD_PATH = Path("PROCEDURE.md")

TEMPLATE = """\
# Procedure card

Blueprint of the production procedure (generated {today} by
`stocks-ml procedure-card` from models/champion_spec.json — edit the spec,
not this file). Rationale and history: AGENTS.md.

## Current champion

| Component | Spec |
|---|---|
| Model | {model_summary} |
| Prediction target | {horizon_label}: {label_text} ({purge_days}-day purge) |
| Training | weekly refit on trailing {train_years} years; early stop on validation rank correlation |
| Features | {features_summary} |
| Price basis / labels | level features on the {price_basis} basis; a delisting's label grades to its {delist_labels} |
| Ensemble | K={k_copies} copies (random_state + whole-week bootstrap), predictions averaged |
| Book | top-{book_size}, equal weight, {sleeves} staggered sleeves rotating weekly, {hold_weeks}-week holds; weekly re-leveling; {cap_summary}; {stop_summary} |
| Ballast | {ballast_row} |
| Decided by | `stocks-ml procedure` on {proc_preds} (K={proc_k}, {proc_weeks} rank weeks of {proc_lo} -> {proc_hi}, world {proc_world}), {proc_at}: book {proc_book} / floor {proc_floor} / stop {proc_stop} / cap {proc_cap}; model fields from the walk's own record: {proc_label} / {proc_years}-year window — tests hold the spec's model and strategy fields and the live job to this record |
| Honest expectation | {honest_expectation} |

## Cadences

| Activity | When | Human involvement |
|---|---|---|
| Refit + rotate one sleeve | weekly (Friday decision, Monday open trade, {cost_bps} bps) | none |
| Re-level weights / check stops / ballast state | weekly | none |
| Full re-selection (any layer) | **never on a calendar** — structural triggers only: {triggers} | pre-registered, owner-approved |

## Selection procedure (how each component is chosen)

{procedure_note}

| step | menu | decided by |
|---|---|---|
{procedure_rows}

{procedure_constraints}

Metric convention: {metric_convention}. Measured selection inflation of this
procedure: {inflation}.

## Standing rules

- {holdout_start}+ is holdout: {holdout_status}.
- Champion changes are owner-approved and recorded here + in the ledger;
  doubts become pre-registered falsification tests, never quiet overrides.
- Every evaluated config enters models/trials_ledger.json.
- All pre-holdout numbers carry design-iteration shine; treat accordingly.
"""


def features_summary(s: dict) -> str:
    """The model's inputs: the panel's standing f_ columns plus the spec's
    bundle, if one was adopted — the generated bundle (g_ columns with their
    formulas) or a screened bundle of hand-written ideas (asterisk)."""
    feats = s.get("features") or []
    if not feats:
        return ("the panel's f_ columns (features/panel.py, Sharadar f_sf_*/f_sfi_*); no screened bundle "
                "(the split-leak bundle retired 2026-09-11, features/bundle.py)")
    formulas = s.get("formulas") or {}
    if formulas and all(f in formulas for f in feats):
        return (f"the panel's f_ columns plus the generated bundle of {len(feats)} (features/bundle.py, "
                f"selected on 2006-2015 alone, no asterisk): "
                + ", ".join(f"{f} = `{formulas[f]}`" for f in feats))
    return (f"the panel's f_ columns plus the screened bundle of {len(feats)}: {', '.join(feats)} "
            f"(feature screen, rule v3.1; asterisk: the candidate ideas were written from the whole "
            f"2006-2024 record)")


def strategy_summary(s: dict) -> tuple[str, str]:
    """The cap and stop clauses of the Book row from the strategy block."""
    cap, stop = s["strategy"]["sector_cap"], s["strategy"]["stop_loss"]
    cap_summary = (f"max {cap}/sector (blocked slots to next-ranked other-sector name)"
                   if cap else "no sector cap")
    stop_summary = f"stop at {stop:+.0%} (to SPY until the sleeve rotates)" if stop else "no stop"
    return cap_summary, stop_summary


def ballast_row(s: dict) -> str:
    """The Ballast row: what the floor does with the money outside the book
    (ledger.floor_split). A fixed floor parks it in SPY and shifts one third
    to IEF per breached SPY moving average; halfgate cuts the book itself
    per breached average and parks the remainder in IEF."""
    floor, mix = s["procedure"]["decision"]["floor"], s["ballast"]["mix"]
    if floor == "halfgate":
        return f"{mix} — the book shrinks one sixth of NAV per breached SPY trailing MA, " \
               "the freed money sits in IEF (no fixed SPY ballast)"
    if floor == "none":
        return mix
    return f"{mix}: ballast in SPY, shifted to IEF one-third per breached trailing MA (30/40/52w)"


def render(spec: dict, today: str | None = None) -> str:
    s = spec
    cap_summary, stop_summary = strategy_summary(s)
    proc = s["procedure"]
    return TEMPLATE.format(
        today=today or str(date.today()),
        model_summary=s["model"]["summary"],
        horizon_label=s["horizon"]["label"],
        label_text=LABELS_4W[s["horizon"]["label"]],
        purge_days=s["horizon"]["purge_days"],
        train_years=s["training_window_years"],
        k_copies=s["ensemble"]["k_copies"],
        features_summary=features_summary(s),
        price_basis=s.get("price_basis", "closeadj"),
        delist_labels={"last_print": "final print (last_print)"}.get(
            s.get("delist_labels", "drop"), "no label; the name is dropped (drop)"),
        book_size=s["strategy"]["book_size"],
        sleeves=s["strategy"]["sleeves"],
        hold_weeks=s["strategy"]["hold_weeks"],
        cap_summary=cap_summary,
        stop_summary=stop_summary,
        ballast_row=ballast_row(s),
        proc_preds=proc["preds"]["path"], proc_k=proc["k_copies"],
        proc_weeks=proc["preds"]["rank_weeks"], proc_lo=proc["selection_window"][0],
        proc_hi=proc["selection_window"][1], proc_world=proc["world"], proc_at=proc["decided_at"],
        proc_book=proc["decision"]["book_size"], proc_floor=proc["decision"]["floor"],
        proc_stop=proc["decision"]["stop_loss"], proc_cap=proc["decision"]["sector_cap"],
        proc_label=proc["model"]["label"], proc_years=proc["model"]["train_years"],
        honest_expectation=s["honest_expectation"],
        cost_bps=s["costs_assumed_bps_oneway"],
        triggers="; ".join(s["retune_policy"]["triggers"]),
        holdout_start=s["holdout_start"],
        holdout_status=s["holdout_status"],
        procedure_note=s["selection_procedure"]["note"],
        procedure_rows="\n".join(
            f"| {st['step']} | {st['menu']} | {st['metric']} |"
            for st in s["selection_procedure"]["steps"]),
        procedure_constraints=s["selection_procedure"]["constraints"],
        metric_convention=s["selection_procedure"]["metric_convention"],
        inflation=s["selection_procedure"]["measured_selection_inflation"],
    )


def write_card(spec_path: Path = SPEC_PATH, card_path: Path = CARD_PATH) -> str:
    spec = json.loads(spec_path.read_text())
    text = render(spec)
    card_path.write_text(text)
    return text
