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
    """The model's inputs: the panel's standing f_ columns plus any extra
    panel columns the spec names (`features`; none since 2026-09-11)."""
    feats = s.get("features") or []
    if not feats:
        return ("the panel's f_ columns (features/panel.py, Sharadar f_sf_*/f_sfi_*); no screened bundle "
                "(the split-leak bundle retired 2026-09-11)")
    return f"the panel's f_ columns plus the extra columns {', '.join(feats)} (asterisk: see features_note)"


VOL_CUT_WORDS = {None: "no volatility cut",
                 "abs": "volatility cut: the most volatile third of the model's top-30 (12-week volatility rank above +0.3) is dropped before the sleeve picks",
                 "abs_or_sector": "volatility cut: the most volatile third of the model's top-30 (12-week volatility rank above +0.3), and any name 0.3 above its sector's median volatility, are dropped before the sleeve picks"}


def strategy_summary(s: dict) -> tuple[str, str]:
    """The cap, stop and volatility-cut clauses of the Book row from the strategy block."""
    cap, stop = s["strategy"]["sector_cap"], s["strategy"]["stop_loss"]
    cap_summary = (f"max {cap}/sector (blocked slots to next-ranked other-sector name)"
                   if cap else "no sector cap")
    stop_summary = f"stop at {stop:+.0%} (to SPY until the sleeve rotates)" if stop else "no stop"
    stop_summary += "; " + VOL_CUT_WORDS[s["strategy"].get("vol_cut")]
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


README_PATH = Path("README.md")
BLOCK_BEGIN, BLOCK_END = "<!-- champion:begin (stocks-ml procedure-card) -->", "<!-- champion:end -->"
SPANS = ("2006-2024", "2016-2024", "2006-2015")


def _row(name: str, r: dict) -> str:
    cells = []
    for w in SPANS:
        cells += [f"${r[w]['terminal_100']:,.0f}, {r[w]['cagr_pct']:+.1f}%",
                  f"{r[w]['sharpe']:.2f}, {r[w]['max_dd']:.0%}"]
    t = ("—" if "paired_t_vs_sp500" not in r else
         " / ".join(f"{r['paired_t_vs_sp500'][w]:+.2f}" for w in SPANS[:2]))
    return f"| {name} | " + " | ".join(cells) + f" | {t} |"


def champion_block(spec: dict, ev: dict) -> str:
    """The README's champion section body: the settings table from the
    spec, the record table and the caveat paragraph from the champion's
    eval.json, the out-of-sample chart. Numbers only; the model's character
    line comes from the spec's `character` prose field when present."""
    s, st = spec, spec["strategy"]
    cap = f"at most {st['sector_cap']} per sector" if st["sector_cap"] else "no sector cap"
    stop = (f"stop-loss at {st['stop_loss']:.0%}" if st["stop_loss"] else "no stop-loss") + "; " + VOL_CUT_WORDS[st.get("vol_cut")]
    feats = f" plus {len(s.get('features') or [])} named columns" if s.get("features") else ""
    settings = [
        "| Component | Setting |", "|---|---|",
        f"| Model | {s['model']['summary']} (`selection.MODEL_PARAMS`) |",
        f"| Target | {LABELS_4W[s['horizon']['label']]}; {s['horizon']['purge_days']}-day purge |",
        f"| Training | refit every week on the trailing {s['training_window_years']} years; "
        f"early stopping on a purged, time-ordered tail |",
        f"| Ensemble | {s['ensemble']['k_copies']} copies (seed + whole-week bootstrap), scores averaged |",
        f"| Features | the point-in-time panel's `f_` columns: prices, fundamentals, filings, insider "
        f"trades, short interest, macro{feats} |",
        f"| Book | top-{st['book_size']} per sleeve, equal weight, {st['sleeves']} staggered sleeves — one "
        f"rotates each week, so every name is held {st['hold_weeks']} weeks; {cap}; {stop} |",
        f"| Ballast | {ballast_row(s)} |",
        f"| Costs | {s['costs_assumed_bps_oneway']} bp one-way, fills at the next session's open |"]
    lab = ev["label"]
    tab = ev["table"]
    record = [f"**Record** (`stocks-ml eval`, [reports/{lab}_eval.md](reports/{lab}_eval.md), "
              f"{ev['evaluated_at'][:10]}), $100 at the start of each span, costs included, holdout excluded:",
              "",
              "| model | " + " | ".join(f"{w}: $100, %/yr | SR, DD" for w in SPANS)
              + " | weekly t vs sp500 (06-24 / 16-24) |",
              "|---|" + "---|" * (2 * len(SPANS) + 1),
              _row("champion", tab[lab]), _row("sp500", tab["sp500"])]
    c16 = ev["confidence"]["2016-2024"]
    ex = c16["excess_cagr_vs_sp500_pct"]
    cav = [f"2006–2015 is where the label, window and strategy layers were chosen. "
           f"2016–2024 was read once, for this grade. The excess over the S&P 500 on 2016–2024 is "
           f"{ex['point']:+.1f}%/yr; resampling both the model's seeds and the history gives a 95% "
           f"interval of {ex['nested_95'][0]:+.1f} to {ex['nested_95'][1]:+.1f}%/yr, and "
           f"{c16['p_excess_positive_nested']:.0%} of the resampled histories beat the index."]
    if ev.get("falsification"):
        f = ev["falsification"]
        cav.append(f"Against the previous champion the paired weekly excess on 2016–2024 has "
                   f"t {f['2016-2024']['t']:+.2f} ({f['verdict']}; t < −2 would have rejected it).")
    cav.append(f"Leak audit {ev['leak_audit']['VERDICT']}.")
    if s.get("character"):
        cav.append(s["character"])
    cav.append(f"Worst drawdowns: {tab[lab]['2006-2015']['max_dd']:.0%} on 2006–2015 and "
               f"{tab[lab]['2016-2024']['max_dd']:.0%} on 2016–2024 (the S&P: "
               f"{tab['sp500']['2006-2015']['max_dd']:.0%} and {tab['sp500']['2016-2024']['max_dd']:.0%}). "
               f"Sizing should assume index-like outcomes in adverse regimes. The holdout "
               f"({s['holdout_start']} onward) has never been graded.")
    chart = (ev.get("charts") or [f"reports/{lab}_vs_sp500_2016_2024.png"])[0]
    return "\n".join(settings + [""] + record + ["", " ".join(cav), "",
                      f"![Growth of $100, 2016-2024, out of sample]({chart})", "",
                      "The chart shows the out-of-sample years only. 2006–2015 chose the settings, "
                      "so a curve that includes it overstates the model."])


def write_readme_block(spec: dict, readme_path: Path = README_PATH) -> bool:
    """Rewrite README's champion block from the spec and the champion walk's
    eval.json (written by `stocks-ml eval`). Returns False, changing
    nothing, when that eval does not exist yet."""
    walk = Path(spec["procedure"]["preds"]["path"]).parent.parent
    ev_path = walk / "eval.json"
    if not ev_path.exists():
        return False
    ev = json.loads(ev_path.read_text())
    text = readme_path.read_text()
    a, b = text.find(BLOCK_BEGIN), text.find(BLOCK_END)
    if a < 0 or b < 0 or b < a:
        raise RuntimeError(f"{readme_path} lacks the champion block markers {BLOCK_BEGIN!r} .. {BLOCK_END!r}")
    new = text[:a + len(BLOCK_BEGIN)] + "\n" + champion_block(spec, ev) + "\n" + text[b:]
    readme_path.write_text(new)
    return True


def write_card(spec_path: Path = SPEC_PATH, card_path: Path = CARD_PATH,
               readme_path: Path | None = README_PATH) -> str:
    """PROCEDURE.md from the spec, and README's champion block from the spec
    and the champion's eval when both exist — the adoption's documents."""
    spec = json.loads(spec_path.read_text())
    text = render(spec)
    card_path.write_text(text)
    if readme_path is not None and Path(readme_path).exists():
        write_readme_block(spec, Path(readme_path))
    return text
