"""Decision explanations: the stored trace, and the sentence it renders to.

BUILD.md Phase 8 step 6 asks every decision to emit a structured explanation
decomposed over *named, validated* state deltas — the example being "difficulty
reduced: matched-difficulty accuracy fell 18 % over the last 20 minutes (fatigue
head, validated, se 0.04) while engagement remained high".

This project cannot emit that sentence, and the reason is the result: there is
no fatigue head (the construct was dropped in Phase 5) and the engagement head
failed the validation gate. So the renderer names what each rule actually read —
a gated state with its standard error, or an observable with the threshold it
was compared against — and refuses to name anything else.

`ponytail:` traces are read from a JSON file written by
``app.research.evaluate_policies``, not from a database. A closed-loop run has
no persistent decision store; Phase 10 puts one behind the same function.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from app.model import validation_gate

ROOT = Path(__file__).resolve().parents[3]
TRACE_PATH = ROOT / "artifacts" / "evaluation" / "decision-explanations.json"


@lru_cache(maxsize=1)
def _traces(mtime: float) -> dict[str, dict]:
    payload = json.loads(TRACE_PATH.read_text(encoding="utf-8"))
    return {trace["id"]: trace for trace in payload.get("decisions", [])}


def load() -> dict[str, dict]:
    if not TRACE_PATH.exists():
        return {}
    return _traces(TRACE_PATH.stat().st_mtime)


def find(decision_id: str) -> dict | None:
    return load().get(decision_id)


def render(trace: dict, gate: validation_gate.Gate | None = None) -> str:
    """A sentence per fired rule, naming only what the gate admits."""
    gate = gate if gate is not None else validation_gate.load()
    explanation = trace.get("explanation", {})
    states = {name: value for name, value in explanation.get("states", {}).items()
              if gate.is_admitted(name)}
    errors = explanation.get("state_standard_error", {})

    parts = []
    action = trace.get("action", {})
    parts.append(
        f"Served difficulty {trace.get('served_difficulty', action.get('difficulty'))} "
        f"on {trace.get('concept_id')} ({action.get('concept_move', 'same_concept')}, "
        f"{action.get('intervention', 'no_intervention')}).")
    if states:
        described = ", ".join(
            f"{name} {value:.2f}" + (f" (se {errors[name]:.2f}, validated)"
                                     if name in errors else " (validated)")
            for name, value in states.items())
        parts.append(f"State read: {described}.")
    else:
        parts.append("State read: none — no attempt had been scored yet.")

    for entry in explanation.get("rules_fired", []):
        read = ", ".join(f"{key} = {value}" for key, value in entry.get("read", {}).items())
        parts.append(f"Rule {entry['rule']} fired on {read or 'no measured input'} "
                     f"[{entry['rung']}; {entry['citation']}].")
    if not explanation.get("rules_fired"):
        parts.append("No interaction rule fired; the difficulty came from the target "
                     "success band alone.")

    excluded = explanation.get("rules_excluded", [])
    by_gate = [entry for entry in excluded if entry.get("reason") == "gate"]
    if by_gate:
        names = ", ".join(entry["rule"] for entry in by_gate)
        parts.append(f"Not available to this decision: {names} — the states they need failed "
                     "the Phase 7 validation gate and are withheld from every policy.")
    return " ".join(parts)
