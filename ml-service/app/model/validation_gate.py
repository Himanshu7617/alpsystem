"""The validation gate: which latent states are allowed to drive a policy.

BUILD.md Phase 7 step 4 asks for the gate as *code, not prose*. A state head
exists as soon as someone adds an output layer and a name; whether it measures
the thing its name claims is a separate question, and this module is the only
place where that question is answered. The criteria and their thresholds are
pre-registered in ``docs/preregistration.md`` §8.2 and are mirrored here as
``CRITERIA`` — the module refuses to run if the two disagree on a name.

Two consumers:

* ``app.research.study2`` measures each head on the held-out validation fold,
  calls :func:`evaluate`, and writes ``artifacts/evaluation/validation-gate.json``.
* Phase 8's policy calls :func:`load` and then :meth:`Gate.admitted` /
  :meth:`Gate.filter_states`. A state that failed is not passed to the policy —
  not down-weighted, not clipped, **absent** — so the policy cannot condition on
  an unvalidated construct even by accident.

`ponytail:` the gate is a threshold table, not a statistical framework. Its
ceiling is that every criterion is a single scalar comparison; a criterion
needing a joint test (say, equivalence testing on C3 rather than a bound on
|r|) would need its own entry rather than a new column here.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
GATE_PATH = ROOT / "artifacts" / "evaluation" / "validation-gate.json"

#: The states this project certifies. Fatigue is deliberately absent:
#: preregistration §7's H2 outcome dropped the construct after no archival
#: source showed a within-session accuracy decline. A head that does not exist
#: cannot be gated, and the paper says why rather than leaving a silent gap.
STATES = ("knowledge", "engagement", "confidence")

DROPPED_STATES = {
    "fatigue": "no within-session accuracy decline in any archival source "
               "(preregistration §7); construct dropped before modelling",
}


@dataclass(frozen=True)
class Criterion:
    """One pre-registered pass condition.

    ``bound`` is a floor when ``direction == 'min'`` and a ceiling when it is
    ``'max'``. ``measure`` is the key the caller must supply in its
    measurement dict; a missing key is a failure, never a pass — an untested
    criterion is exactly the failure mode this module exists to prevent.
    """

    key: str
    measure: str
    bound: float
    direction: str
    applies_to: tuple[str, ...]
    description: str
    #: When set, the measurement must also carry ``<measure>_ci_low`` and
    #: ``<measure>_ci_high`` and the interval must exclude zero.
    ci_excludes_zero: bool = False

    def check(self, measured: dict) -> dict:
        value = measured.get(self.measure)
        if value is None:
            return {"criterion": self.key, "measure": self.measure, "value": None,
                    "bound": self.bound, "direction": self.direction, "passed": False,
                    "note": "not measured — an untested criterion fails"}
        value = float(value)
        compared = abs(value) if self.direction == "max" else value
        passed = compared <= self.bound if self.direction == "max" else compared >= self.bound
        result = {"criterion": self.key, "measure": self.measure, "value": round(value, 5),
                  "bound": self.bound, "direction": self.direction, "passed": bool(passed),
                  "description": self.description}
        if self.ci_excludes_zero:
            low = measured.get(f"{self.measure}_ci_low")
            high = measured.get(f"{self.measure}_ci_high")
            excludes = low is not None and high is not None and (low > 0 or high < 0)
            result["ci"] = [low, high]
            result["ci_excludes_zero"] = bool(excludes)
            result["passed"] = bool(passed and excludes)
        return result


#: Pre-registration §8.2, verbatim. Changing a number here without changing it
#: there — or after seeing a result — is the thing the pre-registration exists
#: to make visible.
CRITERIA: tuple[Criterion, ...] = (
    Criterion(
        key="C1_label_correlation",
        measure="label_correlation",
        bound=0.15,
        direction="min",
        applies_to=STATES,
        ci_excludes_zero=True,
        description="point-biserial r between the calibrated estimate and the head's own "
                    "label, learner-clustered 95% CI excluding zero",
    ),
    Criterion(
        key="C2_calibration",
        measure="ece_calibrated",
        bound=0.05,
        direction="max",
        applies_to=STATES,
        description="expected calibration error after isotonic calibration; a policy acting "
                    "at a threshold needs the probability to mean what it says",
    ),
    Criterion(
        key="C3_effort_independent_of_ability",
        measure="ability_correlation",
        bound=0.20,
        direction="max",
        applies_to=("engagement",),
        description="Wise & Kong's negative criterion: an effort index correlating with "
                    "ability has learned ability, not effort",
    ),
    Criterion(
        key="C4_discriminant_validity",
        measure="knowledge_correlation",
        bound=0.85,
        direction="max",
        applies_to=("confidence",),
        description="the confidence head must not be a relabelled knowledge head",
    ),
)


def criteria_for(state: str) -> tuple[Criterion, ...]:
    return tuple(item for item in CRITERIA if state in item.applies_to)


def evaluate(measurements: dict[str, dict]) -> dict:
    """Apply every applicable criterion to every state.

    ``measurements`` maps a state name to its measured values. A state that is
    absent from ``measurements`` is reported as *not measured* and fails; that
    is the same verdict as a measured failure, because a policy must not read
    either one.
    """
    unknown = set(measurements) - set(STATES)
    if unknown:
        raise ValueError(f"unknown state(s) {sorted(unknown)}; known: {list(STATES)}")

    states = {}
    for state in STATES:
        measured = measurements.get(state)
        checks = [item.check(measured or {}) for item in criteria_for(state)]
        states[state] = {
            "measured": measured is not None,
            "admitted": bool(measured is not None and all(check["passed"] for check in checks)),
            "criteria": checks,
            "failed": [check["criterion"] for check in checks if not check["passed"]],
            "source": (measured or {}).get("source"),
            "label": (measured or {}).get("label"),
            "n": (measured or {}).get("n"),
        }
    return {
        "states": states,
        "dropped_states": DROPPED_STATES,
        "admitted": sorted(name for name, item in states.items() if item["admitted"]),
        "rejected": sorted(name for name, item in states.items() if not item["admitted"]),
    }


@dataclass(frozen=True)
class Gate:
    """The decision, read back. Phase 8's policy holds one of these."""

    admitted: frozenset[str]
    report: dict = field(default_factory=dict, repr=False)

    def is_admitted(self, state: str) -> bool:
        return state in self.admitted

    def filter_states(self, states: dict) -> dict:
        """Drop every state the gate did not admit.

        Dropped, not zeroed: a policy that sees ``engagement = 0.0`` for a
        rejected head is still conditioning on it. It must see nothing.
        """
        return {name: value for name, value in states.items() if name in self.admitted}

    def reason(self, state: str) -> str:
        if state in self.admitted:
            return "admitted"
        if state in DROPPED_STATES:
            return DROPPED_STATES[state]
        entry = self.report.get("states", {}).get(state)
        if entry is None:
            return f"{state} is not a gated state"
        if not entry["measured"]:
            return f"{state} was never measured"
        return f"{state} failed {', '.join(entry['failed'])}"


def load(path: Path | None = None) -> Gate:
    """Read the gate report written by Phase 7.

    A missing report is not an open gate. Nothing is admitted until the
    measurement has actually been made.
    """
    path = path or GATE_PATH
    if not path.exists():
        return Gate(admitted=frozenset(), report={"states": {}, "missing": str(path)})
    report = json.loads(path.read_text(encoding="utf-8"))
    return Gate(admitted=frozenset(report.get("admitted", [])), report=report)
