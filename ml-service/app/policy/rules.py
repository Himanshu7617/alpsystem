"""The state-to-action rules. Explicit, auditable, cited, and gate-aware.

BUILD.md Phase 8 step 4: *"a block diagram is not a research result; a rule that
depends on two states jointly is"*. Every rule below is one object with

* the **signal classes** it needs (``rung``) — a rule cannot fire for an arm
  whose state representation does not include its inputs, which is what makes
  the ladder a mechanism rather than a relabelling;
* the **latent states** it needs (``states``) — a rule needing a state that
  failed Phase 7's validation gate is **disabled**, and says so;
* a **citation** for why the rule exists at all;
* a **decision function** returning the action fields it wants to change plus
  the named quantities it read, which is what the explanation endpoint serves.

Two rules are disabled today. That is the point of writing them down:
``hint_when_uncertain_but_engaged`` and ``ease_off_when_uncertain_and_disengaged``
are the two joint-state rules this project set out to test, and Phase 7 rejected
both states they need. The rules stay in the table with ``disabled_by_gate``
set, because a rule that is silently deleted cannot be reported as a null.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from app.model import validation_gate

#: Action space, shared by every arm so the comparison is fair (Phase 8 step 2).
DIFFICULTIES = tuple(range(1, 11))
CONCEPT_MOVES = ("same_concept", "next_concept", "prerequisite_concept")
INTERVENTIONS = ("no_intervention", "hint", "worked_example", "break_suggestion")

#: The desirable-difficulty band. Not a free parameter: it is the success rate
#: adaptive-practice work converges on (Metcalfe & Kornell 2005's region of
#: proximal learning; Wilson et al. 2019's 85 % rule is stated for a different
#: error definition and gives ~0.70-0.75 here once guessing is included).
TARGET_SUCCESS = (0.70, 0.75)

#: How many attempts on one concept without progress counts as wheel-spinning.
#: Beck & Gong (2013) define wheel-spinning as 10 practice opportunities without
#: mastery; a 200-item closed loop over six concepts uses the shorter window
#: their follow-up work uses for within-session detection.
WHEEL_SPINNING_ATTEMPTS = 8

#: Consecutive incorrect responses before a worked example replaces practice.
WORKED_EXAMPLE_STREAK = 3


@dataclass(frozen=True)
class Rule:
    """One state-to-action rule."""

    key: str
    rung: str
    #: Latent states the rule conditions on. Empty means observables only.
    states: tuple[str, ...]
    citation: str
    rationale: str
    decide: Callable[[dict], dict | None] = field(repr=False, default=None)

    def applies_at(self, rung: str) -> bool:
        """Whether an arm at ``rung`` may run this rule."""
        from app.research.feature_catalogue import LEVELS

        return LEVELS.index(self.rung) <= LEVELS.index(rung)


def _recent(view: dict, window: int = 5) -> tuple[float | None, float]:
    """Recent accuracy and its within-window trend, from correctness alone."""
    outcomes = view["outcomes"][-window:]
    if not outcomes:
        return None, 0.0
    midpoint = (len(outcomes) + 1) // 2
    early, late = outcomes[:midpoint], outcomes[midpoint:]
    trend = (sum(late) / len(late) - sum(early) / len(early)) if late else 0.0
    return sum(outcomes) / len(outcomes), trend


# --------------------------------------------------------------------- rules

def _productive_confusion(view: dict) -> dict | None:
    accuracy, trend = _recent(view)
    if accuracy is None or len(view["outcomes"]) < 4:
        return None
    if accuracy <= 0.5 and trend > 0:
        return {"intervention": "no_intervention", "lock": True,
                "read": {"recent_accuracy": round(accuracy, 3),
                         "accuracy_trend": round(trend, 3)}}
    return None


def _wheel_spinning(view: dict) -> dict | None:
    attempts = view["concept_attempts"]
    accuracy, trend = _recent(view)
    if attempts < WHEEL_SPINNING_ATTEMPTS or accuracy is None:
        return None
    if accuracy <= 0.5 and trend <= 0:
        return {"concept_move": "prerequisite_concept",
                "read": {"attempts_on_concept": attempts,
                         "recent_accuracy": round(accuracy, 3),
                         "accuracy_trend": round(trend, 3)}}
    return None


def _worked_example(view: dict) -> dict | None:
    streak = view["incorrect_streak"]
    if streak >= WORKED_EXAMPLE_STREAK:
        return {"intervention": "worked_example",
                "read": {"incorrect_streak": streak}}
    return None


def _hint_on_slow_decision(view: dict) -> dict | None:
    latency = view["observables"].get("decision_latency")
    if latency is None or view["incorrect_streak"] < 1:
        return None
    # A long gap between the first interaction and the submission is the
    # observable form of "stuck on this item", and it is available at L1.
    if latency >= view["latency_threshold"]:
        return {"intervention": "hint",
                "read": {"decision_latency_s": round(latency, 2),
                         "threshold_s": round(view["latency_threshold"], 2),
                         "incorrect_streak": view["incorrect_streak"]}}
    return None


def _break_on_within_session_decay(view: dict) -> dict | None:
    slope = view["observables"].get("matched_difficulty_speed_slope")
    if slope is None or view["session_position"] < 8:
        return None
    if slope >= view["decay_threshold"]:
        return {"intervention": "break_suggestion",
                "read": {"matched_difficulty_speed_slope": round(slope, 4),
                         "threshold": round(view["decay_threshold"], 4),
                         "session_position": view["session_position"]}}
    return None


def _hint_when_uncertain_but_engaged(view: dict) -> dict | None:
    confidence = view["states"].get("confidence")
    engagement = view["states"].get("engagement")
    if confidence is None or engagement is None:
        return None
    if confidence < 0.4 <= engagement:
        return {"intervention": "hint",
                "read": {"confidence": round(confidence, 3),
                         "engagement": round(engagement, 3)}}
    return None


def _ease_off_when_uncertain_and_disengaged(view: dict) -> dict | None:
    confidence = view["states"].get("confidence")
    engagement = view["states"].get("engagement")
    if confidence is None or engagement is None:
        return None
    if confidence < 0.4 and engagement < 0.4:
        return {"difficulty_delta": -2,
                "read": {"confidence": round(confidence, 3),
                         "engagement": round(engagement, 3)}}
    return None


RULES: tuple[Rule, ...] = (
    Rule(
        key="target_desirable_difficulty_band",
        rung="L0", states=("knowledge",),
        citation="Metcalfe & Kornell (2005), region of proximal learning; "
                 "Wilson et al. (2019), the optimal-error rate for training",
        rationale="Difficulty is chosen so the expected success probability lands in "
                  f"{TARGET_SUCCESS}. This is the selection rule itself, shared by every "
                  "model arm, and it is where the knowledge estimate enters.",
        decide=None,  # applied by the policy, not as a modifier
    ),
    Rule(
        key="productive_confusion",
        rung="L0", states=(),
        citation="VanLehn et al. (2003), impasse-driven learning; "
                 "D'Mello et al. (2014), confusion can be productive",
        rationale="A learner who is struggling but improving is in the state help would "
                  "interrupt. The rule's output is deliberately *no* action.",
        decide=_productive_confusion,
    ),
    Rule(
        key="prerequisite_on_wheel_spinning",
        rung="L0", states=(),
        citation="Beck & Gong (2013), wheel-spinning; Käser et al. (2014), "
                 "prerequisite structure in student modelling",
        rationale="Repeated practice on one concept with a flat trend is the wheel-spinning "
                  "signature; the response is the prerequisite, not more of the same.",
        decide=_wheel_spinning,
    ),
    Rule(
        key="worked_example_on_repeated_failure",
        rung="L0", states=(),
        citation="Sweller & Cooper (1985), the worked-example effect; "
                 "Salden et al. (2010), adaptive fading",
        rationale="After a run of failures, an unsupported problem costs time and teaches "
                  "little; a worked example is the cheaper way through.",
        decide=_worked_example,
    ),
    Rule(
        key="hint_on_slow_decision",
        rung="L1", states=(),
        citation="Aleven et al. (2016), help-seeking and help avoidance; "
                 "Beck et al. (2008), response time as a help signal",
        rationale="A long decision latency after an incorrect run is the observable form of "
                  "being stuck. Needs timing, so an L0 arm cannot fire it — that asymmetry "
                  "is part of what the ladder measures.",
        decide=_hint_on_slow_decision,
    ),
    Rule(
        key="break_on_within_session_decay",
        rung="L3", states=(),
        citation="Wise & Kong (2005), response-time effort; "
                 "docs/preregistration.md §7 — the *observable*, after the fatigue "
                 "construct was dropped",
        rationale="Fatigue is not a validated state in this project, so the break rule "
                  "conditions on the within-session matched-difficulty speed slope directly. "
                  "It is an observable in L3, not a latent construct.",
        decide=_break_on_within_session_decay,
    ),
    Rule(
        key="hint_when_uncertain_but_engaged",
        rung="L1", states=("confidence", "engagement"),
        citation="Aleven et al. (2016); Baker et al. (2008), gaming the system",
        rationale="The joint-state rule this project set out to test: help a learner who is "
                  "unsure but still trying. Needs two states the validation gate rejected.",
        decide=_hint_when_uncertain_but_engaged,
    ),
    Rule(
        key="ease_off_when_uncertain_and_disengaged",
        rung="L1", states=("confidence", "engagement"),
        citation="Baker et al. (2008), help abuse; Wise & Kong (2005)",
        rationale="A hint offered to a disengaged learner is help-abuse bait, so the response "
                  "is a lower difficulty instead. Needs the same two rejected states.",
        decide=_ease_off_when_uncertain_and_disengaged,
    ),
)

RULES_BY_KEY = {rule.key: rule for rule in RULES}


@dataclass(frozen=True)
class RuleSet:
    """The rules one arm may actually run, and why the others are out."""

    rung: str
    gate: validation_gate.Gate
    active: tuple[Rule, ...]
    excluded: tuple[dict, ...]

    def apply(self, view: dict) -> tuple[dict, list[dict]]:
        """Run every active rule over ``view``.

        Returns the accumulated action modifiers and one explanation entry per
        rule that fired. Rules run in table order; a rule that sets ``lock``
        stops later rules from adding an intervention, which is how "do not
        interrupt productive confusion" beats "offer a hint".
        """
        modifiers: dict = {}
        fired: list[dict] = []
        locked = False
        for rule in self.active:
            if rule.decide is None:
                continue
            outcome = rule.decide(view)
            if outcome is None:
                continue
            if outcome.get("lock"):
                locked = True
            if "intervention" in outcome and locked and outcome["intervention"] != "no_intervention":
                continue
            for key in ("intervention", "concept_move", "difficulty_delta"):
                if key in outcome:
                    modifiers[key] = outcome[key]
            fired.append({"rule": rule.key, "citation": rule.citation,
                          "states_used": list(rule.states), "rung": rule.rung,
                          "read": outcome.get("read", {})})
        return modifiers, fired


def ruleset(rung: str, gate: validation_gate.Gate | None = None) -> RuleSet:
    """The rules an arm at ``rung`` may run, after the validation gate.

    A rule is excluded for exactly one of two reasons and the reason is kept:
    its signal class is above the arm's rung, or it needs a state the gate did
    not admit. The second kind is a **result** — see ``docs/rule-policy.md``.
    """
    gate = gate if gate is not None else validation_gate.load()
    active, excluded = [], []
    for rule in RULES:
        missing = [state for state in rule.states if not gate.is_admitted(state)]
        if missing:
            excluded.append({"rule": rule.key, "reason": "gate",
                             "states": missing,
                             "detail": {state: gate.reason(state) for state in missing}})
        elif not rule.applies_at(rung):
            excluded.append({"rule": rule.key, "reason": "rung",
                             "detail": f"needs {rule.rung}, arm sees {rung}"})
        else:
            active.append(rule)
    return RuleSet(rung=rung, gate=gate, active=tuple(active), excluded=tuple(excluded))
