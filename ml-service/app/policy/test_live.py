"""Phase 10 acceptance checks for the serving layer.

Three properties, and the platform is wrong if any of them fails:

1. a state Phase 7's gate rejected never leaves the service — not in the state
   response, not in the explanation the learner is shown;
2. every served decision carries a non-zero propensity, because Study 3 cannot
   be re-run on a log that lacks one; and
3. a slot that is evicted and replayed produces the estimate it had before,
   which is what makes the service restartable without a session store.
"""
from __future__ import annotations

import pytest

from app.policy import live
from app.schemas.serving import Attempt, View

CURRICULUM = ["arrays_and_lists", "stacks_and_queues", "hash_tables"]


def attempt(session: str, index: int, correct: bool = True) -> dict:
    return Attempt(
        session_id=session, item_id="ds-e-1", concept_id="stacks_and_queues",
        correct=correct, difficulty_score=2, response_time_ms=9000 + 100 * index,
        item_b=-1.15, timestamp=1000.0 * (index + 1), position=index,
        question_number=index + 1,
        features={"n_options": 4, "decision_latency": 3.2, "reading_time": 2.0},
    ).model_dump()


def view(position: int = 1) -> dict:
    return View(position=position, session_position=position,
                active_concept="stacks_and_queues", concept_attempts=position,
                outcomes=[1] * position, incorrect_streak=0,
                curriculum=CURRICULUM).model_dump()


@pytest.fixture(scope="module")
def engine():
    return live.Engine(key="rule_improved", slots=2, seed=20260821)


def test_a_rejected_state_is_absent_from_the_estimate(engine):
    index = engine.sync("gate-session", [attempt("gate-session", 0)])
    result = engine.states(index)
    assert "knowledge" in result["states"], "the admitted state should be served"
    for rejected in result["gate"]["rejected"]:
        assert rejected not in result["states"]
        assert rejected not in result["standard_error"]


def test_a_rejected_state_is_absent_from_the_explanation(engine):
    engine.sync("explain-session", [attempt("explain-session", 0)])
    action = engine.decide("explain-session", view())
    explanation = action["explanation"]
    rejected = set(engine.states(engine.acquire("explain-session"))["gate"]["rejected"])
    # A rejected state may be *named* as excluded — that is the audit trail the
    # phase asks for — but it may never appear as a value the decision read.
    for rule in explanation.get("rules_fired", []):
        assert not (rejected & set(rule)), rule
    for key, value in explanation.items():
        if key == "excluded_by_gate":
            continue
        assert not any(name in str(key) for name in rejected), (key, value)


def test_every_decision_carries_a_usable_propensity(engine):
    engine.sync("propensity-session", [attempt("propensity-session", 0)])
    for step in range(1, 6):
        action = engine.decide("propensity-session", view(step))
        assert 0.0 < action["propensity"] <= 1.0
        assert 0 <= action["action_index"] < action["action_space"]


def test_an_evicted_slot_replays_to_the_same_estimate(engine):
    history = [attempt("replayed", index) for index in range(3)]
    first = engine.states(engine.sync("replayed", history))["states"]

    # Two more sessions with a two-slot pool: "replayed" is evicted.
    engine.sync("filler-a", [attempt("filler-a", 0)])
    engine.sync("filler-b", [attempt("filler-b", 0)])
    assert "replayed" not in engine.slot_of, "the fixture's pool should have evicted it"

    replays = engine.replays
    again = engine.states(engine.sync("replayed", history))["states"]
    assert engine.replays == replays + 1
    # The recurrence is deterministic, so the replay recovers the same hidden
    # state; the head is an MC-dropout average over 20 draws from a generator
    # the whole cohort shares, so the mean it reports moves by O(sd/sqrt(20)).
    # Anything larger than that would mean the history was replayed wrongly.
    standard_error = engine.states(engine.slot_of["replayed"])["standard_error"]
    for head, value in first.items():
        assert abs(again[head] - value) < max(3 * standard_error[head], 1e-3)


def test_a_shorter_history_restarts_the_slot(engine):
    engine.sync("shrink", [attempt("shrink", index) for index in range(3)])
    index = engine.sync("shrink", [attempt("shrink", 0)])
    assert engine.observed["shrink"] == 1
    assert engine.states(index)["steps"] == 1
