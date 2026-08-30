"""Phase 8 guardrails.

Three of these exist because BUILD.md Phase 8 asks for them by name: the policy
module must not be able to reach the simulator's response function (Defect 1),
a gate-failed state must not change a policy's output (Phase 7's acceptance
criterion, mirrored here), and the termination criterion must be reachable.
The rest check that the online feature builder agrees with the batch one, since
a policy fed differently-computed features is not the model Study 2 validated.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.model import validation_gate
from app.policy import policies as arms
from app.policy import rules as rules_module
from app.research.closed_loop import (ClosedLoop, MASTERY_TARGET, observable_view,
                                      select_curriculum)
from app.policy.online_features import LearnerFeatures, item_response_stats
from app.research import build_features
from app.research import feature_catalogue as catalogue

POLICY_DIR = Path(__file__).resolve().parents[1] / "policy"
COHORT = 24
BUDGET = 40


@pytest.fixture(scope="module")
def rollout():
    """A short closed loop under a knowledge-only model arm."""
    environment = ClosedLoop("V0", 20260821, COHORT, budget=BUDGET)
    policy = arms.build("model_L0", COHORT, 20260821, budget=BUDGET)
    rows: list[dict | None] = [None] * COHORT
    attempts = []
    for _ in range(BUDGET):
        if environment.done:
            break
        actions = policy.act(environment.views(policy.rung, rows))
        outcomes = environment.step(actions)
        policy.observe(outcomes)
        rows = policy.last_rows
        attempts.append((actions, outcomes))
    return environment, policy, attempts


# ------------------------------------------------------- Defect 1: the oracle

def test_policy_package_never_imports_the_simulator():
    """The ML arm may not read the environment's response function.

    Grep-based on purpose: this is a guardrail against the specific defect that
    invalidated the previous results, not a proof of isolation.
    """
    import ast

    offenders = []
    for path in POLICY_DIR.glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                # A string literal naming a latent column would be a lookup
                # into the environment's ground truth by the back door.
                if node.value.startswith(("latent_", "probability_correct")):
                    offenders.append((path.name, node.value))
                continue
            else:
                continue
            offenders.extend((path.name, name) for name in names
                             if "simulator" in name or "closed_loop" in name)
    assert not offenders, f"policy code reaches into the environment: {offenders}"


def test_policy_never_sees_a_latent_column(rollout):
    _, policy, attempts = rollout
    for _, outcomes in attempts:
        for row in outcomes:
            if row is None:
                continue
            visible = observable_view(row, "L4")
            assert not [name for name in visible if name.startswith(("latent_", "probe_"))]
            assert "probability_correct" not in visible


# -------------------------------------------------- Phase 7's gate, mirrored

def test_rejected_states_are_absent_from_a_policy_view(rollout):
    _, policy, _ = rollout
    gate = validation_gate.load()
    for index in range(COHORT):
        states = policy.estimator.states(index)
        assert set(states) <= gate.admitted
        assert "engagement" not in states and "confidence" not in states


def test_a_gate_failed_state_cannot_change_the_action():
    """Flip a state to failed and assert the policy output does not move.

    The rules needing engagement and confidence are excluded by the live gate,
    so their inputs can be set to anything at all without changing a decision.
    Admitting them again brings the rules back — which is what makes this a
    gate test rather than an absence-of-code test.
    """
    live = validation_gate.load()
    ruleset = rules_module.ruleset("L4", live)
    # Chosen so no observable-only rule fires: the only thing that can move the
    # action here is a latent state, which is exactly what the gate controls.
    view = {"outcomes": [1, 1, 0, 1, 1], "concept_attempts": 2, "incorrect_streak": 1,
            "session_position": 3, "observables": {}, "latency_threshold": math.inf,
            "decay_threshold": math.inf}

    baseline, fired_baseline = ruleset.apply({**view, "states": {"knowledge": 0.5}})
    assert not fired_baseline and not baseline
    for confidence, engagement in ((0.01, 0.99), (0.99, 0.01), (0.2, 0.2)):
        modified, fired = ruleset.apply({**view, "states": {
            "knowledge": 0.5, "confidence": confidence, "engagement": engagement}})
        assert modified == baseline
        assert all("confidence" not in entry["states_used"] for entry in fired)

    admitted = validation_gate.Gate(admitted=frozenset({"knowledge", "confidence", "engagement"}),
                                    report=live.report)
    reopened = rules_module.ruleset("L4", admitted)
    changed, _ = reopened.apply({**view, "states": {
        "knowledge": 0.5, "confidence": 0.1, "engagement": 0.9}})
    assert changed != baseline, "the gate is not what is keeping the rule out"


def test_excluded_rules_say_why():
    excluded = {entry["rule"]: entry for entry in rules_module.ruleset("L4").excluded}
    assert "hint_when_uncertain_but_engaged" in excluded
    for entry in excluded.values():
        assert entry["reason"] in {"gate", "rung"}
        assert entry["detail"]


def test_explanations_never_cite_a_rejected_state(rollout):
    _, _, attempts = rollout
    gate = validation_gate.load()
    for actions, _ in attempts:
        for action in actions:
            if action is None:
                continue
            for entry in action["explanation"].get("rules_fired", []):
                assert set(entry["states_used"]) <= gate.admitted
            assert set(action["explanation"].get("states", {})) <= gate.admitted


# ------------------------------------------- the termination criterion works

def test_curriculum_is_prerequisite_closed_and_covered():
    environment = ClosedLoop("V0", 1, 1, budget=1)
    curriculum = set(environment.curriculum)
    from app.research.concept_graph import CONCEPTS

    for concept in CONCEPTS:
        if concept.concept_id in curriculum:
            assert set(concept.prerequisites) <= curriculum
    for concept in curriculum:
        assert len(environment.by_concept[concept]) >= 2


def test_mastery_is_reachable_and_counts_only_the_curriculum(rollout):
    environment, _, _ = rollout
    fractions = [environment.mastery_fraction(run) for run in environment.runs]
    assert max(fractions) > 0, "no learner mastered anything — the criterion is broken again"
    run = environment.runs[0]
    for concept in environment.curriculum:
        assert (environment.success_probability(run, concept) >= MASTERY_TARGET) == \
               environment.mastered(run, concept)


def test_a_capable_arm_masters_some_learners():
    """The acceptance check from BUILD.md: mastery rate is not zero."""
    environment = ClosedLoop("V0", 20260821, 120, budget=150)
    policy = arms.build("irt_cat_maxinfo", 120, 20260821, budget=150)
    rows: list[dict | None] = [None] * 120
    for _ in range(150):
        if environment.done:
            break
        actions = policy.act(environment.views(policy.rung, rows))
        rows = environment.step(actions)
        policy.observe(rows)
    results = environment.results()
    assert sum(record["mastered"] for record in results) > 0


# ------------------------------------------------------------ action space

def test_every_action_is_in_the_shared_space_with_a_propensity(rollout):
    _, _, attempts = rollout
    for actions, _ in attempts:
        for action in actions:
            if action is None:
                continue
            assert action["difficulty"] in rules_module.DIFFICULTIES
            assert action["concept_move"] in rules_module.CONCEPT_MOVES
            assert action["intervention"] in rules_module.INTERVENTIONS
            assert 0 < action["propensity"] <= 1


def test_propensity_matches_the_epsilon_greedy_rule():
    policy = arms.build("model_L0", 4, 7, budget=20)
    policy.epsilon = 0.1
    views = [{"index": index, "position": 0, "session_position": 0,
              "active_concept": policy.table and next(iter(policy.table)), "outcomes": [],
              "concept_attempts": 0, "incorrect_streak": 0, "observables": {},
              "curriculum": [next(iter(policy.table))], "median_b": {}}
             for index in range(4)]
    for action in policy.act(views):
        expected = (0.1 / arms.ACTION_SPACE + (0.9 if not action["explored"] else 0.0))
        assert action["propensity"] == pytest.approx(expected)


def test_random_arm_is_uniform_over_the_action_space():
    policy = arms.build("random", 2, 3, budget=20)
    view = {"index": 0, "position": 0, "session_position": 0, "active_concept": "x",
            "outcomes": [], "concept_attempts": 0, "incorrect_streak": 0,
            "observables": {}, "curriculum": ["x"], "median_b": {}}
    for action in policy.act([view, view]):
        assert action["propensity"] == pytest.approx(1 / arms.ACTION_SPACE)


# ---------------------------------------- the online features are the batch ones

def test_online_history_features_match_the_batch_build(rollout):
    """The online builder against ``build_features``' own vectorised helpers.

    Both are given the same attempts. If the closed loop's `prior_accuracy` is
    not the batch build's `prior_accuracy`, the encoder is being fed something
    it was never fitted on and every Phase 8 number is about a different model.
    """
    _, policy, attempts = rollout
    rows = [row for _, outcomes in attempts for row in outcomes if row is not None]
    assert rows

    stats = item_response_stats()
    online = {}
    builders = {}
    for row in rows:
        learner = row["learner_id"]
        builders.setdefault(learner, LearnerFeatures(stats))
        online.setdefault(learner, []).append(builders[learner].row(row))
    flat = [row for learner in online for row in online[learner]]

    frame = pd.DataFrame(flat)
    frame = frame.sort_values(["learner_id", "step"], kind="stable").reset_index(drop=True)
    frame["correct"] = frame["correct"].astype(float)
    batch_accuracy, batch_attempts = build_features._prior_mean(frame, ["learner_id"], "correct")
    skill_accuracy, skill_attempts = build_features._prior_mean(
        frame, ["learner_id", "concept_id"], "correct")

    assert np.allclose(frame["prior_attempts"], batch_attempts)
    assert np.allclose(frame["prior_accuracy"], batch_accuracy, equal_nan=True)
    assert np.allclose(frame["skill_prior_attempts"], skill_attempts)
    assert np.allclose(frame["skill_prior_accuracy"], skill_accuracy, equal_nan=True)

    slope = build_features._running_slope(
        frame["session_id"], frame["question_number"].astype(float),
        frame["log_rt_z_item"].astype(float))
    assert np.allclose(frame["matched_difficulty_speed_slope"], slope, equal_nan=True,
                       atol=1e-6)


def test_a_row_never_sees_its_own_outcome():
    """Truncating the stream leaves the earlier rows bit-identical."""
    environment = ClosedLoop("V0", 5, 4, budget=12)
    policy = arms.build("fixed_order", 4, 5, budget=12)
    rows: list[dict | None] = [None] * 4
    collected = []
    for _ in range(12):
        if environment.done:
            break
        rows = environment.step(policy.act(environment.views(policy.rung, rows)))
        policy.observe(rows)
        collected.extend(row for row in rows if row is not None)

    stats = item_response_stats()
    full, truncated = LearnerFeatures(stats), LearnerFeatures(stats)
    mine = [row for row in collected if row["learner_id"] == collected[0]["learner_id"]]
    complete = [full.row(row) for row in mine]
    partial = [truncated.row(row) for row in mine[:len(mine) // 2]]
    for early, late in zip(partial, complete):
        for name in ("prior_accuracy", "rt_drift", "matched_difficulty_speed_slope"):
            assert (early[name] == late[name]) or (math.isnan(early[name]) and math.isnan(late[name]))


def test_observable_view_respects_the_rung():
    row = {name: 1.0 for name in catalogue.names()}
    row.update({"latent_knowledge": 3.0, "probability_correct": 0.5})
    correctness_only = observable_view(row, "L0")
    assert "log_response_time" not in correctness_only
    assert "correct" in correctness_only
    assert "log_response_time" in observable_view(row, "L1")
    assert "max_deviation" in observable_view(row, "L4")


def test_interventions_are_recorded_and_bounded(rollout):
    environment, _, _ = rollout
    for record in environment.results():
        served = sum(record[f"n_{name}"] for name in rules_module.INTERVENTIONS)
        assert served == record["items_served"]


def test_selecting_a_curriculum_is_deterministic():
    environment = ClosedLoop("V0", 11, 1, budget=1)
    assert select_curriculum(environment.simulator.items) == environment.curriculum


# ------------------------------------------------------------- explanations

def test_explanation_narrative_names_only_validated_states():
    """The rendered sentence is the audit surface; it must not name a failed state."""
    from app.policy import explanations

    trace = {
        "id": "v0-model_L4-sim-00001-step007", "concept_id": "sql_basics",
        "served_difficulty": 4,
        "action": {"difficulty": 4, "concept_move": "same_concept",
                   "intervention": "hint", "propensity": 0.9},
        "explanation": {
            "states": {"knowledge": 0.62, "confidence": 0.31},
            "state_standard_error": {"knowledge": 0.02},
            "rules_fired": [{"rule": "hint_on_slow_decision", "rung": "L1",
                             "citation": "Aleven et al. (2016)",
                             "states_used": [], "read": {"decision_latency_s": 31.2}}],
            "rules_excluded": [{"rule": "hint_when_uncertain_but_engaged", "reason": "gate",
                                "states": ["confidence", "engagement"]}],
        },
    }
    narrative = explanations.render(trace)
    assert "knowledge 0.62" in narrative
    assert "confidence" not in narrative.split("Not available")[0]
    assert "hint_on_slow_decision" in narrative
    assert "validation gate" in narrative


# ------------------------------------------------------------ reproducibility

def test_cohort_does_not_depend_on_python_hash_seed():
    """The same seed must give the same cohort in every process.

    It did not, until ``ClosedLoop.__init__`` sorted the concept set it seeds
    each learner's per-concept knowledge from: every ``setdefault`` there draws
    from the simulator's RNG, so hash order silently decided which concept got
    which draw and a rerun of the same seed produced a different cohort.
    """
    import os
    import subprocess
    import sys

    script = (
        "from app.research.closed_loop import ClosedLoop;"
        "e = ClosedLoop('V0', 20260821, 12, budget=1);"
        "print(round(sum(sum(r.learner.knowledge.values()) for r in e.runs), 8))"
    )
    digests = set()
    for hash_seed in ("1", "2"):
        environment = {**os.environ, "PYTHONHASHSEED": hash_seed}
        output = subprocess.run([sys.executable, "-c", script], check=True,
                                capture_output=True, text=True, env=environment)
        digests.add(output.stdout.strip())
    assert len(digests) == 1, f"cohort changed with the hash seed: {digests}"
