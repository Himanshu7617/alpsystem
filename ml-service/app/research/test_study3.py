"""Phase 9 acceptance checks — the ones BUILD.md names, and nothing decorative.

Four things have to be true for a Study 3 number to mean anything:

1. the reward weights in the code are the ones `docs/preregistration.md` §10.1
   registered — a weight that drifts after a bad run is how an RL result gets
   manufactured;
2. a state the Phase 7 gate rejected is **absent** from every learned arm's
   observation, bandit and PPO alike;
3. the estimators are arithmetically right on a case whose answer is known;
4. a killed PPO run resumes from its checkpoint instead of restarting.
"""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from app.model import validation_gate
from app.policy import learned
from app.research import ope
from app.rl import reward as reward_module

ROOT = Path(__file__).resolve().parents[3]
ML_DIR = ROOT / "ml-service"
PREREGISTRATION = ROOT / "docs" / "preregistration.md"


# ------------------------------------------------------- the reward contract

def test_reward_weights_match_the_preregistration():
    """§10.1's table is the source of truth; the code may not quietly disagree."""
    text = PREREGISTRATION.read_text(encoding="utf-8")
    section = text[text.index("### 10.1"):text.index("### 10.2")]
    registered = {
        "MASTERY_SCALE": float(re.search(r"MASTERY_SCALE = ([0-9.]+)", section).group(1)),
        "LAMBDA_TIME": float(re.search(r"LAMBDA_TIME = ([0-9.]+)", section).group(1)),
        "MU_FRUSTRATION": float(re.search(r"MU_FRUSTRATION = ([0-9.]+)", section).group(1)),
    }
    assert registered["MASTERY_SCALE"] == reward_module.MASTERY_SCALE
    assert registered["LAMBDA_TIME"] == reward_module.LAMBDA_TIME
    assert registered["MU_FRUSTRATION"] == reward_module.MU_FRUSTRATION
    # And the formula in the document is the formula in the code.
    assert "100.0 · Δp(curriculum success) − 0.002 · minutes" in section


def test_reward_terms_have_the_registered_signs():
    reward_module.demo()


# ------------------------------------------------------------------ the gate

def test_rejected_states_are_absent_from_the_observation_vector():
    """Not zeroed. A rejected head must have no column at all."""
    gate = validation_gate.load()
    admitted = learned.admitted_states(gate)
    names = learned.feature_names(admitted)
    rejected = [head for head in learned.STATE_HEADS if not gate.is_admitted(head)]
    assert rejected, "this test is vacuous if the gate admitted everything"
    for head in rejected:
        assert not any(name.startswith(head) for name in names), \
            f"{head}: {gate.reason(head)} — it must not be a feature"
    # The vector is built from the admitted list, so a state offered to
    # `context()` that the gate rejected cannot widen it either.
    view = {"index": 0, "position": 1, "session_position": 1, "active_concept": "a",
            "concept_attempts": 1, "outcomes": [1], "incorrect_streak": 0,
            "observables": {}, "curriculum": ["a", "b"], "median_b": {}}
    smuggled = {head: 0.9 for head in learned.STATE_HEADS}
    assert len(learned.context(view, smuggled, smuggled, admitted)) == len(names)


def test_the_environment_observation_matches_the_bandit_context():
    """PPO and the bandits must condition on the same vector, or RQ4 is unfair."""
    from app.rl.env import ALPEnv

    environment = ALPEnv(variant="V0", seed=20260821, budget=4)
    assert environment.feature_names == learned.feature_names(
        learned.admitted_states(validation_gate.load()))
    observation, _ = environment.reset()
    assert environment.observation_space.contains(observation)


# ------------------------------------------------------------- the estimators

def synthetic_log(rows: int = 4000, seed: int = 7) -> tuple[pd.DataFrame, list[str], np.ndarray]:
    """A log whose true policy value is known by construction.

    Actions are drawn uniformly, so every propensity is 1/120 and importance
    weights are bounded; the reward depends on the action only through a fixed
    per-action offset, so the value of any target policy is computable in closed
    form.
    """
    rng = np.random.default_rng(seed)
    names = ["bias", "progress"]
    contexts = np.column_stack([np.ones(rows), rng.random(rows)])
    actions = rng.integers(0, learned.N_ACTIONS, rows)
    offsets = rng.normal(0, 0.05, learned.N_ACTIONS)
    rewards = offsets[actions] + 0.2 * contexts[:, 1] + rng.normal(0, 0.01, rows)
    frame = pd.DataFrame({"action_index": actions, "propensity": 1.0 / learned.N_ACTIONS,
                          "reward": rewards})
    for index, name in enumerate(names):
        frame[f"ctx_{name}"] = contexts[:, index]
    return frame, names, offsets


def test_ips_recovers_a_known_value():
    """A target that copies the logging policy must be estimated at its own mean."""
    frame, names, _ = synthetic_log()
    chosen = frame["action_index"].to_numpy()
    weights = ope.importance_weights(frame, chosen, epsilon=1.0)
    # ε = 1 is the uniform target: every weight is exactly 1 under a uniform log.
    assert np.allclose(weights, 1.0)
    value = float(np.mean(weights * frame["reward"].to_numpy()))
    assert value == pytest.approx(frame["reward"].mean(), abs=1e-12)


def test_effective_sample_size_is_bounded_by_the_decisions():
    frame, names, offsets = synthetic_log()
    chosen = np.full(len(frame), int(np.argmax(offsets)))
    weights = np.minimum(ope.importance_weights(frame, chosen), ope.WEIGHT_CLIP)
    ess = ope.effective_sample_size(weights)
    assert 0 < ess <= len(frame)
    # A deterministic target on a 120-action log leaves an ESS far below n; that
    # is the fact f09-01 exists to show, so it is asserted rather than assumed.
    assert ess < len(frame) / 2


def test_doubly_robust_beats_ips_on_a_model_it_can_fit():
    """DR's whole claim is variance reduction when r̂ is any good."""
    frame, names, offsets = synthetic_log()
    best = int(np.argmax(offsets))
    chosen = np.full(len(frame), best)
    model = ope.fit_reward_model(frame, names)
    estimate = ope.estimate(frame, chosen, names, model, seed=11, draws=200)
    truth = float(offsets[best] + 0.2 * frame["ctx_progress"].mean())
    epsilon = ope.TARGET_EPSILON
    truth = (1 - epsilon) * truth + epsilon * float(
        (offsets.mean() + 0.2 * frame["ctx_progress"].mean()))
    assert abs(estimate["dr"] - truth) < abs(estimate["ips"] - truth)


# ---------------------------------------------------------------- resumption

def _train_rl():
    from app.research import train_rl

    return train_rl


@pytest.mark.skipif(os.environ.get("ALP_SKIP_SLOW_TESTS") == "1",
                    reason="ALP_SKIP_SLOW_TESTS=1")
def test_a_killed_ppo_run_resumes_from_its_checkpoint(tmp_path, monkeypatch):
    """Kill it deliberately, restart it, and require that it picked up the rest.

    BUILD.md Phase 9 acceptance asks for exactly this, and asks for it by
    killing a real run rather than by unit-testing the path arithmetic. It
    trains into `tmp_path`: against the published `artifacts/models/rl` a
    finished run's 50,000-step checkpoint already covers the 8,192 asked for
    here, so nothing would train, nothing would be written, and the test would
    fail for a reason that has nothing to do with resumption.
    """
    train_rl = _train_rl()
    checkpoints = tmp_path / "checkpoints"
    monkeypatch.setattr(train_rl, "RL_DIR", tmp_path)
    monkeypatch.setattr(train_rl, "CHECKPOINT_DIR", checkpoints)
    monkeypatch.setattr(train_rl, "POLICY_PATH", tmp_path / "policy.zip")
    existing: set[str] = set()

    command = [sys.executable, "-m", "app.research.train_rl", "--train-only",
               "--timesteps", "8192", "--budget", "40", "--checkpoint-interval", "1",
               "--profile", "smoke"]
    environment = {**os.environ, "PYTHONPATH": str(ML_DIR), "OMP_NUM_THREADS": "1",
                   "ALP_RL_DIR": str(tmp_path)}
    process = subprocess.Popen(command, cwd=ML_DIR, env=environment,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    deadline = time.time() + 240
    fresh = set()
    while time.time() < deadline:
        if checkpoints.exists():
            fresh = {path.name for path in checkpoints.glob("ppo-*-steps.zip")} - existing
        if fresh:
            break
        if process.poll() is not None:
            break
        time.sleep(1.0)
    process.send_signal(signal.SIGKILL)
    process.wait(timeout=30)
    assert fresh, "no checkpoint was written before the kill — nothing to resume from"

    resumed = train_rl.train(timesteps=8192, seed=20260821, variant="V0", budget=40,
                             checkpoint_interval=1, resume=True)
    assert resumed["resumed_from"] > 0, "the restart ignored the checkpoint"
    assert resumed["trained_this_run"] < 8192, "the restart retrained work already paid for"
