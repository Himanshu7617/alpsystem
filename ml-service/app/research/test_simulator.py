"""Checks on Simulator v2's three non-negotiable properties.

Fast by design: everything below runs on a handful of learners, so `make test`
stays green without the archival downloads or a generated dataset.
"""
from __future__ import annotations

import numpy as np
import pytest

from app.research import simulator as simulator_module
from app.research.simulator import VARIANTS, Params, Simulator


def test_trajectory_features_come_from_the_production_extractor(monkeypatch):
    """BUILD.md Phase 4 acceptance: the simulator must not write motor features itself."""
    calls = []
    original = simulator_module.extract

    def spy(request):
        calls.append(request)
        return original(request)

    monkeypatch.setattr(simulator_module, "extract", spy)
    simulator = Simulator(Params(), seed=7)
    rows = list(simulator.run(simulator.learners(1)))

    assert calls, "extract() was never called — the simulator is emitting features on its own"
    assert len(calls) == len(rows)
    # The payload handed to the extractor carries a raw path, not a feature row.
    segments = [event for event in calls[0]["events"] if event["type"] == "CURSOR_SEGMENT"]
    assert segments and segments[0]["payload"]["n_samples"] >= 3
    assert rows[0]["path_ratio"] == segments[0]["payload"]["path_ratio"]


def test_outcome_only_mode_skips_the_extractor(monkeypatch):
    calls = []
    monkeypatch.setattr(simulator_module, "extract", lambda request: calls.append(request) or {"features": {}})
    simulator = Simulator(Params(), seed=7)
    rows = list(simulator.run(simulator.learners(2), observables=False))
    assert rows and not calls


def test_traits_are_drawn_independently_of_ability():
    """The Defect 3 fix: behaviour must carry variance ability cannot explain."""
    simulator = Simulator(Params(), seed=11)
    learners = simulator.learners(4000)
    ability = np.array([learner.ability for learner in learners])
    for trait in ("speed", "jitter", "reread", "move_speed", "indecision", "distraction"):
        values = np.array([getattr(learner, trait) for learner in learners], dtype=float)
        correlation = float(np.corrcoef(ability, values)[0, 1])
        assert abs(correlation) < 0.06, f"trait {trait} correlates with ability (r = {correlation:.3f})"


def test_generation_is_deterministic_given_the_seed():
    def first_rows(seed):
        simulator = Simulator(Params(), seed=seed)
        rows = simulator.run(simulator.learners(2))
        return [(row["item_id"], row["correct"], round(row["response_time_ms"], 3),
                 row["cursor_samples"]) for row, _ in zip(rows, range(40))]

    assert first_rows(3) == first_rows(3)
    assert first_rows(3) != first_rows(4)


def test_latent_state_is_evaluation_only_and_never_a_feature():
    simulator = Simulator(Params(), seed=5)
    row = next(iter(simulator.run(simulator.learners(1))))
    latents = [key for key in row if key.startswith("latent_")]
    assert set(latents) == {"latent_knowledge", "latent_knowledge_mean", "latent_engagement",
                            "latent_confidence", "latent_fatigue"}
    from app.features.extract import FEATURE_CLASSES

    assert not set(FEATURE_CLASSES) & set(latents)


@pytest.mark.parametrize("variant", sorted(VARIANTS))
def test_every_variant_changes_exactly_what_it_claims(variant):
    base = Params()
    changed = base.as_variant(variant)
    assert changed.variant == variant
    if variant == "V0":
        assert changed.to_dict() | {"variant": "V0"} == base.to_dict() | {"variant": "V0"}
    if variant == "V1":
        assert changed.discrimination_log_sd > 0
    if variant == "V2":
        assert changed.fatigue_accuracy_beta == 0 and changed.rt_beta_fatigue != base.rt_beta_fatigue
    if variant == "V3":
        assert changed.state_weight < base.state_weight < changed.trait_weight
    if variant == "V4":
        assert changed.learning_rate_drift_sd > 0


def test_probes_are_noisy_not_the_latent_value():
    simulator = Simulator(Params(probe_noise_sd=0.8), seed=13)
    rows = [row for row in simulator.run(simulator.learners(6)) if row["probe_confidence"] is not None]
    assert rows, "no confidence probe was ever shown"
    reported = np.array([row["probe_confidence"] for row in rows], dtype=float)
    latent = np.array([row["latent_confidence"] for row in rows], dtype=float)
    # Reported confidence tracks the latent but is not a function of it.
    assert 0.0 < abs(float(np.corrcoef(reported, latent)[0, 1])) < 0.95
    assert len(set(reported.tolist())) > 1
