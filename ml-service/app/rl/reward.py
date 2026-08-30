"""The Study 3 reward. Pre-registered constants, defined once.

`docs/preregistration.md` §10.1 fixes these weights **before** any bandit or PPO
result exists, and ``app.research.test_study3`` asserts the document and this
module still agree. Reward shaping is where an RL result gets manufactured; a
weight that moves after a disappointing run is a deviation and is reported as
one.

    reward = MASTERY_SCALE · Δp(curriculum mastered)
             − LAMBDA_TIME · minutes on task
             − MU_FRUSTRATION · 1[incorrect streak ≥ FRUSTRATION_STREAK]

**The mastery term is a simulation-only signal.** It is the environment's own
response model, the same quantity Phase 8 scores arms with — a learner's mean
probability of answering the median item of each curriculum concept. It reaches
the agent as a *reward*, never as an observation: the observation vector is
built by :func:`app.policy.learned.context`, which admits only gated states and
observables. A real deployment has no such signal, so Phase 10's reward has to
be an observable proxy; that is a limitation of this study, recorded here rather
than discovered later.

Scale, measured rather than chosen. Under `rule_improved` on V0 (20 learners,
60 items) a step moves the curriculum's mean success probability by 0.00008 and
costs 103 s. Multiplying the mastery term by 100 puts it in probability points,
where that step is worth 0.008; λ = 0.002 per minute makes its time cost 0.0034,
about 40 % of the learning it bought, and a 300 s break costs 0.010 — roughly
one item's worth. The learning term stays the larger one **on purpose**: a time
penalty that dominates is a reward for ending the episode, not for teaching.
"""
from __future__ import annotations

#: Probability points. Δp of ~0.00025 per item becomes ~0.025 of reward.
MASTERY_SCALE = 100.0

#: Cost of one minute of on-task time, in the same probability points.
LAMBDA_TIME = 0.002

#: Cost of one attempt made while already on a losing streak. Half an item's
#: typical learning, so frustration is discouraged without being catastrophic.
MU_FRUSTRATION = 0.005

#: Consecutive incorrect answers that count as frustration. Three is the
#: threshold `rules.py` already uses for `recent_struggle`; a second definition
#: of "struggling" in the same repository would be one too many.
FRUSTRATION_STREAK = 3

#: Written into `reward_spec.json` beside every trained artifact.
SPEC = {
    "formula": ("MASTERY_SCALE * delta_curriculum_success_probability "
                "- LAMBDA_TIME * minutes_on_task "
                "- MU_FRUSTRATION * frustrated"),
    "mastery_scale": MASTERY_SCALE,
    "lambda_time": LAMBDA_TIME,
    "mu_frustration": MU_FRUSTRATION,
    "frustration_streak": FRUSTRATION_STREAK,
    "preregistration": "docs/preregistration.md §10.1",
    "mastery_term_is_latent": True,
    "note": ("the mastery term is the simulator's response model and exists only in "
             "simulation; it is a reward, never an observation"),
}


def reward(mastery_delta: float, seconds: float, incorrect_streak: int) -> float:
    """One decision's reward. ``mastery_delta`` is in probability, not logits."""
    frustrated = float(incorrect_streak >= FRUSTRATION_STREAK)
    return (MASTERY_SCALE * float(mastery_delta)
            - LAMBDA_TIME * (float(seconds) / 60.0)
            - MU_FRUSTRATION * frustrated)


def demo() -> None:
    """Self-check: each term has the sign and the size §10.1 claims."""
    assert reward(0.0, 0.0, 0) == 0.0
    assert reward(0.00025, 0.0, 0) > 0, "learning is rewarded"
    assert reward(0.0, 60.0, 0) == -LAMBDA_TIME, "a minute costs lambda"
    assert reward(0.0, 0.0, FRUSTRATION_STREAK) == -MU_FRUSTRATION
    assert reward(0.0, 0.0, FRUSTRATION_STREAK - 1) == 0.0, "below the streak is free"
    typical_item = reward(0.00008, 103.0, 0)
    assert 0.0 < typical_item < 0.008, f"a typical item nets a small gain, got {typical_item}"
    assert abs(reward(0.0, 300.0, 0)) == 0.01, "a 300 s break costs one item of learning"
    assert reward(0.00008, 103.0, 0) > reward(0.00008, 103.0, FRUSTRATION_STREAK)
    print("reward: ok")


if __name__ == "__main__":
    demo()
