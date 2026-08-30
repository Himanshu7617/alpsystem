"""The closed loop as a `gymnasium` environment. One learner, one episode.

    from app.rl.env import ALPEnv
    env = ALPEnv(variant="V0", seed=20260821, budget=200)

The dynamics are **not re-derived here**: an episode is a one-learner
:class:`app.research.closed_loop.ClosedLoop`, so PPO trains against the same
generative model, the same curriculum, the same item bank and the same
instructional-action effects that every Phase 8 arm was scored on. A second copy
of the environment would be a second set of assumptions to keep in sync, and the
first divergence between them would be invisible.

* **Observation** — :func:`app.policy.learned.context`, the vector the bandits
  see: observables plus the states the Phase 7 gate admitted. A rejected head is
  absent from it, not zeroed.
* **Action** — ``MultiDiscrete([10, 3, 4])``, the shared action space, in the
  index order :func:`app.policy.learned.action_from_index` defines.
* **Reward** — :func:`app.rl.reward.reward`, whose weights are pre-registered in
  `docs/preregistration.md` §10.1.

A fresh learner is drawn every episode. The episode seed advances with the
episode counter, so training sees a cohort rather than one learner repeatedly,
and a given ``(seed, episode)`` is reproducible.
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from app.model import validation_gate
from app.policy import learned
from app.policy.state import StateEstimator
from app.research.closed_loop import BUDGET, ClosedLoop
from app.rl.reward import reward as step_reward


class ALPEnv(gym.Env):
    """One simulated learner being taught, as a gymnasium environment."""

    metadata = {"render_modes": []}

    def __init__(self, variant: str = "V0", seed: int = 20260821, budget: int = BUDGET,
                 rung: str = "L4", gate: validation_gate.Gate | None = None):
        super().__init__()
        import torch

        torch.set_num_threads(1)
        self.variant = variant
        self.base_seed = int(seed)
        self.budget = int(budget)
        self.rung = rung
        self.gate = gate if gate is not None else validation_gate.load()
        self.admitted = learned.admitted_states(self.gate)
        self.feature_names = learned.feature_names(self.admitted)
        # The encoder is loaded once and reset per episode: reloading it every
        # reset costs more than every other part of a step put together.
        self.estimator = StateEstimator(rung, 1, gate=self.gate, seed=self.base_seed)
        self.observation_space = spaces.Box(low=-10.0, high=10.0,
                                            shape=(len(self.feature_names),), dtype=np.float32)
        self.action_space = spaces.MultiDiscrete(
            [len(learned.DIFFICULTIES), len(learned.CONCEPT_MOVES), len(learned.INTERVENTIONS)])
        self.episode = 0
        self.loop: ClosedLoop | None = None
        self.rows: list[dict | None] = [None]

    # ------------------------------------------------------------- helpers

    @property
    def run(self):
        return self.loop.runs[0]

    def curriculum_success(self) -> float:
        """Mean probability of the median item across the taught curriculum.

        The environment's own response model — the reward signal, never an
        observation. See `app.rl.reward` on why that distinction is the whole
        honesty of this environment.
        """
        return float(np.mean([self.loop.success_probability(self.run, concept)
                              for concept in self.loop.curriculum]))

    def observation(self) -> np.ndarray:
        views = self.loop.views(self.rung, self.rows)
        view = views[0]
        if view is None:
            return np.zeros(len(self.feature_names), dtype=np.float32)
        vector = learned.context(view, self.estimator.states(0),
                                 self.estimator.standard_error(0), self.admitted)
        return np.clip(vector, -10.0, 10.0).astype(np.float32)

    # --------------------------------------------------------------- gym API

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.episode += 1
        episode_seed = self.base_seed + self.episode if seed is None else int(seed)
        self.loop = ClosedLoop(self.variant, episode_seed, 1, budget=self.budget)
        self.estimator.reset()
        self.rows = [None]
        return self.observation(), {"variant": self.variant, "seed": episode_seed}

    def step(self, action):
        if self.loop is None:
            raise RuntimeError("reset() before step()")
        difficulty, move, intervention = (int(value) for value in np.asarray(action).ravel())
        decision = {"difficulty": int(learned.DIFFICULTIES[difficulty]),
                    "concept_move": learned.CONCEPT_MOVES[move],
                    "intervention": learned.INTERVENTIONS[intervention],
                    "propensity": 1.0, "policy": "rl_ppo"}

        before = self.curriculum_success()
        seconds_before = self.run.on_task_seconds
        self.rows = self.loop.step([decision])
        self.estimator.observe(self.rows)
        after = self.curriculum_success()

        payoff = step_reward(after - before,
                             self.run.on_task_seconds - seconds_before,
                             self.run.incorrect_streak)
        mastered = self.run.mastered_at_items is not None
        terminated = bool(mastered)
        truncated = bool(self.run.finished and not mastered)
        info = {"mastery_delta": after - before, "mastered": mastered,
                "items": self.run.position, "curriculum_success": after}
        return self.observation(), float(payoff), terminated, truncated, info


def demo() -> None:
    """Self-check: the space contract holds and a rejected state never appears."""
    env = ALPEnv(variant="V0", seed=20260821, budget=12)
    observation, info = env.reset()
    assert env.observation_space.contains(observation), observation
    for head in learned.STATE_HEADS:
        if not env.gate.is_admitted(head):
            assert not any(name.startswith(head) for name in env.feature_names), \
                f"{head} was rejected by the gate and must be absent from the observation"
    total = 0.0
    for _ in range(12):
        observation, payoff, terminated, truncated, info = env.step(env.action_space.sample())
        assert env.observation_space.contains(observation)
        assert np.isfinite(payoff)
        total += payoff
        if terminated or truncated:
            break
    assert env.reset()[1]["seed"] != info.get("seed"), "each episode draws a new learner"
    print(f"env: ok ({len(env.feature_names)} features, "
          f"{int(np.prod(env.action_space.nvec))} actions, reward {total:+.4f} over the rollout)")


if __name__ == "__main__":
    demo()
