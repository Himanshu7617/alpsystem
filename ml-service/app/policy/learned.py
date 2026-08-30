"""The learned arms: two linear contextual bandits and the trained PPO policy.

Phase 8's arms decide with a hand-written rule over a gated state. These three
learn the decision instead, from the same context, in the same action space —
which is RQ4's whole comparison, so nothing else may differ.

| arm | how it decides |
|---|---|
| ``bandit_linucb`` | disjoint LinUCB (Li et al. 2010) over the gated context |
| ``bandit_lin_ts`` | linear Thompson sampling (Agrawal & Goyal 2013) |
| ``rl_ppo`` | a `stable-baselines3` PPO policy trained in `app.rl.env` |

**The context is built here and only here**, so the bandit, the PPO
environment, the doubly-robust reward model and the on-policy evaluation all
condition on the same vector. :func:`context` reads gated states and the
observable view; a state the validation gate rejected is **absent from the
vector**, not zeroed, and ``test_study3`` asserts that by name.

Nothing here imports the simulator. The reward arrives through
:meth:`LinearBandit.learn`, which the driver calls with the environment's
number — an agent is told its reward, it does not compute it from a latent
variable it was never shown.
"""
from __future__ import annotations

import numpy as np

from app.policy import rules as rules_module
from app.policy.policies import Policy
from app.policy.state import StateEstimator

#: The action space, flattened. Index = ((difficulty-1) * moves + move) *
#: interventions + intervention, so every learned arm and the RL environment
#: agree on what action 47 means.
DIFFICULTIES = rules_module.DIFFICULTIES
CONCEPT_MOVES = rules_module.CONCEPT_MOVES
INTERVENTIONS = rules_module.INTERVENTIONS
N_ACTIONS = len(DIFFICULTIES) * len(CONCEPT_MOVES) * len(INTERVENTIONS)

#: Every state head Phase 7 trained. The vector is built from this list
#: **intersected with what the gate admitted**, so a rejected head can only ever
#: be missing — there is no code path that reaches one.
STATE_HEADS = ("knowledge", "engagement", "confidence")

#: Observable context, in the order the vector packs it.
OBSERVABLE_FEATURES = ("bias", "progress", "session_position", "concept_attempts",
                       "incorrect_streak", "recent_accuracy", "overall_accuracy",
                       "curriculum_position")


def action_index(action: dict) -> int:
    difficulty = int(np.clip(action["difficulty"], 1, len(DIFFICULTIES))) - 1
    move = CONCEPT_MOVES.index(action.get("concept_move", "same_concept"))
    intervention = INTERVENTIONS.index(action.get("intervention", "no_intervention"))
    return (difficulty * len(CONCEPT_MOVES) + move) * len(INTERVENTIONS) + intervention


def action_from_index(index: int) -> dict:
    index = int(index)
    intervention = index % len(INTERVENTIONS)
    rest = index // len(INTERVENTIONS)
    move = rest % len(CONCEPT_MOVES)
    difficulty = rest // len(CONCEPT_MOVES)
    return {"difficulty": int(DIFFICULTIES[difficulty]),
            "concept_move": CONCEPT_MOVES[move],
            "intervention": INTERVENTIONS[intervention]}


def feature_names(admitted: list[str]) -> list[str]:
    """The vector's columns. Only admitted states appear, ever."""
    names = list(OBSERVABLE_FEATURES)
    for state in admitted:
        names += [state, f"{state}_present", f"{state}_standard_error"]
    return names


def admitted_states(gate) -> list[str]:
    """The gated heads, in a fixed order. Rejected heads are not in this list."""
    return [head for head in STATE_HEADS if gate.is_admitted(head)]


def context(view: dict, states: dict, errors: dict, admitted: list[str]) -> np.ndarray:
    """The decision context: observables the arm's rung allows, plus gated states.

    A state that is admitted but not yet measured (the learner's first item)
    contributes its ``_present`` flag as 0 and its value as 0.5 — "no estimate
    yet" is a fact about the history, not a rejected construct, so unlike a
    gate rejection it is representable.
    """
    outcomes = view["outcomes"]
    recent = outcomes[-5:]
    values = [
        1.0,
        min(view["position"], 200) / 200.0,
        min(view["session_position"], 40) / 40.0,
        min(view["concept_attempts"], 40) / 40.0,
        min(view["incorrect_streak"], 5) / 5.0,
        float(np.mean(recent)) if recent else 0.5,
        float(np.mean(outcomes)) if outcomes else 0.5,
        view["curriculum"].index(view["active_concept"]) / max(1, len(view["curriculum"]) - 1),
    ]
    for state in admitted:
        estimate = states.get(state)
        values += [0.5 if estimate is None else float(estimate),
                   0.0 if estimate is None else 1.0,
                   float(errors.get(state) or 0.0)]
    return np.asarray(values, dtype="float64")


# ------------------------------------------------------------------- bandits

class LinearBandit(Policy):
    """Shared machinery: the gated context, per-action linear models, updates.

    Disjoint models — one ``d × d`` inverse covariance per action, updated by
    Sherman-Morrison so a decision costs one rank-1 update rather than a solve.

    `ponytail:` linear in the context, and disjoint across 120 actions, so it
    cannot share what it learns about difficulty 5 with difficulty 6 and it
    assumes reward is linear in the features. Both ceilings are reported in
    `docs/modeling.md`; a neural or action-featurised bandit is the upgrade, and
    only if the linear one is visibly underfitting.
    """

    rung = "L4"
    needs_observables = True
    #: Ridge prior on each action's covariance.
    RIDGE = 1.0

    def __init__(self, cohort: int, seed: int, rung: str = "L4", **kwargs):
        kwargs.setdefault("epsilon", 0.0)   # exploration is the algorithm's own job
        super().__init__(cohort, seed, **kwargs)
        self.rung = rung
        self.estimator = StateEstimator(rung, cohort, gate=self.gate, seed=seed)
        self.admitted = admitted_states(self.gate)
        self.names = feature_names(self.admitted)
        self.dimension = len(self.names)
        self.inverse = np.repeat((np.eye(self.dimension) / self.RIDGE)[None, :, :],
                                 N_ACTIONS, axis=0)
        self.right = np.zeros((N_ACTIONS, self.dimension))
        self.theta = np.zeros((N_ACTIONS, self.dimension))
        #: The (context, action) each learner is waiting to be paid for.
        self.pending: list[tuple[np.ndarray, int] | None] = [None] * cohort
        self.updates = 0

    # -- context -----------------------------------------------------------
    def observe(self, rows: list[dict | None]) -> None:
        self.last_rows = self.estimator.observe(rows)

    def context_for(self, view: dict) -> np.ndarray:
        index = view["index"]
        return context(view, self.estimator.states(index),
                       self.estimator.standard_error(index), self.admitted)

    # -- the algorithm -----------------------------------------------------
    def score(self, features: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def greedy(self, view: dict) -> tuple[dict, dict]:
        features = self.context_for(view)
        scores = self.score(features)
        chosen = int(np.argmax(scores))
        self.pending[view["index"]] = (features, chosen)
        action = action_from_index(chosen)
        return action, {"rules_fired": [], "action_index": chosen,
                        "score": round(float(scores[chosen]), 5),
                        "updates": self.updates, "note": self.key}

    def learn(self, rewards: list[float | None]) -> None:
        """Pay the arm for the decisions it just made. Sherman-Morrison per pull."""
        for index, reward in enumerate(rewards):
            waiting = self.pending[index]
            if reward is None or waiting is None:
                continue
            features, action = waiting
            inverse = self.inverse[action]
            projected = inverse @ features
            denominator = 1.0 + float(features @ projected)
            self.inverse[action] = inverse - np.outer(projected, projected) / denominator
            self.right[action] += reward * features
            self.theta[action] = self.inverse[action] @ self.right[action]
            self.pending[index] = None
            self.updates += 1


class LinUCBPolicy(LinearBandit):
    """Optimism in the face of uncertainty (Li et al. 2010, disjoint form)."""

    key = "bandit_linucb"
    #: Exploration width. 1.0 is the standard α for a bandit whose rewards are
    #: O(0.1); larger values spend the whole budget exploring 120 actions.
    ALPHA = 1.0

    def score(self, features: np.ndarray) -> np.ndarray:
        mean = self.theta @ features
        spread = np.einsum("i,aij,j->a", features, self.inverse, features)
        return mean + self.ALPHA * np.sqrt(np.maximum(spread, 0.0))


class LinTSPolicy(LinearBandit):
    """Linear Thompson sampling (Agrawal & Goyal 2013)."""

    key = "bandit_lin_ts"
    #: Posterior scale. The reward's own noise is ~0.03, so a unit-variance
    #: posterior would explore far past what the signal supports.
    SIGMA = 0.1

    def score(self, features: np.ndarray) -> np.ndarray:
        # One draw per action from N(theta_a, SIGMA^2 * A_a^-1), scored at x.
        # Sampling the scalar x·θ directly instead of the d-vector θ: the two
        # are the same distribution here and this is 120 normals, not 120
        # Cholesky factorisations.
        mean = self.theta @ features
        variance = np.einsum("i,aij,j->a", features, self.inverse, features)
        return mean + self.SIGMA * np.sqrt(np.maximum(variance, 0.0)) * self.rng.normal(
            size=N_ACTIONS)


# ----------------------------------------------------------------------- PPO

class PPOPolicy(LinearBandit):
    """A trained PPO policy, driven through the same arm interface.

    Inherits the context machinery and nothing else: it does not learn online,
    so :meth:`learn` is a no-op and the environment's reward never reaches it
    during evaluation.
    """

    key = "rl_ppo"

    def __init__(self, cohort: int, seed: int, model_path=None, **kwargs):
        super().__init__(cohort, seed, **kwargs)
        from pathlib import Path

        from stable_baselines3 import PPO

        path = Path(model_path) if model_path is not None else None
        if path is None or not path.exists():
            raise SystemExit(f"no trained PPO artifact at {path}. Run `make rl` first.")
        self.model = PPO.load(str(path), device="cpu")

    def act(self, views: list[dict | None]) -> list[dict | None]:
        """Batched: one `predict` for the whole cohort, not one per learner."""
        live = [view for view in views if view is not None]
        if not live:
            return [None] * len(views)
        batch = np.stack([self.context_for(view) for view in live]).astype("float32")
        chosen, _ = self.model.predict(batch, deterministic=True)
        decisions = iter(np.atleast_2d(chosen))
        actions: list[dict | None] = []
        for view in views:
            if view is None:
                actions.append(None)
                continue
            difficulty, move, intervention = (int(value) for value in next(decisions))
            action = {"difficulty": int(DIFFICULTIES[difficulty]),
                      "concept_move": CONCEPT_MOVES[move],
                      "intervention": INTERVENTIONS[intervention]}
            # The same ε-greedy wrapper every other arm wears. Offline policy
            # evaluation treats the target as ε-greedy, so the on-policy run it
            # is checked against has to explore at the same rate — a
            # deterministic ground truth would be measuring a different policy.
            explored = self.epsilon > 0 and self.rng.random() < self.epsilon
            if explored:
                action = action_from_index(int(self.rng.integers(N_ACTIONS)))
            propensity = self.epsilon / N_ACTIONS + (1 - self.epsilon) * float(not explored)
            actions.append({**action, "propensity": propensity, "policy": self.key,
                            "explored": explored,
                            "explanation": {"rules_fired": [],
                                            "note": "ppo, epsilon-greedy" if self.epsilon
                                            else "ppo, deterministic"}})
        return actions

    def learn(self, rewards: list[float | None]) -> None:
        return None


ARM_KEYS = ("bandit_linucb", "bandit_lin_ts", "rl_ppo")


def build(key: str, cohort: int, seed: int, model_path=None, gate=None, **kwargs) -> Policy:
    if key == "bandit_linucb":
        return LinUCBPolicy(cohort, seed, gate=gate, **kwargs)
    if key == "bandit_lin_ts":
        return LinTSPolicy(cohort, seed, gate=gate, **kwargs)
    if key == "rl_ppo":
        return PPOPolicy(cohort, seed, model_path=model_path, gate=gate, **kwargs)
    raise SystemExit(f"unknown learned arm {key!r}. Known: {', '.join(ARM_KEYS)}")


def demo() -> None:
    """Self-check: the action index round-trips and rejected states are absent."""
    from app.model import validation_gate

    for index in range(N_ACTIONS):
        assert action_index(action_from_index(index)) == index
    gate = validation_gate.load()
    admitted = admitted_states(gate)
    names = feature_names(admitted)
    for head in STATE_HEADS:
        if not gate.is_admitted(head):
            assert not any(name.startswith(head) for name in names), \
                f"{head} failed the gate and must not be a feature"
    view = {"index": 0, "position": 3, "session_position": 2, "active_concept": "a",
            "concept_attempts": 2, "outcomes": [1, 0, 1], "incorrect_streak": 1,
            "observables": {}, "curriculum": ["a", "b"], "median_b": {}}
    vector = context(view, {"knowledge": 0.6}, {"knowledge": 0.02}, admitted)
    assert len(vector) == len(names) and np.isfinite(vector).all()
    cold = context(view, {}, {}, admitted)
    if admitted:
        assert cold[len(OBSERVABLE_FEATURES) + 1] == 0.0, "unmeasured state flags itself"
    print(f"learned: ok ({N_ACTIONS} actions, {len(names)} features: {', '.join(names)})")


if __name__ == "__main__":
    demo()
