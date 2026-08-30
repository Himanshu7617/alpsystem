"""The policy arms. Same action space, same budget, different state.

Every arm answers one question — *which item next, and with what support* — and
emits the same three-part action plus the probability with which it chose it.
The propensity is not decoration: Study 3 (Phase 9) cannot be added to logs that
did not record one, and it cannot be backfilled.

| arm | state it sees |
|---|---|
| ``random`` | none |
| ``fixed_order`` | none |
| ``mastery_threshold_bkt`` | BKT posterior over correctness |
| ``irt_cat_maxinfo`` | Rasch θ from correctness and item difficulty |
| ``rule_legacy`` | the deployed engine's knowledge variable |
| ``rule_improved`` | the deployed engine plus timing, gate-restricted |
| ``model_L0`` … ``model_L4`` | Phase 7's encoder at that rung, gated |

**The `model_L0` vs `model_L4` contrast is the paper's headline.** Same
selection rule, same action space, same target success band, same rule table —
only the rung the knowledge estimate was trained at differs, and which rules
that rung can supply the inputs for.

Nothing here imports the simulator.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from app.model import validation_gate
from app.policy import rules as rules_module
from app.policy.state import StateEstimator

ROOT = Path(__file__).resolve().parents[3]
BANK_PATH = ROOT / "data" / "items" / "item-bank-v1.json"
ITEM_PARAMS_PATH = ROOT / "artifacts" / "datasets" / "item-parameters-v1.csv"
BKT_PATH = ROOT / "artifacts" / "models" / "bkt-params.json"

#: Size of the shared action space: 10 difficulties × 3 concept moves ×
#: 4 interventions. An ε-greedy arm's exploration probability is spread over it.
ACTION_SPACE = (len(rules_module.DIFFICULTIES) * len(rules_module.CONCEPT_MOVES)
                * len(rules_module.INTERVENTIONS))

#: Exploration rate. Every adaptive arm explores at the same rate so the
#: comparison is not confounded by one arm being noisier, and so Phase 9 has
#: support for every action in every log. `random` is already uniform and
#: `fixed_order` is a static curriculum by definition; both keep ε = 0.
EPSILON = 0.1

#: A four-option item is guessable at 1/4 by construction of the interface.
#: This is content metadata, not a simulator parameter.
GUESS = 0.25

#: Target success probability: the midpoint of the desirable-difficulty band.
TARGET_SUCCESS = float(np.mean(rules_module.TARGET_SUCCESS))

#: BKT degeneracy bounds (Baker, Corbett & Aleven 2008): a fitted guess above
#: 0.3 or slip above 0.1 makes an incorrect answer raise the mastery posterior.
SLIP_MAX, GUESS_MAX = 0.1, 0.3

#: Standard error at which a knowledge estimate is shrunk halfway toward the
#: concept's median difficulty. Phase 7 measured a mean MC-dropout SE of 0.023
#: for the knowledge head, so this is roughly "twice as uncertain as usual".
SE_HALF_WEIGHT = 0.05


def item_table() -> dict[str, list[dict]]:
    """Concept → its items, with calibrated difficulty. Content metadata."""
    import csv

    bank = json.loads(BANK_PATH.read_text(encoding="utf-8"))["items"]
    with ITEM_PARAMS_PATH.open(encoding="utf-8") as handle:
        parameters = {row["item_id"]: row for row in csv.DictReader(handle)}
    table: dict[str, list[dict]] = {}
    for item in bank:
        row = parameters[item["item_id"]]
        table.setdefault(item["concept_id"], []).append(
            {"item_id": item["item_id"], "b": float(row["b"]),
             "difficulty": int(row["difficulty_bin"])})
    for items in table.values():
        items.sort(key=lambda entry: entry["b"])
    return table


def logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


# ----------------------------------------------------------------------- BKT

def fit_bkt(sequences: list[list[int]], seed: int = 20260821) -> dict:
    """Maximum-likelihood BKT parameters from correctness sequences.

    The four classic parameters (Corbett & Anderson 1994): prior knowledge,
    transition, slip and guess. pyBKT is in the dependency set but is broken
    against the pinned scikit-learn, and a direct likelihood maximisation over
    four bounded parameters is smaller than the workaround would be.

    Slip and guess are bounded at 0.1 and 0.3 — Baker, Corbett & Aleven's (2008)
    degeneracy bounds. Unbounded, the likelihood walks guess to 0.5, where a
    wrong answer raises the posterior and "knowing" stops meaning knowing.

    `ponytail:` one parameter set pooled over skills, not per skill — a
    per-skill fit needs per-skill archival volume this bank does not have.
    """
    from scipy.optimize import minimize

    def negative_log_likelihood(theta):
        prior, transition, slip, guess = 1 / (1 + np.exp(-np.asarray(theta)))
        slip, guess = slip * SLIP_MAX, guess * GUESS_MAX
        total = 0.0
        for sequence in sequences:
            belief = prior
            for outcome in sequence:
                likelihood = (belief * (1 - slip) + (1 - belief) * guess if outcome
                              else belief * slip + (1 - belief) * (1 - guess))
                total -= math.log(max(likelihood, 1e-12))
                posterior = ((belief * (1 - slip) / likelihood) if outcome
                             else (belief * slip / likelihood))
                belief = posterior + (1 - posterior) * transition
        return total

    start = np.array([0.0, -2.0, -1.5, -1.0])
    fit = minimize(negative_log_likelihood, start, method="Nelder-Mead",
                   options={"maxiter": 800, "xatol": 1e-3, "fatol": 1e-2})
    prior, transition, slip, guess = 1 / (1 + np.exp(-fit.x))
    return {"prior": float(prior), "transition": float(transition),
            "slip": float(slip * SLIP_MAX), "guess": float(guess * GUESS_MAX),
            "negative_log_likelihood": float(fit.fun),
            "sequences": len(sequences), "seed": seed}


def bkt_parameters(source: str = "assistments_2012", learners: int = 400,
                   refresh: bool = False) -> dict:
    """BKT parameters fitted on archival training learners, cached."""
    if BKT_PATH.exists() and not refresh:
        cached = json.loads(BKT_PATH.read_text(encoding="utf-8"))
        if cached.get("source") == source:
            return cached

    import pandas as pd

    frame = pd.read_parquet(ROOT / "data" / "processed" / f"features-{source}.parquet",
                            columns=["learner_id", "skill_id", "correct", "split"])
    frame = frame[frame["split"] != "test"]
    keep = frame["learner_id"].drop_duplicates().head(learners)
    frame = frame[frame["learner_id"].isin(keep)]
    sequences = [group["correct"].astype(int).tolist()[:100]
                 for _, group in frame.groupby(["learner_id", "skill_id"], observed=True)
                 if len(group) >= 3]
    fitted = {"source": source, **fit_bkt(sequences)}
    BKT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BKT_PATH.write_text(json.dumps(fitted, indent=2) + "\n", encoding="utf-8")
    return fitted


# -------------------------------------------------------------------- arms

class Policy:
    """Base arm: exploration, propensity bookkeeping and the action contract."""

    key = "base"
    rung = "L0"
    needs_observables = False

    def __init__(self, cohort: int, seed: int, epsilon: float = EPSILON,
                 gate: validation_gate.Gate | None = None):
        self.cohort = cohort
        self.epsilon = epsilon
        self.rng = np.random.default_rng(seed + 4241)
        self.gate = gate if gate is not None else validation_gate.load()
        self.table = item_table()
        #: The rows the next decision may look at. Model arms replace these
        #: with the enriched rows their online feature builder produced, so a
        #: rule reading `matched_difficulty_speed_slope` sees the same number
        #: the encoder did.
        self.last_rows: list[dict | None] = [None] * cohort

    # -- subclass hooks ----------------------------------------------------
    def observe(self, rows: list[dict | None]) -> None:
        """Learn from the attempts just made. Correctness-only arms use this."""
        self.last_rows = rows

    def greedy(self, view: dict) -> tuple[dict, dict]:
        """The action this arm wants, plus its explanation."""
        raise NotImplementedError

    # -- the contract ------------------------------------------------------
    def act(self, views: list[dict | None]) -> list[dict | None]:
        actions: list[dict | None] = []
        for view in views:
            if view is None:
                actions.append(None)
                continue
            action, explanation = self.greedy(view)
            action = {"difficulty": int(np.clip(action["difficulty"], 1, 10)),
                      "concept_move": action.get("concept_move", "same_concept"),
                      "intervention": action.get("intervention", "no_intervention")}
            explored = False
            if self.epsilon > 0 and self.rng.random() < self.epsilon:
                action = {
                    "difficulty": int(self.rng.integers(1, 11)),
                    "concept_move": str(self.rng.choice(rules_module.CONCEPT_MOVES)),
                    "intervention": str(self.rng.choice(rules_module.INTERVENTIONS)),
                }
                explored = True
            greedy_chosen = not explored
            propensity = (self.epsilon / ACTION_SPACE
                          + (1 - self.epsilon) * float(greedy_chosen))
            actions.append({**action, "propensity": propensity, "policy": self.key,
                            "explored": explored, "explanation": explanation})
        return actions

    # -- shared helpers ----------------------------------------------------
    def difficulty_for_b(self, concept: str, target_b: float) -> int:
        """The difficulty bin of the item in ``concept`` closest to ``target_b``."""
        items = self.table[concept]
        best = min(items, key=lambda entry: abs(entry["b"] - target_b))
        return best["difficulty"]

    def median_b(self, concept: str) -> float:
        return float(np.median([entry["b"] for entry in self.table[concept]]))


class RandomPolicy(Policy):
    """Floor. Uniform over the whole action space, so every log has support."""

    key = "random"

    def __init__(self, *args, **kwargs):
        kwargs["epsilon"] = 0.0
        super().__init__(*args, **kwargs)

    def greedy(self, view: dict) -> tuple[dict, dict]:
        return ({"difficulty": int(self.rng.integers(1, 11)),
                 "concept_move": str(self.rng.choice(rules_module.CONCEPT_MOVES)),
                 "intervention": str(self.rng.choice(rules_module.INTERVENTIONS))},
                {"rules_fired": [], "note": "uniform random"})

    def act(self, views):
        actions = super().act(views)
        for action in actions:
            if action is not None:
                action["propensity"] = 1.0 / ACTION_SPACE
        return actions


class FixedOrderPolicy(Policy):
    """Floor. A static curriculum: difficulty ramps, concepts advance on a clock."""

    key = "fixed_order"

    def __init__(self, *args, budget: int = 200, **kwargs):
        kwargs["epsilon"] = 0.0
        super().__init__(*args, **kwargs)
        self.budget = budget

    def greedy(self, view: dict) -> tuple[dict, dict]:
        per_concept = max(1, self.budget // len(view["curriculum"]))
        move = ("next_concept" if view["position"] and view["position"] % per_concept == 0
                else "same_concept")
        return ({"difficulty": min(10, 1 + view["position"] // 3), "concept_move": move},
                {"rules_fired": [], "note": "static ramp"})


class BKTPolicy(Policy):
    """The deployed real-world standard: mastery learning on a BKT posterior."""

    key = "mastery_threshold_bkt"
    #: Corbett & Anderson's (1994) mastery criterion.
    MASTERY = 0.95

    def __init__(self, cohort: int, seed: int, **kwargs):
        super().__init__(cohort, seed, **kwargs)
        self.params = bkt_parameters()
        self.belief: list[dict[str, float]] = [{} for _ in range(cohort)]

    def observe(self, rows: list[dict | None]) -> None:
        self.last_rows = rows
        prior = self.params["prior"]
        slip, guess, transition = (self.params["slip"], self.params["guess"],
                                   self.params["transition"])
        for index, row in enumerate(rows):
            if row is None:
                continue
            concept = row["concept_id"]
            belief = self.belief[index].get(concept, prior)
            if row["correct"]:
                likelihood = belief * (1 - slip) + (1 - belief) * guess
                posterior = belief * (1 - slip) / max(likelihood, 1e-9)
            else:
                likelihood = belief * slip + (1 - belief) * (1 - guess)
                posterior = belief * slip / max(likelihood, 1e-9)
            self.belief[index][concept] = posterior + (1 - posterior) * transition

    def greedy(self, view: dict) -> tuple[dict, dict]:
        concept = view["active_concept"]
        belief = self.belief[view["index"]].get(concept, self.params["prior"])
        move = "next_concept" if belief >= self.MASTERY else "same_concept"
        difficulty = 3 if belief < 0.4 else 5 if belief < 0.7 else 7 if belief < self.MASTERY else 9
        return ({"difficulty": difficulty, "concept_move": move},
                {"rules_fired": [], "bkt_posterior": round(belief, 4),
                 "note": "mastery learning at p(L) >= 0.95"})


class IRTPolicy(Policy):
    """The principled psychometric rule: adaptive testing at maximum information."""

    key = "irt_cat_maxinfo"

    def __init__(self, cohort: int, seed: int, **kwargs):
        super().__init__(cohort, seed, **kwargs)
        self.grid = np.linspace(-4, 4, 81)
        self.prior = -0.5 * self.grid ** 2          # standard normal, in log space
        self.loglik = [self.prior.copy() for _ in range(cohort)]
        self.by_concept = [dict() for _ in range(cohort)]

    def observe(self, rows: list[dict | None]) -> None:
        self.last_rows = rows
        for index, row in enumerate(rows):
            if row is None:
                continue
            probability = GUESS + (1 - GUESS) / (1 + np.exp(-(self.grid - row["item_b"])))
            update = np.log(probability if row["correct"] else 1 - probability)
            self.loglik[index] = self.loglik[index] + update

    def theta(self, index: int) -> float:
        weights = np.exp(self.loglik[index] - self.loglik[index].max())
        return float((weights * self.grid).sum() / weights.sum())

    def greedy(self, view: dict) -> tuple[dict, dict]:
        concept = view["active_concept"]
        estimate = self.theta(view["index"])
        # Fisher information for a 1PL peaks at b = θ; with a guessing floor the
        # optimum shifts slightly upward, which is the standard CAT correction.
        target_b = estimate + 0.4
        mastered = GUESS + (1 - GUESS) / (1 + math.exp(-(estimate - self.median_b(concept))))
        move = "next_concept" if mastered >= 0.80 else "same_concept"
        return ({"difficulty": self.difficulty_for_b(concept, target_b), "concept_move": move},
                {"rules_fired": [], "theta": round(estimate, 3),
                 "note": "maximum-information item selection"})


class RuleLegacyPolicy(Policy):
    """The existing naive engine: correctness in, three difficulty bands out.

    A direct port of ``updateKnowledge`` and the knowledge bands in
    ``backend/src/utils/ruleEngine.js``, so the baseline is the code that is
    actually deployed rather than a charitable reconstruction of it.
    """

    key = "rule_legacy"

    def __init__(self, cohort: int, seed: int, **kwargs):
        kwargs.setdefault("epsilon", EPSILON)
        super().__init__(cohort, seed, **kwargs)
        self.knowledge = [0.5] * cohort

    @staticmethod
    def _band(difficulty: int) -> str:
        return "easy" if difficulty <= 3 else "medium" if difficulty <= 7 else "hard"

    def observe(self, rows: list[dict | None]) -> None:
        self.last_rows = rows
        for index, row in enumerate(rows):
            if row is None:
                continue
            band = self._band(int(row["difficulty_score"]))
            if row["correct"]:
                delta = {"hard": 0.18, "medium": 0.12, "easy": 0.08}[band]
            else:
                delta = {"easy": -0.15, "medium": -0.10, "hard": -0.05}[band]
            self.knowledge[index] = float(np.clip(self.knowledge[index] + delta, 0.0, 1.0))

    def greedy(self, view: dict) -> tuple[dict, dict]:
        knowledge = self.knowledge[view["index"]]
        difficulty = 1 if knowledge < 0.4 else 5 if knowledge < 0.75 else 9
        move = "next_concept" if knowledge >= 0.75 else "same_concept"
        return ({"difficulty": difficulty, "concept_move": move},
                {"rules_fired": [], "knowledge_variable": round(knowledge, 3),
                 "note": "rule-v1.2 knowledge bands, correctness only"})


class RuleImprovedPolicy(RuleLegacyPolicy):
    """The 5-variable engine, restricted to what the validation gate admits.

    ``buildDifficultyDecision`` adjusts its band by cognitive load, fatigue,
    engagement, recent accuracy and response-time trend. Three of those five are
    latent constructs Phase 7 rejected or dropped, so this arm runs the two that
    survive — both computed from observables — and records the rest as excluded.
    That is the honest version of "the existing engine, restricted to gated
    states"; running it unrestricted would condition a policy on constructs the
    project has just reported as invalid.
    """

    key = "rule_improved"
    rung = "L1"
    needs_observables = True

    EXCLUDED_TERMS = {
        "high_cognitive_load": "cognitive load is not a gated state in this project",
        "high_fatigue": "fatigue construct dropped (preregistration §7)",
        "low_engagement": "engagement head failed C3 (Wise & Kong negative criterion)",
        "sustained_mastery_confidence_term": "confidence head failed C1 and C4",
    }

    def greedy(self, view: dict) -> tuple[dict, dict]:
        knowledge = self.knowledge[view["index"]]
        level = 1 if knowledge < 0.4 else 2 if knowledge < 0.75 else 3
        outcomes = view["outcomes"][-5:]
        accuracy = sum(outcomes) / len(outcomes) if outcomes else None
        reasons = []
        if accuracy is not None and len(outcomes) >= 3 and accuracy <= 0.4:
            level -= 1
            reasons.append("recent_struggle")
        ratio = view["observables"].get("response_time_vs_estimate")
        if ratio is not None and len(outcomes) >= 3 and ratio > 1.25:
            level -= 1
            reasons.append("slowing_response_time")
        if accuracy is not None and knowledge >= 0.4 and accuracy >= 0.5:
            level += 1
            reasons.append("sustained_mastery")
        level = int(np.clip(level, 1, 3))
        difficulty = {1: 1, 2: 5, 3: 9}[level]
        move = "prerequisite_concept" if "recent_struggle" in reasons else (
            "next_concept" if knowledge >= 0.75 else "same_concept")
        return ({"difficulty": difficulty, "concept_move": move},
                {"rules_fired": [{"rule": reason, "citation": "rule-v1.2 (deployed engine)"}
                                 for reason in reasons],
                 "knowledge_variable": round(knowledge, 3),
                 "excluded_by_gate": self.EXCLUDED_TERMS,
                 "note": "deployed 5-variable engine, gate-restricted to 2"})


class ModelPolicy(Policy):
    """The ladder arms. One rung's encoder, the gate, and the rule table.

    The knowledge head is calibrated to *P(correct on the next item)*, so the
    selection rule inverts it: with the recent item difficulty as the reference
    point, ``θ̂ = b̄ + logit(p̂)`` and the item served is the one whose difficulty
    puts the predicted success probability in the desirable-difficulty band.

    `ponytail:` the inversion assumes a 1PL link with no guessing floor, which
    biases θ̂ downward at low ability. It is monotone, identical across rungs and
    therefore cannot create the L0-versus-L4 contrast this arm exists to
    measure; a 3PL inversion is the upgrade if the band itself ever matters.
    """

    def __init__(self, cohort: int, seed: int, rung: str = "L4", window: int = 5,
                 thresholds: dict | None = None, **kwargs):
        super().__init__(cohort, seed, **kwargs)
        self.rung = rung
        self.key = f"model_{rung}"
        self.needs_observables = rung in ("L2", "L3", "L4")
        self.estimator = StateEstimator(rung, cohort, gate=self.gate, seed=seed)
        self.rules = rules_module.ruleset(rung, self.gate)
        self.window = window
        self.recent_b: list[list[float]] = [[] for _ in range(cohort)]
        self.thresholds = thresholds or {}

    def observe(self, rows: list[dict | None]) -> None:
        self.last_rows = self.estimator.observe(rows)
        for index, row in enumerate(rows):
            if row is None:
                continue
            self.recent_b[index] = (self.recent_b[index] + [float(row["item_b"])])[-self.window:]

    def greedy(self, view: dict) -> tuple[dict, dict]:
        index = view["index"]
        concept = view["active_concept"]
        states = self.estimator.states(index)
        errors = self.estimator.standard_error(index)
        knowledge = states.get("knowledge")
        reference = (float(np.mean(self.recent_b[index])) if self.recent_b[index]
                     else self.median_b(concept))

        if knowledge is None:
            # Cold start: no attempt yet, so no estimate. The median item of the
            # concept is the honest default, not a guessed state value.
            target_b = self.median_b(concept)
            uncertainty = None
        else:
            theta = reference + logit(knowledge)
            target_b = theta - logit(TARGET_SUCCESS)
            uncertainty = errors.get("knowledge")
            if uncertainty is not None:
                # An unreliable estimate is pulled toward the concept's median
                # item: the policy is allowed to see that it does not know.
                weight = uncertainty / (uncertainty + SE_HALF_WEIGHT)
                target_b = (1 - weight) * target_b + weight * self.median_b(concept)

        difficulty = self.difficulty_for_b(concept, target_b)
        action = {"difficulty": difficulty, "concept_move": "same_concept",
                  "intervention": "no_intervention"}

        rule_view = dict(view)
        rule_view["states"] = states
        rule_view["latency_threshold"] = self.thresholds.get("decision_latency", math.inf)
        rule_view["decay_threshold"] = self.thresholds.get("matched_difficulty_speed_slope",
                                                           math.inf)
        modifiers, fired = self.rules.apply(rule_view)
        if "difficulty_delta" in modifiers:
            action["difficulty"] += modifiers["difficulty_delta"]
        action["concept_move"] = modifiers.get("concept_move", action["concept_move"])
        action["intervention"] = modifiers.get("intervention", action["intervention"])

        explanation = {
            "rules_fired": fired,
            "rules_excluded": list(self.rules.excluded),
            "states": {name: round(value, 4) for name, value in states.items()},
            "state_standard_error": {name: round(value, 4) for name, value in errors.items()},
            "reference_difficulty": round(reference, 3),
            "target_difficulty_b": round(target_b, 3),
            "target_success": TARGET_SUCCESS,
            "rung": self.rung,
        }
        return action, explanation


#: Every arm Phase 8 runs. `bandit_lin_ts` and `rl_ppo` are Phase 9's.
LADDER_RUNGS = ("L0", "L1", "L3", "L4")
ARM_KEYS = ("random", "fixed_order", "mastery_threshold_bkt", "irt_cat_maxinfo",
            "rule_legacy", "rule_improved") + tuple(f"model_{rung}" for rung in LADDER_RUNGS)


def build(key: str, cohort: int, seed: int, budget: int = 200,
          thresholds: dict | None = None, gate: validation_gate.Gate | None = None) -> Policy:
    if key.startswith("model_"):
        return ModelPolicy(cohort, seed, rung=key.removeprefix("model_"),
                           thresholds=thresholds, gate=gate)
    if key == "random":
        return RandomPolicy(cohort, seed, gate=gate)
    if key == "fixed_order":
        return FixedOrderPolicy(cohort, seed, budget=budget, gate=gate)
    if key == "mastery_threshold_bkt":
        return BKTPolicy(cohort, seed, gate=gate)
    if key == "irt_cat_maxinfo":
        return IRTPolicy(cohort, seed, gate=gate)
    if key == "rule_legacy":
        return RuleLegacyPolicy(cohort, seed, gate=gate)
    if key == "rule_improved":
        return RuleImprovedPolicy(cohort, seed, gate=gate)
    raise SystemExit(f"unknown arm {key!r}. Known: {', '.join(ARM_KEYS)}")
