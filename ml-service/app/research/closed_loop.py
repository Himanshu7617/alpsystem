"""The closed loop: a cohort of simulated learners, one policy, one variant.

This module lives in ``app.research`` and not in ``app.policy`` on purpose. It
imports the simulator; the policy package must never be able to. The
environment owns the learner, the policy owns the decision, and they exchange
exactly two things — an action in the shared action space, and an attempt record.
The policy never sees a latent variable, an outcome probability, or the
simulator's response function. That separation is the fix for Defect 1, and
``test_policies.py`` greps ``app/policy`` for a simulator import to keep it.

**Lockstep, not learner-by-learner.** Every learner in a cohort takes step *t*
together, because the model arms run a GRU and a 1,000-learner batched step
costs the same wall clock as a single-learner one.

**Termination.** A learner has mastered a concept when the calibrated response
model puts their probability of answering that concept's median item correctly
at or above :data:`MASTERY_TARGET`, and the run ends when every concept in the
fixed curriculum is mastered. Two things this fixes, both named in BUILD.md
Phase 8 step 1: mastery is judged over the concepts the learner is actually
taught (a fixed six-concept curriculum, not all thirty-six), and the roadmap
concentrates practice on the active concept rather than round-robining.

**Instructional actions extend the generative model.** ``hint``,
``worked_example`` and ``break_suggestion`` change the attempt: the first two
add logits of support and cost time, the third resets within-session fatigue and
costs more time. Their magnitudes are *not* fitted — no archival source in this
project records a hint or a worked example — so they are pre-registered
constants, identical for every arm, and ``docs/preregistration.md`` §9 says so.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from app.policy.online_features import observable_view  # noqa: F401  (re-exported)
from app.research.concept_graph import CONCEPTS
from app.research.simulator import Params, Simulator

#: Probability of a correct response on a concept's median item that counts as
#: mastery of that concept. Above the 0.75 top of the desirable-difficulty band
#: and below the guess-plus-slip ceiling, so it is reachable and not trivially
#: satisfied by guessing.
MASTERY_TARGET = 0.80

#: How many concepts every learner is taught. Six is the largest
#: prerequisite-closed set this item bank covers with enough items per concept
#: to vary difficulty; `select_curriculum` derives the set, it is not typed in.
CURRICULUM_SIZE = 6

#: Item budget. A learner who has not mastered the curriculum by here is
#: right-censored and counted as unmastered, never dropped.
BUDGET = 200

#: Instructional action effects (pre-registration §9.3). `ponytail:` fixed
#: constants, not fitted — nothing in the archival data records an intervention.
#: Upgrade path is a hint/worked-example effect fitted from a real deployment,
#: which is Phase 10's data, not Phase 8's.
HINT_SUPPORT = 1.0          # logits of support a hint adds to the response model
HINT_GAIN_SCALE = 0.7       # learning from a hinted success is worth less
HINT_SECONDS = 15.0
WORKED_SUPPORT = 2.0
WORKED_GAIN_SCALE = 1.0
WORKED_SECONDS = 45.0
BREAK_SECONDS = 300.0

#: Gap between items inside a session, drawn per attempt. Same distribution the
#: Phase 4 generator uses between logged attempts.
INTER_ITEM_MEAN_MS = 20_000


#: A curriculum concept must be one the difficulty action can actually act on.
#: Below these floors, "serve difficulty 3" and "serve difficulty 9" resolve to
#: the same item and the action is a no-op — `stacks_and_queues` has two items
#: at an identical *b*, so every arm from `random` to `model_L4` was served the
#: same thing there and no policy could differ. Measured, not guessed: the
#: floors are the smallest that exclude the degenerate concepts in this bank.
MIN_ITEMS_PER_CONCEPT = 3
MIN_DIFFICULTY_SPREAD = 1.0     # logits of *b*, i.e. the action must move p


def select_curriculum(items, size: int = CURRICULUM_SIZE) -> list[str]:
    """A prerequisite-closed curriculum the difficulty action can act on.

    Derived from the item bank and the concept graph rather than hand-listed:
    take the best-covered concepts whose *b* actually varies, pull in whatever
    they depend on, and sort the result topologically. A concept the bank cannot
    vary the difficulty of would make "choose a difficulty" meaningless, so a
    candidate whose prerequisite closure contains such a concept is skipped
    entirely rather than dragging it in — which is what put a two-item,
    zero-spread concept into a sixth of every rollout before this floor existed.
    """
    counts: dict[str, int] = {}
    spread: dict[str, list[float]] = {}
    for item in items:
        counts[item.concept_id] = counts.get(item.concept_id, 0) + 1
        spread.setdefault(item.concept_id, []).append(item.b)
    prerequisites = {concept.concept_id: concept.prerequisites for concept in CONCEPTS}
    level = {concept.concept_id: concept.level for concept in CONCEPTS}

    def actionable(name: str) -> bool:
        values = spread.get(name, [])
        return (counts.get(name, 0) >= MIN_ITEMS_PER_CONCEPT
                and max(values) - min(values) >= MIN_DIFFICULTY_SPREAD)

    chosen: list[str] = []
    for concept in sorted(counts, key=lambda name: (-counts[name], name)):
        if not actionable(concept):
            continue
        closure = []
        frontier = [concept]
        while frontier:
            current = frontier.pop()
            if current in chosen or current in closure:
                continue
            closure.append(current)
            frontier.extend(name for name in prerequisites.get(current, ())
                            if counts.get(name, 0) > 0)
        if len(chosen) + len(closure) > size or not all(actionable(name) for name in closure):
            continue
        chosen.extend(closure)
        if len(chosen) == size:
            break
    if len(chosen) < size:
        raise SystemExit(
            f"item bank supports only {len(chosen)} actionable concepts, not {size}: "
            f"a curriculum concept needs ≥{MIN_ITEMS_PER_CONCEPT} items spanning "
            f"≥{MIN_DIFFICULTY_SPREAD} logits of b. Add items before rerunning Phase 8.")
    return sorted(chosen, key=lambda name: (level[name], -counts[name], name))


@dataclass
class LearnerRun:
    """Everything the environment tracks for one learner."""

    index: int
    learner: object
    clock: int
    session_index: int = 1
    session_position: int = 0
    session_length: int = 0
    session_started: int = 0
    active: str = ""
    position: int = 0
    on_task_seconds: float = 0.0
    outcomes: list[int] = field(default_factory=list)
    incorrect_streak: int = 0
    concept_attempts: dict[str, int] = field(default_factory=dict)
    mastered_at_items: int | None = None
    mastered_at_seconds: float | None = None
    finished: bool = False
    interventions: dict[str, int] = field(default_factory=dict)
    difficulties: list[int] = field(default_factory=list)
    success_probabilities: list[float] = field(default_factory=list)
    min_engagement: float = 1.0
    sessions: int = 1

    @property
    def session_id(self) -> str:
        return f"{self.learner.learner_id}-s{self.session_index:03d}"


class ClosedLoop:
    """One (variant, seed, cohort) environment. One policy runs against it."""

    def __init__(self, variant: str, seed: int, learners: int, budget: int = BUDGET,
                 curriculum: list[str] | None = None, observables: bool = True,
                 params: Params | None = None):
        base = params if params is not None else Params.load()
        self.params = base.as_variant(variant)
        self.variant = variant
        self.seed = seed
        self.budget = budget
        self.observables = observables
        self.simulator = Simulator(self.params, seed=seed)
        self.rng = np.random.default_rng(seed + 977)
        self.curriculum = curriculum or select_curriculum(self.simulator.items)
        self.by_concept = {concept: sorted(
            (item for item in self.simulator.items if item.concept_id == concept),
            key=lambda item: item.b) for concept in self.curriculum}
        self.median_b = {concept: float(np.median([item.b for item in items]))
                         for concept, items in self.by_concept.items()}

        start = 1_760_000_000_000
        self.runs: list[LearnerRun] = []
        self.screened = 0
        while len(self.runs) < learners:
            for learner in self.simulator.learners(learners):
                # sorted, not a set: each setdefault below draws from the
                # simulator's RNG, so iteration order decides which concept gets
                # which draw. A bare set iterates in hash order, which varies
                # between processes, and the whole cohort changes with it.
                for concept in sorted({item.concept_id for item in self.simulator.items}):
                    learner.knowledge.setdefault(
                        concept, learner.ability + float(self.simulator.rng.normal(0, 0.5)))
                run = LearnerRun(index=len(self.runs), learner=learner,
                                 clock=start + int(self.rng.integers(0, 7 * 24 * 3600 * 1000)))
                self.screened += 1
                # Inclusion criterion (pre-registration §9.2): a learner who has
                # already mastered the whole curriculum has nothing an
                # instructional policy could do for them, and 41 % of draws from
                # the calibrated ability distribution are such learners. Keeping
                # them would put a ceiling on every arm alike and hide whatever
                # difference exists among the learners who need teaching.
                if all(self.mastered(run, concept) for concept in self.curriculum):
                    continue
                # Screening draws more than one batch, and each batch numbers
                # its learners from one, so accepted learners are renumbered:
                # two learners sharing an id would collide in the decision log
                # and in every per-learner join downstream.
                learner.learner_id = f"sim-{len(self.runs) + 1:05d}"
                run.session_started = run.clock
                run.session_length = self._session_length()
                run.active = self.curriculum[0]
                self._advance_to_unmastered(run)
                self.runs.append(run)
                if len(self.runs) == learners:
                    break
        self.initial_knowledge = [
            {concept: run.learner.knowledge[concept] for concept in self.curriculum}
            for run in self.runs]
        self.step_index = 0
        #: Mean over the cohort at each step, for the mastery and difficulty
        #: curves. Only the means are kept: the per-learner trajectory is
        #: 200,000 floats per cell and no figure asks for it.
        self.mastery_curve: list[float] = []
        self.difficulty_curve: list[float] = []
        self.active_curve: list[int] = []

    # ------------------------------------------------------------- mastery

    def success_probability_at(self, knowledge: float, concept: str,
                               b: float | None = None) -> float:
        """The environment's own response model. Evaluation only, never a policy input."""
        params = self.params
        b = self.median_b[concept] if b is None else b
        base = 1 / (1 + math.exp(-float(np.clip(knowledge - b, -30, 30))))
        return params.guess + (1 - params.guess - params.slip) * base

    def success_probability(self, run: LearnerRun, concept: str, b: float | None = None) -> float:
        return self.success_probability_at(run.learner.knowledge[concept], concept, b)

    def mastered(self, run: LearnerRun, concept: str) -> bool:
        return self.success_probability(run, concept) >= MASTERY_TARGET

    def mastery_fraction(self, run: LearnerRun) -> float:
        return float(np.mean([self.mastered(run, concept) for concept in self.curriculum]))

    def _advance_to_unmastered(self, run: LearnerRun) -> None:
        """Keep the active concept on something still worth practising."""
        if not self.mastered(run, run.active):
            return
        for concept in self.curriculum:
            if not self.mastered(run, concept):
                run.active = concept
                return

    def _session_length(self) -> int:
        quantiles = np.asarray(self.params.session_length_quantiles, dtype=float)
        grid = np.linspace(0, 1, len(quantiles))
        length = float(np.interp(float(self.rng.random()), grid, quantiles))
        return int(max(2, min(60, round(length))))

    # ---------------------------------------------------------------- views

    def views(self, rung: str, last_rows: list[dict | None]) -> list[dict | None]:
        """The decision context handed to the policy, one entry per learner."""
        out: list[dict | None] = []
        for run in self.runs:
            if run.finished:
                out.append(None)
                continue
            out.append({
                "index": run.index,
                "position": run.position,
                "session_position": run.session_position,
                "active_concept": run.active,
                "concept_attempts": run.concept_attempts.get(run.active, 0),
                "outcomes": run.outcomes,
                "incorrect_streak": run.incorrect_streak,
                "observables": observable_view(last_rows[run.index], rung),
                "curriculum": self.curriculum,
                "median_b": self.median_b,
            })
        return out

    # ----------------------------------------------------------------- step

    def resolve(self, run: LearnerRun, action: dict):
        """Action → the item that will actually be served."""
        move = action["concept_move"]
        order = self.curriculum
        current = order.index(run.active)
        if move == "next_concept":
            for concept in order[current + 1:] + order[:current]:
                if not self.mastered(run, concept):
                    run.active = concept
                    break
        elif move == "prerequisite_concept":
            for concept in reversed(order[:current]):
                run.active = concept
                break
        self._advance_to_unmastered(run)

        pool = self.by_concept[run.active]
        wanted = action["difficulty"]
        distance = [abs(item.difficulty_score - wanted) for item in pool]
        best = min(distance)
        candidates = [item for item, gap in zip(pool, distance) if gap == best]
        return candidates[int(self.rng.integers(len(candidates)))]

    def step(self, actions: list[dict | None]) -> list[dict | None]:
        """Execute one action per active learner. Returns the attempt records."""
        rows: list[dict | None] = [None] * len(self.runs)
        served = []
        active = 0
        for run, action in zip(self.runs, actions):
            if run.finished or action is None:
                continue
            active += 1
            rows[run.index] = self._one(run, action)
            served.append(rows[run.index]["difficulty_score"])
        self.step_index += 1
        self.mastery_curve.append(float(np.mean([self.mastery_fraction(run)
                                                 for run in self.runs])))
        self.difficulty_curve.append(float(np.mean(served)) if served else float("nan"))
        self.active_curve.append(active)
        return rows

    def _one(self, run: LearnerRun, action: dict) -> dict:
        intervention = action["intervention"]
        support, gain_scale, extra_seconds = 0.0, 1.0, 0.0

        if intervention == "break_suggestion":
            run.learner.fatigue = 0.0
            extra_seconds += BREAK_SECONDS
            run.session_index += 1
            run.session_position = 0
            run.session_length = self._session_length()
            run.clock += int(BREAK_SECONDS * 1000)
            run.session_started = run.clock
            run.sessions += 1
        elif intervention == "hint":
            support, gain_scale, extra_seconds = HINT_SUPPORT, HINT_GAIN_SCALE, HINT_SECONDS
        elif intervention == "worked_example":
            support, gain_scale, extra_seconds = WORKED_SUPPORT, WORKED_GAIN_SCALE, WORKED_SECONDS

        item = self.resolve(run, action)
        row = self.simulator.attempt(
            run.learner, item, run.session_id, run.session_started, run.session_position,
            run.clock, action["propensity"], observables=self.observables,
            support=support, gain_scale=gain_scale)

        row["intervention"] = intervention
        row["concept_move"] = action["concept_move"]
        row["requested_difficulty"] = action["difficulty"]
        row["policy"] = action.get("policy", "")
        row["variant"] = self.variant
        row["seed"] = self.seed
        row["step"] = run.position
        row["question_number"] = run.session_position + 1
        row["session_index"] = run.session_index

        elapsed = row["response_time_ms"] / 1000.0 + extra_seconds
        gap_ms = float(self.rng.exponential(INTER_ITEM_MEAN_MS))
        run.on_task_seconds += elapsed + gap_ms / 1000.0
        run.clock += int(row["response_time_ms"] + extra_seconds * 1000 + gap_ms)
        run.position += 1
        run.session_position += 1
        run.outcomes.append(int(row["correct"]))
        run.incorrect_streak = 0 if row["correct"] else run.incorrect_streak + 1
        run.concept_attempts[item.concept_id] = run.concept_attempts.get(item.concept_id, 0) + 1
        run.interventions[intervention] = run.interventions.get(intervention, 0) + 1
        run.difficulties.append(int(item.difficulty_score))
        run.success_probabilities.append(float(row["probability_correct"]))
        run.min_engagement = min(run.min_engagement, float(row["latent_engagement"]))

        if run.session_position >= run.session_length:
            # Between sessions: knowledge decays, the clock jumps, fatigue clears.
            for concept in run.learner.knowledge:
                run.learner.knowledge[concept] *= self.params.retention_between_sessions
            run.learner.fatigue = 0.0
            run.session_index += 1
            run.sessions += 1
            run.session_position = 0
            run.session_length = self._session_length()
            run.clock += 30 * 60 * 1000 + int(self.rng.exponential(6 * 3600 * 1000))
            run.session_started = run.clock

        self._advance_to_unmastered(run)
        if all(self.mastered(run, concept) for concept in self.curriculum):
            run.mastered_at_items = run.position
            run.mastered_at_seconds = run.on_task_seconds
            run.finished = True
        elif run.position >= self.budget:
            run.finished = True
        return row

    @property
    def done(self) -> bool:
        return all(run.finished for run in self.runs)

    # -------------------------------------------------------------- outcomes

    def results(self) -> list[dict]:
        """One record per learner. Latent quantities appear here and only here."""
        out = []
        for run in self.runs:
            probabilities = run.success_probabilities or [float("nan")]
            initial = self.initial_knowledge[run.index]
            gain = float(np.mean([run.learner.knowledge[concept] - initial[concept]
                                  for concept in self.curriculum]))
            out.append({
                "learner_id": run.learner.learner_id,
                "learner_index": run.index,
                "variant": self.variant,
                "seed": self.seed,
                "mastered": bool(run.mastered_at_items is not None),
                # Right-censored at the budget: an unmastered learner contributes
                # the budget, and `mastered` says which it was.
                "items_to_mastery": run.mastered_at_items if run.mastered_at_items else run.position,
                "time_to_mastery_seconds": (run.mastered_at_seconds if run.mastered_at_seconds
                                            else run.on_task_seconds),
                "items_served": run.position,
                "final_mastery_fraction": self.mastery_fraction(run),
                "mean_success_probability": float(np.mean(probabilities)),
                # Latent knowledge gain over the taught curriculum. Evaluation
                # only — and the outcome that is *not* censored by the budget,
                # which matters because the calibrated learning rate leaves most
                # learners short of mastery inside it.
                "knowledge_gain": gain,
                "knowledge_gain_per_item": gain / run.position if run.position else float("nan"),
                "initial_mastery_fraction": float(np.mean(
                    [self.success_probability_at(initial[concept], concept) >= MASTERY_TARGET
                     for concept in self.curriculum])),
                "mean_difficulty": float(np.mean(run.difficulties)) if run.difficulties else float("nan"),
                "accuracy": float(np.mean(run.outcomes)) if run.outcomes else float("nan"),
                "sessions": run.sessions,
                "min_engagement": run.min_engagement,
                "disengaged": bool(run.min_engagement < 0.3),
                "ability": float(run.learner.ability),
                "learning_rate": float(run.learner.learning_rate),
                **{f"n_{name}": run.interventions.get(name, 0)
                   for name in ("no_intervention", "hint", "worked_example", "break_suggestion")},
            })
        return out
