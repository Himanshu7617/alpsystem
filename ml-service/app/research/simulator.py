"""Simulator v2 — a calibrated, non-tautological generative learner model.

The full generative model, its DAG and its limitations are in
`docs/simulator.md`. Three properties matter here and each is a direct answer to
an audit finding:

1. **Parameters come from real data.** Everything in :class:`Params` is fitted
   by `calibrate_simulator.py` against archival learner data, or is explicitly
   marked as not identifiable from it.
2. **Observables are not deterministic functions of the latent state**
   (BUILD.md Defect 3). Every behavioural signal is a sum of a state-driven
   term, a *stable learner trait drawn independently of ability*, an item-level
   effect, and a contaminating off-task channel. A model cannot invert the
   observable back to the latent state, because three of the four inputs have
   nothing to do with it.
3. **Trajectory features are never emitted directly.** The simulator produces a
   2-D pointer path and a real event stream, then runs the production extractor
   (`app.features.extract`) over it. There is one definition of a feature and
   the simulator is not allowed its own.

Variants V0–V4 are deliberate mis-specifications of this model; a closed-loop
result that flips ranking across them is not a finding. See :data:`VARIANTS`.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Iterator

import numpy as np

from app.features.extract import SAMPLE_INTERVAL_MS, extract, reduce_trajectory

ROOT = Path(__file__).resolve().parents[3]
BANK_PATH = ROOT / "data" / "items" / "item-bank-v1.json"
ITEM_PARAMS_PATH = ROOT / "artifacts" / "datasets" / "item-parameters-v1.csv"
PARAMS_PATH = ROOT / "artifacts" / "datasets" / "simulator-params-v2.json"

N_OPTIONS = 4
PROBE_RATE = 0.12          # docs/preregistration.md §3
EFFORT_PROBE_EVERY = 15    # docs/preregistration.md §3
SESSION_GAP_MS = 30 * 60 * 1000

#: Option layout the paths are drawn over. The client reports viewport pixels;
#: these are the geometry of the real quiz card in `files/index.html`.
OPTION_RECTS = [
    {"left": 120.0, "right": 720.0, "top": 300.0 + 80.0 * index, "bottom": 352.0 + 80.0 * index}
    for index in range(N_OPTIONS)
]
#: Where the pointer sits when an item is presented: near the submit control,
#: below the option list, but not at a fixed pixel — a pointer that always
#: started at the same point would make every path's geometry identical and the
#: deviation features pure noise.
CURSOR_START = (420.0, 700.0)
CURSOR_START_SPREAD = (240.0, 30.0)

VARIANTS = {
    "V0": "calibrated baseline",
    "V1": "2PL link — item discrimination varies instead of being fixed at 1",
    "V2": "fatigue affects speed only, never accuracy",
    "V3": "behavioural signals are half noise — traits dominate the latent state",
    "V4": "non-stationary learning rate — learners speed up or slow down mid-session",
}


@dataclass(frozen=True)
class Params:
    """Every knob of the generative model. Fitted values live in JSON."""

    # --- ability and outcome (fitted) ---
    ability_mean: float = 0.95
    ability_sd: float = 1.05
    guess: float = 0.25
    slip: float = 0.06
    learning_rate_log_mean: float = -2.9
    learning_rate_log_sd: float = 0.55
    retention_between_sessions: float = 0.96
    fatigue_growth: float = 0.045          # per item within a session
    fatigue_accuracy_beta: float = 0.35    # logits lost at fatigue = 1

    # --- response time, log ms (fitted) ---
    rt_intercept: float = 10.1
    rt_beta_distance: float = 0.10         # per logit of |ability - difficulty|
    rt_beta_fatigue: float = 0.22
    rt_beta_stem: float = 0.25             # per stem length z-score
    rt_sigma: float = 0.95

    # --- session structure (fitted, empirical quantiles) ---
    session_length_quantiles: tuple[float, ...] = tuple(np.linspace(2, 40, 21))
    learner_total_quantiles: tuple[float, ...] = tuple(np.linspace(10, 400, 21))

    # --- learner traits, drawn once per learner, INDEPENDENT of ability ---
    trait_speed_sd: float = 0.45           # personal log-RT offset
    trait_jitter_log_mean: float = 2.4     # cursor noise, px
    trait_jitter_log_sd: float = 0.5
    trait_reread_log_sd: float = 0.35      # reading-time multiplier
    trait_move_speed_mean: float = 0.75    # px/ms of pointer travel, mouse
    trait_move_speed_log_sd: float = 0.35
    trackpad_speed_factor: float = 0.65
    trait_indecision_mean: float = 0.55    # baseline option-change rate
    trait_indecision_sd: float = 0.35
    trait_distraction_mean: float = 0.06   # per-item off-task probability
    trait_distraction_sd: float = 0.05
    trackpad_share: float = 0.45

    # --- observable mixing weights: how much of a signal is state vs trait ---
    state_weight: float = 1.0
    trait_weight: float = 1.0

    # --- probes: self-report is not the latent value (not identifiable offline) ---
    probe_noise_sd: float = 0.55
    probe_bias: float = 0.25

    # --- mis-specification switches ---
    discrimination_log_sd: float = 0.0     # V1 turns this on
    learning_rate_drift_sd: float = 0.0    # V4 turns this on

    variant: str = "V0"
    calibrated_against: str | None = None
    fitted: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data["session_length_quantiles"] = list(self.session_length_quantiles)
        data["learner_total_quantiles"] = list(self.learner_total_quantiles)
        return data

    @classmethod
    def load(cls, path: Path = PARAMS_PATH) -> "Params":
        if not path.exists():
            raise SystemExit(
                f"{path.relative_to(ROOT)} does not exist. Run `make simulate`'s calibration step "
                "first: python -m app.research.calibrate_simulator"
            )
        data = json.loads(path.read_text(encoding="utf-8"))
        known = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        known["session_length_quantiles"] = tuple(known["session_length_quantiles"])
        known["learner_total_quantiles"] = tuple(known["learner_total_quantiles"])
        return cls(**known)

    def as_variant(self, variant: str) -> "Params":
        """Return this calibration deliberately mis-specified in one way."""
        if variant not in VARIANTS:
            raise SystemExit(f"unknown variant {variant}. Known: {', '.join(VARIANTS)}")
        if variant == "V0":
            return replace(self, variant="V0")
        if variant == "V1":
            return replace(self, variant="V1", discrimination_log_sd=0.35)
        if variant == "V2":
            return replace(self, variant="V2", fatigue_accuracy_beta=0.0,
                           rt_beta_fatigue=self.rt_beta_fatigue * 1.5)
        if variant == "V3":
            return replace(self, variant="V3", state_weight=0.5, trait_weight=1.5)
        return replace(self, variant="V4", learning_rate_drift_sd=0.5)


@dataclass(frozen=True)
class Item:
    item_id: str
    concept_id: str
    b: float
    difficulty_score: int
    stem_z: float
    correct_index: int
    discrimination: float = 1.0


@dataclass
class Learner:
    learner_id: str
    ability: float
    learning_rate: float
    fatigue_susceptibility: float
    # traits — drawn independently of ability, so behaviour carries variance no
    # model can decode into knowledge (BUILD.md Defect 3)
    speed: float
    jitter: float
    reread: float
    move_speed: float
    indecision: float
    distraction: float
    trackpad: bool
    total_items: int
    drift: float = 0.0
    knowledge: dict = field(default_factory=dict)
    engagement: float = 0.75
    confidence: float = 0.6
    fatigue: float = 0.0

    @property
    def archetype(self) -> str:
        """Post-hoc label for figures. Learners are drawn from the calibrated
        distributions, never from hand-typed archetype presets — the label
        describes a learner, it does not generate one."""
        fast = self.learning_rate >= math.exp(-2.9)
        tires = self.fatigue_susceptibility >= 1.0
        return {(True, False): "rapid", (True, True): "fatigable",
                (False, False): "steady", (False, True): "careful"}[(fast, tires)]


def load_items(rng: np.random.Generator, params: Params) -> list[Item]:
    """The real item bank, with calibrated difficulty and real stem lengths."""
    import csv

    bank = json.loads(BANK_PATH.read_text(encoding="utf-8"))["items"]
    with ITEM_PARAMS_PATH.open(encoding="utf-8") as handle:
        parameters = {row["item_id"]: row for row in csv.DictReader(handle)}

    lengths = np.array([len(item["stem"]) for item in bank], dtype=float)
    stem_z = (lengths - lengths.mean()) / (lengths.std() or 1.0)
    discrimination = (np.exp(rng.normal(0, params.discrimination_log_sd, len(bank)))
                      if params.discrimination_log_sd else np.ones(len(bank)))

    return [
        Item(
            item_id=item["item_id"],
            concept_id=item["concept_id"],
            b=float(parameters[item["item_id"]]["b"]),
            difficulty_score=int(parameters[item["item_id"]]["difficulty_bin"]),
            stem_z=float(stem_z[index]),
            correct_index=int(item["correct_index"]),
            discrimination=float(discrimination[index]),
        )
        for index, item in enumerate(bank)
    ]


def _quantile_draw(rng: np.random.Generator, quantiles: tuple[float, ...], size: int) -> np.ndarray:
    """Sample from an empirical distribution given as evenly spaced quantiles."""
    grid = np.linspace(0, 1, len(quantiles))
    return np.interp(rng.random(size), grid, np.asarray(quantiles, dtype=float))


class Simulator:
    """Generates learners, sessions and attempts under one parameterisation."""

    def __init__(self, params: Params, seed: int = 20260821):
        self.params = params
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.items = load_items(self.rng, params)
        self.items_by_bin: dict[int, list[Item]] = {}
        for item in self.items:
            self.items_by_bin.setdefault(item.difficulty_score, []).append(item)

    # ------------------------------------------------------------- learners

    def learners(self, count: int) -> list[Learner]:
        params, rng = self.params, self.rng
        ability = rng.normal(params.ability_mean, params.ability_sd, count)
        learning_rate = np.exp(rng.normal(params.learning_rate_log_mean, params.learning_rate_log_sd, count))
        fatigue_susceptibility = np.exp(rng.normal(0, 0.4, count))
        totals = np.clip(_quantile_draw(rng, params.learner_total_quantiles, count), 10, None)

        # Traits are drawn from their own generators. Nothing below reads
        # `ability`: that independence is the point.
        speed = rng.normal(0, params.trait_speed_sd, count)
        jitter = np.exp(rng.normal(params.trait_jitter_log_mean, params.trait_jitter_log_sd, count))
        reread = np.exp(rng.normal(0, params.trait_reread_log_sd, count))
        move_speed = params.trait_move_speed_mean * np.exp(
            rng.normal(0, params.trait_move_speed_log_sd, count))
        indecision = np.clip(rng.normal(params.trait_indecision_mean, params.trait_indecision_sd, count), 0.02, None)
        distraction = np.clip(rng.normal(params.trait_distraction_mean, params.trait_distraction_sd, count), 0.0, 0.6)
        trackpad = rng.random(count) < params.trackpad_share
        drift = (rng.normal(0, params.learning_rate_drift_sd, count)
                 if params.learning_rate_drift_sd else np.zeros(count))

        return [
            Learner(
                learner_id=f"sim-{index + 1:05d}",
                ability=float(ability[index]),
                learning_rate=float(learning_rate[index]),
                fatigue_susceptibility=float(fatigue_susceptibility[index]),
                speed=float(speed[index]),
                jitter=float(jitter[index]),
                reread=float(reread[index]),
                move_speed=float(move_speed[index]),
                indecision=float(indecision[index]),
                distraction=float(distraction[index]),
                trackpad=bool(trackpad[index]),
                total_items=int(round(totals[index])),
                drift=float(drift[index]),
            )
            for index in range(count)
        ]

    # -------------------------------------------------------------- sampling

    def choose_item(self, learner: Learner) -> tuple[Item, float]:
        """Logging policy: a difficulty band uniformly, then an item in it.

        Stochastic and with a known propensity, because Study 3 (Phase 9) cannot
        be added later to logs that did not record one.
        """
        bins = sorted(self.items_by_bin)
        chosen_bin = bins[int(self.rng.integers(len(bins)))]
        pool = self.items_by_bin[chosen_bin]
        item = pool[int(self.rng.integers(len(pool)))]
        return item, 1.0 / (len(bins) * len(pool))

    # ------------------------------------------------------------ generation

    def run(self, learners: list[Learner], start_ms: int = 1_760_000_000_000,
            observables: bool = True) -> Iterator[dict]:
        """Attempt rows for every learner.

        ``observables=False`` runs the same latent dynamics and outcome model but
        skips the cursor path, the event stream and the extractor. It exists so
        calibration can moment-match ability against *this* generative process
        cheaply — the outcome code is shared, not duplicated.
        """
        for learner in learners:
            yield from self._run_learner(learner, start_ms, observables)

    def _run_learner(self, learner: Learner, start_ms: int, observables: bool = True) -> Iterator[dict]:
        params, rng = self.params, self.rng
        remaining = learner.total_items
        clock = start_ms + int(rng.integers(0, 7 * 24 * 3600 * 1000))
        session_index = 0

        for concept in {entry.concept_id for entry in self.items}:
            learner.knowledge.setdefault(
                concept, learner.ability + float(rng.normal(0, 0.5)))

        while remaining > 0:
            session_index += 1
            length = int(round(float(_quantile_draw(rng, params.session_length_quantiles, 1)[0])))
            length = max(1, min(length, remaining))
            learner.fatigue = 0.0
            session_id = f"{learner.learner_id}-s{session_index:03d}"
            session_started = clock

            for position in range(length):
                item, propensity = self.choose_item(learner)
                row = self.attempt(learner, item, session_id, session_started, position, clock,
                                   propensity, observables)
                yield row
                clock += int(row["response_time_ms"] + rng.exponential(20_000))

            remaining -= length
            # Between sessions: knowledge decays a little, the clock jumps.
            for concept in learner.knowledge:
                learner.knowledge[concept] *= params.retention_between_sessions
            clock += SESSION_GAP_MS + int(rng.exponential(6 * 3600 * 1000))

    # --------------------------------------------------------------- attempt

    def attempt(self, learner: Learner, item: Item, session_id: str, session_started: int,
                position: int, clock: int, propensity: float, observables: bool = True,
                support: float = 0.0, gain_scale: float = 1.0) -> dict:
        """One attempt, and the latent transition it causes.

        ``support`` and ``gain_scale`` are the instructional actions Phase 8's
        closed loop can take: a hint or a worked example adds logits of support
        to the response model and scales what the learner takes away from a
        success. They are zero and one for every logged attempt in Phase 4, so
        the datasets this simulator generated are unchanged by their existence.
        """
        params, rng = self.params, self.rng
        knowledge = learner.knowledge[item.concept_id]
        fatigue = learner.fatigue

        # ---- outcome: 2PL link with slip and guess ---------------------------
        distance = knowledge - item.b - params.fatigue_accuracy_beta * fatigue + support
        base = 1 / (1 + math.exp(-float(np.clip(item.discrimination * distance, -30, 30))))
        probability = params.guess + (1 - params.guess - params.slip) * base
        correct = bool(rng.random() < probability)

        # ---- response time: log-normal, location from ability, fatigue, item --
        log_rt = (params.rt_intercept
                  + params.rt_beta_distance * abs(knowledge - item.b)
                  + params.rt_beta_fatigue * fatigue
                  + params.rt_beta_stem * item.stem_z
                  + learner.speed
                  + rng.normal(0, params.rt_sigma))
        response_time_ms = float(np.clip(math.exp(log_rt), 500, 600_000))

        observed = (self._observe(learner, item, session_started, position, clock, base, correct,
                                  fatigue, response_time_ms)
                    if observables else {"features": {}, "response_time_ms": response_time_ms,
                                         "selected_index": None, "distracted": 0,
                                         "probe_confidence": None, "probe_effort": None})

        record = {
            "learner_id": learner.learner_id,
            "archetype": learner.archetype,
            "session_id": session_id,
            "position": position,
            "timestamp": clock,
            "item_id": item.item_id,
            "concept_id": item.concept_id,
            "difficulty_score": item.difficulty_score,
            "item_b": item.b,
            "correct": int(correct),
            "response_time_ms": observed["response_time_ms"],
            "probability_correct": probability,
            "selected_index": observed["selected_index"],
            "propensity": propensity,
            "distracted": observed["distracted"],
            "probe_confidence": observed["probe_confidence"],
            "probe_effort": observed["probe_effort"],
            # Latent state is evaluation-only ground truth. The `latent_` prefix
            # is what Phase 5's leakage audit excludes on (BUILD.md Defect 2).
            "latent_knowledge": knowledge,
            # Knowledge of *this item's* concept jumps as the concept changes;
            # the mean over all concepts is the learner's overall mastery, which
            # is what a policy's objective and the figures actually care about.
            "latent_knowledge_mean": float(np.mean(list(learner.knowledge.values()))),
            "latent_engagement": learner.engagement,
            "latent_confidence": learner.confidence,
            "latent_fatigue": fatigue,
            "variant": params.variant,
        }
        record.update(observed["features"])

        # ---- latent transition ------------------------------------------------
        # The drift multiplier is clipped: a non-stationary learning rate is a
        # mis-specification, not a licence for the knowledge recursion to diverge.
        rate = float(np.clip(learner.learning_rate * math.exp(learner.drift * (position / 20.0)), 0.0, 0.5))
        gain = rate * (1.0 if correct else 0.3) * gain_scale
        learner.knowledge[item.concept_id] = float(np.clip(knowledge + gain * (3.5 - knowledge), -4.0, 4.0))
        learner.fatigue = min(1.0, fatigue + params.fatigue_growth * learner.fatigue_susceptibility)
        learner.confidence = float(np.clip(0.85 * learner.confidence + 0.15 * (1.0 if correct else 0.0), 0, 1))
        learner.engagement = float(np.clip(
            0.9 * learner.engagement + 0.1 * (0.8 - 0.6 * learner.fatigue)
            - 0.15 * observed["distracted"], 0, 1))
        return record

    # ------------------------------------------------------------- observables

    def _observe(self, learner: Learner, item: Item, session_started: int, position: int,
                 clock: int, base: float, correct: bool, fatigue: float,
                 response_time_ms: float) -> dict:
        """Behaviour, cursor path, event stream and probes for one attempt.

        Every signal below is a sum of four independent sources — latent state,
        a stable learner trait, an item effect, and an off-task channel — so no
        observable inverts to the latent state (BUILD.md Defect 3). The
        trajectory is emitted as a path and reduced by the production extractor;
        the simulator never writes a motor feature itself.
        """
        params, rng = self.params, self.rng
        uncertainty = 1 - abs(2 * base - 1)                # state-driven hesitation
        state_term = params.state_weight * (1.4 * uncertainty + 0.8 * fatigue)
        trait_term = params.trait_weight * learner.indecision
        item_term = 0.25 * max(0.0, item.stem_z)           # long stems get re-read
        distracted = bool(rng.random() < learner.distraction)  # off-task, unrelated to state
        option_changes = int(rng.poisson(max(0.02, state_term + trait_term + item_term)))

        reading_ms = float(np.clip(
            response_time_ms * (0.30 + 0.10 * item.stem_z) * learner.reread * math.exp(rng.normal(0, 0.15)),
            200, response_time_ms * 0.8))
        idle_ms = float(rng.exponential(9_000)) if distracted else 0.0
        response_time_ms += idle_ms

        wrong = [index for index in range(N_OPTIONS) if index != item.correct_index]
        selected_index = item.correct_index if correct else wrong[int(rng.integers(len(wrong)))]
        competitor_index = (selected_index + 1 + int(rng.integers(N_OPTIONS - 1))) % N_OPTIONS
        attraction = float(np.clip(params.state_weight * 0.55 * uncertainty
                                   + params.trait_weight * 0.15 * learner.indecision, 0, 1.2))
        samples = self._path(learner, selected_index, competitor_index, attraction,
                             reading_ms, response_time_ms)
        trajectory = reduce_trajectory(samples, OPTION_RECTS, selected_index,
                                       first_selection_ms=samples[-1]["t"] if samples else None)

        # The selection sequence has to end on the option actually submitted, or
        # the extractor's change/revisit features describe a different attempt.
        selections = [int(rng.integers(N_OPTIONS)) for _ in range(option_changes)] + [selected_index]
        events = self._events(clock, response_time_ms, reading_ms, idle_ms, selections,
                              trajectory, distracted)
        row = extract({
            "consent": {"timing": True, "interaction": True, "motor": True, "probes": True},
            "item": {"difficulty_score": item.difficulty_score, "n_options": N_OPTIONS,
                     "estimated_time_seconds": 30},
            "attempt": {
                "correct": correct, "skipped": False, "selected_index": selected_index,
                "response_time_ms": response_time_ms, "seq": position + 1,
                "presented_at": _iso(clock), "submitted_at": _iso(clock + int(response_time_ms)),
                "session_started_at": _iso(session_started),
            },
            "events": events,
        })

        # Probes: self-report, deliberately noisy and biased. It is a criterion,
        # never the latent value (docs/preregistration.md §3).
        probe_confidence = probe_effort = None
        if rng.random() < PROBE_RATE:
            probe_confidence = _to_scale(learner.confidence + params.probe_bias
                                         + rng.normal(0, params.probe_noise_sd), 4)
        if (position + 1) % EFFORT_PROBE_EVERY == 0:
            probe_effort = _to_scale(1 - learner.fatigue + params.probe_bias
                                     + rng.normal(0, params.probe_noise_sd), 5)

        return {"features": row["features"], "response_time_ms": response_time_ms,
                "selected_index": selected_index, "distracted": int(distracted),
                "probe_confidence": probe_confidence, "probe_effort": probe_effort,
                "path": samples}

    # ------------------------------------------------------------------ path

    def _path(self, learner: Learner, selected_index: int, competitor_index: int,
              attraction: float, reading_ms: float, response_time_ms: float) -> list[dict]:
        """A noisy, competition-attracted 2-D drift toward the chosen option.

        Only the movement window produces samples: the client samples on
        `pointermove`, so a learner reading a stem generates nothing.
        """
        params, rng = self.params, self.rng
        target = _centre(OPTION_RECTS[selected_index])
        competitor = _centre(OPTION_RECTS[competitor_index])

        # How fast the pointer travels is a motor trait (and a device property),
        # not a function of how long the learner thought: the browser samples
        # only while the pointer moves, so a long deliberation produces a
        # stationary period and no samples. Deriving the sample count from
        # response time instead would make mean velocity a deterministic
        # function of the latent state — the Defect 3 failure, rebuilt.
        speed = learner.move_speed * (params.trackpad_speed_factor if learner.trackpad else 1.0)
        start_point = (CURSOR_START[0] + float(rng.uniform(-1, 1)) * CURSOR_START_SPREAD[0],
                       CURSOR_START[1] + float(rng.uniform(-1, 1)) * CURSOR_START_SPREAD[1])
        straight = math.dist(start_point, target)
        travel = straight * (1.0 + 0.6 * attraction)
        movement_ms = float(np.clip(travel / max(0.05, speed), 3 * SAMPLE_INTERVAL_MS, 12_000))
        count = max(3, int(movement_ms / SAMPLE_INTERVAL_MS))

        # Quadratic Bezier whose control point is pulled toward the competing
        # option: the more uncertain the decision, the more the path bows.
        start = np.array(start_point)
        end = np.array(target)
        control = (start + end) / 2 + attraction * (np.array(competitor) - (start + end) / 2)

        # Minimum-jerk timing, so speed rises and falls instead of being uniform.
        u = np.linspace(0, 1, count)
        s = 10 * u ** 3 - 15 * u ** 4 + 6 * u ** 5
        curve = ((1 - s)[:, None] ** 2 * start + 2 * (1 - s)[:, None] * s[:, None] * control
                 + (s[:, None] ** 2) * end)
        curve += rng.normal(0, learner.jitter * (0.6 if not learner.trackpad else 0.9), curve.shape)

        decision_window = max(0.0, response_time_ms - reading_ms - movement_ms)
        onset = reading_ms + float(rng.uniform(0, 1)) * decision_window
        times = onset + np.arange(count) * SAMPLE_INTERVAL_MS
        # A hesitation pause: the pointer stops moving mid-path.
        if rng.random() < 0.35 + 0.4 * attraction:
            hold = int(rng.integers(1, max(2, count // 3)))
            stop = int(rng.integers(1, count))
            times[stop:] += hold * SAMPLE_INTERVAL_MS
            curve[stop:stop + hold] = curve[stop]

        return [{"t": float(times[index]), "x": float(curve[index, 0]), "y": float(curve[index, 1])}
                for index in range(count)]

    # ---------------------------------------------------------------- events

    def _events(self, clock: int, response_time_ms: float, reading_ms: float, idle_ms: float,
                selections: list[int], trajectory: dict, distracted: bool) -> list[dict]:
        events: list[dict] = []
        seq = 0

        def add(offset: float, event_type: str, payload: dict | None = None) -> None:
            nonlocal seq
            seq += 1
            events.append({"seq": seq, "type": event_type, "client_ts": _iso(clock + int(offset)),
                           "payload": payload or {}})

        add(0, "ITEM_PRESENTED")
        add(reading_ms, "FIRST_INTERACTION", {"kind": "pointermove"})
        selection_at = reading_ms + (response_time_ms - reading_ms) * 0.5
        add(selection_at, "OPTION_SELECTED", {"to_index": selections[0]})
        changes = selections[1:]
        for change, index in enumerate(changes):
            offset = selection_at + (change + 1) * (response_time_ms - selection_at) / (len(changes) + 2)
            add(offset, "OPTION_CHANGED", {"to_index": index})
        if idle_ms > 0:
            add(selection_at, "IDLE_ENTERED")
            add(selection_at + idle_ms, "IDLE_EXITED")
        if distracted:
            add(selection_at, "VISIBILITY_CHANGED", {"visible": False})
            add(selection_at + idle_ms, "VISIBILITY_CHANGED", {"visible": True})
        add(response_time_ms, "CURSOR_SEGMENT", trajectory)
        add(response_time_ms, "ANSWER_SUBMITTED", {"selected_index": selections[-1]})
        return events


def _centre(rect: dict) -> tuple[float, float]:
    return ((rect["left"] + rect["right"]) / 2, (rect["top"] + rect["bottom"]) / 2)


def _iso(epoch_ms: float) -> str:
    from datetime import datetime, timezone

    return datetime.fromtimestamp(epoch_ms / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def _to_scale(value: float, points: int) -> int:
    """Map a latent 0-1 quantity onto a k-point self-report scale."""
    return int(np.clip(round(value * (points - 1)) + 1, 1, points))
