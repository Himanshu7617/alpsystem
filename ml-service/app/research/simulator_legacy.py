"""LEGACY simulator (pre-Phase-4). Kept only as the "before" side of Defect 3.

**Do not generate research data with this.** Its behavioural observables —
`option_changes`, `pause_duration`, `mouse_distance`, `mouse_speed`,
`tab_switches` — are deterministic functions of the latent state plus Gaussian
noise, so any model trained on them must find that behaviour predicts the latent
state (BUILD.md Defect 3). `app.research.simulator` is the calibrated
replacement; figure `f04-07` compares the two on observable-latent mutual
information. The two remaining importers (`generate_dataset.py`,
`evaluate_policies.py`) are themselves pre-Phase-8 and are replaced there.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from math import exp
from random import Random
from typing import Iterable


@dataclass(frozen=True)
class LearnerProfile:
    learner_id: str
    archetype: str
    learning_rate: float
    initial_knowledge: float
    confidence: float
    guess_probability: float
    reading_speed_wpm: float
    typing_speed_wpm: float
    fatigue_growth: float
    motivation: float
    persistence: float
    retention: float
    attention_span: float


ARCHETYPES = {
    "steady": dict(learning_rate=.075, confidence=.62, guess_probability=.12, reading_speed_wpm=220, typing_speed_wpm=42, fatigue_growth=.025, motivation=.72, persistence=.76, retention=.90, attention_span=.78),
    "rapid": dict(learning_rate=.115, confidence=.72, guess_probability=.10, reading_speed_wpm=270, typing_speed_wpm=55, fatigue_growth=.020, motivation=.80, persistence=.72, retention=.88, attention_span=.82),
    "careful": dict(learning_rate=.060, confidence=.48, guess_probability=.06, reading_speed_wpm=165, typing_speed_wpm=35, fatigue_growth=.030, motivation=.75, persistence=.88, retention=.93, attention_span=.74),
    "fatigable": dict(learning_rate=.070, confidence=.57, guess_probability=.13, reading_speed_wpm=205, typing_speed_wpm=40, fatigue_growth=.060, motivation=.60, persistence=.58, retention=.84, attention_span=.52),
}


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def logistic(value: float) -> float:
    return 1 / (1 + exp(-value))


class StudentSimulator:
    """Generates independent, aggregate behavioural interaction records."""

    def __init__(self, seed: int = 20260726):
        self.random = Random(seed)
        self.seed = seed

    def make_profiles(self, count: int) -> list[LearnerProfile]:
        profiles = []
        names = list(ARCHETYPES)
        for index in range(count):
            archetype = names[index % len(names)]
            base = ARCHETYPES[archetype]
            jitter = lambda key, spread: clamp(base[key] + self.random.uniform(-spread, spread))
            profiles.append(LearnerProfile(
                learner_id=f"sim-{index + 1:05d}", archetype=archetype,
                learning_rate=jitter("learning_rate", .018),
                initial_knowledge=clamp(self.random.gauss(.35 if archetype != "rapid" else .48, .12)),
                confidence=jitter("confidence", .10), guess_probability=jitter("guess_probability", .04),
                reading_speed_wpm=max(80, base["reading_speed_wpm"] + self.random.gauss(0, 22)),
                typing_speed_wpm=max(15, base["typing_speed_wpm"] + self.random.gauss(0, 7)),
                fatigue_growth=jitter("fatigue_growth", .012), motivation=jitter("motivation", .10),
                persistence=jitter("persistence", .10), retention=jitter("retention", .05),
                attention_span=jitter("attention_span", .12),
            ))
        return profiles

    def simulate(self, profiles: Iterable[LearnerProfile], items_per_learner: int = 24):
        for profile in profiles:
            knowledge, fatigue = profile.initial_knowledge, 0.0
            for item_number in range(1, items_per_learner + 1):
                difficulty_score = 1 + ((item_number - 1) % 10)
                difficulty = "easy" if difficulty_score <= 3 else "medium" if difficulty_score <= 7 else "hard"
                yield self.interaction(profile, item_number, difficulty, difficulty_score, knowledge, fatigue)
                event = self._last_event
                knowledge = clamp(knowledge * profile.retention + (profile.learning_rate * (1 if event["correct"] else .25)) - (.012 if not event["correct"] else 0))
                fatigue = clamp(fatigue + profile.fatigue_growth * (1 + difficulty_score / 10))

    def interaction(self, profile, item_number, difficulty, difficulty_score, knowledge, fatigue):
        challenge = difficulty_score / 10
        probability_correct = logistic(4.2 * (knowledge - challenge) + 1.1 * profile.confidence - 1.6 * fatigue)
        correct = self.random.random() < probability_correct
        guessed = not correct and self.random.random() < profile.guess_probability
        response_time = max(4, 18 + difficulty_score * 5 + fatigue * 28 - knowledge * 8 + self.random.gauss(0, 4))
        reading_time = max(1, response_time * (.32 + self.random.uniform(-.08, .08)))
        idle_time = max(0, fatigue * 6 + self.random.gauss(.6, .5))
        option_changes = max(0, int((1 - profile.confidence + fatigue) * 3 + self.random.random()))
        pause_duration = max(0, fatigue * 7 + self.random.gauss(.5, .5))
        attention = clamp(profile.attention_span - fatigue + self.random.gauss(0, .08))
        
        from app.research.concept_graph import CONCEPTS
        concept_id = self.random.choice(CONCEPTS).concept_id
        
        event = {
            "learner_id": profile.learner_id, "archetype": profile.archetype, "event_index": item_number,
            "question_id": f"{concept_id}-item-{item_number}",
            "concept_id": concept_id, "difficulty": difficulty,
            "difficulty_score": difficulty_score, "knowledge_before": round(knowledge, 6),
            "fatigue_before": round(fatigue, 6), "correct": correct, "guessed": guessed,
            "total_response_time": round(response_time, 3), "reading_time": round(reading_time, 3),
            "time_after_last_interaction": round(idle_time, 3), "attempts": 1 if correct else 1 + int(self.random.random() > profile.persistence),
            "skip": attention < .16, "option_changes": option_changes,
            "mouse_distance": round(max(0, attention * response_time * self.random.uniform(25, 65)), 3),
            "mouse_speed": round(max(0, attention * self.random.uniform(20, 120)), 3),
            "hover_time": round(max(0, response_time - reading_time), 3),
            "typing_speed": round(profile.typing_speed_wpm, 3), "backspaces": int(max(0, option_changes + self.random.gauss(1, 1))),
            "delete_frequency": int(max(0, option_changes + self.random.gauss(0, 1))), "pause_duration": round(pause_duration, 3),
            "question_number": item_number, "session_duration": round(response_time * item_number, 3),
            "tab_switches": int(self.random.random() > attention),
            "profile": asdict(profile),
        }
        self._last_event = event
        return event
