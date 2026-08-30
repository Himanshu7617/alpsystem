"""Request contracts for the two serving endpoints, `/v1/state` and `/v1/decide`.

An attempt record here is the same shape the closed-loop environment produces,
because the estimator and the arms are the same objects Phase 8 and Phase 9
ran. The behavioural columns arrive in ``features`` — the extractor's own
output, forwarded verbatim by the backend — and the history columns
(``prior_accuracy`` and the rest) are computed inside the service, so the live
row and the training row are built by one piece of code.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Attempt(BaseModel):
    """One finished item attempt, as the platform observed it."""

    session_id: str
    item_id: str
    concept_id: str
    correct: bool
    #: The item's difficulty bin, 1-10. The rule arms band on it directly.
    difficulty_score: int = 1
    response_time_ms: float = 0.0
    #: The item's IRT difficulty. The estimator reads it as `item_difficulty`.
    item_b: float = 0.0
    #: Client clock, milliseconds. Only differences are used (lag between items).
    timestamp: float = 0.0
    #: Items answered in the whole session so far, and within the current
    #: sitting. The environment calls them `position` and `question_number`.
    position: int = 0
    question_number: int = 0
    idle_time: float | None = None
    #: `app.features.extract`'s output for this attempt, verbatim.
    features: dict = Field(default_factory=dict)

    def model_dump(self, **kwargs) -> dict:  # type: ignore[override]
        """Flatten `features` into the record, which is how a row is packed."""
        data = super().model_dump(**kwargs)
        features = data.pop("features") or {}
        return {**features, **data}


class View(BaseModel):
    """The decision context: where the learner is, not what they scored."""

    position: int = 0
    session_position: int = 0
    active_concept: str
    concept_attempts: int = 0
    outcomes: list[int] = Field(default_factory=list)
    incorrect_streak: int = 0
    curriculum: list[str]


class StateRequest(BaseModel):
    session_id: str
    attempts: list[Attempt] = Field(default_factory=list)


class DecideRequest(BaseModel):
    session_id: str
    view: View
    #: Optional: sending the history with the decision lets a cold service warm
    #: its slot in one round trip instead of two.
    attempts: list[Attempt] | None = None
