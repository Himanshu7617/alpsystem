"""Telemetry event schema — the contract between the browser, the backend and
the feature extractor.

There is one envelope and one payload model per event type. The Prisma models in
`backend/prisma/schema.prisma` mirror this file, and `docs/event-schema.md`
documents it in prose; if you change a field here, change it in all three.
"""
from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class EventType(str, Enum):
    SESSION_STARTED = "SESSION_STARTED"
    SESSION_ENDED = "SESSION_ENDED"
    ITEM_PRESENTED = "ITEM_PRESENTED"
    FIRST_INTERACTION = "FIRST_INTERACTION"
    OPTION_SELECTED = "OPTION_SELECTED"
    OPTION_CHANGED = "OPTION_CHANGED"
    ANSWER_SUBMITTED = "ANSWER_SUBMITTED"
    ITEM_SKIPPED = "ITEM_SKIPPED"
    HINT_REQUESTED = "HINT_REQUESTED"
    IDLE_ENTERED = "IDLE_ENTERED"
    IDLE_EXITED = "IDLE_EXITED"
    VISIBILITY_CHANGED = "VISIBILITY_CHANGED"
    CURSOR_SEGMENT = "CURSOR_SEGMENT"
    PROBE_SHOWN = "PROBE_SHOWN"
    PROBE_ANSWERED = "PROBE_ANSWERED"
    DECISION_MADE = "DECISION_MADE"


#: Which consent class each event type belongs to. `core` cannot be switched
#: off — without it there is no adaptive loop to consent to.
SIGNAL_CLASS: dict[EventType, str] = {
    EventType.SESSION_STARTED: "core",
    EventType.SESSION_ENDED: "core",
    EventType.ITEM_PRESENTED: "core",
    EventType.ANSWER_SUBMITTED: "core",
    EventType.ITEM_SKIPPED: "core",
    EventType.DECISION_MADE: "core",
    EventType.FIRST_INTERACTION: "timing",
    EventType.OPTION_SELECTED: "interaction",
    EventType.OPTION_CHANGED: "interaction",
    EventType.HINT_REQUESTED: "interaction",
    EventType.IDLE_ENTERED: "interaction",
    EventType.IDLE_EXITED: "interaction",
    EventType.VISIBILITY_CHANGED: "interaction",
    EventType.CURSOR_SEGMENT: "motor",
    EventType.PROBE_SHOWN: "probes",
    EventType.PROBE_ANSWERED: "probes",
}


class TrajectoryFeatures(BaseModel):
    """Aggregated cursor trajectory for one item.

    Raw pointer samples never leave the browser: they live in a session-scoped
    buffer, are reduced to these numbers on submit, and are then discarded. The
    definitions are pre-registered in `docs/preregistration.md`.
    """

    auc_toward_nonchosen: float
    max_deviation: float
    x_flips: int
    sample_entropy: float
    velocity_peak: float
    velocity_mean: float
    pause_count: int
    path_ratio: float
    time_to_first_movement_ms: float
    time_to_first_selection_ms: float
    hover_time_ms: list[float] = Field(default_factory=list)
    n_samples: int = 0
    sample_interval_ms: int = 50


class OptionChangePayload(BaseModel):
    from_index: int | None = None
    to_index: int
    change_number: int
    ms_since_presented: float


class ProbeShownPayload(BaseModel):
    probe_id: str
    kind: Literal["confidence", "perceived_difficulty", "effort"]
    scale_points: int = 4
    assign_p: float
    assign_draw: float


class ProbeAnsweredPayload(BaseModel):
    probe_id: str
    response: int | None = None
    skipped: bool = False


class VisibilityPayload(BaseModel):
    visible: bool
    focused: bool
    ms_since_presented: float | None = None


class Event(BaseModel):
    """The envelope every event shares."""

    event_id: str          # client-generated uuid; the idempotency key
    session_id: str | None = None
    learner_id: str | None = None
    item_id: str | None = None
    attempt_id: str | None = None
    type: EventType
    client_ts: str
    server_ts: str | None = None
    seq: int = 0
    payload: dict[str, Any] = Field(default_factory=dict)


class EventBatch(BaseModel):
    session_id: str
    learner_id: str | None = None
    events: list[Event] = Field(default_factory=list)


class Consent(BaseModel):
    correctness: bool = True
    timing: bool = True
    interaction: bool = True
    motor: bool = True
    probes: bool = True


class ItemContext(BaseModel):
    item_id: str
    difficulty: str | None = None
    difficulty_score: float | None = None
    estimated_time_seconds: float | None = None
    n_options: int = 4


class AttemptContext(BaseModel):
    attempt_id: str | None = None
    seq: int = 1
    correct: bool = False
    skipped: bool = False
    selected_index: int | None = None
    response_time_ms: float | None = None
    presented_at: str | None = None
    submitted_at: str | None = None
    session_started_at: str | None = None


class ExtractRequest(BaseModel):
    consent: Consent = Field(default_factory=Consent)
    item: ItemContext
    attempt: AttemptContext
    events: list[Event] = Field(default_factory=list)
