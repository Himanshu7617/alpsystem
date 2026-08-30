"""Raw telemetry events -> the per-attempt feature row.

**This module is the single definition of a feature.** The backend reaches it
over HTTP (`POST /v1/features/extract`), offline training imports `extract()`
directly. Two implementations of one feature is how train/serve skew starts, so
there is only ever one.

Two rules hold here and are checked by `audit_leakage.py` in Phase 5:

1. **No latent ground truth.** Nothing named `knowledge_*`, `fatigue_*`,
   `confidence_*` or `engagement_*` is produced. Those are targets, never
   inputs (BUILD.md Defect 2).
2. **Missing means missing.** A signal class the learner did not consent to
   yields ``None``, never ``0``. A zero is a measurement; a null is an absence,
   and the privacy-utility curve in RQ4 depends on being able to tell them
   apart.
"""
from __future__ import annotations

import math
from collections import Counter
from datetime import datetime
from typing import Any

FEATURE_VERSION = "v1-phase2"

#: Feature -> signal class. The ablation ladder in Phase 5 (L0 correctness,
#: L1 +timing, L2 +interaction, L3 +session, L4 +motor) slices the row with this.
FEATURE_CLASSES: dict[str, str] = {
    # L0 — correctness
    "correct": "correctness",
    "skipped": "correctness",
    "difficulty_score": "correctness",
    "n_options": "correctness",
    # L1 — timing
    "total_response_time": "timing",
    "log_response_time": "timing",
    "reading_time": "timing",
    "decision_latency": "timing",
    "time_after_last_interaction": "timing",
    "response_time_vs_estimate": "timing",
    # L2 — interaction
    "option_changes": "interaction",
    "option_change_entropy": "interaction",
    "option_revisits": "interaction",
    "first_selection_changed": "interaction",
    "hint_count": "interaction",
    "idle_count": "interaction",
    "idle_time": "interaction",
    "visibility_changes": "interaction",
    "focus_fraction": "interaction",
    # L3 — session
    "question_number": "session",
    "session_duration": "session",
    # L4 — motor (pre-registered in docs/preregistration.md)
    "auc_toward_nonchosen": "motor",
    "max_deviation": "motor",
    "x_flips": "motor",
    "sample_entropy": "motor",
    "velocity_peak": "motor",
    "velocity_mean": "motor",
    "pause_count": "motor",
    "path_ratio": "motor",
    "time_to_first_movement": "motor",
    "time_to_first_selection": "motor",
    "hover_time_max": "motor",
    "hover_time_nonchosen": "motor",
    "cursor_samples": "motor",
}

_TIMING_EVENTS = {"FIRST_INTERACTION"}
_INTERACTION_EVENTS = {
    "OPTION_SELECTED", "OPTION_CHANGED", "HINT_REQUESTED",
    "IDLE_ENTERED", "IDLE_EXITED", "VISIBILITY_CHANGED",
}


def _ts(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _seconds(later: datetime | None, earlier: datetime | None) -> float | None:
    if later is None or earlier is None:
        return None
    return (later - earlier).total_seconds()


def _entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total <= 1:
        return 0.0
    return -sum((n / total) * math.log2(n / total) for n in counts.values() if n)


def _as_dict(request: Any) -> dict:
    return request if isinstance(request, dict) else request.model_dump(mode="json")


def extract(request: Any) -> dict:
    """Reduce one item attempt's event stream to a flat feature row.

    ``request`` is an :class:`app.schemas.events.ExtractRequest` or the same
    shape as a plain dict. Returns ``{"feature_version": ..., "features": {...},
    "available_classes": [...]}``.
    """
    payload = _as_dict(request)
    consent = payload.get("consent") or {}
    item = payload.get("item") or {}
    attempt = payload.get("attempt") or {}
    events = sorted(payload.get("events") or [], key=lambda event: (event.get("seq", 0), event.get("client_ts") or ""))

    has_timing = consent.get("timing", True)
    has_interaction = consent.get("interaction", True)
    has_motor = consent.get("motor", True)

    by_type: dict[str, list[dict]] = {}
    for event in events:
        by_type.setdefault(str(event.get("type")), []).append(event)

    presented_at = _ts(attempt.get("presented_at")) or _ts(
        (by_type.get("ITEM_PRESENTED") or [{}])[0].get("client_ts")
    )
    submitted_at = _ts(attempt.get("submitted_at")) or _ts(
        (by_type.get("ANSWER_SUBMITTED") or [{}])[-1].get("client_ts")
    )

    features: dict[str, Any] = {}

    # ---- L0 correctness -----------------------------------------------------
    features["correct"] = int(bool(attempt.get("correct")))
    features["skipped"] = int(bool(attempt.get("skipped")))
    features["difficulty_score"] = item.get("difficulty_score")
    features["n_options"] = item.get("n_options", 4)

    # ---- L1 timing ----------------------------------------------------------
    if has_timing:
        response_ms = attempt.get("response_time_ms")
        total = (response_ms / 1000.0) if response_ms is not None else _seconds(submitted_at, presented_at)
        features["total_response_time"] = round(total, 4) if total is not None else None
        features["log_response_time"] = round(math.log1p(max(0.0, total)), 4) if total is not None else None

        first_interaction = (by_type.get("FIRST_INTERACTION") or [{}])[0].get("client_ts")
        features["reading_time"] = _round(_seconds(_ts(first_interaction), presented_at))

        selections = by_type.get("OPTION_SELECTED", []) + by_type.get("OPTION_CHANGED", [])
        first_selection = min((event.get("client_ts") for event in selections if event.get("client_ts")), default=None)
        features["decision_latency"] = _round(_seconds(submitted_at, _ts(first_selection)))

        interactive = [
            event for event in events
            if str(event.get("type")) in _TIMING_EVENTS | _INTERACTION_EVENTS
        ]
        last_interaction = max((event.get("client_ts") for event in interactive if event.get("client_ts")), default=None)
        features["time_after_last_interaction"] = _round(_seconds(submitted_at, _ts(last_interaction)))

        estimate = item.get("estimated_time_seconds")
        features["response_time_vs_estimate"] = (
            round(total / estimate, 4) if total is not None and estimate else None
        )
    else:
        for name, signal_class in FEATURE_CLASSES.items():
            if signal_class == "timing":
                features[name] = None

    # ---- L2 interaction -----------------------------------------------------
    if has_interaction:
        changes = by_type.get("OPTION_CHANGED", [])
        features["option_changes"] = len(changes)
        chosen_sequence = Counter()
        seen: set[int] = set()
        revisits = 0
        for event in by_type.get("OPTION_SELECTED", []) + changes:
            index = (event.get("payload") or {}).get("to_index")
            if index is None:
                continue
            chosen_sequence[index] += 1
            if index in seen:
                revisits += 1
            seen.add(index)
        features["option_change_entropy"] = round(_entropy(chosen_sequence), 4)
        features["option_revisits"] = revisits
        features["first_selection_changed"] = int(bool(changes))
        features["hint_count"] = len(by_type.get("HINT_REQUESTED", []))

        idle_enters = [_ts(event.get("client_ts")) for event in by_type.get("IDLE_ENTERED", [])]
        idle_exits = [_ts(event.get("client_ts")) for event in by_type.get("IDLE_EXITED", [])]
        idle_total = sum(
            max(0.0, (exit_ts - enter_ts).total_seconds())
            for enter_ts, exit_ts in zip(idle_enters, idle_exits)
            if enter_ts and exit_ts
        )
        # An idle spell still open at submit still counts.
        if len(idle_enters) > len(idle_exits) and submitted_at and idle_enters[-1]:
            idle_total += max(0.0, (submitted_at - idle_enters[-1]).total_seconds())
        features["idle_count"] = len(idle_enters)
        features["idle_time"] = round(idle_total, 4)

        visibility = by_type.get("VISIBILITY_CHANGED", [])
        features["visibility_changes"] = len(visibility)
        hidden = 0.0
        hidden_since: datetime | None = None
        for event in visibility:
            visible = bool((event.get("payload") or {}).get("visible", True))
            stamp = _ts(event.get("client_ts"))
            if not visible and hidden_since is None:
                hidden_since = stamp
            elif visible and hidden_since is not None and stamp:
                hidden += max(0.0, (stamp - hidden_since).total_seconds())
                hidden_since = None
        if hidden_since is not None and submitted_at:
            hidden += max(0.0, (submitted_at - hidden_since).total_seconds())
        span = _seconds(submitted_at, presented_at)
        features["focus_fraction"] = round(max(0.0, 1 - hidden / span), 4) if span and span > 0 else None
    else:
        for name, signal_class in FEATURE_CLASSES.items():
            if signal_class == "interaction":
                features[name] = None

    # ---- L3 session ---------------------------------------------------------
    features["question_number"] = attempt.get("seq", 1)
    features["session_duration"] = _round(_seconds(submitted_at, _ts(attempt.get("session_started_at"))))

    # ---- L4 motor -----------------------------------------------------------
    segments = by_type.get("CURSOR_SEGMENT", [])
    if has_motor and segments:
        trajectory = segments[-1].get("payload") or {}
        selected = attempt.get("selected_index")
        hover = list(trajectory.get("hover_time_ms") or [])
        features["auc_toward_nonchosen"] = _round(trajectory.get("auc_toward_nonchosen"))
        features["max_deviation"] = _round(trajectory.get("max_deviation"))
        features["x_flips"] = trajectory.get("x_flips")
        features["sample_entropy"] = _round(trajectory.get("sample_entropy"))
        features["velocity_peak"] = _round(trajectory.get("velocity_peak"))
        features["velocity_mean"] = _round(trajectory.get("velocity_mean"))
        features["pause_count"] = trajectory.get("pause_count")
        features["path_ratio"] = _round(trajectory.get("path_ratio"))
        features["time_to_first_movement"] = _round(
            (trajectory.get("time_to_first_movement_ms") or 0) / 1000.0
        )
        features["time_to_first_selection"] = _round(
            (trajectory.get("time_to_first_selection_ms") or 0) / 1000.0
        )
        features["hover_time_max"] = _round(max(hover) / 1000.0) if hover else None
        features["hover_time_nonchosen"] = (
            _round(sum(ms for index, ms in enumerate(hover) if index != selected) / 1000.0)
            if hover else None
        )
        features["cursor_samples"] = trajectory.get("n_samples", 0)
    else:
        for name, signal_class in FEATURE_CLASSES.items():
            if signal_class == "motor":
                features[name] = None

    available = ["correctness", "session"]
    if has_timing:
        available.append("timing")
    if has_interaction:
        available.append("interaction")
    if has_motor and segments:
        available.append("motor")

    return {
        "feature_version": FEATURE_VERSION,
        "features": {name: features.get(name) for name in FEATURE_CLASSES},
        "available_classes": available,
    }


def _round(value: Any, places: int = 4) -> float | None:
    if value is None:
        return None
    return round(float(value), places)


# --------------------------------------------------------------- trajectory
# The browser reduces raw pointer samples to the pre-registered aggregate before
# they leave the device (files/telemetry.js, `trajectoryFeatures`). Anything that
# holds a raw 2-D path server-side — the Phase 4 simulator — must produce that
# same aggregate, so the reduction lives here rather than being written a second
# time in the simulator.
#
# ponytail: this is a hand-kept mirror of the JavaScript, not a shared
# implementation, because the browser cannot call Python and raw samples must
# not be uploaded. The definitions are pinned in docs/preregistration.md §1 and
# the constants below are the same four the client uses. If the two ever need to
# be provably identical, the fix is a golden-path fixture checked in both
# languages, not a third implementation.

SAMPLE_INTERVAL_MS = 50
PAUSE_THRESHOLD_MS = 300
PAUSE_VELOCITY = 0.05  # px/ms


def _signed_distance(point: tuple[float, float], start: tuple[float, float],
                     target: tuple[float, float]) -> float:
    """Perpendicular signed distance of ``point`` from the line start -> target."""
    dx, dy = target[0] - start[0], target[1] - start[1]
    length = math.hypot(dx, dy)
    if length == 0:
        return 0.0
    return ((point[0] - start[0]) * dy - (point[1] - start[1]) * dx) / length


def sample_entropy(series: list[float], m: int = 2, r_factor: float = 0.2) -> float:
    """SampEn of the speed series (m = 2, r = 0.2σ), mirroring telemetry.js."""
    n = len(series)
    if n < m + 2:
        return 0.0
    values = _np().asarray(series, dtype=float)
    sd = float(values.std())
    if sd == 0:
        return 0.0
    tolerance = r_factor * sd

    def matches(length: int) -> int:
        rows = n - length + 1
        if rows < 2:
            return 0
        np = _np()
        windows = np.lib.stride_tricks.sliding_window_view(values, length)
        distance = np.abs(windows[:, None, :] - windows[None, :, :]).max(axis=2)
        return int(np.triu(distance <= tolerance, k=1).sum())

    a, b = matches(m + 1), matches(m)
    if a == 0 or b == 0:
        return 0.0
    return round(-math.log(a / b), 4)


def reduce_trajectory(samples: list[dict], option_rects: list[dict | None],
                      selected_index: int | None, option_centres: list[dict | None] | None = None,
                      first_selection_ms: float | None = None) -> dict:
    """Raw 2-D pointer path -> the `CURSOR_SEGMENT` payload the extractor reads.

    ``samples`` are ``{"t": ms since presented, "x": px, "y": px}`` at the
    client's 50 ms cadence; ``option_rects`` are viewport rectangles
    (``left/right/top/bottom``) index-aligned with the item's options. Hover time
    is accumulated per sample exactly as the client's `pointermove` handler does.
    """
    centres = option_centres or [
        None if rect is None else {"x": (rect["left"] + rect["right"]) / 2,
                                   "y": (rect["top"] + rect["bottom"]) / 2}
        for rect in option_rects
    ]

    hover = [0.0] * len(option_rects)
    for sample in samples:
        for index, rect in enumerate(option_rects):
            if rect and rect["left"] <= sample["x"] <= rect["right"] and rect["top"] <= sample["y"] <= rect["bottom"]:
                hover[index] += SAMPLE_INTERVAL_MS

    first_movement = samples[0]["t"] if samples else 0
    base = {
        "time_to_first_movement_ms": first_movement,
        "time_to_first_selection_ms": first_selection_ms or 0,
        "hover_time_ms": [round(value) for value in hover],
        "n_samples": len(samples),
        "sample_interval_ms": SAMPLE_INTERVAL_MS,
    }
    if len(samples) < 2:
        return base | {"auc_toward_nonchosen": 0, "max_deviation": 0, "x_flips": 0,
                       "sample_entropy": 0, "velocity_peak": 0, "velocity_mean": 0,
                       "pause_count": 0, "path_ratio": 1}

    start = (samples[0]["x"], samples[0]["y"])
    chosen_centre = centres[selected_index] if selected_index is not None and selected_index < len(centres) else None
    target = ((chosen_centre["x"], chosen_centre["y"]) if chosen_centre
              else (samples[-1]["x"], samples[-1]["y"]))

    # The competing option: the non-chosen option the pointer lingered on
    # longest. The AUC is signed positive toward it.
    competitor, best_hover = None, -1.0
    for index, centre in enumerate(centres):
        if index == selected_index or centre is None:
            continue
        if hover[index] > best_hover:
            best_hover, competitor = hover[index], centre
    sign = 1.0
    if competitor is not None:
        offset = _signed_distance((competitor["x"], competitor["y"]), start, target)
        sign = math.copysign(1.0, offset) if offset else 1.0

    path_length = max_deviation = auc = 0.0
    flips = pause_count = 0
    last_direction = 0
    pause_run = 0.0
    speeds: list[float] = []

    for previous, current in zip(samples, samples[1:]):
        dt = max(1.0, current["t"] - previous["t"])
        step = math.hypot(current["x"] - previous["x"], current["y"] - previous["y"])
        path_length += step

        speed = step / dt  # px/ms
        speeds.append(speed)
        if speed < PAUSE_VELOCITY:
            pause_run += dt
        else:
            if pause_run > PAUSE_THRESHOLD_MS:
                pause_count += 1
            pause_run = 0.0

        direction = (current["x"] > previous["x"]) - (current["x"] < previous["x"])
        if direction and last_direction and direction != last_direction:
            flips += 1
        if direction:
            last_direction = direction

        deviation = _signed_distance((current["x"], current["y"]), start, target)
        max_deviation = max(max_deviation, abs(deviation))
        auc += sign * deviation * dt / 1000.0  # px-seconds, signed toward the competitor
    if pause_run > PAUSE_THRESHOLD_MS:
        pause_count += 1

    straight = math.hypot(target[0] - start[0], target[1] - start[1])
    return base | {
        "auc_toward_nonchosen": round(auc, 4),
        "max_deviation": round(max_deviation, 4),
        "x_flips": flips,
        "sample_entropy": sample_entropy(speeds),
        "velocity_peak": round(max(speeds) * 1000, 4),
        "velocity_mean": round(sum(speeds) / len(speeds) * 1000, 4),
        "pause_count": pause_count,
        "path_ratio": round(path_length / straight, 4) if straight > 0 else 1,
    }


def _np():
    """numpy, imported lazily: the serving path does not need it."""
    import numpy

    return numpy
