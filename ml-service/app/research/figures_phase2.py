"""Phase 2 figures — the telemetry instrument itself.

Four figures, written to `artifacts/figures/02-telemetry/`:

  f02-01  the event and database model
  f02-02  cursor trajectories: direct vs hesitant  (illustrative, synthetic)
  f02-03  what each trajectory feature measures    (illustrative, synthetic)
  f02-04  one item attempt as a timeline           (from the recorded fixture)

f02-02 and f02-03 illustrate *definitions*, not results: no learner produced
them and every panel says so on its face. f02-04 is drawn from the recorded
event stream in `app/features/testdata/attempt-events.json`, the same fixture
the conformance test asserts against.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from app.research import plotstyle

ROOT = Path(__file__).resolve().parents[3]
FIGURES = ROOT / "artifacts" / "figures" / "02-telemetry"
FIXTURE = ROOT / "ml-service" / "app" / "features" / "testdata" / "attempt-events.json"

SUBTITLE = "Synthetic illustration of a measurement definition — not a result."


#: Every file this script reads. `figure_index` copies it into INDEX.md.
SOURCES = [
    "ml-service/app/features/testdata/attempt-events.json",
]


def _save(figure, name: str) -> None:
    path = plotstyle.save(figure, FIGURES, name)
    print(f"wrote: {path.relative_to(ROOT)} (+ .pdf)")


# --------------------------------------------------------------- f02-01

ENTITIES = {
    "Learner":     (0.06, 0.78, ["learner_id", "username"]),
    "Consent":     (0.06, 0.46, ["correctness*", "timing", "interaction", "motor", "probes"]),
    "Session":     (0.34, 0.78, ["session_id", "learner_id", "version", "started/ended_at"]),
    "ItemAttempt": (0.34, 0.44, ["attempt_id", "item_id", "seq", "correct", "features"]),
    "InteractionEvent": (0.63, 0.78, ["event_id (idem.)", "type", "client/server_ts", "seq", "payload"]),
    "CursorSegment": (0.63, 0.50, ["11 trajectory features", "aggregate only"]),
    "Probe":       (0.63, 0.26, ["kind", "response", "skipped", "assign_p / draw"]),
    "StateEstimate": (0.86, 0.62, ["4 states", "*_se", "*_validated"]),
    "Decision":    (0.86, 0.30, ["policy_name/version", "action_space", "propensity*"]),
}

EDGES = [
    ("Learner", "Session"), ("Learner", "Consent"), ("Session", "ItemAttempt"),
    ("Session", "InteractionEvent"), ("ItemAttempt", "InteractionEvent"),
    ("ItemAttempt", "CursorSegment"), ("ItemAttempt", "Probe"),
    ("ItemAttempt", "StateEstimate"), ("ItemAttempt", "Decision"),
]


def figure_event_schema() -> None:
    figure, axes = plt.subplots(figsize=(13, 7))
    axes.set_xlim(0, 1.06)
    axes.set_ylim(0.06, 0.94)
    axes.axis("off")

    boxes = {}
    for name, (x, y, fields) in ENTITIES.items():
        height = 0.055 + 0.032 * len(fields)
        box = FancyBboxPatch((x, y - height), 0.19, height, boxstyle="round,pad=0.006",
                             linewidth=1.2, edgecolor="#334155", facecolor="#eef2ff")
        axes.add_patch(box)
        axes.text(x + 0.095, y - 0.028, name, ha="center", va="center", fontsize=10.5, fontweight="bold")
        for index, field in enumerate(fields):
            axes.text(x + 0.012, y - 0.055 - 0.03 * index, field, ha="left", va="center", fontsize=7.6, color="#334155")
        boxes[name] = (x, y, height)

    for source, target in EDGES:
        sx, sy, sh = boxes[source]
        tx, ty, th = boxes[target]
        start = (sx + 0.19, sy - sh / 2) if tx > sx else (sx + 0.095, sy - sh)
        end = (tx, ty - th / 2) if tx > sx else (tx + 0.095, ty)
        axes.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                       linewidth=0.9, color="#64748b", shrinkA=2, shrinkB=2))

    axes.set_title("f02-01 · Event and storage model for the telemetry instrument", fontsize=13, fontweight="bold")
    figure.text(0.02, 0.02,
                "* propensity on Decision and correctness on Consent are not optional: Study 3 dies without logged "
                "propensities, and they cannot\n   be backfilled. Raw pointer samples are never stored — "
                "CursorSegment holds the aggregate only.",
                fontsize=8.4, color="#475569")
    _save(figure, "f02-01_event-schema.png")


# --------------------------------------------------------------- trajectories

def synthetic_path(rng, hesitant: bool, target=(0.82, 0.72), competitor=(0.18, 0.72)):
    """A cursor path from the item stem to the chosen option.

    Direct paths bow slightly; hesitant paths detour toward the competing
    option and reverse direction before committing.
    """
    start = np.array([0.5, 0.08])
    end = np.array(target)
    steps = 60
    t = np.linspace(0, 1, steps)

    if hesitant:
        pull = np.array(competitor)
        control = 0.5 * (start + end) + 0.75 * (pull - 0.5 * (start + end))
        path = ((1 - t) ** 2)[:, None] * start + (2 * (1 - t) * t)[:, None] * control + (t ** 2)[:, None] * end
        path[:, 0] += 0.02 * np.sin(t * 9 * np.pi) * (1 - t)
    else:
        control = 0.5 * (start + end) + np.array([0.05, 0.06])
        path = ((1 - t) ** 2)[:, None] * start + (2 * (1 - t) * t)[:, None] * control + (t ** 2)[:, None] * end

    path += rng.normal(0, 0.004, path.shape)
    return path


def figure_trajectory_examples(seed: int) -> None:
    rng = np.random.default_rng(seed)
    figure, axes_grid = plt.subplots(2, 3, figsize=(12, 7.2), sharex=True, sharey=True)

    for column in range(3):
        for row, hesitant in enumerate((False, True)):
            axes = axes_grid[row][column]
            path = synthetic_path(rng, hesitant)
            axes.plot([0.5, 0.82], [0.08, 0.72], linestyle="--", linewidth=1, color="#94a3b8")
            axes.plot(path[:, 0], path[:, 1], linewidth=1.8,
                      color="#dc2626" if hesitant else "#0f766e")
            axes.scatter(*path[0], s=22, color="#334155", zorder=3)
            for x, label in ((0.18, "B"), (0.82, "A")):
                axes.add_patch(plt.Rectangle((x - 0.13, 0.68), 0.26, 0.09, facecolor="#e2e8f0", edgecolor="#94a3b8"))
                axes.text(x, 0.73, f"option {label}", ha="center", va="center", fontsize=8)
            axes.set_xlim(0, 1)
            axes.set_ylim(0, 0.9)
            axes.set_xticks([])
            axes.set_yticks([])
            axes.set_title(("hesitant" if hesitant else "direct") + f" · example {column + 1}", fontsize=9)

    figure.suptitle("f02-02 · Cursor trajectories: direct (top) vs hesitant (bottom)", fontsize=13, fontweight="bold")
    figure.text(0.5, 0.02, SUBTITLE + "  The dashed line is the ideal straight path to the chosen option.",
                ha="center", fontsize=8.6, color="#475569")
    _save(figure, "f02-02_cursor-trajectory-examples.png")


def figure_feature_definitions(seed: int) -> None:
    rng = np.random.default_rng(seed)
    path = synthetic_path(rng, hesitant=True)
    start, end = path[0], path[-1]

    figure, axes = plt.subplots(figsize=(10.5, 7))
    ideal = np.linspace(start, end, len(path))
    axes.fill(np.concatenate([path[:, 0], ideal[::-1, 0]]),
              np.concatenate([path[:, 1], ideal[::-1, 1]]),
              color="#fca5a5", alpha=0.35, label="auc_toward_nonchosen (area vs the ideal line)")
    axes.plot(ideal[:, 0], ideal[:, 1], "--", color="#94a3b8", linewidth=1.2, label="ideal straight path")
    axes.plot(path[:, 0], path[:, 1], color="#dc2626", linewidth=2, label="observed path (path_ratio = length / straight)")

    direction = end - start
    normal = np.array([-direction[1], direction[0]]) / np.linalg.norm(direction)
    offsets = (path - ideal) @ normal
    peak = int(np.argmax(np.abs(offsets)))
    axes.annotate("", xy=tuple(path[peak]), xytext=tuple(ideal[peak]),
                  arrowprops=dict(arrowstyle="<->", color="#1d4ed8", linewidth=1.6))
    axes.text(*(ideal[peak] + (path[peak] - ideal[peak]) * 0.5 + np.array([0.015, -0.03])),
              "max_deviation", color="#1d4ed8", fontsize=9)

    flips = np.where(np.diff(np.sign(np.diff(path[:, 0]))) != 0)[0] + 1
    axes.scatter(path[flips, 0], path[flips, 1], s=40, facecolor="none", edgecolor="#7c3aed",
                 linewidth=1.4, label="x_flips (direction reversals)", zorder=4)
    pauses = [12, 34]
    axes.scatter(path[pauses, 0], path[pauses, 1], s=60, marker="s", color="#f59e0b",
                 label="pause_count (speed < threshold for > 300 ms)", zorder=4)
    axes.scatter(*start, s=45, color="#334155", zorder=5,
                 label="time_to_first_movement / time_to_first_selection anchor")

    for x, label in ((0.18, "competing option"), (0.82, "chosen option")):
        axes.add_patch(plt.Rectangle((x - 0.13, 0.68), 0.26, 0.09, facecolor="#e2e8f0", edgecolor="#94a3b8"))
        axes.text(x, 0.725, f"{label}\n(hover_time_ms)", ha="center", va="center", fontsize=8)

    axes.set_xlim(0, 1)
    axes.set_ylim(0, 0.92)
    axes.set_xticks([])
    axes.set_yticks([])
    axes.legend(loc="upper center", fontsize=8.2, frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.02))
    axes.set_title("f02-03 · What each pre-registered trajectory feature measures", fontsize=13, fontweight="bold")
    figure.text(0.5, -0.09, SUBTITLE + "  Definitions are fixed in docs/preregistration.md before any collection.",
                ha="center", fontsize=8.6, color="#475569")
    _save(figure, "f02-03_trajectory-feature-definitions.png")


# --------------------------------------------------------------- f02-04

EVENT_COLOURS = {
    "ITEM_PRESENTED": "#0f766e", "FIRST_INTERACTION": "#2563eb", "OPTION_SELECTED": "#7c3aed",
    "OPTION_CHANGED": "#c026d3", "IDLE_ENTERED": "#f59e0b", "IDLE_EXITED": "#f59e0b",
    "VISIBILITY_CHANGED": "#64748b", "CURSOR_SEGMENT": "#059669", "ANSWER_SUBMITTED": "#dc2626",
}


def figure_timeline() -> None:
    request = json.loads(FIXTURE.read_text())
    events = sorted(request["events"], key=lambda event: event["seq"])
    base = datetime.fromisoformat(events[0]["client_ts"].replace("Z", "+00:00"))
    seconds = [(datetime.fromisoformat(event["client_ts"].replace("Z", "+00:00")) - base).total_seconds()
               for event in events]
    types = [event["type"] for event in events]

    figure, axes = plt.subplots(figsize=(12.5, 5.2))
    lanes = list(dict.fromkeys(types))
    lane_index = {name: index for index, name in enumerate(lanes)}

    # Idle and hidden bands first, so the markers sit on top of them.
    idle_pairs = [(seconds[types.index("IDLE_ENTERED")], seconds[types.index("IDLE_EXITED")])] \
        if "IDLE_ENTERED" in types and "IDLE_EXITED" in types else []
    for start, stop in idle_pairs:
        axes.axvspan(start, stop, color="#fef3c7", zorder=0, label="idle (no interaction > 3 s)")
    hidden = [(index, event) for index, event in enumerate(events) if event["type"] == "VISIBILITY_CHANGED"]
    if len(hidden) >= 2:
        axes.axvspan(seconds[hidden[0][0]], seconds[hidden[1][0]], color="#e2e8f0", zorder=0,
                     label="tab hidden / unfocused")

    for second, event_type in zip(seconds, types):
        axes.scatter(second, lane_index[event_type], s=90, zorder=3,
                     color=EVENT_COLOURS.get(event_type, "#334155"))
        axes.annotate(f"{second:.1f}s", (second, lane_index[event_type]), textcoords="offset points",
                      xytext=(0, 10), ha="center", fontsize=7.5, color="#475569")

    axes.set_yticks(range(len(lanes)))
    axes.set_yticklabels(lanes, fontsize=9)
    axes.invert_yaxis()
    axes.set_xlabel("seconds since the item was presented")
    axes.grid(axis="x", linestyle=":", alpha=0.5)
    handles, labels = axes.get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    axes.legend(unique.values(), unique.keys(), loc="upper left", fontsize=8.2, frameon=False)
    axes.set_title("f02-04 · One item attempt as a telemetry timeline", fontsize=13, fontweight="bold")
    figure.text(0.5, -0.02,
                "Recorded event stream from app/features/testdata/attempt-events.json — the same fixture the "
                "conformance test asserts against.",
                ha="center", fontsize=8.6, color="#475569")
    _save(figure, "f02-04_telemetry-timeline.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260821)
    arguments = parser.parse_args()

    figure_event_schema()
    figure_trajectory_examples(arguments.seed)
    figure_feature_definitions(arguments.seed)
    figure_timeline()


if __name__ == "__main__":
    main()
