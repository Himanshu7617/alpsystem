"""Phase 10 figure pack — the deployed system, measured rather than drawn.

    python -m app.research.figures_phase10 [--dashboard http://127.0.0.1:4000]

Five of the six charts are generated here. Every latency number comes from
`artifacts/evaluation/load-test.json` (written by `scripts/loadtest.py`) and the
live trace comes from the running backend's dashboard JSON, so no figure states
a number this repository cannot reproduce. `f10-05` is a contact sheet of real
browser screenshots and is assembled by this script from the PNGs captured
under `artifacts/figures/10-system/screenshots/`; if they are absent the sheet
is skipped and the reason is recorded rather than a placeholder being drawn.

Whatever a figure computes that is not already in an artifact is written to
`system-figure-findings.json`, so no document has to quote a picture.
"""
from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from app.research import plotstyle

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "artifacts" / "figures" / "10-system"
SHOTS_DIR = FIG_DIR / "screenshots"
EVAL_DIR = ROOT / "artifacts" / "evaluation"
LOAD_PATH = EVAL_DIR / "load-test.json"
FINDINGS = EVAL_DIR / "system-figure-findings.json"

#: BUILD.md Phase 10 acceptance: the decision path must stay under this.
LATENCY_BUDGET_MS = 300

BLUE, GREEN, RED, GREY = "#0072B2", "#009E73", "#D55E00", "#8c8c8c"


#: Every file this script reads. `figure_index` copies it into INDEX.md.
SOURCES = [
    "artifacts/evaluation/load-test.json",
    "artifacts/evaluation/system-figure-findings.json",
]


def save(fig: plt.Figure, name: str) -> None:
    path = plotstyle.save(fig, FIG_DIR, name)
    print(f"wrote: {path.relative_to(ROOT)} (+ .pdf)")


def box(axis, x, y, width, height, text, colour=BLUE, alpha=0.12):
    axis.add_patch(plt.Rectangle((x, y), width, height, facecolor=colour, alpha=alpha,
                                 edgecolor=colour, linewidth=1.4, zorder=2))
    axis.text(x + width / 2, y + height / 2, text, ha="center", va="center",
              fontsize=8.5, zorder=3)


def arrow(axis, start, end, label="", colour=GREY):
    axis.annotate("", xy=end, xytext=start,
                  arrowprops=dict(arrowstyle="->", color=colour, linewidth=1.2))
    if label:
        axis.text((start[0] + end[0]) / 2, (start[1] + end[1]) / 2 + 0.12, label,
                  ha="center", fontsize=7.5, color="#333")


# ------------------------------------------------------------------- figures

def f01_architecture(serving: dict) -> dict:
    """The deployed topology, with what each process owns."""
    fig, axis = plt.subplots(figsize=(11, 5.6))
    axis.set_xlim(0, 12)
    axis.set_ylim(0, 7)
    axis.axis("off")

    box(axis, 0.3, 4.4, 2.6, 1.6, "Browser\nfiles/ — telemetry.js\nitem, probes, state panel", BLUE)
    box(axis, 3.6, 4.4, 3.0, 1.6,
        "backend (Express)\n/v1 consent · sessions · next-item\nevents · attempts", GREEN)
    box(axis, 7.4, 4.4, 3.4, 1.6,
        f"ml-service (FastAPI)\n/v1/features/extract · /v1/state\n/v1/decide · /model-info", RED)

    box(axis, 3.6, 2.2, 3.0, 1.4,
        "Postgres\nSession · ItemAttempt · InteractionEvent\nStateEstimate · Decision · Probe", GREEN)
    box(axis, 7.4, 2.2, 3.4, 1.4,
        "artifacts/models/\nstate-encoders-by-rung.pt\nbandit_*.npz · rl/policy.zip", RED)
    box(axis, 0.3, 2.2, 2.6, 1.4, "Researcher dashboard\n/research/dashboard", GREY)

    box(axis, 3.6, 0.4, 7.2, 1.1,
        "research pipeline — same modules: app.policy.* (arms, gate, estimator), "
        "app.features.extract (one feature definition)", "#CC79A7")

    arrow(axis, (2.9, 5.2), (3.6, 5.2), "events, attempts")
    arrow(axis, (6.6, 5.4), (7.4, 5.4), "features, state, decide")
    arrow(axis, (7.4, 4.9), (6.6, 4.9), "action + propensity")
    arrow(axis, (5.1, 4.4), (5.1, 3.6), "persist")
    arrow(axis, (9.1, 4.4), (9.1, 3.6), "load once")
    arrow(axis, (3.6, 2.9), (2.9, 2.9), "read-only")
    arrow(axis, (9.1, 1.5), (9.1, 2.2))
    arrow(axis, (5.1, 1.5), (5.1, 2.2))

    policy = serving.get("policy", "unknown")
    admitted = ", ".join(serving.get("gate", {}).get("admitted", [])) or "none"
    axis.set_title("f10-01 — deployed topology "
                   f"(serving {policy}; states admitted by the gate: {admitted})",
                   fontsize=10.5)
    save(fig, "f10-01_system-architecture.png")
    return {"policy": policy, "admitted_states": admitted}


def f02_sequence(load: dict) -> dict:
    """One item attempt end to end, with the measured median at each hop."""
    endpoints = load["headline"]["endpoints"]
    steps = [
        ("browser", "backend", "GET /v1/next-item", endpoints.get("next-item", {}).get("p50")),
        ("backend", "ml-service", "POST /v1/decide", None),
        ("ml-service", "backend", "action + propensity", None),
        ("backend", "browser", "item + explanation", None),
        ("browser", "backend", "POST /v1/events", endpoints.get("events", {}).get("p50")),
        ("browser", "backend", "POST /v1/attempts", endpoints.get("attempt", {}).get("p50")),
        ("backend", "ml-service", "POST /v1/features/extract", None),
        ("backend", "ml-service", "POST /v1/state", None),
        ("ml-service", "backend", "gated state + uncertainty", None),
        ("backend", "browser", "result + estimate", None),
    ]
    lanes = {"browser": 1.0, "backend": 3.0, "ml-service": 5.0}

    fig, axis = plt.subplots(figsize=(10, 6.4))
    axis.set_xlim(0.2, 6.4)
    axis.set_ylim(-len(steps) - 0.8, 1.2)
    axis.axis("off")
    for name, x in lanes.items():
        axis.text(x, 0.7, name, ha="center", fontsize=10, fontweight="bold")
        axis.plot([x, x], [0.4, -len(steps) - 0.4], color="#d0d4da", linewidth=1, zorder=1)

    for index, (source, target, label, latency) in enumerate(steps):
        y = -index - 0.6
        axis.annotate("", xy=(lanes[target], y), xytext=(lanes[source], y),
                      arrowprops=dict(arrowstyle="->", color=BLUE if latency else GREY,
                                      linewidth=1.4 if latency else 1.0))
        text = label if latency is None else f"{label} — median {latency:.1f} ms"
        axis.text((lanes[source] + lanes[target]) / 2, y + 0.14, text,
                  ha="center", fontsize=8)

    axis.set_title("f10-02 — one item attempt, end to end\n"
                   "(measured medians are the client-observed times; the internal hops "
                   "are inside them)", fontsize=10.5)
    save(fig, "f10-02_request-sequence-diagram.png")
    return {"medians_ms": {name: stats.get("p50") for name, stats in endpoints.items()}}


def f03_latency(load: dict) -> dict:
    """p50/p95/p99 per endpoint at the busiest concurrency, against the budget."""
    run = load["headline"]
    endpoints = run["endpoints"]
    names = [name for name in ("next-item", "attempt", "events", "session", "consent", "session-end")
             if name in endpoints]
    positions = np.arange(len(names))
    width = 0.26

    fig, axis = plt.subplots(figsize=(9, 4.4))
    for offset, key, colour in ((-width, "p50", GREEN), (0.0, "p95", BLUE), (width, "p99", RED)):
        values = [endpoints[name][key] for name in names]
        axis.bar(positions + offset, values, width, label=key.upper(), color=colour)

    axis.axhline(LATENCY_BUDGET_MS, color=RED, linestyle="--", linewidth=1.2)
    axis.text(len(names) - 0.5, LATENCY_BUDGET_MS * 1.05,
              f"budget {LATENCY_BUDGET_MS} ms", ha="right", fontsize=8, color=RED)
    axis.set_yscale("log")
    axis.set_xticks(positions)
    axis.set_xticklabels(names, rotation=15)
    axis.set_ylabel("latency (ms, log scale)")
    axis.legend(frameon=False, ncol=3)
    axis.set_title(f"f10-03 — endpoint latency at concurrency {run['concurrency']} "
                   f"({run['decisions']} decisions)", fontsize=10.5)
    save(fig, "f10-03_latency-distribution.png")
    decision = endpoints["next-item"]
    return {"decision_p95_ms": decision["p95"], "decision_p99_ms": decision["p99"],
            "budget_ms": LATENCY_BUDGET_MS, "within_budget": decision["p95"] < LATENCY_BUDGET_MS}


def f04_throughput(load: dict) -> dict:
    """Throughput and decision latency as concurrency rises."""
    runs = sorted(load["runs"], key=lambda run: run["concurrency"])
    concurrency = [run["concurrency"] for run in runs]
    throughput = [run["decisions_per_second"] for run in runs]
    p95 = [run["endpoints"]["next-item"]["p95"] for run in runs]

    fig, axis = plt.subplots(figsize=(8, 4.4))
    axis.plot(concurrency, throughput, marker="o", color=GREEN, label="decisions / second")
    axis.set_xlabel("concurrent learners")
    axis.set_ylabel("decisions / second", color=GREEN)
    axis.set_xticks(concurrency)

    twin = axis.twinx()
    twin.plot(concurrency, p95, marker="s", color=BLUE, label="decision p95")
    twin.axhline(LATENCY_BUDGET_MS, color=RED, linestyle="--", linewidth=1.0)
    twin.set_ylabel("decision p95 (ms)", color=BLUE)
    twin.set_ylim(0, max(LATENCY_BUDGET_MS * 1.1, max(p95) * 1.3))

    axis.set_title("f10-04 — throughput and decision latency against concurrency", fontsize=10.5)
    save(fig, "f10-04_load-test-throughput.png")
    return {"concurrency": concurrency, "decisions_per_second": throughput,
            "decision_p95_ms": p95}


def f05_screenshots() -> dict:
    """Contact sheet of the real UI. Skipped, loudly, when nothing was captured."""
    shots = sorted(path for path in SHOTS_DIR.glob("*")
                   if path.suffix.lower() in (".png", ".jpg", ".jpeg"))
    if not shots:
        print(f"skipped f10-05: no screenshots in {SHOTS_DIR.relative_to(ROOT)}")
        return {"captured": 0, "note": "no screenshots captured; run the browser session first"}

    columns = min(3, len(shots))
    rows = int(np.ceil(len(shots) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.6 * columns, 3.2 * rows))
    for axis, shot in zip(np.atleast_1d(axes).ravel(), shots):
        axis.imshow(mpimg.imread(shot))
        axis.set_title(shot.stem.replace("-", " "), fontsize=8)
        axis.axis("off")
    for axis in np.atleast_1d(axes).ravel()[len(shots):]:
        axis.axis("off")
    fig.suptitle("f10-05 — the learner-facing loop, captured from the running platform",
                 fontsize=11)
    save(fig, "f10-05_ui-screenshots.png")
    return {"captured": len(shots), "files": [shot.name for shot in shots]}


def f06_live_trace(dashboard: dict) -> dict:
    """A real session: the knowledge estimate with its uncertainty, decisions overlaid."""
    sessions = [session for session in dashboard.get("sessions", [])
                if len(session.get("estimates", [])) >= 5]
    if not sessions:
        print("skipped f10-06: no session with at least five estimates")
        return {"note": "no live session long enough to trace"}
    session = max(sessions, key=lambda entry: len(entry["estimates"]))
    estimates = session["estimates"]
    knowledge = np.array([row["knowledge"] for row in estimates], dtype=float)
    error = np.array([row.get("knowledge_se") or 0.0 for row in estimates], dtype=float)
    steps = np.arange(1, len(knowledge) + 1)

    # The page's decision list is the latest hundred across every session, so
    # the trace asks for this session's own log rather than plotting a slice.
    focused = fetch(f"{dashboard.get('url')}/research/dashboard?format=json"
                    f"&session_id={session['session_id']}") or {}
    decisions = sorted(focused.get("decisions", []),
                       key=lambda decision: decision["created_at"])
    difficulty = [decision["action"].get("difficulty") for decision in decisions]
    difficulty = [value for value in difficulty if isinstance(value, (int, float))]

    fig, axis = plt.subplots(figsize=(9.5, 4.6))
    axis.plot(steps, knowledge, marker="o", color=BLUE, label="knowledge estimate")
    axis.fill_between(steps, knowledge - error, knowledge + error, color=BLUE, alpha=0.18,
                      label="± MC-dropout standard error")
    axis.set_xlabel("item")
    axis.set_ylabel("P(next item correct)")
    axis.set_ylim(0, 1)

    twin = axis.twinx()
    twin.step(np.arange(1, len(difficulty) + 1), difficulty, where="mid", color=GREEN,
              alpha=0.8, label="difficulty served")
    twin.set_ylabel("difficulty bin served", color=GREEN)
    twin.set_ylim(0, 11)

    handles, labels = axis.get_legend_handles_labels()
    extra = twin.get_legend_handles_labels()
    axis.legend(handles + extra[0], labels + extra[1], frameon=False, fontsize=8, loc="lower right")

    rejected = ", ".join(dashboard.get("serving", {}).get("gate", {}).get("rejected", {}))
    axis.set_title("f10-06 — a live session, as the platform recorded it\n"
                   f"only the admitted state is plotted; withheld: {rejected or 'none'}",
                   fontsize=10.5)
    save(fig, "f10-06_live-session-state-trace.png")
    return {"session_id": session["session_id"], "items": len(knowledge),
            "knowledge_start": round(float(knowledge[0]), 4),
            "knowledge_end": round(float(knowledge[-1]), 4),
            "difficulty_served": difficulty}


def fetch(url: str) -> dict | None:
    """GET the dashboard JSON, from the host or from inside the compose network.

    `make system` runs this in the ml-service container when one is up, where
    the host's 127.0.0.1:4000 is the container itself; the compose service name
    is tried as well so the same command works either way.
    """
    candidates = [url]
    for local in ("127.0.0.1", "localhost"):
        if local in url:
            candidates.append(url.replace(local, "backend"))
    last = None
    for candidate in candidates:
        try:
            with urllib.request.urlopen(candidate, timeout=10) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            last = error
    print(f"dashboard unreachable ({last}); f10-06 needs a running backend")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dashboard", default="http://127.0.0.1:4000")
    parser.add_argument("--seed", type=int, default=20260821)   # accepted for symmetry
    arguments = parser.parse_args()

    if not LOAD_PATH.exists():
        raise SystemExit(f"{LOAD_PATH.relative_to(ROOT)} does not exist. "
                         "Run `make loadtest` against a running stack first.")
    load = json.loads(LOAD_PATH.read_text(encoding="utf-8"))
    dashboard = fetch(f"{arguments.dashboard}/research/dashboard?format=json&limit=25") or {}
    dashboard.setdefault("url", arguments.dashboard)

    findings = {
        "f10-01": f01_architecture(load.get("serving") or dashboard.get("serving") or {}),
        "f10-02": f02_sequence(load),
        "f10-03": f03_latency(load),
        "f10-04": f04_throughput(load),
        "f10-05": f05_screenshots(),
        "f10-06": f06_live_trace(dashboard),
    }
    FINDINGS.write_text(json.dumps(findings, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote: {FINDINGS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
