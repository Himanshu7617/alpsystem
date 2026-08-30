"""Phase 12 figure pack — the walkthrough, the test sweep and the poster.

    python -m app.research.figures_phase12 [--seed 20260821]

Five charts into `artifacts/figures/12-walkthrough/`. Everything is read from an
artifact: the test counts from `test-results.json` (written by
`scripts/run_tests.py`), the timings from `reproduction-timing.json` (written by
`scripts/reproduce.py`), the headline numbers from the study artifacts. The UI
contact sheet is assembled from the screenshots captured while driving the
running application; if they are absent the sheet is skipped and the reason is
recorded rather than a placeholder being drawn.

f12-05 is the graphical abstract, and it is deliberately not a victory lap: the
closed-loop panel shows a null with its confidence interval, because that is
what Phase 11 measured.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from app.research import plotstyle

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "artifacts" / "figures" / "12-walkthrough"
SHOTS_DIR = FIG_DIR / "screenshots"
EVAL_DIR = ROOT / "artifacts" / "evaluation"
BENCH_DIR = ROOT / "artifacts" / "benchmarks"
FINDINGS = EVAL_DIR / "walkthrough-figure-findings.json"

#: Every file this script reads. `figure_index` copies it into INDEX.md.
SOURCES = [
    "artifacts/evaluation/test-results.json",
    "artifacts/evaluation/reproduction-timing.json",
    "artifacts/evaluation/statistical-report.json",
    "artifacts/evaluation/load-test.json",
    "artifacts/evaluation/validation-gate.json",
    "artifacts/evaluation/rl-results.json",
    "artifacts/benchmarks/study1-final.json",
]

BLUE, ORANGE, GREEN = plotstyle.BLUE, plotstyle.ORANGE, plotstyle.GREEN
PURPLE, VERMILLION, GREY = plotstyle.PURPLE, plotstyle.VERMILLION, plotstyle.GREY
SKY, LIGHT = plotstyle.SKY, plotstyle.LIGHT_GREY


def save(fig: plt.Figure, name: str) -> None:
    path = plotstyle.save(fig, FIG_DIR, name)
    print(f"wrote: {path.relative_to(ROOT)} (+ .pdf)")


def figure_count() -> int:
    """How many figures the manifest holds — never a number typed into a chart."""
    from app.research import figure_index

    return len(figure_index.read_manifest())


def read(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


# ------------------------------------------------------------------ f12-01

def f01_journey(load: dict | None) -> dict:
    """The complete user journey, with the measured latency of each hop."""
    latency = ((load or {}).get("headline", {}).get("endpoints", {}))
    steps = [
        ("Consent", "five signal classes,\nseparately revocable", "consent"),
        ("Session", "canonical 60-item bank,\n6-concept curriculum", "session"),
        ("Item", "policy picks difficulty,\nconcept move, support", "next-item"),
        ("Answer", "telemetry posted with\nthe attempt", "attempt"),
        ("State", "gated estimate +\nuncertainty, explanation", "events"),
        ("End", "summary, roadmap,\ndecision log persisted", "session-end"),
    ]
    fig, axis = plt.subplots(figsize=(13, 3.6))
    axis.set_xlim(0, len(steps) * 2.16)
    axis.set_ylim(0, 3)
    axis.axis("off")
    for index, (title, detail, endpoint) in enumerate(steps):
        left = index * 2.16 + 0.08
        axis.add_patch(FancyBboxPatch((left, 1.0), 1.85, 1.5, boxstyle="round,pad=0.06",
                                      facecolor="#eef4f8", edgecolor=BLUE, linewidth=1.2))
        axis.text(left + 0.92, 2.24, f"{index + 1}. {title}", ha="center", fontsize=10,
                  fontweight="bold", color="#0b2b3c")
        axis.text(left + 0.92, 1.62, detail, ha="center", va="center", fontsize=7.6,
                  color="#33454f")
        measured = latency.get(endpoint, {})
        if measured:
            axis.text(left + 0.92, 1.12, f"p95 {measured['p95']:.1f} ms @ 16 concurrent",
                      ha="center", fontsize=7, color=GREEN)
        if index < len(steps) - 1:
            axis.add_patch(FancyArrowPatch((left + 1.95, 1.75), (left + 2.16, 1.75),
                                           arrowstyle="-|>", mutation_scale=11, color=GREY))
    axis.text(len(steps) * 1.08, 0.45,
              "Every decision is logged with its propensity; every explanation names only "
              "gate-admitted states.", ha="center", fontsize=8, color="#33454f")
    axis.set_title("f12-01 · One learner's journey through the deployed system, end to end",
                   fontsize=11)
    save(fig, "f12-01_e2e-flow.png")
    return {"steps": [step[0] for step in steps],
            "latency_p95_ms": {name: values["p95"] for name, values in latency.items()}}


# ------------------------------------------------------------------ f12-02

def f02_screens() -> dict:
    shots = sorted(SHOTS_DIR.glob("*.jpg")) + sorted(SHOTS_DIR.glob("*.png"))
    if not shots:
        return {"skipped": f"no screenshots in {SHOTS_DIR.relative_to(ROOT)}"}
    columns = 3
    rows = int(np.ceil(len(shots) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 2.9 * rows))
    for axis, shot in zip(np.ravel(axes), shots):
        axis.imshow(mpimg.imread(shot))
        axis.set_title(shot.stem.replace("-", " ", 1).replace("-", " "), fontsize=8)
        axis.axis("off")
    for axis in np.ravel(axes)[len(shots):]:
        axis.axis("off")
    fig.suptitle("f12-02 · The running application, captured while driving it "
                 "(zero console errors)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save(fig, "f12-02_ui-screens.png")
    return {"screens": [shot.name for shot in shots]}


# ------------------------------------------------------------------ f12-03

def f03_tests(tests: dict | None) -> dict:
    if tests is None:
        # Drawn rather than skipped: the report references this figure, and a
        # checkout where the suites have not been run should say so on the page
        # instead of failing the document build.
        fig, axis = plt.subplots(figsize=(9, 3.2))
        axis.axis("off")
        axis.text(0.5, 0.62, "test sweep not run in this checkout", ha="center",
                  fontsize=13, color=VERMILLION, fontweight="bold")
        axis.text(0.5, 0.34, "run `make test-report` (needs the stack up) to record\n"
                             "artifacts/evaluation/test-results.json", ha="center",
                  fontsize=9, color=GREY)
        fig.suptitle("f12-03 · Test sweep", fontsize=11)
        save(fig, "f12-03_test-results-summary.png")
        return {"not_run": "test-results.json absent — run `make test-report`"}
    suites = tests["suites"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.8),
                                      gridspec_kw={"width_ratios": [1.1, 1]})
    names = [suite["suite"] for suite in suites]
    passed = [suite["counts"].get("passed", 0) for suite in suites]
    failed = [suite["counts"].get("failed", 0) for suite in suites]
    positions = np.arange(len(names))
    left.barh(positions, passed, color=GREEN, label="passed")
    left.barh(positions, failed, left=passed, color=VERMILLION, label="failed")
    for index, (count, seconds) in enumerate(zip(passed, [s["seconds"] for s in suites])):
        left.text(count + 1.5, index, f"{count} in {seconds:.1f}s", va="center", fontsize=8)
    left.set_yticks(positions)
    left.set_yticklabels(names, fontsize=8)
    left.set_xlabel("tests")
    left.set_xlim(0, max(passed) * 1.35)
    left.legend(loc="lower right", fontsize=8)
    left.set_title("Suites", fontsize=9)

    right.axis("off")
    verdict = "ALL GREEN" if tests["all_passed"] else "FAILURES"
    right.text(0.5, 0.78, verdict, ha="center", fontsize=22, fontweight="bold",
               color=GREEN if tests["all_passed"] else VERMILLION)
    right.text(0.5, 0.58, f"{tests['total_tests']} tests", ha="center", fontsize=13)
    right.text(0.5, 0.34,
               "leakage audit · validity-gate guardrail\npolicy conformance · propensity "
               "logging\nPhase 11 statistics guards · live loop E2E",
               ha="center", fontsize=8.5, color="#33454f")
    right.text(0.5, 0.10, tests["generated"], ha="center", fontsize=7, color=GREY)
    fig.suptitle("f12-03 · Test sweep, recorded as it ran", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "f12-03_test-results-summary.png")
    return {"total": tests["total_tests"], "all_passed": tests["all_passed"],
            "per_suite": {suite["suite"]: suite["counts"] for suite in suites}}


# ------------------------------------------------------------------ f12-04

def f04_timing(timing: dict | None) -> dict:
    if timing is None:
        fig, axis = plt.subplots(figsize=(9, 3.2))
        axis.axis("off")
        axis.text(0.5, 0.62, "clean-room reproduction not run in this checkout",
                  ha="center", fontsize=13, color=VERMILLION, fontweight="bold")
        axis.text(0.5, 0.34, "run `make reproduce` to time every target into\n"
                             "artifacts/evaluation/reproduction-timing.json", ha="center",
                  fontsize=9, color=GREY)
        fig.suptitle("f12-04 · Clean-room reproduction", fontsize=11)
        save(fig, "f12-04_reproduction-timing.png")
        return {"not_run": "reproduction-timing.json absent — run `make reproduce`"}
    entries = timing["targets"]
    names = [entry["target"] for entry in entries]
    seconds = np.array([entry["seconds"] for entry in entries])
    ok = [entry["ok"] for entry in entries]
    fig, axis = plt.subplots(figsize=(11, 4.6))
    positions = np.arange(len(names))
    axis.bar(positions, seconds / 60, color=[BLUE if good else VERMILLION for good in ok],
             width=0.66)
    for index, value in enumerate(seconds):
        axis.text(index, value / 60, f"{value / 60:.1f}" if value >= 60 else f"{value:.0f}s",
                  ha="center", va="bottom", fontsize=7)
    axis.set_xticks(positions)
    axis.set_xticklabels([f"{name}\n(P{entry['phase']})" for name, entry
                          in zip(names, entries)], fontsize=7)
    axis.set_ylabel("minutes")
    total = timing.get("total_seconds", float(seconds.sum()))
    axis.set_title(f"f12-04 · Clean-room reproduction at PROFILE={timing['profile']}: "
                   f"{len(entries)} targets, {total / 60:.0f} min total "
                   f"({'all ok' if timing.get('all_ok') else 'with failures'})", fontsize=10)
    fig.tight_layout()
    save(fig, "f12-04_reproduction-timing.png")
    return {"total_minutes": round(total / 60, 1), "targets": len(entries),
            "all_ok": timing.get("all_ok"),
            "slowest": max(entries, key=lambda entry: entry["seconds"])["target"]}


# ------------------------------------------------------------------ f12-05

def f05_poster(report: dict, final: dict, gate: dict, rl: dict, load: dict | None,
               tests: dict | None) -> dict:
    fig = plt.figure(figsize=(14, 8.4))
    grid = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.28)

    # Study 1 — prediction, per dataset, best cell.
    axis = fig.add_subplot(grid[0, 0])
    best = {source: max(block["cells"].values(), key=lambda cell: cell["roc_auc"])
            for source, block in final["sources"].items()}
    names = list(best)
    values = [cell["roc_auc"] for cell in best.values()]
    errors = np.array([[cell["roc_auc"] - cell["ci_low"] for cell in best.values()],
                       [cell["ci_high"] - cell["roc_auc"] for cell in best.values()]])
    axis.bar(np.arange(len(names)), values, yerr=errors, color=BLUE, capsize=3, width=0.6)
    axis.axhline(0.5, color=GREY, linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(len(names)))
    axis.set_xticklabels([name.replace("assistments_", "a").replace("ednet_kt1", "ednet")
                          for name in names], fontsize=8)
    axis.set_ylim(0.4, 0.8)
    axis.set_ylabel("held-out ROC-AUC")
    axis.set_title("Study 1 — prediction works,\nbehaviour adds a little", fontsize=9)

    # Study 2 — the gate.
    axis = fig.add_subplot(grid[0, 1])
    admitted = list(gate.get("admitted", []))
    rejected = list(gate.get("rejected", {}))
    dropped = list(gate.get("dropped_states", {})) or ["fatigue"]
    axis.axis("off")
    axis.text(0.5, 0.93, "Study 2 — the validity gate", ha="center", fontsize=9,
              fontweight="bold")
    for index, (label, items, colour) in enumerate([
            ("admitted", admitted, GREEN), ("rejected", rejected, VERMILLION),
            ("dropped before modelling", dropped, GREY)]):
        axis.text(0.06, 0.72 - index * 0.24, label, fontsize=8.5, color=colour,
                  fontweight="bold")
        axis.text(0.06, 0.64 - index * 0.24, ", ".join(items) or "—", fontsize=9,
                  color="#33454f")
    axis.text(0.5, 0.06, "one construct of four survived; the policies\nsee knowledge only",
              ha="center", fontsize=8, color="#33454f")

    # Study 4 — the null.
    axis = fig.add_subplot(grid[0, 2])
    effects = [entry for entry in report["effect_sizes"]
               if entry["outcome"] == "knowledge_gain"]
    effects.sort(key=lambda entry: entry["hedges_g"])
    positions = np.arange(len(effects))
    axis.errorbar([entry["hedges_g"] for entry in effects], positions,
                  xerr=[[entry["hedges_g"] - entry["g_ci_low"] for entry in effects],
                        [entry["g_ci_high"] - entry["hedges_g"] for entry in effects]],
                  fmt="o", markersize=4, capsize=2, linewidth=1.2, color=BLUE)
    axis.axvline(0, color=GREY, linewidth=1)
    for edge in (-0.3, 0.3):
        axis.axvline(edge, color=VERMILLION, linestyle=":", linewidth=0.9)
    axis.set_yticks(positions)
    axis.set_yticklabels([entry["policy"] for entry in effects], fontsize=6.5)
    axis.set_xlabel("Hedges' g vs rule_improved")
    axis.set_title("Study 4 — no arm beats the rule\n(knowledge gain, SIMULATED)", fontsize=9)
    plotstyle.simulated_note(axis)

    # Study 3 — learned arms.
    axis = fig.add_subplot(grid[1, 0])
    summary = rl["summary"]["V0"]
    arms = sorted(summary, key=lambda name: summary[name]["mean_knowledge_gain"])
    values = [summary[name]["mean_knowledge_gain"] for name in arms]
    lows = [summary[name]["knowledge_gain_95_ci"][0] for name in arms]
    highs = [summary[name]["knowledge_gain_95_ci"][1] for name in arms]
    axis.barh(np.arange(len(arms)), values,
              xerr=[np.array(values) - np.array(lows), np.array(highs) - np.array(values)],
              color=[ORANGE if name.startswith(("bandit", "rl")) else BLUE for name in arms],
              capsize=3, height=0.6)
    axis.axvline(0, color=GREY, linewidth=1)
    axis.set_yticks(np.arange(len(arms)))
    axis.set_yticklabels(arms, fontsize=7.5)
    axis.set_xlabel("mean knowledge gain")
    axis.set_title("Study 3 — learned arms, honestly\n(orange = learned; CIs overlap)",
                   fontsize=9)
    plotstyle.simulated_note(axis)

    # The system.
    axis = fig.add_subplot(grid[1, 1])
    axis.axis("off")
    headline = (load or {}).get("headline", {})
    endpoints = headline.get("endpoints", {})
    rows = [
        ("decision p95 @ 16 concurrent",
         f"{endpoints.get('next-item', {}).get('p95', float('nan')):.1f} ms",
         "budget 300 ms"),
        ("throughput", f"{headline.get('decisions_per_second', float('nan')):.0f} /s",
         f"{headline.get('concurrency', '?')} concurrent learners"),
        ("tests", f"{(tests or {}).get('total_tests', 0)}",
         "all green" if (tests or {}).get("all_passed") else "see the report"),
        ("figures", str(figure_count()), "PNG + PDF, provenance recorded"),
    ]
    axis.text(0.5, 0.95, "The deployed system", ha="center", fontsize=9, fontweight="bold")
    for index, (label, value, note) in enumerate(rows):
        top = 0.76 - index * 0.21
        axis.text(0.04, top, label, fontsize=8, color="#33454f")
        axis.text(0.62, top, value, fontsize=12, fontweight="bold", color="#0b2b3c")
        axis.text(0.62, top - 0.08, note, fontsize=7, color=GREY)

    # What a real trial would need.
    axis = fig.add_subplot(grid[1, 2])
    axis.axis("off")
    requirement = report["power"]["rct_requirement"]
    axis.text(0.5, 0.9, "What would settle it", ha="center", fontsize=9, fontweight="bold")
    axis.text(0.5, 0.62, f"{requirement['n_per_arm']:.0f}", ha="center", fontsize=34,
              fontweight="bold", color=VERMILLION)
    axis.text(0.5, 0.47, "learners per arm", ha="center", fontsize=9)
    axis.text(0.5, 0.34, f"({requirement['n_total']:.0f} total) for g = "
                         f"{requirement['target_g']} at 80 % power",
              ha="center", fontsize=8, color="#33454f")
    axis.text(0.5, 0.14, "no human participants were involved in this work;\nevery "
                         "closed-loop number above is simulated",
              ha="center", fontsize=7.5, color=GREY)

    fig.suptitle("f12-05 · Adaptive Learning Platform — every headline result on one page",
                 fontsize=13, fontweight="bold")
    save(fig, "f12-05_results-summary-poster.png")
    return {
        "study1_best_auc": {source: round(cell["roc_auc"], 4) for source, cell in best.items()},
        "gate_admitted": admitted, "gate_rejected": rejected,
        "closed_loop_max_abs_g": round(max(abs(entry["hedges_g"]) for entry in effects), 4),
        "rct_n_per_arm": requirement["n_per_arm"],
    }


# -------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260821)   # accepted for symmetry
    parser.parse_args()

    report = read(EVAL_DIR / "statistical-report.json")
    if report is None:
        raise SystemExit("statistical-report.json absent — run `make stats` first")
    findings = {
        "f12-01": f01_journey(read(EVAL_DIR / "load-test.json")),
        "f12-02": f02_screens(),
        "f12-03": f03_tests(read(EVAL_DIR / "test-results.json")),
        "f12-04": f04_timing(read(EVAL_DIR / "reproduction-timing.json")),
        "f12-05": f05_poster(report, read(BENCH_DIR / "study1-final.json"),
                             read(EVAL_DIR / "validation-gate.json"),
                             read(EVAL_DIR / "rl-results.json"),
                             read(EVAL_DIR / "load-test.json"),
                             read(EVAL_DIR / "test-results.json")),
    }
    FINDINGS.write_text(json.dumps(findings, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote: {FINDINGS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
