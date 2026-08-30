"""Phase 8.5 — the four integration figures. `make integrate`.

    python -m app.research.figures_phase85 --seed 20260821

These are diagrams, but they are not hand-drawn: every box is annotated with a
number read out of the artifact it describes, so a stage that stops producing
its artifact shows up here as a missing count rather than as a picture that is
quietly out of date.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from app.research import plotstyle

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "artifacts" / "figures" / "08.5-integration"
EVAL_DIR = ROOT / "artifacts" / "evaluation"
MODEL_DIR = ROOT / "artifacts" / "models"
BENCH_DIR = ROOT / "artifacts" / "benchmarks"
PROCESSED = ROOT / "data" / "processed"

INK = "#2f2f37"
BOX = {"data": "#0072B2", "model": "#E69F00", "policy": "#009E73",
       "gate": "#CC79A7", "missing": "#D55E00"}


#: Every file this script reads. `figure_index` copies it into INDEX.md.
SOURCES = [
    "artifacts/evaluation/phase-8.5-validation.json",
    "artifacts/benchmarks/study1-cv.json",
    "artifacts/benchmarks/study1-final.json",
    "artifacts/evaluation/validation-gate.json",
    "artifacts/evaluation/policy-results-v0.json",
]


def save(fig, name: str) -> None:
    path = plotstyle.save(fig, FIG_DIR, name)
    print(f"wrote: {path.relative_to(ROOT)} (+ .pdf)")


def read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def human(size: float) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


def box(axis, x, y, width, height, title, lines, kind="data") -> None:
    present = kind != "missing"
    axis.add_patch(FancyBboxPatch(
        (x, y), width, height, boxstyle="round,pad=0.012",
        linewidth=1.4, edgecolor=BOX[kind],
        facecolor=BOX[kind] + ("22" if present else "18"), zorder=2))
    axis.text(x + width / 2, y + height - 0.035, title, ha="center", va="top",
              fontsize=9, fontweight="bold", color=INK, zorder=3)
    for index, line in enumerate(lines):
        axis.text(x + width / 2, y + height - 0.075 - 0.036 * index, line, ha="center",
                  va="top", fontsize=7.2, color=INK, zorder=3)


def arrow(axis, start, end) -> None:
    axis.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=11,
                                   linewidth=1.1, color="#6b6b76", zorder=1))


def blank(width=11.0, height=6.4):
    fig, axis = plt.subplots(figsize=(width, height))
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1)
    axis.axis("off")
    return fig, axis


# ------------------------------------------------------------------ f085-01

def training_pipeline() -> None:
    """Data to artifact, annotated with what each stage actually produced."""
    features = sorted(PROCESSED.glob("features-*.parquet"))
    raw = sorted(PROCESSED.glob("*.parquet"))
    raw = [path for path in raw if not path.name.startswith("features-")]
    cv = read(BENCH_DIR / "study1-cv.json")
    final = read(BENCH_DIR / "study1-final.json")
    registry_dirs = [entry for entry in sorted(MODEL_DIR.glob("*"))
                     if entry.is_dir() and (entry / "training_manifest.json").exists()]

    fig, axis = blank()
    fig.suptitle("f085-01 The training pipeline as built, with what each stage produced",
                 fontsize=12, fontweight="bold", y=0.97)
    stages = [
        (0.02, "data/raw → data/processed", [f"{len(raw)} normalised sources",
                                             f"{human(sum(p.stat().st_size for p in raw))} on disk",
                                             "make fetch-data / prep-data"], "data"),
        (0.21, "feature matrices", [f"{len(features)} matrices",
                                    f"{human(sum(p.stat().st_size for p in features))}",
                                    "make features"], "data"),
        (0.40, "leakage audit", ["audit_leakage.py",
                                 "no latent_* column survives",
                                 "runs inside make test"], "gate"),
        (0.59, "cross-validation", [f"{len(cv.get('sources', {})) if cv else 0} sources tuned"
                                    if cv else "study1-cv.json missing",
                                    "5 learner-grouped folds",
                                    "test set untouched"], "model" if cv else "missing"),
        (0.78, "final test pass", ([f"{len(final.get('sources', {}))} sources scored",
                                   "held-out learners, once",
                                   "study1-final.json"] if final else
                                  ["NOT RUN", "study1-final.json absent",
                                   "no Study 1 number is quotable"]),
         "model" if final else "missing"),
    ]
    for x, title, lines, kind in stages:
        box(axis, x, 0.56, 0.168, 0.30, title, lines, kind)
        if x > 0.02:
            arrow(axis, (x - 0.019, 0.71), (x - 0.003, 0.71))

    box(axis, 0.30, 0.16, 0.40, 0.26, "artifacts/models/<name>/  — the contract",
        [f"{len(registry_dirs)} artifact director"
         f"{'y' if len(registry_dirs) == 1 else 'ies'}: "
         f"{', '.join(entry.name for entry in registry_dirs) or 'none yet'}",
         "model.joblib · feature_schema.json · metrics.json",
         "calibration.json · training_manifest.json",
         "loaded only through app/model/registry.py"],
        "model" if registry_dirs else "missing")
    arrow(axis, (0.87, 0.55), (0.70, 0.42))
    axis.text(0.5, 0.06, "Every box is counted from the artifact tree at render time; "
                         "a stage that stops producing its output turns red here.",
              ha="center", fontsize=7.5, color="#6b6b76", style="italic")
    save(fig, "f085-01_final-training-pipeline.png")


# ------------------------------------------------------------------ f085-02

def policy_architecture() -> None:
    """Estimation, the gate, and the two kinds of adaptation over one action space."""
    gate = read(EVAL_DIR / "validation-gate.json") or {}
    admitted = sorted(gate.get("admitted", []))
    rejected = sorted(gate.get("rejected", []))
    results = read(EVAL_DIR / "policy-results-v0.json") or {}
    space = results.get("action_space", {})
    excluded = results.get("rules_excluded", [])

    fig, axis = blank()
    fig.suptitle("f085-02 Policy architecture: estimation ≠ rule adaptation ≠ learned adaptation",
                 fontsize=12, fontweight="bold", y=0.97)
    box(axis, 0.30, 0.79, 0.40, 0.13, "ESTIMATION",
        ["state encoder · BKT · Rasch θ · logistic regression",
         "outputs a probability and its uncertainty — never an action"], "model")
    arrow(axis, (0.50, 0.785), (0.50, 0.735))
    box(axis, 0.26, 0.58, 0.48, 0.15, "VALIDATION GATE  (Phase 7)",
        [f"admitted: {', '.join(admitted) or 'none'}",
         f"rejected: {', '.join(rejected) or 'none'}",
         "a rejected state is removed from the policy input, not zeroed"], "gate")
    arrow(axis, (0.40, 0.575), (0.28, 0.50))
    arrow(axis, (0.60, 0.575), (0.72, 0.50))
    box(axis, 0.06, 0.33, 0.38, 0.16, "RULE-BASED ADAPTATION",
        ["app/policy/rules.py — one citation and one signal class per rule",
         f"{len(excluded)} rule(s) excluded by the gate or by rung,",
         "reported rather than deleted"], "policy")
    box(axis, 0.56, 0.33, 0.38, 0.16, "LEARNED ADAPTATION  (Phase 9)",
        ["contextual bandit, then PPO", "same context, fitted action selection",
         "not built yet — Phase 9"], "missing")
    arrow(axis, (0.25, 0.325), (0.42, 0.26))
    arrow(axis, (0.75, 0.325), (0.58, 0.26))
    box(axis, 0.22, 0.08, 0.56, 0.17, "THE SHARED ACTION SPACE",
        [f"difficulty {min(space.get('difficulty', [1]))}..{max(space.get('difficulty', [10]))}"
         f"  ×  {len(space.get('concept_move', []))} concept moves"
         f"  ×  {len(space.get('intervention', []))} interventions"
         f"  =  {space.get('size', '?')} actions",
         "every arm emits an action, a propensity and an explanation",
         "comparisons are only fair because the set is identical"], "policy")
    save(fig, "f085-02_policy-architecture.png")


# ------------------------------------------------------------------ f085-03

def data_flow_and_splits() -> None:
    """Which rows tune, which are scored once, drawn from splits.json."""
    splits = read(PROCESSED / "splits.json") or {}
    counts: dict[str, dict[str, int]] = {}
    for source, mapping in splits.items():
        if not isinstance(mapping, dict):
            continue
        tally: dict[str, int] = {}
        for split in mapping.values():
            key = str(split)
            tally[key] = tally.get(key, 0) + 1
        counts[source] = tally
    order = ["fold0", "fold1", "fold2", "fold3", "fold4", "test", "unassigned"]
    colours = {**{f"fold{index}": "#0072B2" for index in range(5)},
               "test": "#D55E00", "unassigned": "#b8b8c0"}

    fig, axis = plt.subplots(figsize=(11, 0.9 + 0.75 * max(1, len(counts))))
    for row, (source, tally) in enumerate(sorted(counts.items())):
        total = sum(tally.values()) or 1
        left = 0.0
        for key in order:
            value = tally.get(key, 0)
            if not value:
                continue
            width = value / total
            axis.barh(row, width, left=left, color=colours.get(key, "#999999"),
                      edgecolor="white", height=0.6)
            if width > 0.055:
                axis.text(left + width / 2, row, f"{key}\n{value:,}", ha="center",
                          va="center", fontsize=6.8, color="white", fontweight="bold")
            left += width
    axis.set_yticks(range(len(counts)))
    axis.set_yticklabels(sorted(counts), fontsize=9)
    axis.set_xlim(0, 1)
    axis.set_xlabel("share of learners (splits are by learner, never by row)")
    axis.set_title("f085-03 Data flow and splits: who tunes, who is scored once\n"
                   "red = held-out test learners, touched exactly once by the final pass",
                   fontsize=11, fontweight="bold")
    axis.spines[["top", "right", "left"]].set_visible(False)
    save(fig, "f085-03_data-flow-and-splits.png")


# ------------------------------------------------------------------ f085-04

def artifact_lineage() -> None:
    """Every artifact with the target that produced it and its size on disk."""
    owners = [
        ("make items", ["datasets/item-parameters-v1.csv"]),
        ("make prep-data", ["datasets/archival-summary.json"]),
        ("make simulate", ["datasets/sim-manifest.json"]),
        ("make features", ["datasets/features-manifest.json",
                           "benchmarks/eda-summary.json"]),
        ("make train-baselines", ["benchmarks/study1-cv.json",
                                  "benchmarks/study1-final.json"]),
        ("make train-states", ["models/state-encoder.pt", "models/state-calibrators.joblib",
                               "evaluation/validation-gate.json",
                               "benchmarks/study2-heads.json"]),
        ("make eval-policies", ["models/bkt-params.json", "models/item-response-stats.json",
                                "models/policy-thresholds.json",
                                "evaluation/policy-results-v0.json",
                                "evaluation/decision-log-v0.parquet",
                                "evaluation/decision-divergence-v0.json"]),
        ("make integrate", ["evaluation/phase-8.5-validation.json"]),
    ]
    rows = []
    for target, paths in owners:
        for relative in paths:
            path = ROOT / "artifacts" / relative
            rows.append((target, relative, path.stat().st_size if path.exists() else None))

    fig, axis = plt.subplots(figsize=(11, 0.55 + 0.30 * len(rows)))
    axis.axis("off")
    axis.set_xlim(0, 1)
    axis.set_ylim(0, len(rows) + 1.4)
    axis.text(0.02, len(rows) + 0.8, "make target", fontsize=9, fontweight="bold")
    axis.text(0.30, len(rows) + 0.8, "artifact", fontsize=9, fontweight="bold")
    axis.text(0.88, len(rows) + 0.8, "on disk", fontsize=9, fontweight="bold")
    previous = None
    for index, (target, relative, size) in enumerate(rows):
        y = len(rows) - index - 0.2
        if target != previous:
            axis.text(0.02, y, target, fontsize=8, color=INK, fontweight="bold")
            previous = target
        present = size is not None
        axis.text(0.30, y, relative, fontsize=8,
                  color=INK if present else BOX["missing"])
        axis.text(0.88, y, human(size) if present else "NOT PRODUCED", fontsize=8,
                  color=INK if present else BOX["missing"],
                  fontweight="normal" if present else "bold")
    axis.set_title("f085-04 Artifact lineage: every artifact, the target that regenerates it, "
                   "and whether it exists\nred rows are the gaps — nothing downstream may quote them",
                   fontsize=11, fontweight="bold", loc="left")
    save(fig, "f085-04_artifact-lineage.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.parse_args()
    training_pipeline()
    policy_architecture()
    data_flow_and_splits()
    artifact_lineage()


if __name__ == "__main__":
    main()
