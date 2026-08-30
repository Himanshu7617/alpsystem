"""Phase 4 figure pack: the generative model, its fit, and its honesty checks.

    python -m app.research.figures_phase4

Writes artifacts/figures/04-simulator/f04-0{1..9}_*.png. Numbers come from
artifacts/datasets/sim-v2-*.parquet, sim-v2-manifest.json and
simulator-params-v2.json; the legacy side of f04-07 comes from the quarantined
artifacts-legacy-invalid/ dataset, which is read but never cited as a result.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from app.research import plotstyle
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.feature_selection import mutual_info_regression
from sklearn.model_selection import cross_val_score

from app.research.calibrate_simulator import load_real, session_slopes
from app.research.simulator import OPTION_RECTS, Params, Simulator

ROOT = Path(__file__).resolve().parents[3]
DATASETS = ROOT / "artifacts" / "datasets"
LEGACY = ROOT / "artifacts-legacy-invalid" / "datasets" / "interactions-long.csv"
FIG_DIR = ROOT / "artifacts" / "figures" / "04-simulator"

VARIANT_COLOURS = {"V0": "#0072B2", "V1": "#E69F00", "V2": "#009E73", "V3": "#D55E00", "V4": "#CC79A7"}
REAL_COLOUR = "#333333"

#: Legacy observable -> its Simulator v2 counterpart. Same behavioural signal,
#: same question: how much does it tell you about the latent state?
MI_PAIRS = [
    ("total_response_time", "total_response_time", "response time"),
    ("reading_time", "reading_time", "reading time"),
    ("option_changes", "option_changes", "option changes"),
    ("time_after_last_interaction", "time_after_last_interaction", "time after last action"),
    ("pause_duration", "idle_time", "idle / pause time"),
    ("mouse_speed", "velocity_mean", "pointer velocity"),
    ("mouse_distance", "path_ratio", "path length ratio"),
    ("hover_time", "hover_time_max", "hover time"),
    ("tab_switches", "visibility_changes", "tab switches"),
]


#: Every file this script reads. `figure_index` copies it into INDEX.md.
SOURCES = [
    "artifacts/datasets/sim-v2-manifest.json",
    "artifacts/datasets/sim-v2-V0.parquet",
    "artifacts/datasets/simulator-params-v2.json",
    "artifacts/datasets/simulator-calibration-ks.json",
    "artifacts/datasets/simulator-recoverability.json",
    "artifacts/datasets/simulator-mi-comparison.csv",
]


def save(fig: plt.Figure, name: str) -> None:
    path = plotstyle.save(fig, FIG_DIR, name)
    print(f"wrote: {path.relative_to(ROOT)} (+ .pdf)")


# ------------------------------------------------------------------ f04-01

def fig_dag(params: Params) -> None:
    fig, ax = plt.subplots(figsize=(12, 7.5))
    ax.axis("off")

    nodes = {
        "ability θ": (0.08, 0.80, "#0072B2"),
        "learning rate η": (0.08, 0.62, "#0072B2"),
        "traits\n(speed, jitter, device,\nindecision, distraction)": (0.08, 0.30, "#D55E00"),
        "knowledge Kc": (0.34, 0.80, "#0072B2"),
        "fatigue F": (0.34, 0.62, "#0072B2"),
        "engagement E": (0.34, 0.46, "#0072B2"),
        "confidence C": (0.34, 0.30, "#0072B2"),
        "item (b, a, stem)": (0.34, 0.12, "#E69F00"),
        "correct": (0.62, 0.83, "#009E73"),
        "response time": (0.62, 0.66, "#009E73"),
        "behaviour\n(changes, idle, hover)": (0.62, 0.44, "#009E73"),
        "cursor path (2-D)": (0.62, 0.22, "#009E73"),
        "probes\n(noisy self-report)": (0.62, 0.05, "#CC79A7"),
        "extract()\nfeature row": (0.88, 0.33, "#333333"),
        "distraction\n(off-task channel)": (0.08, 0.10, "#D55E00"),
    }
    for label, (x, y, colour) in nodes.items():
        ax.add_patch(plt.Rectangle((x - 0.075, y - 0.045), 0.15, 0.09, facecolor=colour,
                                   alpha=0.18, edgecolor=colour, linewidth=1.4))
        ax.text(x, y, label, ha="center", va="center", fontsize=8)

    edges = [
        ("ability θ", "knowledge Kc"), ("learning rate η", "knowledge Kc"),
        ("knowledge Kc", "correct"), ("item (b, a, stem)", "correct"),
        ("fatigue F", "correct"), ("knowledge Kc", "response time"),
        ("fatigue F", "response time"), ("item (b, a, stem)", "response time"),
        ("traits\n(speed, jitter, device,\nindecision, distraction)", "response time"),
        ("knowledge Kc", "behaviour\n(changes, idle, hover)"),
        ("fatigue F", "behaviour\n(changes, idle, hover)"),
        ("traits\n(speed, jitter, device,\nindecision, distraction)", "behaviour\n(changes, idle, hover)"),
        ("item (b, a, stem)", "behaviour\n(changes, idle, hover)"),
        ("distraction\n(off-task channel)", "behaviour\n(changes, idle, hover)"),
        ("traits\n(speed, jitter, device,\nindecision, distraction)", "cursor path (2-D)"),
        ("knowledge Kc", "cursor path (2-D)"),
        ("confidence C", "probes\n(noisy self-report)"),
        ("engagement E", "probes\n(noisy self-report)"),
        ("correct", "extract()\nfeature row"), ("response time", "extract()\nfeature row"),
        ("behaviour\n(changes, idle, hover)", "extract()\nfeature row"),
        ("cursor path (2-D)", "extract()\nfeature row"),
        ("correct", "knowledge Kc"), ("correct", "confidence C"), ("fatigue F", "engagement E"),
    ]
    for source, target in edges:
        x1, y1, _ = nodes[source]
        x2, y2, _ = nodes[target]
        ax.annotate("", xy=(x2 - 0.078, y2), xytext=(x1 + 0.078, y1),
                    arrowprops=dict(arrowstyle="->", color="#888888", linewidth=0.9,
                                    connectionstyle="arc3,rad=0.08"))

    ax.set_xlim(0, 1)
    ax.set_ylim(-0.02, 0.92)
    ax.set_title("Simulator v2 generative model — blue: latent, orange: item, red: learner traits and "
                 "contamination,\ngreen: observables, purple: self-report. Traits and distraction are drawn "
                 "independently of the latent state:\nthat is what stops an observable from inverting back "
                 "to it (BUILD.md Defect 3).", fontsize=10, loc="left")
    ax.text(0.5, -0.01, f"calibrated against {params.calibrated_against}; "
                        f"parameters in artifacts/datasets/simulator-params-v2.json",
            ha="center", fontsize=8, style="italic")
    save(fig, "f04-01_generative-dag.png")


# -------------------------------------------------------------- f04-02..05

def _overlay(ax, real: np.ndarray, simulated: np.ndarray, bins, label: str, statistic) -> None:
    ax.hist(real, bins=bins, density=True, histtype="step", linewidth=2, color=REAL_COLOUR, label="real")
    ax.hist(simulated, bins=bins, density=True, histtype="stepfilled", alpha=0.35,
            color=VARIANT_COLOURS["V0"], label="simulated V0")
    ax.set_xlabel(label)
    ax.set_ylabel("density")
    limit = "≤ 0.15 ✓" if statistic is not None and statistic <= 0.15 else "> 0.15 ✗"
    ax.set_title(f"{label}\nKS D = {statistic}  ({limit})")
    ax.legend(fontsize=8)


def fig_sim_vs_real(real: pd.DataFrame, simulated: pd.DataFrame, checks: dict) -> None:
    def statistic(name: str):
        return (checks.get(name) or {}).get("ks_d")

    fig, ax = plt.subplots(figsize=(7, 5))
    _overlay(ax, real.groupby("learner_id")["correct"].mean().to_numpy(),
             simulated.groupby("learner_id")["correct"].mean().to_numpy(),
             np.linspace(0, 1, 41), "per-learner accuracy", statistic("accuracy"))
    save(fig, "f04-02_sim-vs-real-accuracy.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    _overlay(ax, np.log(real["response_time_ms"].clip(lower=1).to_numpy()),
             np.log(simulated["response_time_ms"].clip(lower=1).to_numpy()),
             np.linspace(5, 16, 60), "log response time (ms)", statistic("log_response_time"))
    save(fig, "f04-03_sim-vs-real-logrt.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    real_lengths = real.groupby("learner_id").size().to_numpy()
    sim_lengths = simulated.groupby("learner_id").size().to_numpy()
    bins = np.logspace(1, np.log10(max(real_lengths.max(), sim_lengths.max())), 40)
    _overlay(ax, real_lengths, sim_lengths, bins, "interactions per learner", statistic("sequence_length"))
    ax.set_xscale("log")
    save(fig, "f04-04_sim-vs-real-seqlen.png")

    fig, ax = plt.subplots(figsize=(7, 5))
    simulated = simulated.rename(columns={"session_id": "session"})
    simulated["position"] = simulated.groupby(["learner_id", "session"]).cumcount()
    _overlay(ax, session_slopes(real), session_slopes(simulated), np.linspace(-0.08, 0.08, 50),
             "within-session accuracy slope (matched difficulty)", statistic("session_decay_slope"))
    save(fig, "f04-05_sim-vs-real-session-decay.png")


# ------------------------------------------------------------------ f04-06

def fig_latent_trajectories(params: Params, seed: int) -> None:
    simulator = Simulator(params, seed=seed)
    pool = simulator.learners(600)
    # Sessions are short by calibration (median 7 items), so a randomly picked
    # learner shows almost no trajectory. Take several candidates per archetype
    # and keep the one whose first session is longest — the dynamics are what
    # this figure is for, and the session-length distribution is figure f04-04's
    # job, not this one's.
    candidates: dict[str, list] = {}
    for learner in pool:
        candidates.setdefault(learner.archetype, []).append(learner)
    selected = [learner for archetype in sorted(candidates) for learner in candidates[archetype][:12]]
    frame = pd.DataFrame(simulator.run(selected))

    first_sessions = (frame.groupby(["archetype", "learner_id", "session_id"]).size()
                      .reset_index(name="length"))
    first_sessions = first_sessions.sort_values("length", ascending=False).groupby("archetype").head(1)

    latents = [("latent_knowledge_mean", "mean knowledge over concepts (logits)"),
               ("latent_engagement", "engagement"),
               ("latent_confidence", "confidence"), ("latent_fatigue", "fatigue")]
    colours = dict(zip(sorted(candidates), ["#0072B2", "#E69F00", "#009E73", "#D55E00"]))

    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for axis, (column, label) in zip(axes.ravel(), latents):
        for archetype, learner_id, session_id, length in first_sessions.itertuples(index=False):
            session = frame[(frame["learner_id"] == learner_id) & (frame["session_id"] == session_id)]
            axis.plot(session["position"], session[column], marker="o", markersize=3,
                      color=colours[archetype], label=f"{archetype} ({length} items)")
        axis.set_xlabel("item within the session")
        axis.set_ylabel(label)
        axis.set_title(label)
    axes[0][0].legend(fontsize=8, title="archetype (post-hoc label)")
    fig.suptitle("Latent state over one simulated session, one learner per archetype (their longest "
                 "session, since the calibrated median is 7 items).\nArchetypes label learners drawn "
                 "from the calibrated distributions; they are not hand-typed presets.")
    fig.tight_layout()
    save(fig, "f04-06_latent-trajectories.png")


# ------------------------------------------------------------------ f04-07

def _recoverability(frame: pd.DataFrame, observables: list[str], latent: str) -> float:
    """Cross-validated R² of predicting a latent from *all* observables at once.

    Mutual information is per-signal; this is the whole point of Defect 3. If the
    latent state can be read straight back out of the behaviour vector, the
    behaviour was a re-encoding of the state and any 'behaviour helps' result is
    circular by construction.
    """
    features = frame[observables].fillna(0).to_numpy()
    target = frame[latent].to_numpy()
    model = HistGradientBoostingRegressor(max_iter=120, random_state=0)
    scores = cross_val_score(model, features, target, cv=3, scoring="r2")
    return float(np.mean(scores))


def fig_mutual_information(simulated: pd.DataFrame) -> dict:
    legacy = pd.read_csv(LEGACY)
    rng = np.random.default_rng(20260821)
    sample = simulated.sample(min(len(simulated), 20_000), random_state=0)
    legacy_sample = legacy.sample(min(len(legacy), 20_000), random_state=0)

    rows = []
    for latent_new, latent_old, latent_label in [("latent_knowledge", "knowledge_before", "knowledge"),
                                                 ("latent_fatigue", "fatigue_before", "fatigue")]:
        for legacy_column, new_column, label in MI_PAIRS:
            rows.append({
                "latent": latent_label,
                "signal": label,
                "legacy": float(mutual_info_regression(
                    legacy_sample[[legacy_column]].fillna(0), legacy_sample[latent_old],
                    random_state=0)[0]),
                "v2": float(mutual_info_regression(
                    sample[[new_column]].fillna(0), sample[latent_new], random_state=0)[0]),
            })
    table = pd.DataFrame(rows)

    legacy_observables = [column for column, _, _ in MI_PAIRS]
    new_observables = [column for _, column, _ in MI_PAIRS]
    recoverability = {
        "legacy_knowledge_r2": _recoverability(legacy_sample, legacy_observables, "knowledge_before"),
        "v2_knowledge_r2": _recoverability(sample, new_observables, "latent_knowledge"),
        "legacy_fatigue_r2": _recoverability(legacy_sample, legacy_observables, "fatigue_before"),
        "v2_fatigue_r2": _recoverability(sample, new_observables, "latent_fatigue"),
        "legacy_mean_mi": float(table["legacy"].mean()),
        "v2_mean_mi": float(table["v2"].mean()),
    }

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=True)
    for axis, latent in zip(axes, ["knowledge", "fatigue"]):
        subset = table[table["latent"] == latent]
        y = np.arange(len(subset))
        axis.barh(y - 0.2, subset["legacy"], height=0.4, color="#D55E00", label="legacy simulator")
        axis.barh(y + 0.2, subset["v2"], height=0.4, color="#0072B2", label="simulator v2")
        axis.set_yticks(y, subset["signal"])
        axis.set_xlabel("mutual information with the latent (nats)")
        legacy_r2 = recoverability[f"legacy_{latent}_r2"]
        v2_r2 = recoverability[f"v2_{latent}_r2"]
        axis.set_title(f"{latent}\nlatent recoverable from all observables: "
                       f"legacy R² = {legacy_r2:.2f} → v2 R² = {v2_r2:.2f}")
    axes[0].legend(fontsize=9, loc="lower right")
    fig.suptitle("Defect 3, before and after. Mean MI falls from "
                 f"{recoverability['legacy_mean_mi']:.3f} to {recoverability['v2_mean_mi']:.3f} nats; "
                 "what matters more is that the latent state is no longer\nrecoverable from the behaviour "
                 "vector, because traits, item effects and off-task events now carry variance the state "
                 "does not explain.")
    fig.tight_layout()
    save(fig, "f04-07_behaviour-latent-mutual-information.png")

    table_path = DATASETS / "simulator-mi-comparison.csv"
    table.to_csv(table_path, index=False)
    print(f"wrote: {table_path.relative_to(ROOT)}")
    return recoverability


# ------------------------------------------------------------------ f04-08

def fig_cursor_paths(params: Params, seed: int) -> None:
    simulator = Simulator(params, seed=seed)
    learner = simulator.learners(1)[0]

    fig, axes = plt.subplots(1, 2, figsize=(13, 6), sharex=True, sharey=True)
    for axis, (attraction, label) in zip(axes, [(0.05, "confident (low uncertainty)"),
                                                (0.9, "hesitant (high uncertainty)")]):
        for rect_index, rect in enumerate(OPTION_RECTS):
            axis.add_patch(plt.Rectangle((rect["left"], rect["top"]),
                                         rect["right"] - rect["left"], rect["bottom"] - rect["top"],
                                         facecolor="#eeeeee", edgecolor="#999999"))
            axis.text(rect["left"] + 10, (rect["top"] + rect["bottom"]) / 2,
                      f"option {rect_index}", va="center", fontsize=8, color="#666666")
        for run in range(6):
            samples = simulator._path(learner, selected_index=2, competitor_index=0,
                                      attraction=attraction, reading_ms=2_000, response_time_ms=22_000)
            axis.plot([point["x"] for point in samples], [point["y"] for point in samples],
                      linewidth=1.2, alpha=0.85)
            axis.plot(samples[0]["x"], samples[0]["y"], marker="o", color="black", markersize=4)
        axis.set_title(f"{label} — attraction {attraction}")
        axis.invert_yaxis()
        axis.set_xlabel("viewport x (px)")
    axes[0].set_ylabel("viewport y (px)")
    fig.suptitle("Simulated 2-D pointer paths toward option 2, competing option 0. The extractor reduces "
                 "these paths;\nthe simulator never writes a trajectory feature itself.")
    fig.tight_layout()
    save(fig, "f04-08_simulated-cursor-paths.png")


# ------------------------------------------------------------------ f04-09

def fig_variants(manifest: dict) -> None:
    frames = {}
    for entry in manifest["variants"]:
        path = ROOT / entry["file"]
        if path.exists():
            frames[entry["variant"]] = pd.read_parquet(
                path, columns=["learner_id", "correct", "response_time_ms", "option_changes",
                               "latent_knowledge", "latent_fatigue"])
    if not frames:
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for variant, frame in frames.items():
        colour = VARIANT_COLOURS[variant]
        axes[0][0].hist(frame.groupby("learner_id")["correct"].mean(), bins=np.linspace(0, 1, 41),
                        histtype="step", density=True, linewidth=1.6, color=colour, label=variant)
        axes[0][1].hist(np.log(frame["response_time_ms"].clip(lower=1)), bins=60, histtype="step",
                        density=True, linewidth=1.6, color=colour, label=variant)
        axes[1][0].hist(frame["option_changes"].clip(upper=10), bins=np.arange(-0.5, 11, 1),
                        histtype="step", density=True, linewidth=1.6, color=colour, label=variant)
        knowledge = frame.groupby(frame.groupby("learner_id").cumcount() // 10)["latent_knowledge"].mean()
        axes[1][1].plot(knowledge.index * 10, knowledge.to_numpy(), color=colour, label=variant)

    titles = ["per-learner accuracy", "log response time (ms)", "option changes per item",
              "mean latent knowledge by attempt number"]
    for axis, title in zip(axes.ravel(), titles):
        axis.set_title(title)
        axis.legend(fontsize=8)
    fig.suptitle("Mis-specification variants. V0 calibrated · V1 2PL link · V2 fatigue affects speed only · "
                 "V3 behaviour half noise · V4 non-stationary learning rate.\n"
                 "Every closed-loop result in Phases 8-9 is reported against all five.")
    fig.tight_layout()
    save(fig, "f04-09_variant-comparison.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    params = Params.load()
    manifest = json.loads((DATASETS / "sim-v2-manifest.json").read_text(encoding="utf-8"))
    simulated = pd.read_parquet(DATASETS / "sim-v2-V0.parquet")
    real = load_real(params.calibrated_against)

    fig_dag(params)
    fig_sim_vs_real(real, simulated, manifest.get("v0_vs_real") or {})
    fig_latent_trajectories(params, args.seed)
    recoverability = fig_mutual_information(simulated)
    fig_cursor_paths(params, args.seed)
    fig_variants(manifest)

    path = DATASETS / "simulator-recoverability.json"
    path.write_text(json.dumps(recoverability, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
