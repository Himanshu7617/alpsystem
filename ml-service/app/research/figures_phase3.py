"""Phase 3 figure pack: what the archival data is, and whether it can be trusted.

    python -m app.research.figures_phase3

Writes artifacts/figures/03-archival/f03-0{1..8}_*.png. Every number is read from
data/processed/*.parquet, data/processed/splits.json or
artifacts/datasets/archival-summary.json — nothing here is hand-typed.
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

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"
SUMMARY_PATH = ROOT / "artifacts" / "datasets" / "archival-summary.json"
SPLITS_PATH = PROCESSED_DIR / "splits.json"
FIG_DIR = ROOT / "artifacts" / "figures" / "03-archival"

SOURCE_COLOURS = {
    "assistments_2009": "#0072B2",
    "assistments_2012": "#E69F00",
    "ednet_kt1": "#009E73",
}
RTE_DISENGAGEMENT = 0.90


#: Every file this script reads. `figure_index` copies it into INDEX.md.
SOURCES = [
    "data/processed/assistments_2009.parquet",
    "data/processed/assistments_2012.parquet",
    "data/processed/ednet_kt1.parquet",
    "data/processed/splits.json",
    "artifacts/datasets/archival-summary.json",
]


def save(fig: plt.Figure, name: str) -> None:
    path = plotstyle.save(fig, FIG_DIR, name)
    print(f"wrote: {path.relative_to(ROOT)} (+ .pdf)")


def load(sources: list[str], columns: list[str]) -> dict[str, pd.DataFrame]:
    return {
        name: pd.read_parquet(PROCESSED_DIR / f"{name}.parquet", columns=columns)
        for name in sources
    }


def fig_summary_table(summary: dict) -> None:
    rows, labels = [], []
    for name, notes in summary.items():
        labels.append(name)
        rows.append([
            f"{notes['learners']:,}",
            f"{notes['items']:,}",
            f"{notes['skills']:,}",
            f"{notes['interactions']:,}",
            f"{notes['mean_sequence_length']:.1f}",
            f"{notes['accuracy']:.3f}",
            f"{notes['response_time_available_pct']:.0f}%",
            f"{notes['lag_time_available_pct']:.0f}%",
            f"{notes['timestamp_available_pct']:.0f}%",
        ])
    columns = ["learners", "items", "skills", "interactions", "mean seq len",
               "accuracy", "RT", "lag time", "timestamp"]

    fig, ax = plt.subplots(figsize=(12, 1.2 + 0.5 * len(rows)))
    ax.axis("off")
    table = ax.table(cellText=rows, rowLabels=labels, colLabels=columns, loc="center", cellLoc="right")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.6)
    for index, name in enumerate(labels):
        table[index + 1, -1].set_facecolor(SOURCE_COLOURS.get(name, "#cccccc"))
        table[index + 1, -1].set_alpha(0.35)
    ax.set_title("Archival sources after normalisation, filtering and learner-level splitting", pad=18)
    save(fig, "f03-01_dataset-summary-table.png")


def fig_sequence_lengths(frames: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, frame in frames.items():
        lengths = frame.groupby("learner_id", sort=False).size()
        bins = np.logspace(np.log10(lengths.min()), np.log10(lengths.max()), 40)
        ax.hist(lengths, bins=bins, histtype="step", linewidth=1.8,
                color=SOURCE_COLOURS[name], label=f"{name} (median {lengths.median():.0f})")
    ax.set_xscale("log")
    ax.set_xlabel("interactions per learner (log)")
    ax.set_ylabel("learners")
    ax.set_title(f"Sequence length per learner, after the {'≥10 interactions'} filter")
    ax.legend()
    save(fig, "f03-02_sequence-length-distribution.png")


def fig_response_times(frames: dict[str, pd.DataFrame], summary: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, frame in frames.items():
        raw = frame["response_time_ms_raw"].dropna()
        sample = raw.sample(min(len(raw), 200_000), random_state=0)
        axes[0].hist(sample / 1000, bins=np.linspace(0, 120, 60), histtype="step",
                     linewidth=1.8, color=SOURCE_COLOURS[name], label=name, density=True)
        positive = sample[sample > 0]
        axes[1].hist(np.log10(positive), bins=60, histtype="step", linewidth=1.8,
                     color=SOURCE_COLOURS[name], label=name, density=True)
        low, high = summary[name]["source_level_bounds_ms"]
        for bound in (low, high):
            if bound > 0:
                axes[1].axvline(np.log10(bound), color=SOURCE_COLOURS[name], linestyle="--", linewidth=1)

    axes[0].set_xlabel("response time (s, first 120 s)")
    axes[0].set_ylabel("density")
    axes[0].set_title("Raw response time")
    axes[1].set_xlabel("log10 response time (ms)")
    axes[1].set_title("Log response time; dashed = source-level winsorisation bounds")
    for axis in axes:
        axis.legend(fontsize=8)
    percentages = ", ".join(f"{name} {summary[name]['rows_winsorised_pct']:.1f}%" for name in frames)
    fig.suptitle(f"Response time before winsorisation. Rows clipped at the per-item 1st/99th pct: {percentages}")
    save(fig, "f03-03_response-time-distribution.png")


def fig_accuracy_by_attempt(frames: dict[str, pd.DataFrame]) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    plotted = []
    for name, frame in frames.items():
        if frame["attempt_count"].isna().all():
            continue
        attempts = frame["attempt_count"].clip(upper=8)
        grouped = frame.assign(_a=attempts).groupby("_a", observed=True)["correct"].agg(["mean", "size"])
        grouped = grouped[grouped["size"] >= 100]
        ax.plot(grouped.index, grouped["mean"], marker="o", color=SOURCE_COLOURS[name], label=name)
        plotted.append(name)
    ax.set_xlabel("attempts on the item (clipped at 8)")
    ax.set_ylabel("proportion correct")
    ax.set_ylim(0, 1)
    missing = [name for name in frames if name not in plotted]
    note = f"  ({', '.join(missing)}: no attempt count in the release)" if missing else ""
    ax.set_title(f"Accuracy by attempt number{note}")
    ax.legend()
    save(fig, "f03-04_accuracy-by-attempt-number.png")


def fig_rte(frames: dict[str, pd.DataFrame], summary: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    for name, frame in frames.items():
        rte = frame.groupby("learner_id", sort=False)["learner_rte"].first().dropna()
        below = summary[name]["learners_below_rte_090"]
        ax.hist(rte, bins=np.linspace(0, 1, 51), histtype="step", linewidth=1.8,
                color=SOURCE_COLOURS[name], density=True,
                label=f"{name} — {below:,} learners below {RTE_DISENGAGEMENT}")
    ax.axvline(RTE_DISENGAGEMENT, color="black", linestyle="--", linewidth=1)
    ax.text(RTE_DISENGAGEMENT - 0.01, ax.get_ylim()[1] * 0.9, "disengagement threshold",
            ha="right", fontsize=9)
    ax.set_xlabel("learner response-time effort (proportion of solution-behaviour responses)")
    ax.set_ylabel("density")
    ax.set_title("Wise & Kong response-time effort per learner")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "f03-05_rte-distribution.png")


def fig_rapid_guess(summary: dict) -> None:
    names = list(summary)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), gridspec_kw={"width_ratios": [1, 1.2]})

    x = np.arange(len(names))
    axes[0].bar(x - 0.2, [summary[n]["rapid_guess_accuracy"] for n in names], width=0.4,
                color=[SOURCE_COLOURS[n] for n in names], label="rapid guess")
    axes[0].bar(x + 0.2, [summary[n]["solution_behaviour_accuracy"] for n in names], width=0.4,
                color=[SOURCE_COLOURS[n] for n in names], alpha=0.45, label="solution behaviour")
    for index, name in enumerate(names):
        chance = summary[name].get("chance_level")
        if chance:
            axes[0].hlines(chance, index - 0.45, index + 0.45, color="black", linestyle="--")
            axes[0].text(index, chance + 0.02, f"chance {chance:.2f}", ha="center", fontsize=8)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=15, fontsize=8)
    axes[0].set_ylabel("proportion correct")
    axes[0].set_ylim(0, 1)
    axes[0].legend(fontsize=8)
    axes[0].set_title("Rapid-guess vs solution behaviour accuracy")

    # The right panel is the evidence behind the recorded deviation: where the
    # pooled check fails, the ability split says whether the threshold is wrong
    # or the population breaks the rapid-guessing assumption.
    deviating = [name for name in names if summary[name].get("rapid_guess_check") == "deviation"] or names
    for name in deviating:
        by_ability = summary[name].get("rapid_guess_accuracy_by_ability_quartile", {})
        if not by_ability:
            continue
        labels = list(by_ability)
        axes[1].plot(labels, [by_ability[label]["accuracy"] for label in labels], marker="o",
                     color=SOURCE_COLOURS[name], label=name)
        chance = summary[name].get("chance_level")
        if chance:
            axes[1].axhline(chance, color="black", linestyle="--", linewidth=1)
            axes[1].fill_between(labels, chance - 0.10, chance + 0.10, color="black", alpha=0.08)
    axes[1].set_ylim(0, 1)
    axes[1].set_ylabel("rapid-guess accuracy")
    axes[1].set_xlabel("learner ability quartile")
    axes[1].set_title("Rapid-guess accuracy by ability (shaded: chance ± 0.10)")
    axes[1].legend(fontsize=8)
    save(fig, "f03-06_rapid-guess-accuracy.png")


def fig_split_balance(frames: dict[str, pd.DataFrame], splits: dict) -> None:
    fig, axes = plt.subplots(2, len(frames), figsize=(4.6 * len(frames), 8.6), squeeze=False,
                             gridspec_kw={"hspace": 0.55})
    for column, (name, frame) in enumerate(frames.items()):
        split = splits["sources"][name]
        assignment = {learner: "test" for learner in split["test"]}
        for index, fold in enumerate(split["cv_folds"]):
            assignment.update({learner: f"fold {index + 1}" for learner in fold})

        by_learner = frame.groupby("learner_id", sort=False).agg(
            accuracy=("correct", "mean"), length=("correct", "size")
        )
        by_learner["split"] = by_learner.index.map(assignment)
        order = ["test"] + [f"fold {i + 1}" for i in range(len(split["cv_folds"]))]

        axes[0][column].boxplot([by_learner.loc[by_learner["split"] == part, "accuracy"] for part in order],
                                tick_labels=order, showfliers=False)
        axes[0][column].set_title(f"{name}\nlearner accuracy by split")
        axes[0][column].set_ylim(0, 1)
        axes[1][column].boxplot([by_learner.loc[by_learner["split"] == part, "length"] for part in order],
                                tick_labels=order, showfliers=False)
        axes[1][column].set_yscale("log")
        axes[1][column].set_title("sequence length by split (log)")
        for axis in (axes[0][column], axes[1][column]):
            axis.tick_params(axis="x", rotation=45, labelsize=8)
    fig.suptitle("Split balance — learners are grouped, so a split that differs systematically would bias every fold")
    save(fig, "f03-07_learner-split-balance.png")


def fig_missingness(summary: dict) -> None:
    fields = ["response_time_available_pct", "lag_time_available_pct", "attempt_count_available_pct",
              "hint_count_available_pct", "timestamp_available_pct"]
    labels = [field.replace("_available_pct", "").replace("_", " ") for field in fields]
    names = list(summary)
    matrix = np.array([[summary[name][field] for field in fields] for name in names])

    fig, ax = plt.subplots(figsize=(8, 1.4 + 0.7 * len(names)))
    image = ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(labels)), labels, rotation=25, ha="right")
    ax.set_yticks(range(len(names)), names)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            ax.text(column, row, f"{value:.0f}%", ha="center", va="center",
                    color="white" if value > 55 else "black", fontsize=9)
    fig.colorbar(image, ax=ax, label="rows with the field present (%)")
    ax.set_title("Feature availability per source")
    save(fig, "f03-08_missingness-heatmap.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260821, help="recorded upstream; sampling here is fixed at 0")
    parser.parse_args()

    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))["sources"]
    splits = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    sources = list(summary)
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig_summary_table(summary)
    fig_rapid_guess(summary)
    fig_missingness(summary)

    frames = load(sources, ["learner_id", "correct", "attempt_count", "response_time_ms_raw", "learner_rte"])
    fig_sequence_lengths(frames)
    fig_response_times(frames, summary)
    fig_accuracy_by_attempt(frames)
    fig_rte(frames, summary)
    fig_split_balance(frames, splits)


if __name__ == "__main__":
    main()
