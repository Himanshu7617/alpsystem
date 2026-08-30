"""Phase 1 figure pack: content coverage, the prerequisite DAG, difficulty.

    python -m app.research.figures_phase1

Writes artifacts/figures/01-content/f01-0{1,2,3,4}_*.png. Every number is read
from data/items/ or artifacts/datasets/ — nothing here is hand-typed.
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
BANK_PATH = ROOT / "data" / "items" / "item-bank-v1.json"
MANIFEST_PATH = ROOT / "data" / "items" / "item-bank-v1.manifest.json"
GRAPH_PATH = ROOT / "backend" / "src" / "data" / "concept_graph.json"
PARAMS_PATH = ROOT / "artifacts" / "datasets" / "item-parameters-v1.csv"
FIG_DIR = ROOT / "artifacts" / "figures" / "01-content"

MIN_ITEMS_PER_CONCEPT = 5
SUBJECT_COLOURS = {
    "Data Structures": "#0072B2",
    "Algorithms": "#E69F00",
    "OOP": "#009E73",
    "DBMS": "#D55E00",
    "Operating Systems": "#CC79A7",
    "Computer Networks": "#56B4E9",
}


#: Every file this script reads. `figure_index` copies it into INDEX.md.
SOURCES = [
    "data/items/item-bank-v1.json",
    "data/items/item-bank-v1.manifest.json",
    "backend/src/data/concept_graph.json",
    "artifacts/datasets/item-parameters-v1.csv",
]


def save(fig: plt.Figure, name: str) -> None:
    path = plotstyle.save(fig, FIG_DIR, name)
    print(f"wrote: {path.relative_to(ROOT)} (+ .pdf)")


def fig_item_counts(concepts: dict, coverage: dict) -> None:
    order = sorted(coverage, key=lambda cid: (-coverage[cid], cid))
    counts = [coverage[cid] for cid in order]
    colours = [SUBJECT_COLOURS[concepts[cid]["subject"]] for cid in order]

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.bar(range(len(order)), counts, color=colours)
    ax.axhline(MIN_ITEMS_PER_CONCEPT, color="black", linestyle="--", linewidth=1)
    ax.text(len(order) - 0.5, MIN_ITEMS_PER_CONCEPT + 0.4, f"{MIN_ITEMS_PER_CONCEPT}-item floor", ha="right")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([concepts[cid]["name"] for cid in order], rotation=90, fontsize=7)
    ax.set_ylabel("items in bank")
    short = sum(1 for count in counts if count < MIN_ITEMS_PER_CONCEPT)
    ax.set_title(f"Item count per concept ({sum(counts)} items; {short}/{len(order)} concepts below the floor)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colour) for colour in SUBJECT_COLOURS.values()]
    ax.legend(handles, SUBJECT_COLOURS.keys(), fontsize=7, ncol=3)
    save(fig, "f01-01_item-count-per-concept.png")


def fig_dag(concepts: dict) -> None:
    levels: dict[int, list[str]] = {}
    for cid in sorted(concepts, key=lambda c: (concepts[c]["subject"], c)):
        levels.setdefault(concepts[cid]["level"], []).append(cid)

    position = {}
    for level, members in levels.items():
        for index, cid in enumerate(members):
            position[cid] = (index - (len(members) - 1) / 2, -level)

    fig, ax = plt.subplots(figsize=(14, 7))
    for cid, info in concepts.items():
        x1, y1 = position[cid]
        for prereq in info["prerequisites"]:
            x0, y0 = position[prereq]
            ax.annotate(
                "",
                xy=(x1, y1 + 0.06),
                xytext=(x0, y0 - 0.06),
                arrowprops=dict(arrowstyle="-|>", color="#999999", linewidth=0.8, shrinkA=0, shrinkB=0),
            )
    for cid, info in concepts.items():
        x, y = position[cid]
        ax.scatter([x], [y], s=160, color=SUBJECT_COLOURS[info["subject"]], zorder=3, edgecolors="white")
        ax.text(x, y - 0.14, info["name"], ha="center", va="top", fontsize=6, rotation=25)

    ax.set_yticks(sorted(-level for level in levels))
    ax.set_yticklabels([f"level {level}" for level in sorted(levels, reverse=True)])
    ax.set_xticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Prerequisite DAG ({len(concepts)} concepts, acyclic, laid out by level)")
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=c) for c in SUBJECT_COLOURS.values()]
    ax.legend(handles, SUBJECT_COLOURS.keys(), fontsize=7, ncol=3, loc="lower right")
    save(fig, "f01-02_concept-dag.png")


def fig_difficulty(bank_items: list[dict], parameters: pd.DataFrame, calibration: dict) -> None:
    author = pd.Series({item["item_id"]: item["author_difficulty"] for item in bank_items})
    merged = parameters.set_index("item_id").join(author.rename("author_difficulty"))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(merged["author_difficulty"], bins=np.arange(0.5, 11.5, 1), color="#0072B2", edgecolor="white")
    axes[0].set_xlabel("author_difficulty (1–10)")
    axes[0].set_ylabel("items")
    axes[0].set_title("Author-asserted difficulty")

    axes[1].hist(merged["b"], bins=20, color="#E69F00", edgecolor="white")
    axes[1].set_xlabel("Rasch b (logit)")
    axes[1].set_title("Calibrated difficulty b")
    axes[1].set_ylim(top=axes[1].get_ylim()[1] * 1.35)
    fitted = calibration.get("items_fitted", 0)
    if not fitted:
        axes[1].text(
            0.5,
            0.9,
            "PLACEHOLDER: no responses yet.\nb is z-scored author difficulty.\nRe-run after Phase 3.",
            transform=axes[1].transAxes,
            ha="center",
            va="top",
            fontsize=8,
            bbox=dict(facecolor="#ffe6e6", edgecolor="#D55E00"),
        )
    fig.suptitle(f"Item difficulty: asserted vs estimated ({fitted}/{len(merged)} items Rasch-calibrated)")
    save(fig, "f01-03_item-difficulty-distribution.png")


def fig_bins(parameters: pd.DataFrame, bins: dict) -> None:
    ordered = parameters.sort_values("b")
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.step(ordered["b"], ordered["difficulty_bin"], where="post", color="#0072B2")
    ax.scatter(ordered["b"], ordered["difficulty_bin"], s=8, color="#333333", zorder=3)
    for edge in bins["edges"]:
        ax.axvline(edge, color="#cccccc", linewidth=0.8, zorder=0)
    ax.set_xlabel("Rasch b (logit)")
    ax.set_ylabel("assigned difficulty_score bin")
    ax.set_yticks(range(1, bins["n_bins"] + 1))
    ax.set_title(f"b → difficulty bin ({bins['n_bins']} bins, {bins['method']}, {len(bins['edges'])} edges)")
    save(fig, "f01-04_difficulty-bin-mapping.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260821)   # accepted for symmetry
    parser.parse_args()
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    bank = json.loads(BANK_PATH.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    parameters = pd.read_csv(PARAMS_PATH)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig_item_counts(graph["concepts"], manifest["concept_coverage"])
    fig_dag(graph["concepts"])
    fig_difficulty(bank["items"], parameters, manifest["calibration"])
    fig_bins(parameters, manifest["difficulty_bins"])


if __name__ == "__main__":
    main()
