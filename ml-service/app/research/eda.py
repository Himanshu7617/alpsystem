"""Phase 5 exploratory analysis — the notebook, written as a script.

    python -m app.research.eda [--seed 20260821] [--sample 300000]

Writes ``artifacts/figures/05-eda/f05-{01..13}_*.png``,
``artifacts/benchmarks/eda-summary.json`` and
``artifacts/benchmarks/feature-shortlist.json``. Every number a doc or a figure
quotes comes from those two JSON files; nothing here is hand-typed and no
figure computes a statistic the summary does not also record.

No Jupyter. A notebook's outputs are not reproducible from a `make` target and
its cells run in whatever order the author last clicked, which is the opposite
of what a leakage-sensitive pipeline needs.

Two findings this script exists to decide rather than illustrate:

* **f05-06 decides the fatigue construct.** `docs/preregistration.md` H2 says
  that if the within-session matched-difficulty accuracy curve is flat, the
  construct is reduced or dropped rather than defended. Phase 4 already found
  the calibration source's residual slope to be *positive*. The figure and the
  bootstrap CI in the summary are what settle it, on all three real sources.
* **f05-09 is RQ2 in one picture.** Does trajectory curvature carry anything
  response time does not? The partial correlation in the summary is the number;
  the scatter is how it looks.
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

from app.research import build_features, feature_catalogue as catalogue

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "artifacts" / "figures" / "05-eda"
BENCH_DIR = ROOT / "artifacts" / "benchmarks"
SUMMARY_PATH = BENCH_DIR / "eda-summary.json"
SHORTLIST_PATH = BENCH_DIR / "feature-shortlist.json"

REAL_SOURCES = build_features.ARCHIVAL_SOURCES
SIM_SOURCE = "sim-V0"
LABEL = "next_correct"

CLASS_COLOURS = {
    "correctness": "#0072B2",
    "timing": "#E69F00",
    "interaction": "#009E73",
    "session": "#D55E00",
    "motor": "#CC79A7",
}
SOURCE_COLOURS = {
    "assistments_2009": "#0072B2",
    "assistments_2012": "#E69F00",
    "ednet_kt1": "#009E73",
    "sim-V0": "#CC79A7",
}

#: Rows drawn per source for every estimate in this script. Mutual information
#: on 4.9 M rows costs minutes and buys nothing: the standard error of an MI
#: estimate at 300 k is already far below the differences being read off it.
#: The draw is seeded and the size is recorded in the summary.
SAMPLE_ROWS = 300_000

#: A pair of features this correlated is one feature twice. The survivor is
#: whichever carries more mutual information with the label.
REDUNDANT_ABS_CORRELATION = 0.95

#: Below this, a feature is not carrying enough to justify collecting it. It is
#: deliberately low — the point of the shortlist is to drop what is *empty*,
#: not to pre-empt Phase 6's ablation ladder, which is where value is decided.
MIN_MUTUAL_INFORMATION = 1e-4

BOOTSTRAP_DRAWS = 400


#: Every file this script reads. `figure_index` copies it into INDEX.md.
SOURCES = [
    "data/processed/features-assistments_2009.parquet",
    "data/processed/features-assistments_2012.parquet",
    "data/processed/features-ednet_kt1.parquet",
    "data/processed/features-sim-V0.parquet",
    "artifacts/datasets/features-manifest.json",
]


def save(fig: plt.Figure, name: str) -> None:
    path = plotstyle.save(fig, FIG_DIR, name)
    print(f"wrote: {path.relative_to(ROOT)} (+ .pdf)")


def simulated_note(axis: plt.Axes) -> None:
    """Standing guardrail 6: simulated results are labelled simulated."""
    axis.text(0.99, 0.02, "SIMULATED", transform=axis.transAxes, ha="right", va="bottom",
              fontsize=7, color="#CC79A7", alpha=0.8, fontweight="bold")


# --------------------------------------------------------------------- loading

def load(source: str, rows: int, seed: int) -> pd.DataFrame:
    """A seeded sample of one feature matrix, reindexed to the full schema."""
    frame = build_features.read_features(source)
    if len(frame) > rows:
        frame = frame.sample(rows, random_state=seed).sort_index()
    frame["_source"] = source
    return frame


def numeric(frame: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    return frame[names].apply(pd.to_numeric, errors="coerce")


def usable(frame: pd.DataFrame, source: str) -> list[str]:
    """Catalogue features that actually carry a value for this source."""
    return [name for name in catalogue.available(source)
            if frame[name].notna().any() and frame[name].nunique(dropna=True) > 1]


# ------------------------------------------------------------------ statistics

def mutual_information(frame: pd.DataFrame, names: list[str], seed: int) -> dict[str, float]:
    """MI between each feature and the label, medians filled for the estimator.

    ponytail: median imputation for the estimator only — never written back to
    the matrix. sklearn's MI estimator cannot take NaN, and the alternative
    (dropping rows) would silently restrict the sample to whichever learners
    happen to have every signal class, which is a different population.
    """
    matrix = numeric(frame, names)
    filled = matrix.fillna(matrix.median(numeric_only=True))
    filled = filled.fillna(0.0)
    scores = mutual_info_classif(filled, frame[LABEL].astype(int), random_state=seed)
    return {name: float(score) for name, score in zip(names, scores)}


def variance_inflation(frame: pd.DataFrame, names: list[str]) -> dict[str, float]:
    """VIF per feature, from the R² of regressing it on the others.

    ponytail: computed through the correlation matrix's inverse rather than by
    fitting one OLS per feature — same number, one matrix decomposition.
    """
    matrix = numeric(frame, names)
    matrix = matrix.loc[:, matrix.std() > 0]
    correlation = matrix.corr().to_numpy()
    if not np.isfinite(correlation).all():
        return {}
    try:
        inverse = np.linalg.pinv(correlation)
    except np.linalg.LinAlgError:
        return {}
    return {name: float(round(inverse[index, index], 3))
            for index, name in enumerate(matrix.columns)}


def quick_baseline(frame: pd.DataFrame, names: list[str], seed: int) -> tuple[dict[str, float], float]:
    """Permutation importance on a small gradient-boosting fit, plus its AUC.

    Grouped by learner so the split matches `splits.json`'s protocol. This is
    evidence for the shortlist, not a result: Phase 6 owns model comparison.
    """
    matrix = numeric(frame, names)
    label = frame[LABEL].astype(int)
    learners = frame["learner_id"]
    holdout = set(pd.Series(learners.unique()).sample(
        frac=0.25, random_state=seed))
    train = ~learners.isin(holdout)
    if train.sum() < 500 or (~train).sum() < 500:
        return {}, float("nan")

    model = HistGradientBoostingClassifier(max_iter=120, random_state=seed)
    model.fit(matrix[train], label[train])
    auc = float(roc_auc_score(label[~train], model.predict_proba(matrix[~train])[:, 1]))

    subset = matrix[~train].head(20_000)
    importance = permutation_importance(
        model, subset, label[~train].head(len(subset)),
        n_repeats=3, random_state=seed, scoring="roc_auc", n_jobs=1,
    )
    return ({name: float(round(value, 6))
             for name, value in zip(names, importance.importances_mean)}, auc)


def bootstrap_slope(x: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int,
                    draws: int = BOOTSTRAP_DRAWS) -> dict:
    """Slope of y on x with a learner-clustered bootstrap CI.

    Resampling *learners*, not rows: attempts within a learner are not
    independent, and a row bootstrap would report a confidence interval several
    times narrower than the data supports.
    """
    valid = np.isfinite(x) & np.isfinite(y)
    x, y, groups = x[valid], y[valid], groups[valid]
    if len(x) < 100:
        return {"slope": None, "reason": "fewer than 100 usable points"}

    slope = float(np.polyfit(x, y, 1)[0])
    unique, inverse = np.unique(groups, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(unique)))
    ends = np.append(starts[1:], len(order))

    rng = np.random.default_rng(seed)
    draws_out = []
    for _ in range(draws):
        picked = rng.integers(0, len(unique), len(unique))
        index = np.concatenate([order[starts[k]:ends[k]] for k in picked])
        if len(index) < 10:
            continue
        draws_out.append(np.polyfit(x[index], y[index], 1)[0])
    low, high = np.percentile(draws_out, [2.5, 97.5]) if draws_out else (np.nan, np.nan)
    return {
        "slope": round(slope, 6),
        "ci_low": round(float(low), 6),
        "ci_high": round(float(high), 6),
        "n": int(len(x)),
        "learners": int(len(unique)),
        "flat": bool(low <= 0 <= high),
    }


# ---------------------------------------------------------------------- figures

def fig_distributions(frames: dict[str, pd.DataFrame]) -> None:
    """f05-01 — every feature, real and simulated overlaid."""
    names = catalogue.names()
    columns = 6
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3 * columns, 2.1 * rows))
    for axis, name in zip(axes.ravel(), names):
        drawn = False
        for source, frame in frames.items():
            values = pd.to_numeric(frame[name], errors="coerce").dropna()
            if values.empty or values.nunique() < 2:
                continue
            low, high = values.quantile([0.01, 0.99])
            clipped = values.clip(low, high)
            axis.hist(clipped, bins=40, density=True, histtype="step", linewidth=1.2,
                      color=SOURCE_COLOURS[source], label=source)
            drawn = True
        axis.set_title(f"{name}\n{catalogue.BY_NAME[name].signal_class}", fontsize=7)
        axis.tick_params(labelsize=6)
        axis.set_yticks([])
        if not drawn:
            constant = any(frame[name].notna().any() for frame in frames.values())
            axis.text(0.5, 0.5, "constant" if constant else "no source supplies it",
                      ha="center", va="center", fontsize=7, color="grey")
    for axis in axes.ravel()[len(names):]:
        axis.axis("off")
    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower right", ncol=4, fontsize=8,
               bbox_to_anchor=(0.98, 0.005))
    fig.suptitle("f05-01 Feature distributions — real archival vs simulated (V0)", fontsize=12)
    fig.tight_layout(rect=(0, 0.02, 1, 0.98))
    save(fig, "f05-01_feature-distributions-grid.png")


def fig_correlation(frame: pd.DataFrame, names: list[str]) -> dict:
    """f05-02 — correlation, ordered by signal class so the blocks are visible."""
    ordered = sorted(names, key=lambda name: (catalogue.LEVELS.index(catalogue.BY_NAME[name].level), name))
    correlation = numeric(frame, ordered).corr()

    fig, axis = plt.subplots(figsize=(0.32 * len(ordered) + 4, 0.32 * len(ordered) + 3))
    image = axis.imshow(correlation, cmap="RdBu_r", vmin=-1, vmax=1)
    axis.set_xticks(range(len(ordered)), ordered, rotation=90, fontsize=6)
    axis.set_yticks(range(len(ordered)), ordered, fontsize=6)
    boundary = 0
    for level in catalogue.LEVELS:
        size = sum(1 for name in ordered if catalogue.BY_NAME[name].level == level)
        if not size:
            continue
        axis.add_patch(plt.Rectangle((boundary - 0.5, boundary - 0.5), size, size,
                                     fill=False, edgecolor=CLASS_COLOURS[catalogue.LEVEL_CLASS[level]],
                                     linewidth=2))
        axis.text(boundary + size / 2 - 0.5, -1.4, level, ha="center", fontsize=8,
                  color=CLASS_COLOURS[catalogue.LEVEL_CLASS[level]], fontweight="bold")
        boundary += size
    fig.colorbar(image, ax=axis, shrink=0.6, label="Pearson r")
    axis.set_title("f05-02 Feature correlation, blocked by signal class (simulated V0)", fontsize=11)
    simulated_note(axis)
    save(fig, "f05-02_correlation-heatmap.png")

    upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool)).stack()
    strong = upper[upper.abs() > REDUNDANT_ABS_CORRELATION]
    return {
        "simulated_strong_pairs": [
            {"a": a, "b": b, "r": round(float(value), 4)}
            for (a, b), value in strong.sort_values(key=abs, ascending=False).items()
        ],
        "max_abs_correlation": round(float(upper.abs().max()), 4) if len(upper) else None,
    }


def redundant_pairs(frames: dict[str, pd.DataFrame]) -> list[dict]:
    """Pairs that are one feature twice **in every source that has both**.

    Judging this from the simulator alone would be a mistake with a name: the
    generative model couples signals the real world does not. It emits idle
    time and tab visibility from one off-task channel, so their correlation is
    exactly 1.0 — a property of the simulator, not of the constructs. A pair is
    only called redundant here if no source separates it, and the per-source
    correlations are recorded so a later reader can see which sources voted.
    """
    per_source: dict[str, dict[tuple[str, str], float]] = {}
    for source, frame in frames.items():
        names = usable(frame, source)
        if len(names) < 2:
            continue
        correlation = numeric(frame, names).corr()
        upper = correlation.where(np.triu(np.ones(correlation.shape), k=1).astype(bool)).stack()
        per_source[source] = {pair: float(value) for pair, value in upper.items() if np.isfinite(value)}

    seen: dict[tuple[str, str], dict[str, float]] = {}
    for source, pairs in per_source.items():
        for pair, value in pairs.items():
            seen.setdefault(pair, {})[source] = round(value, 4)

    found = []
    for (a, b), by_source in seen.items():
        if all(abs(value) > REDUNDANT_ABS_CORRELATION for value in by_source.values()):
            found.append({"a": a, "b": b,
                          "r_by_source": by_source,
                          "r": max(by_source.values(), key=abs),
                          "sources_checked": sorted(by_source),
                          "simulated_only": all(source.startswith("sim") for source in by_source)})
    return sorted(found, key=lambda pair: -abs(pair["r"]))


def fig_mutual_information(scores: dict[str, dict[str, float]]) -> None:
    """f05-03 — MI ranking, coloured by signal class, simulated vs real."""
    combined = scores[SIM_SOURCE]
    ordered = sorted(combined, key=combined.get, reverse=True)
    fig, axis = plt.subplots(figsize=(9, 0.24 * len(ordered) + 2))
    positions = np.arange(len(ordered))
    axis.barh(positions, [combined[name] for name in ordered],
              color=[CLASS_COLOURS[catalogue.BY_NAME[name].signal_class] for name in ordered])
    for source in REAL_SOURCES:
        real = scores.get(source, {})
        marks = [real.get(name, np.nan) for name in ordered]
        axis.scatter(marks, positions, s=14, zorder=3, label=source,
                     color=SOURCE_COLOURS[source], edgecolor="white", linewidth=0.4)
    axis.set_yticks(positions, ordered, fontsize=7)
    axis.invert_yaxis()
    axis.set_xlabel("mutual information with next_correct (nats)")
    axis.set_title("f05-03 Mutual information ranking — bars simulated (V0), dots real", fontsize=11)
    handles = [plt.Line2D([], [], color=colour, lw=6, label=name) for name, colour in CLASS_COLOURS.items()]
    axis.legend(handles=handles + axis.get_legend_handles_labels()[0], fontsize=7, ncol=2)
    save(fig, "f05-03_mutual-information-ranking.png")


def fig_rt_by_difficulty(frames: dict[str, pd.DataFrame]) -> dict:
    """f05-04 — the core behavioural relationship, faceted by item difficulty."""
    usable_sources = [source for source, frame in frames.items()
                      if frame["log_response_time"].notna().any()]
    fig, axes = plt.subplots(1, len(usable_sources), figsize=(4.2 * len(usable_sources), 3.6),
                             squeeze=False, sharey=True)
    summary = {}
    for axis, source in zip(axes[0], usable_sources):
        frame = frames[source]
        difficulty = pd.qcut(frame["item_difficulty"], 3,
                             labels=["easy tercile", "medium", "hard tercile"], duplicates="drop")
        for tercile, colour in zip(difficulty.cat.categories, ["#009E73", "#E69F00", "#D55E00"]):
            subset = frame[difficulty == tercile]
            bins = pd.qcut(subset["log_response_time"], 10, duplicates="drop")
            grouped = subset.groupby(bins, observed=True)["correct"]
            centres = subset.groupby(bins, observed=True)["log_response_time"].mean()
            axis.plot(centres, grouped.mean(), marker="o", ms=3, color=colour, label=str(tercile))
        axis.set_title(source, fontsize=10)
        axis.set_xlabel("log response time")
        summary[source] = {
            "correlation_logrt_correct": round(float(
                frame[["log_response_time", "correct"]].corr().iloc[0, 1]), 4),
        }
        if source.startswith("sim"):
            simulated_note(axis)
    axes[0][0].set_ylabel("accuracy")
    axes[0][0].legend(fontsize=8)
    fig.suptitle("f05-04 Accuracy against response time, by item-difficulty tercile", fontsize=12)
    fig.tight_layout()
    save(fig, "f05-04_rt-vs-correctness-by-difficulty.png")
    return summary


def fig_option_changes(frame: pd.DataFrame) -> dict:
    """f05-05 — answer changing, with the literature's wrong->right split.

    On a single-key multiple-choice item exactly one option is correct, so an
    attempt with **exactly one** change that ends correct must have moved
    wrong -> right; one that ends incorrect is right -> wrong or wrong -> wrong
    and cannot be separated further without the first-selection index, which
    the pre-registered aggregate deliberately does not store. The determinate
    half is plotted solid and the indeterminate half is hatched.
    """
    changes = frame["option_changes"].fillna(0).astype(int).clip(upper=4)
    accuracy = frame.groupby(changes, observed=True)["correct"].agg(["mean", "size"])
    single = frame[changes == 1]
    wrong_to_right = float(single["correct"].mean()) if len(single) else float("nan")

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    axes[0].bar(accuracy.index, accuracy["mean"], color="#009E73")
    for position, (mean, size) in accuracy.iterrows():
        axes[0].text(position, mean + 0.01, f"n={int(size):,}", ha="center", fontsize=7)
    axes[0].axhline(float(frame["correct"].mean()), ls="--", color="grey", lw=1,
                    label="overall accuracy")
    axes[0].set_xlabel("option changes before submission (4 = 4 or more)")
    axes[0].set_ylabel("accuracy")
    axes[0].legend(fontsize=8)
    simulated_note(axes[0])

    axes[1].bar(["wrong→right\n(determinate)", "right→wrong or\nwrong→wrong"],
                [wrong_to_right, 1 - wrong_to_right], color=["#009E73", "#D55E00"],
                hatch=["", "//"])
    axes[1].set_ylabel("share of single-change attempts")
    axes[1].set_title(f"single-change attempts, n = {len(single):,}", fontsize=9)
    simulated_note(axes[1])
    fig.suptitle("f05-05 Answer changing and outcome (simulated V0)", fontsize=12)
    fig.tight_layout()
    save(fig, "f05-05_option-changes-vs-correctness.png")
    return {
        "accuracy_by_option_changes": {int(k): round(float(v), 4) for k, v in accuracy["mean"].items()},
        "single_change_wrong_to_right_rate": round(wrong_to_right, 4),
        "single_change_n": int(len(single)),
    }


def _within_session(frame: pd.DataFrame, response: str, residual_of: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(position, residual, learner) for the within-session decay figures."""
    position = frame["question_number"].to_numpy(dtype=float)
    values = pd.to_numeric(frame[response], errors="coerce")
    # Match on difficulty by removing the item's own mean before looking at
    # position: raw accuracy falls when later items are harder, and that is a
    # selection effect, not fatigue.
    item_mean = frame.groupby(residual_of, observed=True)[response].transform("mean")
    residual = (values - item_mean).to_numpy(dtype=float)
    return position, residual, frame["learner_id"].to_numpy()


def fig_within_session(frames: dict[str, pd.DataFrame], seed: int) -> dict:
    """f05-06 and f05-07 — the fatigue construct's only observable.

    `docs/preregistration.md` H2 pre-committed: **if this is flat, the fatigue
    construct is reduced or dropped rather than defended.**
    """
    results = {}
    for name, response, title, filename in [
        ("accuracy", "correct", "f05-06 Within-session matched-difficulty accuracy",
         "f05-06_within-session-accuracy-decay.png"),
        ("speed", "log_response_time", "f05-07 Within-session matched-difficulty speed",
         "f05-07_within-session-speed-drift.png"),
    ]:
        sources = [source for source in frames
                   if frames[source]["question_number"].notna().any()
                   and frames[source][response].notna().any()]
        fig, axis = plt.subplots(figsize=(8, 4.4))
        results[name] = {}
        for source in sources:
            frame = frames[source].dropna(subset=["question_number", response])
            position, residual, learners = _within_session(frame, response, "item_id")
            fit = bootstrap_slope(position, residual, learners, seed)
            results[name][source] = fit

            bins = pd.cut(position, bins=[0, 5, 10, 15, 20, 30, 40, 60, 100, np.inf])
            grouped = pd.DataFrame({"bin": bins, "residual": residual}).groupby("bin", observed=True)
            centre = grouped["residual"].mean()
            error = grouped["residual"].sem() * 1.96
            centres = [interval.mid if np.isfinite(interval.mid) else 120 for interval in centre.index]
            axis.plot(centres, centre, marker="o", ms=4, color=SOURCE_COLOURS[source],
                      label=f"{source}  slope {fit.get('slope')} "
                            f"[{fit.get('ci_low')}, {fit.get('ci_high')}]")
            axis.fill_between(centres, centre - error, centre + error, alpha=0.15,
                              color=SOURCE_COLOURS[source])
        axis.axhline(0, ls="--", lw=1, color="grey")
        if name == "accuracy":
            # The figure states its own verdict: H2 pre-committed to reducing
            # the construct if this is flat, so the reader should not have to
            # open a JSON file to find out which way it went.
            real = [fit for source, fit in results[name].items()
                    if source in REAL_SOURCES and fit.get("slope") is not None]
            declining = [fit for fit in real if (fit.get("ci_high") or 0) < 0]
            axis.text(0.02, 0.04,
                      "H2: no real source declines — reduce or drop the fatigue construct"
                      if real and not declining else
                      "H2: a real source declines within session — construct retained",
                      transform=axis.transAxes, fontsize=8, color="#D55E00", fontweight="bold")
        axis.set_xlabel("position within session (item number)")
        axis.set_ylabel(f"{response} minus item mean")
        axis.set_title(f"{title}\nlearner-clustered bootstrap 95 % CI on the slope", fontsize=11)
        axis.legend(fontsize=7)
        save(fig, filename)
    return results


def fig_trajectory_violins(frame: pd.DataFrame) -> None:
    """f05-08 — every motor feature, correct against incorrect."""
    names = [name for name in catalogue.names(level="L4") if frame[name].notna().any()]
    columns = 5
    rows = int(np.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(2.9 * columns, 2.6 * rows))
    for axis, name in zip(axes.ravel(), names):
        groups = []
        for outcome in (0, 1):
            values = pd.to_numeric(frame.loc[frame["correct"] == outcome, name], errors="coerce").dropna()
            low, high = values.quantile([0.01, 0.99])
            groups.append(values.clip(low, high).to_numpy())
        parts = axis.violinplot(groups, showmedians=True)
        for body, colour in zip(parts["bodies"], ["#D55E00", "#009E73"]):
            body.set_facecolor(colour)
            body.set_alpha(0.6)
        axis.set_xticks([1, 2], ["incorrect", "correct"], fontsize=7)
        axis.set_title(name, fontsize=8)
        axis.tick_params(labelsize=6)
    for axis in axes.ravel()[len(names):]:
        axis.axis("off")
    simulated_note(axes.ravel()[0])
    fig.suptitle("f05-08 Trajectory features by outcome (simulated V0)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    save(fig, "f05-08_trajectory-features-by-outcome.png")


def fig_trajectory_vs_rt(frame: pd.DataFrame) -> dict:
    """f05-09 — RQ2 in one picture: does curvature carry anything RT does not?"""
    motor = [name for name in catalogue.names(level="L4") if frame[name].notna().any()]
    rt = pd.to_numeric(frame["log_response_time"], errors="coerce")
    label = frame[LABEL].astype(int)

    partials = {}
    for name in motor:
        values = pd.to_numeric(frame[name], errors="coerce")
        mask = values.notna() & rt.notna()
        if mask.sum() < 500 or values[mask].nunique() < 3:
            continue
        # Partial correlation with the label, response time held out of both
        # sides. This is exactly RQ2: information beyond response time.
        residual_feature = values[mask] - np.poly1d(np.polyfit(rt[mask], values[mask], 1))(rt[mask])
        residual_label = label[mask] - np.poly1d(np.polyfit(rt[mask], label[mask], 1))(rt[mask])
        partials[name] = {
            "raw_correlation_with_label": round(float(np.corrcoef(values[mask], label[mask])[0, 1]), 4),
            "partial_correlation_given_rt": round(float(
                np.corrcoef(residual_feature, residual_label)[0, 1]), 4),
            "correlation_with_rt": round(float(np.corrcoef(values[mask], rt[mask])[0, 1]), 4),
        }

    headline = max(partials, key=lambda name: abs(partials[name]["partial_correlation_given_rt"])) \
        if partials else None
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    if headline:
        values = pd.to_numeric(frame[headline], errors="coerce")
        mask = values.notna() & rt.notna()
        sample = frame[mask].sample(min(6000, int(mask.sum())), random_state=0)
        axes[0].scatter(pd.to_numeric(sample["log_response_time"]), pd.to_numeric(sample[headline]),
                        s=4, alpha=0.25,
                        c=np.where(sample["correct"] == 1, "#009E73", "#D55E00"))
        axes[0].set_xlabel("log response time")
        axes[0].set_ylabel(headline)
        axes[0].set_title(f"{headline} against response time\ngreen = correct", fontsize=10)
        simulated_note(axes[0])

    ordered = sorted(partials, key=lambda name: abs(partials[name]["partial_correlation_given_rt"]))
    positions = np.arange(len(ordered))
    axes[1].barh(positions - 0.2, [partials[name]["raw_correlation_with_label"] for name in ordered],
                 height=0.4, color="#bbbbbb", label="raw correlation with label")
    axes[1].barh(positions + 0.2, [partials[name]["partial_correlation_given_rt"] for name in ordered],
                 height=0.4, color="#CC79A7", label="partial, response time removed")
    axes[1].set_yticks(positions, ordered, fontsize=7)
    axes[1].axvline(0, color="grey", lw=1)
    axes[1].set_xlabel("correlation with next_correct")
    axes[1].legend(fontsize=8)
    simulated_note(axes[1])
    fig.suptitle("f05-09 RQ2 — what trajectory features carry beyond response time (simulated V0)",
                 fontsize=12)
    fig.tight_layout()
    save(fig, "f05-09_trajectory-vs-rt-scatter.png")
    return {"headline_feature": headline, "partial_correlations": partials}


def fig_stability(frame: pd.DataFrame, names: list[str]) -> dict:
    """f05-10 — is a feature a property of the learner or of the attempt?"""
    matrix = numeric(frame, names)
    matrix["learner_id"] = frame["learner_id"].to_numpy()
    grouped = matrix.groupby("learner_id", observed=True)
    between = grouped.mean(numeric_only=True).var()
    within = grouped.var(numeric_only=True).mean()
    ratio = (between / (between + within)).dropna().sort_values()

    fig, axis = plt.subplots(figsize=(8, 0.24 * len(ratio) + 2))
    axis.barh(range(len(ratio)), ratio,
              color=[CLASS_COLOURS[catalogue.BY_NAME[name].signal_class] for name in ratio.index])
    axis.set_yticks(range(len(ratio)), ratio.index, fontsize=7)
    axis.set_xlabel("between-learner variance / total variance  (1 = a trait, 0 = a state)")
    axis.axvline(0.5, ls="--", color="grey", lw=1)
    axis.set_title("f05-10 Feature stability across learners (simulated V0)", fontsize=11)
    handles = [plt.Line2D([], [], color=colour, lw=6, label=name) for name, colour in CLASS_COLOURS.items()]
    axis.legend(handles=handles, fontsize=7)
    simulated_note(axis)
    save(fig, "f05-10_feature-stability-across-learners.png")
    return {name: round(float(value), 4) for name, value in ratio.items()}


def fig_class_overlap(frame: pd.DataFrame, seed: int) -> dict:
    """f05-11 — how much of each signal class's information is already elsewhere.

    A four-set Venn diagram of continuous mutual information does not exist as
    an honest picture, so this is the same question answered as a matrix. Each
    class is compressed to a single logistic score, and the redundancy of a
    pair is ``I(A;Y) + I(B;Y) - I(A,B;Y)`` — positive means the two classes
    tell the model the same thing twice.
    """
    label = frame[LABEL].astype(int)
    scores, own = {}, {}
    for level in catalogue.LEVELS:
        names = [name for name in catalogue.names(level=level) if frame[name].notna().any()]
        if not names:
            continue
        matrix = numeric(frame, names)
        matrix = matrix.fillna(matrix.median()).fillna(0.0)
        model = LogisticRegression(max_iter=400)
        model.fit(StandardScaler().fit_transform(matrix), label)
        scores[level] = model.decision_function(StandardScaler().fit_transform(matrix))
        own[level] = float(mutual_info_classif(
            scores[level].reshape(-1, 1), label, random_state=seed)[0])

    levels = list(scores)
    overlap = pd.DataFrame(np.nan, index=levels, columns=levels, dtype=float)
    for a in levels:
        overlap.loc[a, a] = own[a]
        for b in levels:
            if a >= b:
                continue
            joint = float(mutual_info_classif(
                np.column_stack([scores[a], scores[b]]), label, random_state=seed).sum())
            redundancy = own[a] + own[b] - joint
            overlap.loc[a, b] = overlap.loc[b, a] = redundancy

    fig, axis = plt.subplots(figsize=(6.4, 5.2))
    image = axis.imshow(overlap, cmap="viridis")
    axis.set_xticks(range(len(levels)), [f"{level}\n{catalogue.LEVEL_CLASS[level]}" for level in levels],
                    fontsize=8)
    axis.set_yticks(range(len(levels)), [f"{level} {catalogue.LEVEL_CLASS[level]}" for level in levels],
                    fontsize=8)
    for i, a in enumerate(levels):
        for j, b in enumerate(levels):
            axis.text(j, i, f"{overlap.iloc[i, j]:.4f}", ha="center", va="center",
                      fontsize=7, color="white")
    fig.colorbar(image, ax=axis, shrink=0.7, label="nats")
    axis.set_title("f05-11 Signal-class information (diagonal)\nand pairwise redundancy (off-diagonal)",
                   fontsize=11)
    simulated_note(axis)
    save(fig, "f05-11_signal-class-venn-mi.png")
    return {
        "own_information_nats": {level: round(value, 5) for level, value in own.items()},
        "pairwise_redundancy_nats": {
            f"{a}|{b}": round(float(overlap.loc[a, b]), 5)
            for i, a in enumerate(levels) for b in levels[i + 1:]
        },
    }


def fig_archetypes(frame: pd.DataFrame) -> dict:
    """f05-12 — behavioural fingerprints of the simulated archetypes."""
    axes_features = ["log_response_time", "option_changes", "pause_count", "path_ratio",
                     "hover_time_nonchosen", "idle_fraction", "x_flips", "prior_accuracy"]
    axes_features = [name for name in axes_features if frame[name].notna().any()]
    matrix = numeric(frame, axes_features)
    standardised = (matrix - matrix.mean()) / matrix.std().replace(0, np.nan)
    standardised["archetype"] = frame["archetype"].to_numpy()
    profile = standardised.groupby("archetype", observed=True).mean(numeric_only=True)

    angles = np.linspace(0, 2 * np.pi, len(axes_features), endpoint=False)
    closed = np.append(angles, angles[0])
    fig, axis = plt.subplots(figsize=(6.6, 6.2), subplot_kw={"projection": "polar"})
    for (name, row), colour in zip(profile.iterrows(), plt.cm.tab10.colors):
        values = np.append(row.to_numpy(), row.to_numpy()[0])
        axis.plot(closed, values, label=name, color=colour, lw=1.6)
        axis.fill(closed, values, color=colour, alpha=0.08)
    axis.set_xticks(angles, axes_features, fontsize=7)
    axis.set_title("f05-12 Archetype behavioural fingerprints (simulated V0)\n"
                   "standardised means, 0 = population average", fontsize=11)
    axis.legend(loc="upper right", bbox_to_anchor=(1.28, 1.12), fontsize=8)
    save(fig, "f05-12_archetype-behavioural-fingerprints.png")
    return {name: {feature: round(float(value), 3) for feature, value in row.items()}
            for name, row in profile.iterrows()}


def fig_consent(frames: dict[str, pd.DataFrame], manifest: dict) -> dict:
    """f05-13 — what survives each consent configuration, per source."""
    configurations = {
        "correctness only": ["L0"],
        "+ timing": ["L0", "L1"],
        "+ interaction": ["L0", "L1", "L2"],
        "+ session": ["L0", "L1", "L2", "L3"],
        "+ motor (full)": catalogue.LEVELS,
    }
    sources = list(frames)
    counts = pd.DataFrame(0, index=list(configurations), columns=sources, dtype=int)
    for source, frame in frames.items():
        present = set(usable(frame, source))
        for label, levels in configurations.items():
            counts.loc[label, source] = sum(
                1 for level in levels for name in catalogue.names(level=level) if name in present)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4),
                             gridspec_kw={"width_ratios": [1.1, 1.4]})
    image = axes[0].imshow(counts, cmap="Blues", aspect="auto")
    axes[0].set_xticks(range(len(sources)), sources, rotation=25, ha="right", fontsize=8)
    axes[0].set_yticks(range(len(configurations)), list(configurations), fontsize=8)
    for i in range(counts.shape[0]):
        for j in range(counts.shape[1]):
            axes[0].text(j, i, counts.iat[i, j], ha="center", va="center", fontsize=8,
                         color="white" if counts.iat[i, j] > counts.to_numpy().max() * 0.6 else "black")
    fig.colorbar(image, ax=axes[0], shrink=0.8, label="features available")
    axes[0].set_title("features surviving each consent configuration", fontsize=10)

    null_rates = pd.DataFrame({
        source: pd.Series(manifest["sources"][source]["null_rate"])
        for source in sources if source in manifest["sources"]
    })
    null_rates = null_rates.reindex(catalogue.names()).fillna(1.0)
    image = axes[1].imshow(null_rates.T, cmap="magma_r", aspect="auto", vmin=0, vmax=1)
    axes[1].set_xticks(range(len(null_rates)), null_rates.index, rotation=90, fontsize=5)
    axes[1].set_yticks(range(null_rates.shape[1]), null_rates.columns, fontsize=8)
    fig.colorbar(image, ax=axes[1], shrink=0.8, label="null rate")
    axes[1].set_title("missingness per feature (1.0 = the source cannot supply it)", fontsize=10)
    fig.suptitle("f05-13 Missingness and the privacy–utility configurations", fontsize=12)
    fig.tight_layout()
    save(fig, "f05-13_missingness-and-consent-impact.png")
    return {
        "features_per_configuration": {
            label: {source: int(counts.loc[label, source]) for source in sources}
            for label in configurations
        },
    }


# ------------------------------------------------------------------- shortlist

def constant_features(frames: dict[str, pd.DataFrame]) -> dict[str, list[str]]:
    """Features that are present but take one value, per source.

    A constant is not an uninformative feature, it is a feature this deployment
    cannot vary — `n_options` is 4 for every item in a bank that is 100 % MCQ.
    Saying so is more useful than reporting its mutual information as zero.
    """
    constants: dict[str, list[str]] = {}
    for source, frame in frames.items():
        for name in catalogue.available(source):
            if frame[name].notna().any() and frame[name].nunique(dropna=True) <= 1:
                constants.setdefault(name, []).append(source)
    return constants


def shortlist(mi: dict[str, dict[str, float]], vif: dict[str, float],
              importance: dict[str, float], redundant: list[dict],
              manifest: dict, constants: dict[str, list[str]], examined: list[str]) -> dict:
    """Which features earn their place, and why the rest were dropped.

    Evidence, not intuition (BUILD.md Phase 5 step 5). The bar is deliberately
    low: this drops what is *empty*, not what is merely small. Deciding what is
    worth its privacy cost is the ablation ladder's job in Phase 6, and doing it
    here on the same data would be selection on the outcome.
    """
    kept, dropped = [], {}
    sim_mi = mi.get(SIM_SOURCE, {})
    best_real = {
        name: max((mi[source].get(name, 0.0) for source in REAL_SOURCES if source in mi), default=0.0)
        for name in catalogue.names()
    }
    redundant_of = {}
    for pair in redundant:
        loser = min((pair["a"], pair["b"]), key=lambda name: sim_mi.get(name, 0.0))
        winner = pair["a"] if loser == pair["b"] else pair["b"]
        redundant_of[loser] = pair | {"winner": winner}

    for feature in catalogue.CATALOGUE:
        name = feature.name
        # Only the sources this run actually loaded may vote: the constancy
        # check has not looked at the others, so counting them as evidence of
        # variation would be asserting something unmeasured.
        available_anywhere = [
            source for source in examined
            if manifest["sources"].get(source, {}).get("null_rate", {}).get(name, 1.0) < 1.0
        ]
        varies_anywhere = [source for source in available_anywhere
                           if source not in constants.get(name, [])]
        information = max(sim_mi.get(name, 0.0), best_real.get(name, 0.0))
        if not available_anywhere:
            dropped[name] = "no source supplies it — every matrix is null"
        elif not varies_anywhere:
            dropped[name] = (f"constant in every source that supplies it "
                             f"({', '.join(constants.get(name, []))}) — this deployment cannot vary it")
        elif name in redundant_of:
            pair = redundant_of[name]
            caveat = (" — measured on simulated data only, so this is a property of the generative "
                      "model until real telemetry can separate them"
                      if pair["simulated_only"] else "")
            dropped[name] = (f"redundant with {pair['winner']} (|r| > {REDUNDANT_ABS_CORRELATION} in "
                             f"{', '.join(pair['sources_checked'])}: {pair['r_by_source']}), which "
                             f"carries more information{caveat}")
        elif information < MIN_MUTUAL_INFORMATION and importance.get(name, 0.0) <= 0:
            dropped[name] = (f"mutual information {information:.2e} below {MIN_MUTUAL_INFORMATION:.0e} "
                             f"and permutation importance {importance.get(name, 0.0):+.6f}")
        else:
            kept.append(name)

    return {
        "criteria": {
            "min_mutual_information": MIN_MUTUAL_INFORMATION,
            "redundant_abs_correlation": REDUNDANT_ABS_CORRELATION,
            "sources_examined": examined,
            "note": "a feature survives if it is available somewhere, not duplicated by a "
                    "better-informed twin, and not simultaneously uninformative and unimportant. "
                    "Value is decided by Phase 6's ablation ladder, not here.",
        },
        "kept": kept,
        "kept_by_class": {
            signal: [name for name in kept if catalogue.BY_NAME[name].signal_class == signal]
            for signal in CLASS_COLOURS
        },
        "dropped": dropped,
        "high_collinearity_vif_above_10": {
            name: value for name, value in sorted(vif.items(), key=lambda item: -item[1]) if value > 10
        },
    }


# ------------------------------------------------------------------------ main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--sample", type=int, default=SAMPLE_ROWS)
    args = parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((build_features.MANIFEST_PATH).read_text(encoding="utf-8"))

    frames = {source: load(source, args.sample, args.seed)
              for source in REAL_SOURCES + [SIM_SOURCE]}
    simulated = frames[SIM_SOURCE]
    sim_features = usable(simulated, SIM_SOURCE)

    print(f"loaded {len(frames)} sources, {sum(len(f) for f in frames.values()):,} sampled rows")

    scores = {source: mutual_information(frame, usable(frame, source), args.seed)
              for source, frame in frames.items()}
    vif = variance_inflation(simulated, sim_features)
    importance, baseline_auc = quick_baseline(simulated, sim_features, args.seed)

    fig_distributions(frames)
    correlation = fig_correlation(simulated, sim_features)
    fig_mutual_information(scores)
    rt_summary = fig_rt_by_difficulty(frames)
    option_summary = fig_option_changes(simulated)
    session_summary = fig_within_session(frames, args.seed)
    fig_trajectory_violins(simulated)
    rq2 = fig_trajectory_vs_rt(simulated)
    stability = fig_stability(simulated, sim_features)
    overlap = fig_class_overlap(simulated, args.seed)
    archetypes = fig_archetypes(simulated)
    consent = fig_consent(frames, manifest)

    duplicates = redundant_pairs(frames)
    constants = constant_features(frames)
    chosen = shortlist(scores, vif, importance, duplicates, manifest, constants, list(frames))

    # H2's verdict may only be voted on by sources that produced a fit.
    # `assistments_2009` has no wall clock, so it has no sessions and no slope;
    # counting its silence as agreement either way would be inventing evidence.
    accuracy_fits = session_summary["accuracy"]
    measured = [source for source in REAL_SOURCES
                if accuracy_fits.get(source, {}).get("slope") is not None]
    declining = [source for source in measured if (accuracy_fits[source]["ci_high"] or 0) < 0]
    not_measurable = [source for source in REAL_SOURCES if source not in measured]
    summary = {
        "seed": args.seed,
        "sample_rows_per_source": args.sample,
        "git_sha": build_features.git_sha(),
        "features_in_catalogue": len(catalogue.CATALOGUE),
        "sources": {source: {"rows_sampled": int(len(frame)),
                             "features_usable": len(usable(frame, source)),
                             "accuracy": round(float(frame["correct"].mean()), 4),
                             "next_correct_rate": round(float(frame[LABEL].mean()), 4)}
                    for source, frame in frames.items()},
        "mutual_information": {source: {name: round(value, 6) for name, value in scores[source].items()}
                               for source in scores},
        "variance_inflation": vif,
        "permutation_importance_simulated": importance,
        "quick_baseline_auc_simulated": round(baseline_auc, 4) if np.isfinite(baseline_auc) else None,
        "correlation": correlation | {"redundant_across_sources": duplicates},
        "constant_features": constants,
        "response_time_vs_correctness": rt_summary,
        "option_changes": option_summary,
        "within_session": session_summary,
        "rq2_trajectory_beyond_response_time": rq2,
        "feature_stability": stability,
        "signal_class_overlap": overlap,
        "archetype_profiles": archetypes,
        "consent_configurations": consent,
        "h2_fatigue_verdict": {
            "preregistered_rule": "docs/preregistration.md H2 — if f05-06 is flat, the fatigue "
                                  "construct is reduced or dropped rather than defended",
            "real_sources_measured": measured,
            "real_sources_not_measurable": {
                source: "no wall-clock column, so no sessions and no within-session position"
                for source in not_measurable
            },
            "real_sources_declining": declining,
            "slopes": {source: accuracy_fits[source] for source in measured},
            "decision": ("retain — at least one real source declines within session"
                         if declining else
                         "reduce or drop — no real source shows a within-session accuracy decline"),
        },
    }
    SUMMARY_PATH.write_text(json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote: {SUMMARY_PATH.relative_to(ROOT)}")
    SHORTLIST_PATH.write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {SHORTLIST_PATH.relative_to(ROOT)} "
          f"({len(chosen['kept'])} kept, {len(chosen['dropped'])} dropped)")
    print(f"H2: {summary['h2_fatigue_verdict']['decision']}")


if __name__ == "__main__":
    main()
