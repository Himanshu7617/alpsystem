"""Phase 6 figure pack — Study 1's fourteen charts.

    python -m app.research.figures_phase6 [--seed 20260821]

Reads `artifacts/benchmarks/study1-cv.json`,
`artifacts/benchmarks/study1-final.json` and
`artifacts/benchmarks/study1-predictions.parquet`. Every number drawn here is
already in one of those files — this script computes nothing that the summary
does not also record, so a figure and the JSON can never disagree.

The headline is **f06-02**. BUILD.md Phase 6 step 3 fixes how to read it: pyKT
(K12) attributes 1–2 % AUC to protocol variation alone, so the figure draws a
shaded band at ±2 % and every rung is marked *clears* or *inside protocol
noise*. An increment inside the band is reported as a null, not as a small win.
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
from sklearn.metrics import (auc as auc_of, confusion_matrix, precision_recall_curve,
                             roc_auc_score, roc_curve)

from app.research import build_features, feature_catalogue as catalogue, study1

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "artifacts" / "figures" / "06-study1"
BENCH_DIR = ROOT / "artifacts" / "benchmarks"

LABEL = study1.LABEL
BAND = study1.PROTOCOL_NOISE_BAND
SAINT_PLUS = study1.SAINT_PLUS_REFERENCE

CLASS_COLOURS = {
    "correctness": "#0072B2", "timing": "#E69F00", "interaction": "#009E73",
    "session": "#D55E00", "motor": "#CC79A7",
}
RUNG_COLOURS = {level: CLASS_COLOURS[catalogue.LEVEL_CLASS[level]] for level in catalogue.LEVELS}
SOURCE_COLOURS = {
    "assistments_2009": "#0072B2", "assistments_2012": "#E69F00",
    "ednet_kt1": "#009E73", "sim-V0": "#CC79A7",
}


#: Every file this script reads. `figure_index` copies it into INDEX.md.
SOURCES = [
    "artifacts/benchmarks/study1-cv.json",
    "artifacts/benchmarks/study1-final.json",
    "artifacts/benchmarks/study1-predictions.parquet",
]


def save(fig: plt.Figure, name: str) -> None:
    path = plotstyle.save(fig, FIG_DIR, name)
    print(f"wrote: {path.relative_to(ROOT)} (+ .pdf)")


def simulated_note(axis: plt.Axes) -> None:
    """Standing guardrail 6: simulated results are labelled simulated."""
    axis.text(0.99, 0.02, "SIMULATED", transform=axis.transAxes, ha="right", va="bottom",
              fontsize=7, color="#CC79A7", alpha=0.85, fontweight="bold")


def _column(predictions: pd.DataFrame, source: str, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = predictions[predictions["source"] == source]
    scored = frame[key].notna()
    return (frame.loc[scored, LABEL].to_numpy(), frame.loc[scored, key].to_numpy(),
            frame.loc[scored, "learner_id"].to_numpy())


# ------------------------------------------------------------------- figures

def fig_model_comparison(report: dict) -> None:
    """f06-01 — every model on every dataset at its top rung, with CIs."""
    sources = list(report["sources"])
    rows = []
    for source in sources:
        result = report["sources"][source]
        top = result["live_rungs"][-1]
        for key, cell in result["cells"].items():
            if cell["rung"] == top:
                rows.append({"source": source, "model": cell["model"], "auc": cell["auc"],
                             "low": cell["ci_low"], "high": cell["ci_high"], "kind": "tabular"})
        for name, cell in (result.get("baselines") or {}).items():
            if cell.get("auc") is not None:
                rows.append({"source": source, "model": name, "auc": cell["auc"],
                             "low": cell.get("ci_low", cell["auc"]),
                             "high": cell.get("ci_high", cell["auc"]), "kind": "baseline"})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    models = frame.groupby("model")["auc"].mean().sort_values().index.tolist()

    fig, axis = plt.subplots(figsize=(10, 0.32 * len(models) + 2.5))
    width = 0.8 / max(1, len(sources))
    for index, source in enumerate(sources):
        subset = frame[frame["source"] == source].set_index("model").reindex(models)
        positions = np.arange(len(models)) + index * width - 0.4 + width / 2
        axis.barh(positions, subset["auc"].fillna(0), height=width,
                  color=SOURCE_COLOURS.get(source, "grey"), label=source)
        axis.errorbar(subset["auc"], positions,
                      xerr=[subset["auc"] - subset["low"], subset["high"] - subset["auc"]],
                      fmt="none", ecolor="black", elinewidth=0.7, capsize=1.5)
    axis.axvline(0.5, ls="--", color="grey", lw=1)
    axis.set_yticks(range(len(models)),
                    [f"{name}  ({'baseline' if frame[frame['model'] == name]['kind'].iloc[0] == 'baseline' else 'tabular'})"
                     for name in models], fontsize=8)
    axis.set_xlim(0.45, max(0.8, float(frame["high"].max()) + 0.02))
    axis.set_xlabel("ROC-AUC, out of fold (learner-clustered 95 % CI)")
    axis.set_title("f06-01 Every model at its top rung, on every dataset", fontsize=12)
    axis.legend(fontsize=8)
    save(fig, "f06-01_model-comparison-auc.png")


def _ladder_rows(report: dict, model: str | None = None) -> pd.DataFrame:
    rows = []
    for source, result in report["sources"].items():
        chosen = model or result["selected_model"]
        for name, step in (result["ladder"].get(chosen) or {}).items():
            if step.get("delta_auc") is None:
                continue
            rows.append({"source": source, "model": chosen, "step": name, **step})
    return pd.DataFrame(rows)


def fig_ablation_ladder(report: dict) -> None:
    """f06-02 — the headline. ΔAUC per rung against the protocol-noise band."""
    frame = _ladder_rows(report)
    if frame.empty:
        return
    frame = frame.sort_values(["step", "source"]).reset_index(drop=True)

    fig, axis = plt.subplots(figsize=(10, 0.42 * len(frame) + 3))
    positions = np.arange(len(frame))
    axis.axvspan(-BAND, BAND, color="#cccccc", alpha=0.45, zorder=0,
                 label=f"±{BAND:.0%} protocol-noise band (pyKT, K12)")
    axis.axvline(0, color="black", lw=1)
    axis.axvline(SAINT_PLUS, color="#E69F00", ls=":", lw=1.4,
                 label=f"SAINT+ reference +{SAINT_PLUS:.2%} AUC (K8)")
    axis.barh(positions, frame["delta_auc"], height=0.55,
              color=[RUNG_COLOURS[step.split("->")[1]] for step in frame["step"]], zorder=2)
    axis.errorbar(frame["delta_auc"], positions,
                  xerr=[frame["delta_auc"] - frame["ci_low"], frame["ci_high"] - frame["delta_auc"]],
                  fmt="none", ecolor="black", elinewidth=0.9, capsize=2, zorder=3)
    for position, row in frame.iterrows():
        verdict = "CLEARS the band" if row["clears_protocol_band"] else "inside protocol noise"
        # A bootstrap p of exactly 0 means "no draw crossed zero", not "p = 0".
        holm = row.get("p_holm", float("nan"))
        formatted = f"< {1 / study1.BOOTSTRAP_DRAWS:.3f}" if holm == 0 else f"= {holm:.3g}"
        axis.text(max(row["ci_high"], BAND) + 0.002, position,
                  f"{verdict}   p(Holm) {formatted}",
                  va="center", fontsize=7,
                  color="#2a7f2a" if row["clears_protocol_band"] else "#b03030")
    axis.set_yticks(positions, [f"{row['source']}  {row['step']}  ({row['model']})"
                                for _, row in frame.iterrows()], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(min(-BAND * 1.4, frame["ci_low"].min() - 0.005), BAND * 3.2)
    axis.set_xlabel("Δ ROC-AUC from adding the rung (paired, learner-clustered 95 % CI)")
    axis.set_title("f06-02 The ablation ladder — does any behavioural rung clear protocol noise?\n"
                   "An increment whose CI does not exclude ±2 % is reported as a null "
                   "(BUILD.md Phase 6 step 3)", fontsize=11)
    axis.legend(fontsize=8, loc="lower right")
    save(fig, "f06-02_ablation-ladder.png")


def fig_ladder_per_dataset(report: dict) -> None:
    """f06-03 — the same ladder faceted, to show whether simulation inflates it."""
    sources = list(report["sources"])
    fig, axes = plt.subplots(1, len(sources), figsize=(3.6 * len(sources), 4.2),
                             squeeze=False, sharey=True)
    for axis, source in zip(axes[0], sources):
        result = report["sources"][source]
        model = result["selected_model"]
        steps = result["ladder"].get(model) or {}
        names = [name for name in steps if steps[name].get("delta_auc") is not None]
        values = [steps[name]["delta_auc"] for name in names]
        errors = [[steps[name]["delta_auc"] - steps[name]["ci_low"] for name in names],
                  [steps[name]["ci_high"] - steps[name]["delta_auc"] for name in names]]
        axis.axhspan(-BAND, BAND, color="#cccccc", alpha=0.45, zorder=0)
        axis.axhline(0, color="black", lw=1)
        axis.bar(range(len(names)), values,
                 color=[RUNG_COLOURS[name.split("->")[1]] for name in names], zorder=2)
        axis.errorbar(range(len(names)), values, yerr=errors, fmt="none",
                      ecolor="black", elinewidth=0.9, capsize=2, zorder=3)
        axis.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7)
        axis.set_title(f"{source}\n({model})", fontsize=9)
        if result["skipped_rungs"]:
            axis.text(0.5, 0.97, "not run: " + ", ".join(result["skipped_rungs"]),
                      transform=axis.transAxes, ha="center", va="top", fontsize=6.5, color="grey")
        if source.startswith("sim"):
            simulated_note(axis)
    axes[0][0].set_ylabel("Δ ROC-AUC (shaded = ±2 % protocol noise)")
    fig.suptitle("f06-03 Ablation ladder per dataset — real against simulated", fontsize=12)
    fig.tight_layout()
    save(fig, "f06-03_ablation-ladder-per-dataset.png")


def fig_curves(report: dict, predictions: pd.DataFrame) -> None:
    """f06-04 and f06-05 — ROC and precision-recall, top rung, per dataset."""
    sources = list(report["sources"])
    for filename, title, kind in [
        ("f06-04_roc-curves.png", "f06-04 ROC curves — top rung, out of fold", "roc"),
        ("f06-05_pr-curves.png", "f06-05 Precision-recall curves — top rung, out of fold", "pr"),
    ]:
        fig, axes = plt.subplots(1, len(sources), figsize=(3.5 * len(sources), 3.6),
                                 squeeze=False, sharey=True)
        for axis, source in zip(axes[0], sources):
            result = report["sources"][source]
            top = result["live_rungs"][-1]
            for key, cell in sorted(result["cells"].items(), key=lambda item: -item[1]["auc"]):
                if cell["rung"] != top or key not in predictions.columns:
                    continue
                labels, scores, _ = _column(predictions, source, key)
                if len(labels) < 100:
                    continue
                if kind == "roc":
                    x, y, _ = roc_curve(labels, scores)
                    score = roc_auc_score(labels, scores)
                else:
                    y, x, _ = precision_recall_curve(labels, scores)
                    score = auc_of(x, y)
                axis.plot(x, y, lw=1.2, label=f"{cell['model']} {score:.3f}")
            if kind == "roc":
                axis.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
                axis.set_xlabel("false positive rate")
            else:
                axis.axhline(float(predictions.loc[predictions["source"] == source, LABEL].mean()),
                             ls="--", color="grey", lw=1)
                axis.set_xlabel("recall")
            axis.set_title(source, fontsize=9)
            axis.legend(fontsize=6.5, loc="lower right" if kind == "roc" else "upper right")
            if source.startswith("sim"):
                simulated_note(axis)
        axes[0][0].set_ylabel("true positive rate" if kind == "roc" else "precision")
        fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        save(fig, filename)


def fig_calibration(report: dict, predictions: pd.DataFrame) -> None:
    """f06-06 — reliability before and after isotonic, ECE annotated.

    The metric BUILD.md Phase 6 step 4 puts ahead of AUC, because the output
    drives a decision and AUC cannot see a miscalibrated one.
    """
    sources = [source for source in report["sources"] if report["sources"][source].get("calibration")]
    if not sources:
        return
    fig, axes = plt.subplots(1, len(sources), figsize=(3.6 * len(sources), 3.8),
                             squeeze=False, sharey=True)
    for axis, source in zip(axes[0], sources):
        calibration = report["sources"][source]["calibration"]
        top, model = calibration["rung"], calibration["model"]
        for key, style, colour, name in [
            (f"{top}|{model}", "-", "#0072B2", "raw"),
            (f"{top}|{model}|calibrated", "-", "#D55E00", "isotonic"),
        ]:
            if key not in predictions.columns:
                continue
            labels, scores, _ = _column(predictions, source, key)
            edges = np.linspace(0, 1, 16)
            index = np.clip(np.digitize(scores, edges[1:-1]), 0, 14)
            observed, predicted = [], []
            for bucket in range(15):
                mask = index == bucket
                if mask.sum() < 30:
                    continue
                observed.append(labels[mask].mean())
                predicted.append(scores[mask].mean())
            ece = calibration["raw" if name == "raw" else "calibrated"]["ece"]
            axis.plot(predicted, observed, style, marker="o", ms=3, color=colour,
                      label=f"{name}  ECE {ece:.4f}")
        axis.plot([0, 1], [0, 1], ls="--", color="grey", lw=1)
        axis.set_xlabel("predicted probability")
        axis.set_title(f"{source}\n{model} at {top}", fontsize=9)
        axis.legend(fontsize=7.5)
        if source.startswith("sim"):
            simulated_note(axis)
    axes[0][0].set_ylabel("observed frequency")
    fig.suptitle("f06-06 Calibration — before and after isotonic, fitted on training folds only",
                 fontsize=12)
    fig.tight_layout()
    save(fig, "f06-06_calibration-curves.png")


def _shap_values(source: str, seed: int, rows: int = 4_000):
    """SHAP on the selected tree model, refitted on training folds.

    Refitted here rather than persisted from `study1.py`: SHAP needs the model
    *and* a background sample, and one extra fit is cheaper than carrying a
    pickle whose only consumer is three figures.
    """
    import shap
    from sklearn.base import clone

    report = json.loads((BENCH_DIR / "study1-cv.json").read_text(encoding="utf-8"))
    result = report["sources"][source]
    models, _ = study1.tabular_models(seed)
    name = result["selected_model"]
    if name not in models:
        return None
    columns = catalogue.available(source, upto=result["live_rungs"][-1])
    frame = study1.load(source, seed)
    train = frame[~frame["split"].isin({"test"})]

    estimator = clone(models[name])
    if result["tuned_hyperparameters"].get(name):
        estimator.set_params(**result["tuned_hyperparameters"][name])
    pipeline = study1.pipeline_for(estimator, columns)
    pipeline.fit(train[columns], train[LABEL])

    sample = train.sample(min(rows, len(train)), random_state=seed)
    prepared = pipeline.named_steps["prepare"].transform(sample[columns])
    try:
        explainer = shap.TreeExplainer(pipeline.named_steps["model"])
        values = explainer.shap_values(prepared)
    except Exception:  # noqa: BLE001 - a non-tree winner simply has no TreeExplainer
        return None
    if isinstance(values, list):
        values = values[1]
    if values.ndim == 3:
        values = values[:, :, -1]
    return {"values": np.asarray(values), "prepared": np.asarray(prepared),
            "columns": columns, "model": name, "sample": sample}


def fig_shap(source: str, seed: int) -> dict:
    """f06-07, f06-08, f06-09 — global, by signal class, and the RQ2 dependence."""
    explained = _shap_values(source, seed)
    if explained is None:
        print(f"skipped SHAP figures: no tree explainer for {source}")
        return {}
    values, prepared = explained["values"], explained["prepared"]
    columns = explained["columns"]
    magnitude = np.abs(values).mean(axis=0)
    order = np.argsort(magnitude)[-25:]

    # ---- f06-07 beeswarm
    fig, axis = plt.subplots(figsize=(9, 0.3 * len(order) + 2))
    rng = np.random.default_rng(seed)
    for row, index in enumerate(order):
        column = prepared[:, index]
        finite = np.isfinite(column)
        if finite.sum() > 1:
            span = np.nanpercentile(column[finite], [5, 95])
            normalised = np.clip((column - span[0]) / max(1e-9, span[1] - span[0]), 0, 1)
        else:
            normalised = np.zeros_like(column)
        axis.scatter(values[:, index], row + rng.normal(0, 0.11, len(values)),
                     c=normalised, cmap="coolwarm", s=3, alpha=0.35, linewidths=0)
    axis.set_yticks(range(len(order)), [columns[index] for index in order], fontsize=7)
    axis.axvline(0, color="grey", lw=1)
    axis.set_xlabel("SHAP value (log-odds of the next answer being correct)")
    axis.set_title(f"f06-07 SHAP beeswarm — {explained['model']} on {source}\n"
                   "colour = feature value, low to high", fontsize=11)
    if source.startswith("sim"):
        simulated_note(axis)
    save(fig, "f06-07_shap-summary-beeswarm.png")

    # ---- f06-08 by signal class
    by_class: dict[str, float] = {}
    for index, name in enumerate(columns):
        signal = catalogue.BY_NAME[name].signal_class
        by_class[signal] = by_class.get(signal, 0.0) + float(magnitude[index])
    total = sum(by_class.values()) or 1.0
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1, 1.5]})
    signals = list(by_class)
    axes[0].bar(signals, [by_class[name] / total for name in signals],
                color=[CLASS_COLOURS[name] for name in signals])
    for position, name in enumerate(signals):
        axes[0].text(position, by_class[name] / total + 0.01,
                     f"{by_class[name] / total:.1%}", ha="center", fontsize=8)
    axes[0].set_ylabel("share of total |SHAP|")
    axes[0].set_title("information budget by signal class", fontsize=10)
    top = np.argsort(magnitude)[-18:]
    axes[1].barh(range(len(top)), magnitude[top],
                 color=[CLASS_COLOURS[catalogue.BY_NAME[columns[index]].signal_class]
                        for index in top])
    axes[1].set_yticks(range(len(top)), [columns[index] for index in top], fontsize=7)
    axes[1].set_xlabel("mean |SHAP|")
    axes[1].set_title("per feature, coloured by class", fontsize=10)
    if source.startswith("sim"):
        simulated_note(axes[0])
    fig.suptitle(f"f06-08 Where the model's information comes from — {source}", fontsize=12)
    fig.tight_layout()
    save(fig, "f06-08_shap-bar-by-signal-class.png")

    # ---- f06-09 the RQ2 dependence plot
    motor = [index for index, name in enumerate(columns)
             if catalogue.BY_NAME[name].signal_class == "motor"]
    summary = {"shap_share_by_class": {name: round(value / total, 5) for name, value in by_class.items()},
               "top_features": {columns[index]: round(float(magnitude[index]), 6)
                                for index in reversed(np.argsort(magnitude)[-10:])}}
    if motor:
        best = max(motor, key=lambda index: magnitude[index])
        rt = columns.index("log_response_time") if "log_response_time" in columns else None
        fig, axis = plt.subplots(figsize=(7.5, 4.6))
        colour = prepared[:, rt] if rt is not None else None
        scatter = axis.scatter(prepared[:, best], values[:, best], c=colour, cmap="viridis",
                               s=6, alpha=0.5, linewidths=0)
        if colour is not None:
            fig.colorbar(scatter, ax=axis, label="log response time (standardised)")
        axis.axhline(0, color="grey", lw=1)
        axis.set_xlabel(f"{columns[best]} (standardised)")
        axis.set_ylabel("SHAP value")
        axis.set_title(f"f06-09 RQ2 in one picture — {columns[best]} against its SHAP value\n"
                       f"mean |SHAP| {magnitude[best]:.5f}, "
                       f"{magnitude[best] / total:.2%} of the model's total", fontsize=11)
        simulated_note(axis)
        save(fig, "f06-09_shap-dependence-trajectory.png")
        summary["rq2_top_motor_feature"] = {
            "feature": columns[best],
            "mean_abs_shap": round(float(magnitude[best]), 6),
            "share_of_total": round(float(magnitude[best] / total), 5),
        }
    return summary


def fig_learning_curves(report: dict) -> None:
    """f06-10 — AUC against training-set size. Is deep KT even justified here?"""
    fig, axis = plt.subplots(figsize=(8, 4.4))
    drawn = False
    for source, result in report["sources"].items():
        points = result.get("learning_curve") or []
        if not points:
            continue
        axis.plot([point["train_learners"] for point in points],
                  [point["auc"] for point in points], marker="o", ms=4,
                  color=SOURCE_COLOURS.get(source, "grey"),
                  label=f"{source} ({result['selected_model']})")
        drawn = True
    if not drawn:
        plt.close(fig)
        return
    axis.set_xscale("log")
    axis.set_xlabel("training learners (log scale)")
    axis.set_ylabel("ROC-AUC on a held-out fold")
    axis.set_title("f06-10 Learning curves — a flat tail means more data is not what is missing\n"
                   "(K13: when is deep learning the best approach to knowledge tracing?)",
                   fontsize=11)
    axis.legend(fontsize=8)
    save(fig, "f06-10_learning-curves.png")


def fig_subgroups(report: dict, predictions: pd.DataFrame, seed: int) -> dict:
    """f06-11 — performance by prior-ability tercile.

    The literature notes an RL tutor helped **lower performers specifically**
    (M76), so an aggregate null can hide a real subgroup effect. Always report
    by tercile.
    """
    rows = []
    for source, result in report["sources"].items():
        top, model = result["live_rungs"][-1], result["selected_model"]
        key = f"{top}|{model}"
        if key not in predictions.columns:
            continue
        frame = predictions[(predictions["source"] == source) & predictions[key].notna()].copy()
        ability = frame.groupby("learner_id")["prior_accuracy"].transform("mean")
        try:
            frame["tercile"] = pd.qcut(ability, 3, labels=["low", "middle", "high"], duplicates="drop")
        except ValueError:
            continue
        for tercile, subset in frame.groupby("tercile", observed=True):
            if subset[LABEL].nunique() < 2:
                continue
            rows.append({
                "source": source, "tercile": str(tercile),
                "auc": round(float(roc_auc_score(subset[LABEL], subset[key])), 5),
                "n": int(len(subset)),
                "base_rate": round(float(subset[LABEL].mean()), 4),
            })
    frame = pd.DataFrame(rows)
    if frame.empty:
        return {}

    fig, axis = plt.subplots(figsize=(8.5, 4.2))
    terciles = ["low", "middle", "high"]
    width = 0.8 / max(1, frame["source"].nunique())
    for index, (source, subset) in enumerate(frame.groupby("source")):
        ordered = subset.set_index("tercile").reindex(terciles)
        axis.bar(np.arange(3) + index * width - 0.4 + width / 2, ordered["auc"].fillna(0),
                 width=width, color=SOURCE_COLOURS.get(source, "grey"), label=source)
    axis.axhline(0.5, ls="--", color="grey", lw=1)
    axis.set_xticks(range(3), [f"{name} prior ability" for name in terciles])
    axis.set_ylim(0.45, max(0.8, float(frame["auc"].max()) + 0.03))
    axis.set_ylabel("ROC-AUC, out of fold")
    axis.set_title("f06-11 Subgroup performance by prior-ability tercile\n"
                   "an aggregate null can hide a subgroup effect", fontsize=11)
    axis.legend(fontsize=8)
    save(fig, "f06-11_subgroup-performance.png")
    return {row["source"] + "|" + row["tercile"]: row for row in rows}


def fig_confusion(report: dict, predictions: pd.DataFrame) -> None:
    """f06-12 — confusion matrices at the 0.5 threshold, top rung."""
    sources = list(report["sources"])
    fig, axes = plt.subplots(1, len(sources), figsize=(3.1 * len(sources), 3.3), squeeze=False)
    for axis, source in zip(axes[0], sources):
        result = report["sources"][source]
        key = f"{result['live_rungs'][-1]}|{result['selected_model']}"
        if key not in predictions.columns:
            axis.axis("off")
            continue
        labels, scores, _ = _column(predictions, source, key)
        matrix = confusion_matrix(labels, (scores >= 0.5).astype(int), normalize="true")
        axis.imshow(matrix, cmap="Blues", vmin=0, vmax=1)
        for i in range(2):
            for j in range(2):
                axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center",
                          fontsize=10, color="white" if matrix[i, j] > 0.6 else "black")
        axis.set_xticks([0, 1], ["pred wrong", "pred right"], fontsize=8)
        axis.set_yticks([0, 1], ["was wrong", "was right"], fontsize=8)
        axis.set_title(f"{source}\n{result['selected_model']}", fontsize=9)
        if source.startswith("sim"):
            simulated_note(axis)
    fig.suptitle("f06-12 Confusion at the 0.5 threshold, row-normalised", fontsize=12)
    fig.tight_layout()
    save(fig, "f06-12_confusion-matrices.png")


def fig_cost_benefit(report: dict) -> None:
    """f06-13 — training seconds against AUC. Expect deep models to lose."""
    rows = []
    for source, result in report["sources"].items():
        top = result["live_rungs"][-1]
        for cell in result["cells"].values():
            if cell["rung"] == top:
                rows.append({"source": source, "model": cell["model"], "auc": cell["auc"],
                             "seconds": max(cell.get("train_seconds", 0.0), 0.05),
                             "kind": "tabular"})
        for name, cell in (result.get("baselines") or {}).items():
            if cell.get("auc") is not None:
                rows.append({"source": source, "model": name, "auc": cell["auc"],
                             "seconds": max(cell.get("train_seconds", 0.0), 0.05),
                             "kind": "deep" if name in ("dkt", "sakt", "akt", "saint") else "baseline"})
    frame = pd.DataFrame(rows)
    if frame.empty:
        return

    markers = {"tabular": "o", "baseline": "s", "deep": "^"}
    fig, axis = plt.subplots(figsize=(8.5, 5))
    for kind, subset in frame.groupby("kind"):
        axis.scatter(subset["seconds"], subset["auc"], marker=markers[kind], s=48,
                     c=[SOURCE_COLOURS.get(source, "grey") for source in subset["source"]],
                     edgecolor="black", linewidth=0.4, label=kind)
    for _, row in frame.iterrows():
        axis.annotate(row["model"], (row["seconds"], row["auc"]), fontsize=6,
                      xytext=(3, 3), textcoords="offset points")
    axis.set_xscale("log")
    axis.axhline(0.5, ls="--", color="grey", lw=1)
    axis.set_xlabel("training seconds (log scale)")
    axis.set_ylabel("ROC-AUC, out of fold")
    axis.set_title("f06-13 Cost against benefit — what does the extra compute buy?", fontsize=11)
    axis.legend(fontsize=8)
    save(fig, "f06-13_train-time-vs-auc.png")


def fig_incremental_trajectory(report: dict, predictions: pd.DataFrame, seed: int) -> dict:
    """f06-14 — the direct RQ2 test: RT-only against RT-plus-trajectory, per learner.

    Paired by learner, so the question is not "is the trajectory model better on
    average" but "is it better *for this learner*, and how often". A mean that
    hides an even split of winners and losers is not an effect.
    """
    sources = [source for source, result in report["sources"].items()
               if "L4" in result["live_rungs"]]
    if not sources:
        print("skipped f06-14: no source carries motor features")
        return {}
    summary = {}
    fig, axes = plt.subplots(1, len(sources) * 2, figsize=(5.5 * len(sources), 4.2), squeeze=False)
    for index, source in enumerate(sources):
        result = report["sources"][source]
        model = result["selected_model"]
        lower_rung = result["live_rungs"][result["live_rungs"].index("L4") - 1]
        low_key, high_key = f"{lower_rung}|{model}", f"L4|{model}"
        if low_key not in predictions.columns or high_key not in predictions.columns:
            continue
        frame = predictions[(predictions["source"] == source)
                            & predictions[low_key].notna()
                            & predictions[high_key].notna()].copy()

        deltas = []
        for learner, subset in frame.groupby("learner_id", observed=True):
            if subset[LABEL].nunique() < 2 or len(subset) < 20:
                continue
            deltas.append(roc_auc_score(subset[LABEL], subset[high_key])
                          - roc_auc_score(subset[LABEL], subset[low_key]))
        deltas = np.asarray(deltas)
        if not len(deltas):
            continue

        left, right = axes[0][index * 2], axes[0][index * 2 + 1]
        left.hist(deltas, bins=40, color="#CC79A7", alpha=0.8)
        left.axvline(0, color="black", lw=1.2)
        left.axvline(float(deltas.mean()), color="#D55E00", lw=1.4,
                     label=f"mean {deltas.mean():+.4f}")
        left.set_xlabel(f"per-learner ΔAUC, {lower_rung} → L4")
        left.set_ylabel("learners")
        left.legend(fontsize=8)
        simulated_note(left)

        improved = float((deltas > 0).mean())
        right.bar(["improved", "unchanged", "worse"],
                  [improved, float((deltas == 0).mean()), float((deltas < 0).mean())],
                  color=["#009E73", "#cccccc", "#D55E00"])
        right.axhline(0.5, ls="--", color="grey", lw=1)
        right.set_ylabel("share of learners")
        right.set_title(f"{improved:.1%} of learners improved", fontsize=9)
        simulated_note(right)
        summary[source] = {
            "learners_compared": int(len(deltas)),
            "mean_delta_auc": round(float(deltas.mean()), 5),
            "median_delta_auc": round(float(np.median(deltas)), 5),
            "share_improved": round(improved, 4),
            "from_rung": lower_rung,
        }
    fig.suptitle("f06-14 Adding trajectory features on top of response time, paired by learner",
                 fontsize=12)
    fig.tight_layout()
    save(fig, "f06-14_incremental-value-trajectory-over-rt.png")
    return summary


# ---------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--shap-source", default="sim-V0",
                        help="the source SHAP is computed on — must carry motor features for f06-09")
    args = parser.parse_args()

    report = json.loads((BENCH_DIR / "study1-cv.json").read_text(encoding="utf-8"))
    predictions = pd.read_parquet(BENCH_DIR / "study1-predictions.parquet")
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig_model_comparison(report)
    fig_ablation_ladder(report)
    fig_ladder_per_dataset(report)
    fig_curves(report, predictions)
    fig_calibration(report, predictions)
    shap_summary = fig_shap(args.shap_source, args.seed) if args.shap_source in report["sources"] else {}
    fig_learning_curves(report)
    subgroups = fig_subgroups(report, predictions, args.seed)
    fig_confusion(report, predictions)
    fig_cost_benefit(report)
    incremental = fig_incremental_trajectory(report, predictions, args.seed)

    # The figures compute three things the cross-validation pass does not, so
    # they are written back rather than left living only inside a PNG — no
    # number in a doc may come from a picture.
    report["figure_findings"] = {
        "shap": shap_summary, "subgroups": subgroups,
        "incremental_trajectory": incremental,
    }
    (BENCH_DIR / "study1-cv.json").write_text(
        json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote: {(BENCH_DIR / 'study1-cv.json').relative_to(ROOT)} (figure findings appended)")


if __name__ == "__main__":
    main()
