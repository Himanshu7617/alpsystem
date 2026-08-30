"""Phase 7 figure pack — Study 2's ten charts.

    python -m app.research.figures_phase7 [--seed 20260821]

Reads `artifacts/benchmarks/study2-heads.json`,
`artifacts/benchmarks/study2-predictions.parquet` and
`artifacts/evaluation/validation-gate.json`. Everything drawn here is already in
one of those files, or is computed here and written back into the JSON before
the run ends — no number in a doc may come from a picture.

Two of the ten figures are deliberately negative. **f07-04** prints failures
next to passes because a scorecard that only shows passes is an advertisement,
and **f07-10** documents that no fatigue head exists: the construct was dropped
in Phase 5 when no archival source showed a within-session accuracy decline.
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

from app.model import validation_gate
from app.research import study2

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "artifacts" / "figures" / "07-study2"
BENCH_DIR = ROOT / "artifacts" / "benchmarks"

HEAD_COLOURS = {"knowledge": "#0072B2", "engagement": "#009E73", "confidence": "#E69F00"}
PASS_COLOUR, FAIL_COLOUR = "#009E73", "#D55E00"
SIM_SOURCES = {"sim-V0"}


#: Every file this script reads. `figure_index` copies it into INDEX.md.
SOURCES = [
    "artifacts/benchmarks/study2-heads.json",
    "artifacts/benchmarks/study2-predictions.parquet",
    "artifacts/benchmarks/eda-summary.json",
    "artifacts/evaluation/validation-gate.json",
]


def save(fig: plt.Figure, name: str) -> None:
    path = plotstyle.save(fig, FIG_DIR, name)
    print(f"wrote: {path.relative_to(ROOT)} (+ .pdf)")


def simulated_note(axis: plt.Axes) -> None:
    """Standing guardrail 6: simulated results are labelled simulated."""
    axis.text(0.99, 0.02, "SIMULATED", transform=axis.transAxes, ha="right", va="bottom",
              fontsize=7, color="#CC79A7", alpha=0.85, fontweight="bold")


def scored(predictions: pd.DataFrame, head: str) -> pd.DataFrame:
    """The rows a head is judged on: its own label, on its gate source."""
    rows = predictions[(predictions["_half"] == "scoring")
                       & predictions["source"].eq(study2.GATE_SOURCE[head])
                       & predictions[f"y_{head}"].notna()]
    return rows


def reliability(labels: np.ndarray, probabilities: np.ndarray, bins: int = 10) -> tuple:
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    x, y, weight = [], [], []
    for bucket in range(bins):
        mask = index == bucket
        if mask.sum() < 20:
            continue
        x.append(probabilities[mask].mean())
        y.append(labels[mask].mean())
        weight.append(mask.sum())
    return np.array(x), np.array(y), np.array(weight)


# ------------------------------------------------------------------- figures

def fig_architecture(heads: dict) -> None:
    """f07-01 — what is shared, what is separately supervised, and by what.

    The figure exists to make one claim checkable at a glance: every head has
    its own label. A head supervised by correctness and *named* engagement is
    the failure mode this phase was built to eliminate.
    """
    fig, axis = plt.subplots(figsize=(11, 5.5))
    axis.set_xlim(0, 10)
    axis.set_ylim(0, 6)
    axis.axis("off")

    def box(x, y, w, h, text, colour, size=9):
        axis.add_patch(plt.Rectangle((x, y), w, h, facecolor=colour, edgecolor="#333",
                                     alpha=0.85, linewidth=1.2))
        axis.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
                  color="white", fontweight="bold", wrap=True)

    box(0.3, 2.4, 1.9, 1.2, f"interaction\nsequence\n{heads['ladder'][max(heads['ladder'])]['features']} features\n"
                            f"(value + present)", "#666666", 8)
    box(2.7, 2.4, 1.8, 1.2, f"shared GRU\nhidden {study2.HIDDEN}\ndropout {study2.DROPOUT}", "#333333", 8)
    axis.annotate("", xy=(2.7, 3.0), xytext=(2.2, 3.0), arrowprops=dict(arrowstyle="->", lw=1.5))

    for index, head in enumerate(study2.HEADS):
        y = 4.3 - index * 1.6
        box(5.2, y, 1.6, 1.0, f"{head}\nhead", HEAD_COLOURS[head], 9)
        axis.annotate("", xy=(5.2, y + 0.5), xytext=(4.5, 3.0),
                      arrowprops=dict(arrowstyle="->", lw=1.2, color="#888"))
        axis.text(7.0, y + 0.72, "supervised by", fontsize=7.5, color="#555", style="italic")
        axis.text(7.0, y + 0.30, heads["labels"][head], fontsize=7.5, color="#222", wrap=True)
        axis.text(7.0, y - 0.02, f"source: {heads['gate_source'][head]}", fontsize=7, color="#777")

    axis.text(5.2, 0.35, "fatigue: no head. Within-session accuracy decline not detected in any "
                         "archival source (preregistration §7) — the construct was dropped before "
                         "modelling, not modelled and hidden.",
              fontsize=8, color=FAIL_COLOUR, ha="left", va="center", wrap=True)
    axis.set_title("f07-01 — shared encoder, separately supervised heads", fontsize=11, loc="left")
    save(fig, "f07-01_state-head-architecture.png")


def fig_state_vs_label(predictions: pd.DataFrame, gate: dict) -> None:
    """f07-02 — the estimate against its own label, with r and its CI.

    Binary labels, so the panel bins the estimate and plots the observed rate:
    a scatter of 0/1 against a probability shows nothing.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, head in zip(axes, study2.HEADS):
        rows = scored(predictions, head)
        measurement = gate["measurements"].get(head, {})
        if rows.empty:
            axis.text(0.5, 0.5, f"{head}: not measured", ha="center", transform=axis.transAxes)
            axis.axis("off")
            continue
        probability = rows[f"{head}_p"].to_numpy()
        label = rows[f"y_{head}"].to_numpy()
        deciles = pd.qcut(probability, 10, duplicates="drop")
        grouped = pd.DataFrame({"p": probability, "y": label}).groupby(deciles, observed=True).mean()
        axis.plot(grouped["p"], grouped["y"], "o-", color=HEAD_COLOURS[head])
        axis.plot([0, 1], [0, 1], "--", color="#999", linewidth=1)
        r = measurement.get("label_correlation")
        low, high = (measurement.get("label_correlation_ci_low"),
                     measurement.get("label_correlation_ci_high"))
        axis.set_title(f"{head}\nr = {r} [{low}, {high}]", fontsize=9)
        axis.set_xlabel("estimated probability (decile mean)")
        axis.set_ylabel("observed rate")
        axis.set_xlim(0, 1)
        axis.set_ylim(0, 1)
        if study2.GATE_SOURCE[head] in SIM_SOURCES:
            simulated_note(axis)
    axes[3].axis("off")
    axes[3].text(0.5, 0.5, "fatigue\n\nno head.\nConstruct dropped in Phase 5:\nno archival source shows a\n"
                           "within-session accuracy decline\n(preregistration §7).",
                 ha="center", va="center", fontsize=9, color=FAIL_COLOUR)
    fig.suptitle("f07-02 — each state estimate against its own label (scoring half of fold4)",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "f07-02_state-vs-label-scatter.png")


def fig_calibration(predictions: pd.DataFrame, gate: dict) -> None:
    """f07-03 — reliability before and after isotonic, ECE annotated.

    Calibration is ahead of AUC in this phase because the policy acts on a
    threshold: 0.7 has to mean 0.7.
    """
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for axis, head in zip(axes, study2.HEADS):
        rows = scored(predictions, head)
        measurement = gate["measurements"].get(head, {})
        if rows.empty:
            axis.axis("off")
            continue
        label = rows[f"y_{head}"].to_numpy()
        for key, colour, style, name in (("p_raw", "#999999", "--", "raw"),
                                         (f"p", HEAD_COLOURS[head], "-", "isotonic")):
            column = f"{head}_{key}" if key != "p_raw" else f"{head}_p_raw"
            x, y, _ = reliability(label, rows[column].to_numpy())
            axis.plot(x, y, style, marker="o", color=colour, label=name, markersize=4)
        axis.plot([0, 1], [0, 1], ":", color="#333", linewidth=1)
        axis.set_title(f"{head}\nECE {measurement.get('ece_raw')} → {measurement.get('ece_calibrated')}"
                       f"  (bound {validation_gate.CRITERIA[1].bound})", fontsize=9)
        axis.set_xlabel("predicted")
        axis.set_ylabel("observed")
        axis.legend(fontsize=7)
        if study2.GATE_SOURCE[head] in SIM_SOURCES:
            simulated_note(axis)
    axes[3].axis("off")
    axes[3].text(0.5, 0.5, "fatigue\n\nno head, so nothing to calibrate.",
                 ha="center", va="center", fontsize=9, color=FAIL_COLOUR)
    fig.suptitle("f07-03 — calibration per head, before and after isotonic", fontsize=11)
    fig.tight_layout()
    save(fig, "f07-03_state-calibration-curves.png")


def fig_scorecard(gate: dict) -> None:
    """f07-04 — the gate, failures included.

    **The headline figure of the phase.** A state that fails is excluded from
    the policy in Phase 8, and that exclusion is a result.
    """
    criteria = [item.key for item in validation_gate.CRITERIA]
    states = list(validation_gate.STATES)
    fig, axis = plt.subplots(figsize=(11, 3.6))
    for row, state in enumerate(states):
        entry = gate["states"][state]
        checks = {check["criterion"]: check for check in entry["criteria"]}
        for column, criterion in enumerate(criteria):
            check = checks.get(criterion)
            if check is None:
                axis.add_patch(plt.Rectangle((column, row), 1, 1, facecolor="#eeeeee",
                                             edgecolor="white"))
                axis.text(column + 0.5, row + 0.5, "n/a", ha="center", va="center",
                          fontsize=8, color="#999")
                continue
            colour = PASS_COLOUR if check["passed"] else FAIL_COLOUR
            axis.add_patch(plt.Rectangle((column, row), 1, 1, facecolor=colour,
                                         edgecolor="white", alpha=0.85))
            value = "not measured" if check["value"] is None else f"{check['value']:.3f}"
            bound = ("≥" if check["direction"] == "min" else "≤") + f"{check['bound']}"
            axis.text(column + 0.5, row + 0.58, value, ha="center", va="center",
                      fontsize=9, color="white", fontweight="bold")
            axis.text(column + 0.5, row + 0.28, bound, ha="center", va="center",
                      fontsize=7.5, color="white")
        verdict = "ADMITTED" if entry["admitted"] else "EXCLUDED"
        axis.text(len(criteria) + 0.15, row + 0.5, verdict, va="center", fontsize=9,
                  color=PASS_COLOUR if entry["admitted"] else FAIL_COLOUR, fontweight="bold")
    axis.set_xlim(0, len(criteria) + 1.4)
    axis.set_ylim(0, len(states))
    axis.set_xticks(np.arange(len(criteria)) + 0.5)
    axis.set_xticklabels([key.split("_", 1)[0] + "\n" + key.split("_", 1)[1].replace("_", " ")
                          for key in criteria], fontsize=8)
    axis.set_yticks(np.arange(len(states)) + 0.5)
    axis.set_yticklabels(states, fontsize=10)
    axis.set_title("f07-04 — validation gate scorecard (thresholds: preregistration §8.2)",
                   fontsize=11, loc="left")
    for spine in axis.spines.values():
        spine.set_visible(False)
    axis.tick_params(length=0)
    fig.text(0.01, -0.06, "fatigue: not gated — construct dropped in Phase 5 (preregistration §7).",
             fontsize=8, color=FAIL_COLOUR)
    fig.tight_layout()
    save(fig, "f07-04_validation-gate-scorecard.png")


def fig_engagement_vs_ability(predictions: pd.DataFrame, gate: dict) -> dict:
    """f07-05 — Wise & Kong's negative criterion. A flat line is the pass.

    An effort index that tracks ability has learned ability. The criterion is
    the reason this project can claim the engagement head measures effort
    rather than restating competence.
    """
    rows = scored(predictions, "engagement")
    fig, axis = plt.subplots(figsize=(6.5, 5))
    finding = {}
    if rows.empty:
        axis.text(0.5, 0.5, "engagement head not measured", ha="center", transform=axis.transAxes)
    else:
        ability = study2.ability_by_learner(rows)
        estimate = rows.groupby("learner_id", observed=True)["engagement_p"].mean()
        paired = pd.concat([estimate.rename("estimate"), ability.rename("ability")],
                           axis=1, join="inner").dropna()
        axis.scatter(paired["ability"], paired["estimate"], s=14, alpha=0.55,
                     color=HEAD_COLOURS["engagement"])
        slope, intercept = np.polyfit(paired["ability"], paired["estimate"], 1)
        grid = np.linspace(paired["ability"].min(), paired["ability"].max(), 20)
        axis.plot(grid, slope * grid + intercept, color=FAIL_COLOUR, linewidth=1.6)
        measured = gate["measurements"].get("engagement", {})
        r = measured.get("ability_correlation")
        bound = validation_gate.CRITERIA[2].bound
        passed = r is not None and abs(r) <= bound
        axis.set_title(f"f07-05 — engagement estimate vs ability\nr = {r} (pass: |r| ≤ {bound}) — "
                       f"{'PASS' if passed else 'FAIL'}",
                       fontsize=10, color=PASS_COLOUR if passed else FAIL_COLOUR, loc="left")
        finding = {"r": r, "bound": bound, "passed": bool(passed), "slope": round(float(slope), 5),
                   "learners": int(len(paired))}
    axis.set_xlabel("ability — mean correctness on solution-behaviour responses")
    axis.set_ylabel("mean engagement estimate")
    fig.tight_layout()
    save(fig, "f07-05_engagement-vs-ability.png")
    return finding


def fig_discriminant(predictions: pd.DataFrame) -> dict:
    """f07-06 — inter-head correlation. Is confidence a relabelled knowledge head?

    One panel per source, because the heads are not scored on the same rows: a
    pooled heatmap would mix a real-data correlation with a simulated one and
    the number in the C4 cell would match nothing in the gate report. The cell
    the gate actually judges is boxed.
    """
    rows = predictions[predictions["_half"] == "scoring"]
    sources = [source for source in rows["source"].unique()]
    columns = [f"{head}_p" for head in study2.HEADS]
    fig, axes = plt.subplots(1, len(sources), figsize=(5.4 * len(sources), 4.6))
    axes = np.atleast_1d(axes)
    finding = {}
    ceiling = validation_gate.CRITERIA[3].bound
    for axis, source in zip(axes, sources):
        matrix = rows[rows["source"] == source][columns].corr()
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
        axis.set_xticks(range(len(study2.HEADS)))
        axis.set_xticklabels(study2.HEADS, rotation=30, ha="right", fontsize=9)
        axis.set_yticks(range(len(study2.HEADS)))
        axis.set_yticklabels(study2.HEADS, fontsize=9)
        for i, a in enumerate(study2.HEADS):
            for j, b in enumerate(study2.HEADS):
                value = matrix.iloc[i, j]
                axis.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=9,
                          color="white" if abs(value) > 0.5 else "#222")
                judged = ({a, b} == {"confidence", "knowledge"}
                          and source == study2.GATE_SOURCE["confidence"])
                if judged:
                    axis.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False,
                                                 edgecolor="#111", linewidth=2.2))
        fig.colorbar(image, ax=axis, shrink=0.8)
        axis.set_title(f"{source}", fontsize=10)
        if source in SIM_SOURCES:
            simulated_note(axis)
        finding[source] = {f"{a}_vs_{b}": round(float(matrix.loc[f"{a}_p", f"{b}_p"]), 5)
                           for i, a in enumerate(study2.HEADS) for b in study2.HEADS[i + 1:]}
    fig.suptitle(f"f07-06 — discriminant validity; boxed cell is C4 (ceiling |r| ≤ {ceiling})",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "f07-06_state-discriminant-validity.png")
    return finding


def fig_trajectories(predictions: pd.DataFrame, gate: dict) -> None:
    """f07-07 — one session, three states, uncertainty bands, policy annotations.

    The annotations are what a Phase 8 policy would *do*, drawn from the gate:
    a state the gate rejected cannot appear in a decision, so it is drawn
    greyed and labelled excluded rather than quietly omitted.
    """
    rows = predictions[(predictions["source"] == "sim-V0") & predictions["session_id"].notna()]
    if rows.empty:
        return
    sessions = rows.groupby("session_id", observed=True).size()
    chosen = sessions[sessions >= 12].index[0] if (sessions >= 12).any() else sessions.index[0]
    session = rows[rows["session_id"] == chosen].sort_values("order_index")
    step = np.arange(len(session))

    fig, axis = plt.subplots(figsize=(11, 5))
    # An excluded head is drawn greyed *and* dashed: a policy cannot read it, so
    # it must not look like the states that drive the annotations below.
    excluded_dashes = {head: pattern for head, pattern
                       in zip(study2.HEADS, [(4, 2), (1, 2), (6, 2, 1, 2)])}
    for head in study2.HEADS:
        admitted = head in gate["admitted"]
        colour = HEAD_COLOURS[head] if admitted else "#9a9a9a"
        probability = session[f"{head}_p"].to_numpy()
        error = session[f"{head}_se"].to_numpy()
        label = head if admitted else f"{head} (excluded by the gate)"
        line, = axis.plot(step, probability, color=colour, label=label, linewidth=1.8)
        if not admitted:
            line.set_dashes(excluded_dashes[head])
        axis.fill_between(step, probability - error, probability + error, color=colour, alpha=0.18)
    for position in np.linspace(0, len(session) - 1, min(4, len(session))).astype(int):
        knowledge = session[f"knowledge_p"].to_numpy()[position]
        action = "harder item" if knowledge > 0.75 else ("easier item" if knowledge < 0.45
                                                         else "hold difficulty")
        axis.annotate(action, xy=(position, knowledge), xytext=(position, min(1.02, knowledge + 0.16)),
                      fontsize=7.5, ha="center", color="#333",
                      arrowprops=dict(arrowstyle="->", lw=0.8, color="#888"))
    axis.set_xlabel("item within session")
    axis.set_ylabel("state estimate ± MC-dropout SE")
    axis.set_ylim(0, 1.15)
    axis.legend(fontsize=8, loc="lower left")
    axis.set_title("f07-07 — state trajectories over one session, with the action each implies",
                   fontsize=11, loc="left")
    simulated_note(axis)
    fig.tight_layout()
    save(fig, "f07-07_state-trajectories-example.png")


def fig_state_ablation(heads: dict) -> None:
    """f07-08 — which signal classes each state actually needs.

    The privacy–utility curve at the *state* level: if a head is flat above L1,
    the motor telemetry it would cost a learner's privacy to collect buys that
    state nothing.
    """
    rungs = list(heads["ladder"])
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
    for axis, head in zip(axes, study2.HEADS):
        source = heads["gate_source"][head]
        values, errors, present = [], [], []
        for rung in rungs:
            entry = heads["ladder"][rung]["heads"].get(head, {}).get(source)
            if entry is None or entry.get("label_correlation") is None:
                continue
            present.append(rung)
            values.append(entry["label_correlation"])
            errors.append([entry["label_correlation"] - entry["label_correlation_ci_low"],
                           entry["label_correlation_ci_high"] - entry["label_correlation"]])
        if not values:
            axis.text(0.5, 0.5, f"{head}: no rung measured", ha="center", transform=axis.transAxes)
            axis.axis("off")
            continue
        axis.errorbar(range(len(values)), values, yerr=np.array(errors).T, fmt="o-",
                      color=HEAD_COLOURS[head], capsize=3)
        axis.axhline(validation_gate.CRITERIA[0].bound, color=FAIL_COLOUR, linestyle="--",
                     linewidth=1, label=f"C1 floor {validation_gate.CRITERIA[0].bound}")
        axis.set_xticks(range(len(present)))
        axis.set_xticklabels(present)
        axis.set_title(f"{head}  ({source})", fontsize=10)
        axis.set_xlabel("rung")
        axis.set_ylabel("r with own label")
        axis.legend(fontsize=7)
        if source in SIM_SOURCES:
            simulated_note(axis)
    fig.suptitle("f07-08 — state validity by signal class: the privacy–utility curve per state",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "f07-08_state-ablation-by-signal-class.png")


def fig_uncertainty(predictions: pd.DataFrame) -> dict:
    """f07-09 — how uncertain the estimates are, so Phase 8 can refuse to act."""
    rows = predictions[predictions["_half"] == "scoring"]
    fig, axis = plt.subplots(figsize=(7.5, 4.4))
    finding = {}
    for head in study2.HEADS:
        error = rows[f"{head}_se"].dropna().to_numpy()
        if not len(error):
            continue
        axis.hist(error, bins=50, histtype="step", linewidth=1.6, label=head,
                  color=HEAD_COLOURS[head], density=True)
        finding[head] = {"mean_se": round(float(error.mean()), 5),
                         "p90_se": round(float(np.percentile(error, 90)), 5)}
    axis.set_xlabel(f"MC-dropout standard error ({study2.MC_PASSES} passes)")
    axis.set_ylabel("density")
    axis.legend(fontsize=8)
    axis.set_title("f07-09 — uncertainty per state estimate", fontsize=11, loc="left")
    fig.tight_layout()
    save(fig, "f07-09_state-uncertainty-distribution.png")
    return finding


def fig_fatigue(eda: dict) -> None:
    """f07-10 — the figure BUILD.md asked for, answered with the reason there is no head.

    BUILD.md Phase 7 asks for "fatigue head estimate against the raw
    observable". There is no fatigue head: the pre-registered H2 rule fired in
    Phase 5 and dropped the construct. Drawing the observable alone, with its
    CI crossing zero, is the honest version of this figure.
    """
    slopes = eda["h2_fatigue_verdict"]["slopes"]
    speed = eda["within_session"]["speed"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))

    for axis, (title, data, unit) in zip(axes, [
            ("accuracy slope (the fatigue claim)", slopes, "Δ accuracy per item"),
            ("speed slope (the surviving observable)", speed, "Δ log response time per item")]):
        names = list(data)
        centres = [data[name]["slope"] for name in names]
        errors = [[data[name]["slope"] - data[name]["ci_low"] for name in names],
                  [data[name]["ci_high"] - data[name]["slope"] for name in names]]
        colours = [FAIL_COLOUR if data[name].get("flat") else "#0072B2" for name in names]
        axis.errorbar(centres, range(len(names)), xerr=errors, fmt="o", capsize=4,
                      ecolor="#888", linestyle="none")
        for index, colour in enumerate(colours):
            axis.plot(centres[index], index, "o", color=colour, markersize=7)
        axis.axvline(0, color="#333", linewidth=1, linestyle=":")
        axis.set_yticks(range(len(names)))
        axis.set_yticklabels(names, fontsize=9)
        axis.set_xlabel(unit)
        axis.set_title(title, fontsize=10)

    fig.suptitle("f07-10 — no fatigue head exists: the within-session accuracy decline it would "
                 "estimate was not detected (preregistration §7)", fontsize=10.5)
    fig.tight_layout()
    save(fig, "f07-10_fatigue-head-vs-observed-decay.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    heads = json.loads((BENCH_DIR / "study2-heads.json").read_text(encoding="utf-8"))
    gate = json.loads(validation_gate.GATE_PATH.read_text(encoding="utf-8"))
    predictions = pd.read_parquet(BENCH_DIR / "study2-predictions.parquet")
    eda = json.loads((BENCH_DIR / "eda-summary.json").read_text(encoding="utf-8"))
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    fig_architecture(heads)
    fig_state_vs_label(predictions, gate)
    fig_calibration(predictions, gate)
    fig_scorecard(gate)
    ability = fig_engagement_vs_ability(predictions, gate)
    discriminant = fig_discriminant(predictions)
    fig_trajectories(predictions, gate)
    fig_state_ablation(heads)
    uncertainty = fig_uncertainty(predictions)
    fig_fatigue(eda)

    # Two of these the training pass does not compute. Written back so no doc
    # ever has to quote a picture.
    heads["figure_findings"] = {"engagement_vs_ability": ability,
                                "inter_head_correlation": discriminant,
                                "uncertainty": uncertainty}
    (BENCH_DIR / "study2-heads.json").write_text(
        json.dumps(heads, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote: {(BENCH_DIR / 'study2-heads.json').relative_to(ROOT)} (figure findings appended)")


if __name__ == "__main__":
    main()
