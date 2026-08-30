"""Phase 8 figure pack — the closed loop's fourteen charts.

    python -m app.research.figures_phase8 [--seed 20260821]

Reads only what `make eval-policies` wrote: `artifacts/evaluation/policy-*.json`,
the per-learner CSVs, `decision-log-v0.parquet`, `decision-divergence-v0.json`,
`probe-noise-sensitivity.json` and `decision-explanations.json`. Anything a
figure computes that is not already in one of those files is written back into
`policy-figure-findings.json` before the run ends, so no doc ever quotes a
picture.

Every chart is labelled SIMULATED (standing guardrail 6). Three of them are
built to be able to show a null: **f08-03** (the ladder), **f08-09** (the
decision divergence that bounds any ladder effect at all) and **f08-14** (probe
noise, which the gate's rejection of the confidence head predicts is flat).
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
FIG_DIR = ROOT / "artifacts" / "figures" / "08-policies"
EVAL_DIR = ROOT / "artifacts" / "evaluation"

#: Arms in reporting order: floors, deployed baselines, then the ladder.
ARM_ORDER = ["random", "fixed_order", "rule_legacy", "mastery_threshold_bkt",
             "irt_cat_maxinfo", "rule_improved", "model_L0", "model_L1", "model_L3", "model_L4"]
ARM_GROUP = {"random": "floor", "fixed_order": "floor", "rule_legacy": "baseline",
             "mastery_threshold_bkt": "baseline", "irt_cat_maxinfo": "baseline",
             "rule_improved": "baseline", "model_L0": "ladder", "model_L1": "ladder",
             "model_L3": "ladder", "model_L4": "ladder"}
GROUP_COLOUR = {"floor": "#56B4E9", "baseline": "#0072B2", "ladder": "#E69F00"}
LADDER = ["model_L0", "model_L1", "model_L3", "model_L4"]
RUNG_PRIVACY = {"model_L0": "correctness", "model_L1": "+ timing",
                "model_L3": "+ session", "model_L4": "+ motor"}
BAND = (0.70, 0.75)


#: Every file this script reads. `figure_index` copies it into INDEX.md.
SOURCES = [
    "artifacts/evaluation/policy-results-v0.json",
    "artifacts/evaluation/policy-per-learner-v0.csv",
    "artifacts/evaluation/policy-curves.json",
    "artifacts/evaluation/decision-log-v0.parquet",
    "artifacts/evaluation/decision-divergence-v0.json",
    "artifacts/evaluation/decision-explanations.json",
    "artifacts/evaluation/probe-noise-sensitivity.json",
]


def save(fig: plt.Figure, name: str) -> None:
    path = plotstyle.save(fig, FIG_DIR, name)
    print(f"wrote: {path.relative_to(ROOT)} (+ .pdf)")


def simulated_note(axis: plt.Axes) -> None:
    axis.text(0.99, 0.02, "SIMULATED", transform=axis.transAxes, ha="right", va="bottom",
              fontsize=7, color="#CC79A7", alpha=0.85, fontweight="bold")


def load_results() -> dict:
    out = {}
    for path in sorted(EVAL_DIR.glob("policy-results-v*.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        out[report["variant"]] = report
    if not out:
        raise SystemExit("no policy-results-v*.json — run `make eval-policies` first")
    return out


def load_learners() -> pd.DataFrame:
    frames = [pd.read_csv(path) for path in sorted(EVAL_DIR.glob("policy-per-learner-v*.csv"))]
    return pd.concat(frames, ignore_index=True)


def present(frame: pd.DataFrame) -> list[str]:
    return [arm for arm in ARM_ORDER if arm in set(frame["policy"])]


def colours(arms: list[str]) -> list[str]:
    return [GROUP_COLOUR[ARM_GROUP[arm]] for arm in arms]


# ------------------------------------------------------------------ figures

def fig_items_to_mastery(learners: pd.DataFrame, results: dict) -> None:
    """f08-01 — the primary outcome, every arm, V0.

    Censoring is drawn, not hidden: the dashed line is the budget, and the
    annotation is the share of learners who hit it. A box plot whose upper half
    is all censored observations means something different from one that is not.
    """
    rows = learners[learners["variant"] == "V0"]
    arms = present(rows)
    fig, axis = plt.subplots(figsize=(11, 5.5))
    data = [rows[rows["policy"] == arm]["items_to_mastery"].to_numpy() for arm in arms]
    boxes = axis.boxplot(data, patch_artist=True, showfliers=False,
                         medianprops={"color": "black"})
    for patch, arm in zip(boxes["boxes"], arms):
        patch.set_facecolor(GROUP_COLOUR[ARM_GROUP[arm]])
        patch.set_alpha(0.75)
    budget = results["V0"]["budget_items"]
    axis.axhline(budget, color="#D55E00", linestyle="--", linewidth=1)
    axis.text(0.4, budget, f" budget = {budget} (right-censored)", color="#D55E00",
              va="bottom", fontsize=8)
    for index, arm in enumerate(arms, start=1):
        censored = float((rows[rows["policy"] == arm]["mastered"] == False).mean())  # noqa: E712
        axis.text(index, budget * 0.03, f"{censored:.0%}\ncensored", ha="center",
                  fontsize=7, color="#444444")
    axis.set_xticks(range(1, len(arms) + 1))
    axis.set_xticklabels(arms, rotation=30, ha="right")
    axis.set_ylabel("items to mastery of the six-concept curriculum")
    axis.set_title("f08-01 Items to mastery by policy arm (V0, calibrated baseline)")
    simulated_note(axis)
    save(fig, "f08-01_policy-comparison-items-to-mastery.png")


def fig_time_to_mastery(learners: pd.DataFrame) -> None:
    """f08-02 — the same comparison in on-task seconds."""
    rows = learners[learners["variant"] == "V0"]
    arms = present(rows)
    fig, axis = plt.subplots(figsize=(11, 5.5))
    data = [rows[rows["policy"] == arm]["time_to_mastery_seconds"].to_numpy() / 60 for arm in arms]
    boxes = axis.boxplot(data, patch_artist=True, showfliers=False,
                         medianprops={"color": "black"})
    for patch, arm in zip(boxes["boxes"], arms):
        patch.set_facecolor(GROUP_COLOUR[ARM_GROUP[arm]])
        patch.set_alpha(0.75)
    axis.set_xticks(range(1, len(arms) + 1))
    axis.set_xticklabels(arms, rotation=30, ha="right")
    axis.set_ylabel("on-task minutes (response time, gaps and intervention cost)")
    axis.set_title("f08-02 Time to mastery by policy arm (V0)")
    simulated_note(axis)
    save(fig, "f08-02_policy-comparison-time-to-mastery.png")


def fig_ladder(learners: pd.DataFrame) -> dict:
    """f08-03 — the headline: outcome against state representation.

    Two panels because the primary outcome is censored and the co-primary is
    not. If the right-hand panel is flat, the state representation did not
    change what the learner got out of the session, and that is the finding.
    """
    rows = learners[(learners["variant"] == "V0") & learners["policy"].isin(LADDER)]
    arms = [arm for arm in LADDER if arm in set(rows["policy"])]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    findings = {}
    for axis, (column, label) in zip(axes, [
            ("items_to_mastery", "items to mastery (censored)"),
            ("knowledge_gain", "latent knowledge gain over the curriculum")]):
        means, lows, highs = [], [], []
        rng = np.random.default_rng(20260821)
        for arm in arms:
            values = rows[rows["policy"] == arm][column].to_numpy(dtype=float)
            draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
            means.append(values.mean())
            lows.append(np.percentile(draws, 2.5))
            highs.append(np.percentile(draws, 97.5))
        positions = np.arange(len(arms))
        axis.errorbar(positions, means,
                      yerr=[np.array(means) - np.array(lows), np.array(highs) - np.array(means)],
                      fmt="o-", color=GROUP_COLOUR["ladder"], capsize=4)
        axis.set_xticks(positions)
        axis.set_xticklabels([f"{arm}\n{RUNG_PRIVACY[arm]}" for arm in arms], fontsize=8)
        axis.set_ylabel(label)
        axis.grid(alpha=0.25)
        simulated_note(axis)
        findings[column] = {arm: {"mean": float(mean), "ci": [float(low), float(high)]}
                            for arm, mean, low, high in zip(arms, means, lows, highs)}
    fig.suptitle("f08-03 The state ladder: same policy architecture, different state (V0)")
    save(fig, "f08-03_state-ladder-effect.png")
    return findings


def fig_robustness(results: dict) -> dict:
    """f08-04 — the credibility figure: does the ranking survive the variants?"""
    variants = sorted(results)
    arms = [arm for arm in ARM_ORDER if arm in results[variants[0]]["policies"]]
    ranks = {}
    for variant in variants:
        summary = results[variant]["policies"]
        ordered = sorted(arms, key=lambda arm: -summary[arm]["mean_knowledge_gain"])
        ranks[variant] = {arm: ordered.index(arm) + 1 for arm in arms}

    fig, axis = plt.subplots(figsize=(11, 6))
    for arm in arms:
        series = [ranks[variant][arm] for variant in variants]
        axis.plot(range(len(variants)), series, marker="o",
                  color=GROUP_COLOUR[ARM_GROUP[arm]],
                  alpha=0.9 if ARM_GROUP[arm] == "ladder" else 0.5,
                  linewidth=2 if ARM_GROUP[arm] == "ladder" else 1.2, label=arm)
        if max(series) - min(series) >= 2:
            axis.annotate("rank flip", (len(variants) - 1, series[-1]), fontsize=7,
                          color="#D55E00", xytext=(4, 0), textcoords="offset points")
    axis.set_xticks(range(len(variants)))
    axis.set_xticklabels([f"{variant}\n{results[variant]['variant_description'][:28]}"
                          for variant in variants], fontsize=7)
    axis.invert_yaxis()
    axis.set_ylabel("rank by mean knowledge gain (1 = best)")
    axis.set_title("f08-04 Policy ranking under deliberate simulator mis-specification")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.25)
    simulated_note(axis)
    save(fig, "f08-04_robustness-across-variants.png")
    return {"ranks": ranks,
            "max_rank_range": {arm: int(max(ranks[v][arm] for v in variants)
                                        - min(ranks[v][arm] for v in variants)) for arm in arms}}


def fig_mastery_curves(curves: dict) -> None:
    """f08-05 — mean mastery against items served."""
    entries = curves["curves"].get("V0", {})
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for arm in ARM_ORDER:
        if arm not in entries:
            continue
        series = entries[arm]["mastery"]
        axis.plot(range(1, len(series) + 1), series, color=GROUP_COLOUR[ARM_GROUP[arm]],
                  alpha=0.9 if ARM_GROUP[arm] == "ladder" else 0.45,
                  linewidth=2 if ARM_GROUP[arm] == "ladder" else 1.2, label=arm)
    axis.set_xlabel("items served")
    axis.set_ylabel("mean fraction of the curriculum mastered")
    axis.set_title("f08-05 Mastery curves (V0, mean over learners and seeds)")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.25)
    simulated_note(axis)
    save(fig, "f08-05_mastery-curves.png")


def fig_difficulty_trajectories(curves: dict) -> None:
    """f08-06 — what difficulty each arm actually serves as a run proceeds."""
    entries = curves["curves"].get("V0", {})
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for arm in ARM_ORDER:
        if arm not in entries:
            continue
        series = entries[arm]["difficulty"]
        axis.plot(range(1, len(series) + 1), series, color=GROUP_COLOUR[ARM_GROUP[arm]],
                  alpha=0.9 if ARM_GROUP[arm] == "ladder" else 0.45,
                  linewidth=2 if ARM_GROUP[arm] == "ladder" else 1.2, label=arm)
    axis.set_xlabel("items served")
    axis.set_ylabel("mean served difficulty (1-10)")
    axis.set_title("f08-06 Difficulty trajectories (V0)")
    axis.legend(fontsize=7, ncol=2)
    axis.grid(alpha=0.25)
    simulated_note(axis)
    save(fig, "f08-06_difficulty-trajectories.png")


def fig_success_band(learners: pd.DataFrame) -> None:
    """f08-07 — realised success rate against the desirable-difficulty band."""
    rows = learners[learners["variant"] == "V0"]
    arms = present(rows)
    fig, axis = plt.subplots(figsize=(10, 5.5))
    means = [rows[rows["policy"] == arm]["mean_success_probability"].mean() for arm in arms]
    axis.bar(range(len(arms)), means, color=colours(arms), alpha=0.8)
    axis.axhspan(*BAND, color="#009E73", alpha=0.18)
    axis.text(len(arms) - 0.4, BAND[1], " target band", color="#3f7a4f", fontsize=8, va="bottom")
    axis.set_xticks(range(len(arms)))
    axis.set_xticklabels(arms, rotation=30, ha="right")
    axis.set_ylabel("mean realised probability of a correct response")
    axis.set_title("f08-07 Did the arms hit the desirable-difficulty band? (V0)")
    axis.grid(alpha=0.25, axis="y")
    simulated_note(axis)
    save(fig, "f08-07_success-probability-band.png")


def fig_action_distribution(log: pd.DataFrame) -> None:
    """f08-08 — what each policy actually does, not what it is described as doing."""
    arms = [arm for arm in ARM_ORDER if arm in set(log["policy"])]
    interventions = ["no_intervention", "hint", "worked_example", "break_suggestion"]
    moves = ["same_concept", "next_concept", "prerequisite_concept"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, (column, order, title) in zip(axes, [
            ("intervention", interventions, "instructional action"),
            ("concept_move", moves, "concept move")]):
        shares = (log.groupby(["policy", column], observed=True).size()
                  .unstack(fill_value=0).reindex(index=arms, columns=order, fill_value=0))
        shares = shares.div(shares.sum(axis=1), axis=0)
        bottom = np.zeros(len(arms))
        for index, name in enumerate(order):
            axis.bar(range(len(arms)), shares[name], bottom=bottom, label=name,
                     color=plt.cm.tab20(index * 2 + 1))
            bottom += shares[name].to_numpy()
        axis.set_xticks(range(len(arms)))
        axis.set_xticklabels(arms, rotation=35, ha="right", fontsize=8)
        axis.set_ylabel("share of decisions")
        axis.set_title(title)
        axis.legend(fontsize=7)
        simulated_note(axis)
    fig.suptitle("f08-08 Action distribution by arm (V0, first seed)")
    save(fig, "f08-08_action-distribution.png")


def fig_divergence(divergence: dict, table: pd.DataFrame) -> None:
    """f08-09 — the mechanism figure: how often L0 and L4 disagree, and where.

    Pre-registration §9.5 fixes what this figure decides: below 5 % divergence,
    no outcome difference is attributable to the state representation whatever
    a p-value says.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["identical action", "different difficulty", "different intervention",
              "different concept move"]
    values = [divergence["identical_rate"], 1 - divergence["same_difficulty_rate"],
              1 - divergence["same_intervention_rate"], 1 - divergence["same_concept_move_rate"]]
    axes[0].barh(labels, values, color=["#009E73", "#E69F00", "#D55E00", "#CC79A7"], alpha=0.85)
    axes[0].axvline(0.05, color="#333333", linestyle="--", linewidth=1)
    axes[0].text(0.05, 3.4, " 5 % pre-registered floor", fontsize=8, rotation=90, va="top")
    axes[0].set_xlabel("share of model_L0's decisions")
    axes[0].set_title("what model_L4 would have done instead")
    for index, value in enumerate(values):
        axes[0].text(value, index, f" {value:.1%}", va="center", fontsize=8)

    gap = (table["difficulty_a"] - table["difficulty_b"])
    axes[1].hist(gap, bins=np.arange(-9.5, 10.5, 1), color="#E69F00", alpha=0.85)
    axes[1].set_xlabel("difficulty chosen by model_L0 minus model_L4")
    axes[1].set_ylabel("decisions")
    axes[1].set_title(f"mean |gap| = {gap.abs().mean():.2f} difficulty levels")
    for axis in axes:
        simulated_note(axis)
    fig.suptitle("f08-09 Decision divergence, L0 versus L4 on L0's own trajectory (V0)")
    save(fig, "f08-09_decision-divergence-L0-vs-L4.png")


def fig_intervention_timing(log: pd.DataFrame) -> None:
    """f08-10 — when interventions fire, against session position.

    BUILD.md asks for this against fatigue estimates. There is no fatigue head:
    the construct was dropped in Phase 5 when no archival source showed a
    within-session accuracy decline, so the x-axis is the observable the break
    rule actually conditions on — position inside the session.
    """
    arms = [arm for arm in LADDER if arm in set(log["policy"])]
    fig, axis = plt.subplots(figsize=(10, 5.5))
    bins = np.arange(1, 22)
    for arm in arms:
        rows = log[(log["policy"] == arm) & (log["intervention"] != "no_intervention")]
        total = log[log["policy"] == arm]
        if not len(total):
            continue
        counts, _ = np.histogram(rows["question_number"].clip(upper=20), bins=bins)
        denominator, _ = np.histogram(total["question_number"].clip(upper=20), bins=bins)
        rate = np.divide(counts, denominator, out=np.zeros(len(counts)), where=denominator > 0)
        axis.plot(bins[:-1], rate, marker="o", label=arm, alpha=0.9)
    axis.set_xlabel("position within the session (capped at 20)")
    axis.set_ylabel("share of decisions carrying an intervention")
    axis.set_title("f08-10 Intervention timing within a session (V0, first seed)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    simulated_note(axis)
    save(fig, "f08-10_intervention-timing.png")


def fig_subgroups(learners: pd.DataFrame) -> dict:
    """f08-11 — by prior-ability tercile. The effect should concentrate low."""
    rows = learners[learners["variant"] == "V0"].copy()
    rows["tercile"] = pd.qcut(rows["ability"], 3, labels=["low", "middle", "high"])
    arms = [arm for arm in LADDER if arm in set(rows["policy"])]
    fig, axis = plt.subplots(figsize=(10, 5.5))
    width = 0.8 / max(1, len(arms))
    findings = {}
    for index, arm in enumerate(arms):
        means = [rows[(rows["policy"] == arm) & (rows["tercile"] == level)]["knowledge_gain"].mean()
                 for level in ["low", "middle", "high"]]
        axis.bar(np.arange(3) + index * width, means, width=width, label=arm,
                 color=plt.cm.copper(0.2 + 0.2 * index))
        findings[arm] = {level: float(value) for level, value in
                         zip(["low", "middle", "high"], means)}
    axis.set_xticks(np.arange(3) + 0.4 - width / 2)
    axis.set_xticklabels(["low prior ability", "middle", "high"])
    axis.set_ylabel("mean latent knowledge gain")
    axis.set_title("f08-11 Ladder effect by prior-ability tercile (V0)")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25, axis="y")
    simulated_note(axis)
    save(fig, "f08-11_subgroup-effects.png")
    return findings


def fig_privacy_utility(learners: pd.DataFrame) -> None:
    """f08-12 — RQ4: outcome against how much telemetry was collected."""
    rows = learners[(learners["variant"] == "V0") & learners["policy"].isin(LADDER)]
    arms = [arm for arm in LADDER if arm in set(rows["policy"])]
    fig, axis = plt.subplots(figsize=(10, 5.5))
    rng = np.random.default_rng(20260821)
    means, errors = [], []
    for arm in arms:
        values = rows[rows["policy"] == arm]["knowledge_gain"].to_numpy(dtype=float)
        draws = rng.choice(values, size=(2000, len(values)), replace=True).mean(axis=1)
        means.append(values.mean())
        errors.append([values.mean() - np.percentile(draws, 2.5),
                       np.percentile(draws, 97.5) - values.mean()])
    axis.errorbar(range(len(arms)), means, yerr=np.array(errors).T, fmt="o-",
                  color="#CC79A7", capsize=4)
    axis.set_xticks(range(len(arms)))
    axis.set_xticklabels([f"{RUNG_PRIVACY[arm]}\n({arm})" for arm in arms], fontsize=8)
    axis.set_xlabel("signal classes collected — increasing privacy sensitivity →")
    axis.set_ylabel("mean latent knowledge gain")
    axis.set_title("f08-12 Privacy–utility curve in outcomes, not AUC (V0)")
    axis.grid(alpha=0.25)
    simulated_note(axis)
    save(fig, "f08-12_privacy-utility-curve.png")


def fig_explanation(traces: dict) -> None:
    """f08-13 — one real decision, decomposed over validated state deltas."""
    decisions = traces.get("decisions", [])
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.axis("off")
    if not decisions:
        axis.text(0.5, 0.5, "no decision fired a rule in the sampled window",
                  ha="center", fontsize=11)
        simulated_note(axis)
        save(fig, "f08-13_explanation-example.png")
        return
    trace = decisions[0]
    explanation = trace["explanation"]
    lines = [
        f"decision {trace['id']}",
        f"learner {trace['learner_id']}   step {trace['step']}   concept {trace['concept_id']}",
        "",
        f"action: difficulty {trace['action']['difficulty']} (served "
        f"{trace['served_difficulty']}), {trace['action']['concept_move']}, "
        f"{trace['action']['intervention']}   p = {trace['action']['propensity']:.3f}",
        "",
        "state read (validated only):",
    ]
    for name, value in explanation.get("states", {}).items():
        error = explanation.get("state_standard_error", {}).get(name)
        lines.append(f"   {name} = {value:.3f}" + (f"  (se {error:.3f})" if error else ""))
    lines += ["", "rules that fired:"]
    for entry in explanation.get("rules_fired", []):
        lines.append(f"   {entry['rule']}  [{entry['rung']}]")
        lines.append(f"      read: {entry['read']}")
        lines.append(f"      cite: {entry['citation'][:88]}")
    lines += ["", "rules excluded (reported, not deleted):"]
    for entry in explanation.get("rules_excluded", [])[:4]:
        detail = entry.get("states") or entry.get("detail")
        lines.append(f"   {entry['rule']} — {entry['reason']}: {detail}")
    axis.text(0.01, 0.98, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=8)
    axis.set_title("f08-13 One decision, explained over validated states and cited rules")
    simulated_note(axis)
    save(fig, "f08-13_explanation-example.png")


def fig_probe_noise(sweep: dict) -> None:
    """f08-14 — outcome against self-report noise. Expected flat, and measured."""
    rows = pd.DataFrame(sweep["rows"])
    fig, axis = plt.subplots(figsize=(10, 5.5))
    for arm, group in rows.groupby("policy", observed=True):
        group = group.sort_values("probe_noise")
        axis.plot(group["probe_noise"], group["knowledge_gain"], marker="o", label=arm)
    axis.axvline(sweep["calibrated_level"], color="#333333", linestyle="--", linewidth=1)
    axis.text(sweep["calibrated_level"], axis.get_ylim()[1], " calibrated", fontsize=8,
              va="top")
    axis.set_xlabel("self-report probe noise (sd of the latent-to-report map)")
    axis.set_ylabel("mean latent knowledge gain")
    axis.set_title("f08-14 Sensitivity to probe noise — the confidence head is gate-excluded, "
                   "so the policy reads no probe")
    axis.legend(fontsize=8)
    axis.grid(alpha=0.25)
    simulated_note(axis)
    save(fig, "f08-14_sensitivity-to-probe-noise.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    learners = load_learners()
    findings: dict = {}

    fig_items_to_mastery(learners, results)
    fig_time_to_mastery(learners)
    findings["ladder"] = fig_ladder(learners)
    findings["robustness"] = fig_robustness(results)

    curves_path = EVAL_DIR / "policy-curves.json"
    if curves_path.exists():
        curves = json.loads(curves_path.read_text(encoding="utf-8"))
        fig_mastery_curves(curves)
        fig_difficulty_trajectories(curves)

    fig_success_band(learners)

    log_path = EVAL_DIR / "decision-log-v0.parquet"
    if log_path.exists():
        log = pd.read_parquet(log_path)
        fig_action_distribution(log)
        fig_intervention_timing(log)

    divergence_path = EVAL_DIR / "decision-divergence-v0.json"
    if divergence_path.exists():
        divergence = json.loads(divergence_path.read_text(encoding="utf-8"))
        table = pd.read_parquet(EVAL_DIR / "decision-divergence-v0.parquet")
        fig_divergence(divergence, table)
        findings["divergence"] = divergence

    findings["subgroups"] = fig_subgroups(learners)
    fig_privacy_utility(learners)

    traces_path = EVAL_DIR / "decision-explanations.json"
    fig_explanation(json.loads(traces_path.read_text(encoding="utf-8"))
                    if traces_path.exists() else {})

    sweep_path = EVAL_DIR / "probe-noise-sensitivity.json"
    if sweep_path.exists():
        fig_probe_noise(json.loads(sweep_path.read_text(encoding="utf-8")))

    path = EVAL_DIR / "policy-figure-findings.json"
    path.write_text(json.dumps(findings, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote: {path.relative_to(ROOT)} (numbers a figure computed, so no doc quotes a picture)")


if __name__ == "__main__":
    main()
