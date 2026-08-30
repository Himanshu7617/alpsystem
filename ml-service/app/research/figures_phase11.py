"""Phase 11 figure pack — the eight statistics charts. `make stats` first.

    python -m app.research.figures_phase11 [--seed 20260821]

Reads `artifacts/evaluation/statistical-report.json`, `sensitivity.json` and the
per-learner outcome CSVs. Anything a figure computes that is not already in one
of those — the bootstrap draws behind f11-07, the residuals behind f11-08 — is
written back to `stats-figure-findings.json`, so no document has to quote a
picture.

f11-01 is the one a reviewer reads first: every comparison this project makes,
on one forest plot, as an effect size with a confidence interval. If a claim in
the paper is not on it, the claim is not supported by a measured effect.
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
from scipy import stats as scipy_stats

from app.research import plotstyle, stats

ROOT = Path(__file__).resolve().parents[3]
FIG_DIR = ROOT / "artifacts" / "figures" / "11-stats"
EVAL_DIR = ROOT / "artifacts" / "evaluation"
REPORT_PATH = EVAL_DIR / "statistical-report.json"
SENSITIVITY_PATH = EVAL_DIR / "sensitivity.json"
FINDINGS = EVAL_DIR / "stats-figure-findings.json"

#: Every file this script reads. `figure_index` copies it into INDEX.md.
SOURCES = [
    "artifacts/evaluation/statistical-report.json",
    "artifacts/evaluation/sensitivity.json",
    "artifacts/evaluation/policy-per-learner-v0.csv",
]

BLUE, ORANGE, GREEN = plotstyle.BLUE, plotstyle.ORANGE, plotstyle.GREEN
PURPLE, VERMILLION, GREY = plotstyle.PURPLE, plotstyle.VERMILLION, plotstyle.GREY

OUTCOME_COLOUR = {"knowledge_gain": BLUE, "items_to_mastery": ORANGE,
                  "time_to_mastery_seconds": GREEN}
OUTCOME_LABEL = {"knowledge_gain": "knowledge gain (uncensored co-primary)",
                 "items_to_mastery": "items to mastery (censored)",
                 "time_to_mastery_seconds": "time to mastery"}

#: The comparisons f11-07 draws in full. Three is what fits legibly.
HEADLINE = [("model_L0", "knowledge_gain"), ("model_L4", "knowledge_gain"),
            ("irt_cat_maxinfo", "knowledge_gain")]


def save(fig: plt.Figure, name: str) -> None:
    path = plotstyle.save(fig, FIG_DIR, name)
    print(f"wrote: {path.relative_to(ROOT)} (+ .pdf)")


# ------------------------------------------------------------------ f11-01

def f01_forest(report: dict) -> dict:
    effects = pd.DataFrame([{key: value for key, value in entry.items()
                             if key != "assumptions"} for entry in report["effect_sizes"]])
    order = (effects[effects["outcome"] == "knowledge_gain"]
             .sort_values("hedges_g")["policy"].tolist())
    outcomes = [name for name in OUTCOME_COLOUR if name in set(effects["outcome"])]
    fig, axes = plt.subplots(1, len(outcomes), figsize=(4.2 * len(outcomes), 5.4), sharey=True)
    for axis, outcome in zip(np.atleast_1d(axes), outcomes):
        block = effects[effects["outcome"] == outcome].set_index("policy").loc[order]
        positions = np.arange(len(block))
        axis.errorbar(block["hedges_g"], positions,
                      xerr=[block["hedges_g"] - block["g_ci_low"],
                            block["g_ci_high"] - block["hedges_g"]],
                      fmt="o", markersize=5, capsize=3, linewidth=1.4,
                      color=OUTCOME_COLOUR[outcome])
        for index, (_, row) in enumerate(block.iterrows()):
            if row.get("significant_holm"):
                axis.text(row["hedges_g"], index + 0.28, "*", ha="center", fontsize=11,
                          color=OUTCOME_COLOUR[outcome])
        axis.axvline(0, color=GREY, linewidth=1)
        for edge in (-0.3, 0.3):
            axis.axvline(edge, color=GREY, linestyle=":", linewidth=0.8)
        axis.set_yticks(positions)
        axis.set_yticklabels(order)
        axis.set_xlabel("Hedges' g (unpaired) vs " + report["reference_arm"])
        axis.set_title(OUTCOME_LABEL[outcome], fontsize=9)
        plotstyle.simulated_note(axis)
    fig.suptitle("Every closed-loop comparison as an effect size with a 95 % CI "
                 f"(SIMULATED; {report['bootstrap_resamples']:,} bootstrap resamples; "
                 "* = Holm-significant)", fontsize=10)
    fig.text(0.5, -0.02, "dotted lines mark g = ±0.3, the effect this literature reports "
                         "for adaptive instruction", ha="center", fontsize=8, color=GREY)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "f11-01_effect-sizes-forest-plot.png")
    largest = effects.loc[effects["hedges_g"].abs().idxmax()]
    return {"comparisons": int(len(effects)),
            "largest": {"comparison": largest["comparison"], "outcome": largest["outcome"],
                        "hedges_g": float(largest["hedges_g"])},
            "holm_significant": int(effects["significant_holm"].sum()),
            "exceeding_g_0.3": int((effects["hedges_g"].abs() > 0.3).sum())}


# ------------------------------------------------------------------ f11-02

def f02_coefficients(report: dict) -> dict:
    correctness = report["mixed_effects"]["correctness"]
    outcomes = report["mixed_effects"]["outcomes"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 5))

    rows = []
    for source, block in correctness.items():
        if "fixed_effects" not in block:
            continue
        for name in ("base_logit", "rung_increment"):
            entry = block["fixed_effects"][name]
            rows.append({"label": f"{source}\n{name}", "term": name, **entry})
    frame = pd.DataFrame(rows)
    frame["position"] = np.arange(len(frame))
    for term, colour in (("base_logit", BLUE), ("rung_increment", ORANGE)):
        block = frame[frame["term"] == term]
        left.errorbar(block["coefficient"], block["position"],
                      xerr=[block["coefficient"] - block["ci_low"],
                            block["ci_high"] - block["coefficient"]],
                      fmt="o", markersize=5, capsize=3, linewidth=1.3, color=colour,
                      label=term)
    positions = frame["position"].to_numpy()
    left.legend(fontsize=7)
    left.axvline(0, color=GREY, linewidth=1)
    left.set_yticks(positions)
    left.set_yticklabels(frame["label"], fontsize=7)
    left.set_xlabel("change in p(correct) per sd of the predictor")
    left.set_title("Correctness: crossed learner + item random intercepts\n"
                   "orange = what the higher rung adds to the lowest", fontsize=9)

    knowledge = outcomes["knowledge_gain"]
    terms = {name: values for name, values in knowledge["fixed_effects"].items()
             if name != "Intercept"}
    labels = [name.replace("C(policy)[T.", "").rstrip("]") for name in terms]
    values = pd.DataFrame(terms).T
    positions = np.arange(len(values))
    right.errorbar(values["coefficient"], positions,
                   xerr=[values["coefficient"] - values["ci_low"],
                         values["ci_high"] - values["coefficient"]],
                   fmt="o", markersize=5, capsize=3, linewidth=1.3, color=GREEN)
    right.axvline(0, color=GREY, linewidth=1)
    right.set_yticks(positions)
    right.set_yticklabels(labels, fontsize=8)
    right.set_xlabel(f"knowledge gain vs {knowledge['reference']} (logits)")
    right.set_title("Closed loop: learner-level outcome,\nrandom intercept per learner",
                    fontsize=9)
    plotstyle.simulated_note(right)
    fig.suptitle("Mixed-effects fixed effects with 95 % CIs", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "f11-02_mixed-effects-coefficients.png")
    return {"correctness_rung_increment": {
        source: block["fixed_effects"]["rung_increment"]["coefficient"]
        for source, block in correctness.items() if "fixed_effects" in block}}


# ------------------------------------------------------------------ f11-03

def f03_variance(report: dict) -> dict:
    models = {}
    for source, block in report["mixed_effects"]["correctness"].items():
        if "variance_share" in block:
            models[f"correctness\n{source}"] = block["variance_share"]
    items = report["mixed_effects"].get("closed_loop_items", {})
    if "variance_share" in items:
        models["closed loop\nper item"] = items["variance_share"]
    for outcome, block in report["mixed_effects"]["outcomes"].items():
        models[f"outcome\n{outcome}"] = block["variance_share"]

    order = ["learner", "item", "residual"]
    colour = {"learner": BLUE, "item": ORANGE, "residual": GREY}
    fig, axis = plt.subplots(figsize=(10, 4.6))
    positions = np.arange(len(models))
    bottom = np.zeros(len(models))
    for component in order:
        heights = np.array([block.get(component, 0.0) for block in models.values()])
        axis.bar(positions, heights, bottom=bottom, color=colour[component],
                 label=component, width=0.62)
        for index, (height, base) in enumerate(zip(heights, bottom)):
            if height > 0.04:
                axis.text(index, base + height / 2, f"{height:.0%}", ha="center",
                          va="center", fontsize=7,
                          color="white" if component != "residual" else "#eeeeee")
        bottom += heights
    axis.set_xticks(positions)
    axis.set_xticklabels(models, fontsize=7)
    axis.set_ylabel("share of total variance")
    axis.set_ylim(0, 1)
    axis.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.10))
    axis.set_title("Where the variance lives: learner, item, residual", fontsize=10, pad=30)
    fig.tight_layout()
    save(fig, "f11-03_random-effects-variance.png")
    return {name: {key: float(value) for key, value in block.items()}
            for name, block in models.items()}


# ------------------------------------------------------------------ f11-04

def f04_power(report: dict) -> dict:
    power = report["power"]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.4))
    for (effect, points), colour in zip(power["curves"].items(), (GREY, BLUE, GREEN)):
        frame = pd.DataFrame(points)
        left.semilogx(frame["n_per_arm"], frame["power"], color=colour,
                      label=f"g = {effect}", linewidth=1.8)
    left.axhline(0.8, color=VERMILLION, linestyle="--", linewidth=1)
    required = power["rct_requirement"]
    left.axvline(required["n_per_arm"], color=VERMILLION, linestyle=":", linewidth=1)
    left.annotate(f"an RCT for g = {required['target_g']} needs\n"
                  f"{required['n_per_arm']:.0f} learners per arm "
                  f"({required['n_total']:.0f} total)",
                  xy=(required["n_per_arm"], 0.8), xytext=(1.6 * required["n_per_arm"], 0.42),
                  fontsize=8, color=VERMILLION,
                  arrowprops={"arrowstyle": "->", "color": VERMILLION, "linewidth": 0.9})
    left.set_xlabel("learners per arm")
    left.set_ylabel("power")
    left.set_ylim(0, 1.02)
    left.legend(loc="lower right")
    left.set_title("Power against sample size", fontsize=9)

    achieved = pd.DataFrame(power["achieved"])
    achieved = achieved[achieved["outcome"] == "knowledge_gain"].sort_values("power")
    positions = np.arange(len(achieved))
    right.barh(positions, achieved["power"], color=BLUE, height=0.62)
    right.axvline(0.8, color=VERMILLION, linestyle="--", linewidth=1)
    right.set_yticks(positions)
    right.set_yticklabels([name.replace(" vs ", "\nvs ") for name in achieved["comparison"]],
                          fontsize=7)
    right.set_xlim(0, 1.02)
    right.set_xlabel("achieved power")
    right.set_title(f"What the simulated runs achieved\n(knowledge gain, "
                    f"n = {int(achieved['n_per_arm'].max()):,} per arm)", fontsize=9)
    plotstyle.simulated_note(right)
    fig.suptitle("Power: what was achieved, and what a real trial would need", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "f11-04_power-curves.png")
    return {"rct_requirement": required,
            "underpowered_comparisons": int((achieved["power"] < 0.8).sum()),
            "total_comparisons": int(len(achieved))}


# ------------------------------------------------------------------ f11-05

def _panel_note(axis: plt.Axes, block: dict) -> bool:
    if "skipped" in block:
        axis.text(0.5, 0.5, "\n".join(block["skipped"][index:index + 42]
                                      for index in range(0, len(block["skipped"]), 42)),
                  ha="center", va="center", fontsize=7, color=GREY, transform=axis.transAxes)
        axis.set_xticks([])
        axis.set_yticks([])
        return False
    return True


def f05_sensitivity(sensitivity: dict) -> dict:
    fig, axes = plt.subplots(2, 3, figsize=(14, 7.6))
    findings = {}

    axis = axes[0][0]
    block = sensitivity["probe_noise"]
    if _panel_note(axis, block):
        frame = pd.DataFrame(block["rows"])
        for (policy, rows), colour in zip(frame.groupby("policy"), (BLUE, ORANGE)):
            axis.plot(rows["probe_noise"], rows["knowledge_gain"], "o-", color=colour,
                      label=policy, linewidth=1.6)
        axis.axvline(block["calibrated"], color=GREY, linestyle=":", linewidth=1)
        axis.set_xlabel("probe noise sd")
        axis.set_ylabel("knowledge gain")
        axis.legend(fontsize=7)
        findings["probe_noise"] = block["max_spread_in_knowledge_gain"]
    axis.set_title("1. Self-report probe noise", fontsize=9)

    axis = axes[0][1]
    block = sensitivity["mastery_threshold"]
    if _panel_note(axis, block):
        frame = pd.DataFrame(block["rows"])
        for (policy, rows), colour in zip(frame.groupby("policy"), (BLUE, ORANGE)):
            axis.plot(rows["tau"], rows["knowledge_gain"], "o-", color=colour,
                      label=policy, linewidth=1.6)
        axis.axvline(block["preregistered"], color=GREY, linestyle=":", linewidth=1)
        axis.set_xlabel("mastery threshold τ")
        axis.set_ylabel("knowledge gain")
        axis.legend(fontsize=7)
        axis.text(0.02, 0.02, f"{block['learners_per_cell']} learners, "
                              f"budget {block['budget']}", transform=axis.transAxes,
                  fontsize=6, color=GREY)
        findings["mastery_threshold"] = block["best_arm_by_tau"]
    axis.set_title("2. Mastery threshold τ", fontsize=9)

    axis = axes[0][2]
    block = sensitivity["item_budget"]
    frame = pd.DataFrame(block["rows"])
    for policy, rows in frame.groupby("policy"):
        axis.plot(rows["budget"], rows["mastery_rate"], linewidth=1.2,
                  color=BLUE if policy == "model_L0" else GREY,
                  alpha=1.0 if policy == "model_L0" else 0.45,
                  label=policy if policy == "model_L0" else None)
    axis.plot([], [], color=GREY, alpha=0.45, linewidth=1.2, label="other arms")
    axis.axvline(block["preregistered"], color=VERMILLION, linestyle=":", linewidth=1)
    axis.set_xlabel("item budget")
    axis.set_ylabel("mastery rate")
    axis.legend(fontsize=7)
    axis.set_title("3. Item budget (re-cut, not re-run)", fontsize=9)
    findings["item_budget"] = block["best_arm_by_budget"]

    axis = axes[1][0]
    block = sensitivity["simulator_variant"]
    frame = pd.DataFrame(block["rows"])
    pivot = frame.pivot(index="policy", columns="variant", values="knowledge_gain")
    image = axis.imshow(pivot.rank(ascending=False, axis=0), cmap="viridis_r", aspect="auto")
    axis.set_xticks(range(len(pivot.columns)))
    axis.set_xticklabels(pivot.columns, fontsize=7)
    axis.set_yticks(range(len(pivot.index)))
    axis.set_yticklabels(pivot.index, fontsize=6)
    axis.grid(False)
    fig.colorbar(image, ax=axis, label="rank on knowledge gain (1 = best)")
    axis.set_title("4. Simulator variant", fontsize=9)
    findings["simulator_variant"] = block["best_arm_by_variant"]

    axis = axes[1][1]
    block = sensitivity["target_epsilon"]
    if _panel_note(axis, block):
        frame = pd.DataFrame(block["rows"])
        for (policy, rows), colour in zip(frame.groupby("policy"), (BLUE, ORANGE)):
            axis.plot(rows["target_epsilon"], rows["snips"], "o-", color=colour,
                      label=f"{policy} SNIPS", linewidth=1.6)
            axis.plot(rows["target_epsilon"], rows["ips"], "s--", color=colour, alpha=0.6,
                      label=f"{policy} IPS", linewidth=1.1, markersize=4)
        axis.axvline(block["preregistered"], color=GREY, linestyle=":", linewidth=1)
        axis.set_xlabel("target policy ε")
        axis.set_ylabel("estimated value")
        axis.legend(fontsize=6)
        findings["target_epsilon"] = {"levels": block["levels"]}
    axis.set_title("5. Exploration assumed of the target", fontsize=9)

    axis = axes[1][2]
    block = sensitivity["reward_weights"]
    if _panel_note(axis, block):
        frame = pd.DataFrame(block["rows"])
        for (knob, rows), style in zip(frame.groupby("knob"), ("o-", "s--")):
            grouped = rows.groupby("multiplier")["snips"].max()
            axis.plot(grouped.index, grouped.to_numpy(), style,
                      color=BLUE if knob == "lambda_time" else ORANGE,
                      label=knob, linewidth=1.5)
        axis.set_xscale("log")
        axis.set_xticks(sorted(frame["multiplier"].unique()))
        axis.set_xticklabels([f"×{value:g}" for value in
                              sorted(frame["multiplier"].unique())], fontsize=7)
        axis.minorticks_off()
        axis.axvline(1.0, color=GREY, linestyle=":", linewidth=1)
        axis.set_xlabel("multiplier on the pre-registered weight")
        axis.set_ylabel("best arm's SNIPS value")
        axis.legend(fontsize=7)
        findings["reward_weights"] = block["best_arm_by_weights"]
    axis.set_title("6. RL reward weights", fontsize=9)

    for row in axes:
        for axis in row:
            plotstyle.simulated_note(axis)
    fig.suptitle("Sensitivity: every constant somebody chose, moved (SIMULATED)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    save(fig, "f11-05_sensitivity-grid.png")
    findings["ordering_stable"] = sensitivity["summary"]["ordering_stable"]
    return findings


# ------------------------------------------------------------------ f11-06

def f06_correction(report: dict) -> dict:
    families = report["multiple_comparisons"]
    fig, axes = plt.subplots(1, len(families), figsize=(4.4 * len(families), 4.4), sharey=True)
    for axis, (outcome, block) in zip(np.atleast_1d(axes), families.items()):
        raw = np.array([max(block["raw"][name], 1e-12) for name in block["family"]])
        holm = np.array([max(block["holm"][name], 1e-12) for name in block["family"]])
        positions = np.arange(len(raw))
        axis.barh(positions - 0.19, -np.log10(raw), height=0.36, color=BLUE, label="raw")
        axis.barh(positions + 0.19, -np.log10(holm), height=0.36, color=ORANGE, label="Holm")
        axis.axvline(-np.log10(0.05), color=VERMILLION, linestyle="--", linewidth=1)
        axis.set_yticks(positions)
        axis.set_yticklabels([name.replace(" vs ", "\nvs ") for name in block["family"]],
                             fontsize=6)
        axis.set_xlabel("−log₁₀ p")
        axis.set_title(f"{OUTCOME_LABEL[outcome]}\nfamily of {block['size']}", fontsize=8)
        axis.legend(fontsize=7)
        plotstyle.simulated_note(axis)
    fig.suptitle("Raw against Holm-corrected p-values, one family per outcome "
                 "(dashed line: α = 0.05)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    save(fig, "f11-06_multiple-comparison-correction.png")
    return {outcome: {"significant_raw": len(block["significant_raw"]),
                      "significant_holm": len(block["significant_holm"]),
                      "lost_to_correction": sorted(set(block["significant_raw"])
                                                   - set(block["significant_holm"]))}
            for outcome, block in families.items()}


# ------------------------------------------------------------------ f11-07

def f07_bootstrap(learners: pd.DataFrame, report: dict, seed: int, draws: int) -> dict:
    reference = report["reference_arm"]
    fig, axes = plt.subplots(1, len(HEADLINE), figsize=(4.2 * len(HEADLINE), 4.0))
    findings = {}
    for axis, (arm, outcome) in zip(np.atleast_1d(axes), HEADLINE):
        treatment = learners.loc[learners["policy"] == arm, outcome].to_numpy(dtype=float)
        control = learners.loc[learners["policy"] == reference, outcome].to_numpy(dtype=float)
        rng = np.random.default_rng(seed)
        left = rng.choice(treatment, size=(draws, len(treatment)), replace=True).mean(axis=1)
        right = rng.choice(control, size=(draws, len(control)), replace=True).mean(axis=1)
        difference = left - right
        low, high = np.percentile(difference, [2.5, 97.5])
        axis.hist(difference, bins=60, color=BLUE, alpha=0.85)
        axis.axvline(0, color=GREY, linewidth=1.2)
        axis.axvline(float(np.mean(difference)), color=VERMILLION, linewidth=1.4)
        axis.axvspan(low, high, color=ORANGE, alpha=0.18)
        axis.set_xlabel(f"{outcome}: {arm} − {reference}")
        axis.set_ylabel("bootstrap draws")
        axis.set_title(f"{arm}\n95 % CI [{low:+.4f}, {high:+.4f}]", fontsize=8)
        plotstyle.simulated_note(axis)
        findings[f"{arm}|{outcome}"] = {
            "mean_difference": float(np.mean(difference)),
            "ci_low": float(low), "ci_high": float(high),
            "share_favouring_treatment": float((difference > 0).mean()),
            "excludes_zero": bool(low > 0 or high < 0)}
    fig.suptitle(f"Bootstrap distributions of the headline differences "
                 f"({draws:,} resamples, SIMULATED)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save(fig, "f11-07_bootstrap-distributions.png")
    return findings


# ------------------------------------------------------------------ f11-08

def f08_assumptions(learners: pd.DataFrame, report: dict) -> dict:
    outcomes = [name for name in stats.OUTCOMES if name in learners.columns]
    fig, axes = plt.subplots(2, len(outcomes), figsize=(4.2 * len(outcomes), 7.2))
    findings = {}
    for column, outcome in enumerate(outcomes):
        fitted = learners.groupby("policy", observed=True)[outcome].transform("mean")
        residual = learners[outcome].to_numpy(dtype=float) - fitted.to_numpy(dtype=float)
        standardised = residual / (residual.std(ddof=1) or 1.0)

        axis = axes[0][column]
        scipy_stats.probplot(standardised, dist="norm", plot=axis)
        axis.get_lines()[0].set(markersize=2, color=BLUE, alpha=0.4)
        axis.get_lines()[1].set(color=VERMILLION, linewidth=1.2)
        axis.set_title(f"{outcome}\nnormal QQ", fontsize=8)
        axis.set_xlabel("theoretical quantiles")
        axis.set_ylabel("standardised residual")

        axis = axes[1][column]
        axis.scatter(fitted, standardised, s=4, alpha=0.18, color=BLUE)
        axis.axhline(0, color=GREY, linewidth=1)
        axis.set_xlabel(f"fitted arm mean ({outcome})")
        axis.set_ylabel("standardised residual")
        by_arm = (pd.DataFrame({"policy": learners["policy"], "residual": residual})
                  .groupby("policy", observed=True)["residual"].std(ddof=1))
        ratio = float(by_arm.max() / by_arm.min()) if by_arm.min() > 0 else float("nan")
        axis.set_title(f"residual spread by arm: ×{ratio:.2f} widest/narrowest", fontsize=8)
        plotstyle.simulated_note(axis)
        findings[outcome] = {
            "skew": float(scipy_stats.skew(standardised)),
            "kurtosis": float(scipy_stats.kurtosis(standardised)),
            "residual_sd_ratio_across_arms": ratio,
            "reading": "the outcome is a bounded, censored quantity, so normality is "
                       "not expected; Welch and the rank test are reported for that reason",
        }
    fig.suptitle("Assumption checks on the learner-level outcomes (SIMULATED)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    save(fig, "f11-08_assumption-checks.png")
    return findings


# -------------------------------------------------------------------- main

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    args = parser.parse_args()

    if not REPORT_PATH.exists():
        raise SystemExit(f"{REPORT_PATH.relative_to(ROOT)} absent — run `make stats` first")
    report = json.loads(REPORT_PATH.read_text(encoding="utf-8"))
    sensitivity = (json.loads(SENSITIVITY_PATH.read_text(encoding="utf-8"))
                   if SENSITIVITY_PATH.exists() else None)
    learners = pd.read_csv(EVAL_DIR / "policy-per-learner-v0.csv")

    findings = {
        "seed": args.seed,
        "f11-01": f01_forest(report),
        "f11-02": f02_coefficients(report),
        "f11-03": f03_variance(report),
        "f11-04": f04_power(report),
        "f11-06": f06_correction(report),
        "f11-07": f07_bootstrap(learners, report, args.seed, args.bootstrap),
        "f11-08": f08_assumptions(learners, report),
    }
    if sensitivity is None:
        print("note: sensitivity.json absent — f11-05 skipped, run `make stats`")
    else:
        findings["f11-05"] = f05_sensitivity(sensitivity)

    FINDINGS.write_text(json.dumps(findings, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote: {FINDINGS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
