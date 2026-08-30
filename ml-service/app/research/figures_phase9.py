"""Phase 9 figure pack — Study 3's ten charts. `make ope` and `make rl` first.

    python -m app.research.figures_phase9 [--seed 20260821]

Reads only what Study 3 wrote: `artifacts/evaluation/ope-results.json`,
`artifacts/evaluation/rl-results.json`, `artifacts/datasets/logged-bandit-data.parquet`
and the bandit θ artifacts. Anything a figure computes that is not already in
one of those is written back to `study3-figure-findings.json`, so no document
ever has to quote a picture.

Two of these are built to be able to show a failure rather than a result:
**f09-02** (are the OPE estimates anywhere near the truth?) and **f09-08** (does
a policy trained on V0 survive V1–V4?). A flat or losing learned arm is a
reportable Study 3 outcome, not a bug to be tuned away.
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
FIG_DIR = ROOT / "artifacts" / "figures" / "09-ope-rl"
EVAL_DIR = ROOT / "artifacts" / "evaluation"
DATA_DIR = ROOT / "artifacts" / "datasets"
MODEL_DIR = ROOT / "artifacts" / "models"
FINDINGS = EVAL_DIR / "study3-figure-findings.json"

ARM_COLOUR = {"bandit_linucb": "#0072B2", "bandit_lin_ts": "#009E73", "rl_ppo": "#D55E00"}
ESTIMATOR_COLOUR = {"ips": "#56B4E9", "snips": "#CC79A7", "dr": "#E69F00"}


#: Every file this script reads. `figure_index` copies it into INDEX.md.
SOURCES = [
    "artifacts/evaluation/ope-results.json",
    "artifacts/evaluation/rl-results.json",
    "artifacts/datasets/logged-bandit-data.parquet",
]


def save(fig: plt.Figure, name: str) -> None:
    path = plotstyle.save(fig, FIG_DIR, name)
    print(f"wrote: {path.relative_to(ROOT)} (+ .pdf)")


def simulated_note(axis: plt.Axes) -> None:
    axis.text(0.99, 0.02, "SIMULATED", transform=axis.transAxes, ha="right", va="bottom",
              fontsize=7, color="#CC79A7", alpha=0.85, fontweight="bold")


def colour(arm: str) -> str:
    return ARM_COLOUR.get(arm, "#56B4E9")


# ------------------------------------------------------------------- figures

def f01_propensity(log: pd.DataFrame, ope: dict) -> dict:
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4))
    left.hist(log["propensity"], bins=40, color="#0072B2", edgecolor="white")
    left.set_yscale("log")
    left.set_xlabel("logging propensity")
    left.set_ylabel("decisions (log scale)")
    left.set_title(f"Logged propensities — {ope['logging_policy']} at ε = {ope['logging_epsilon']}")
    simulated_note(left)

    names = [row["policy"] for row in ope["estimates"]]
    ess = [row["effective_sample_size"] for row in ope["estimates"]]
    raw = [row.get("effective_sample_size_unclipped", np.nan) for row in ope["estimates"]]
    position = np.arange(len(names))
    right.bar(position - 0.2, ess, 0.4, color="#E69F00", label="clipped weights")
    right.bar(position + 0.2, raw, 0.4, color="#56B4E9", label="raw weights")
    right.axhline(len(log), color="#333333", linestyle=":", linewidth=1,
                  label=f"logged decisions ({len(log):,})")
    right.set_yscale("log")
    right.set_xticks(position, names, rotation=15)
    right.set_ylabel("effective sample size (log scale)")
    right.set_title("ESS — how many decisions the estimate really rests on")
    right.legend(fontsize=7)
    simulated_note(right)
    fig.tight_layout()
    save(fig, "f09-01_propensity-distribution.png")
    return {"logged_decisions": int(len(log)),
            "ess": {name: value for name, value in zip(names, ess)},
            "ess_unclipped": {name: value for name, value in zip(names, raw)},
            "explored_share": float(log["explored"].mean())}


def f02_ope_vs_truth(ope: dict) -> dict:
    rows = ope["estimates"]
    fig, axis = plt.subplots(figsize=(9, 4.6))
    position = np.arange(len(rows))
    width = 0.22
    for offset, name in zip((-width, 0, width), ("ips", "snips", "dr")):
        axis.bar(position + offset, [row[name] for row in rows], width,
                 color=ESTIMATOR_COLOUR[name], label=name.upper())
    axis.plot(position, [row["true_value"] for row in rows], "D", color="#333333",
              markersize=9, label="true on-policy value")
    for index, row in enumerate(rows):
        axis.annotate(f"ESS {row['effective_sample_size']:.0f}",
                      (index, max(row["ips"], row["snips"], row["dr"], row["true_value"])),
                      textcoords="offset points", xytext=(0, 8), ha="center", fontsize=7)
    axis.axhline(0, color="#666666", linewidth=0.8)
    axis.set_xticks(position, [row["policy"] for row in rows])
    axis.set_ylabel("mean reward per decision")
    axis.set_title("Offline estimates against the truth the simulator can compute")
    axis.legend(fontsize=8)
    simulated_note(axis)
    fig.tight_layout()
    save(fig, "f09-02_ope-vs-true-value.png")
    return {"mean_absolute_error": ope["mean_absolute_error_vs_truth"],
            "dr_beats_ips": ope["dr_beats_ips"]}


def f03_estimator_variance(ope: dict) -> dict:
    rows = ope["estimates"]
    fig, axis = plt.subplots(figsize=(9, 4.6))
    position = np.arange(len(rows))
    width = 0.22
    for offset, name in zip((-width, 0, width), ("ips", "snips", "dr")):
        centres = np.array([row[name] for row in rows])
        low = np.array([row[f"{name}_95_ci"][0] for row in rows])
        high = np.array([row[f"{name}_95_ci"][1] for row in rows])
        axis.errorbar(position + offset, centres,
                      yerr=[centres - low, high - centres], fmt="o",
                      color=ESTIMATOR_COLOUR[name], capsize=4, label=name.upper())
    axis.plot(position, [row["true_value"] for row in rows], "D", color="#333333",
              markersize=8, label="truth")
    axis.set_xticks(position, [row["policy"] for row in rows])
    axis.set_ylabel("mean reward per decision")
    axis.set_title("Bootstrap 95 % intervals per estimator")
    axis.legend(fontsize=8)
    simulated_note(axis)
    fig.tight_layout()
    save(fig, "f09-03_ope-estimator-variance.png")
    widths = {name: float(np.mean([row[f"{name}_95_ci"][1] - row[f"{name}_95_ci"][0]
                                   for row in rows])) for name in ("ips", "snips", "dr")}
    return {"mean_ci_width": widths}


def curves(ope: dict) -> dict[str, list[float]]:
    """Each bandit's per-step mean reward, as recorded during its training run."""
    return {arm: entry["reward_curve"]
            for arm, entry in ope.get("bandit_training", {}).items()}


def f04_cumulative(ope: dict, rl: dict | None) -> dict:
    fig, axis = plt.subplots(figsize=(9, 4.6))
    series = curves(ope)
    baseline_curve = None
    if rl:
        rule = rl["best_phase8_arm"]
        baseline = rl["summary"].get(rl["train_variant"], {}).get(rule, {}).get("mean_reward")
        if baseline is not None:
            longest = max((len(values) for values in series.values()), default=0)
            baseline_curve = np.cumsum(np.full(longest, baseline))
            axis.plot(baseline_curve, color="#333333", linestyle="--",
                      label=f"{rule} (best Phase 8 arm)")
    for arm, values in series.items():
        axis.plot(np.nancumsum(values), color=colour(arm), label=arm)
    axis.set_xlabel("decision step")
    axis.set_ylabel("cumulative mean reward")
    axis.set_title("Bandit cumulative reward against the best rule")
    axis.legend(fontsize=8)
    simulated_note(axis)
    fig.tight_layout()
    save(fig, "f09-04_bandit-cumulative-reward.png")
    return {"final_cumulative": {arm: float(np.nancumsum(values)[-1]) if len(values) else None
                                for arm, values in series.items()},
            "baseline_final": float(baseline_curve[-1]) if baseline_curve is not None else None}


def f05_regret(ope: dict, rl: dict | None) -> dict:
    fig, axis = plt.subplots(figsize=(9, 4.6))
    series = curves(ope)
    baseline = None
    if rl:
        rule = rl["best_phase8_arm"]
        baseline = rl["summary"].get(rl["train_variant"], {}).get(rule, {}).get("mean_reward")
    finals = {}
    if baseline is None:
        axis.text(0.5, 0.5, "no rule baseline yet — run `make rl`", ha="center",
                  transform=axis.transAxes)
    else:
        for arm, values in series.items():
            regret = np.nancumsum(baseline - np.asarray(values, dtype=float))
            finals[arm] = float(regret[-1]) if regret.size else None
            axis.plot(regret, color=colour(arm), label=arm)
        axis.axhline(0, color="#333333", linewidth=0.8)
        axis.legend(fontsize=8)
    axis.set_xlabel("decision step")
    axis.set_ylabel("cumulative regret against the best rule")
    axis.set_title("Regret — positive means the rule is still ahead")
    simulated_note(axis)
    fig.tight_layout()
    save(fig, "f09-05_bandit-regret-curve.png")
    return {"final_regret": finals, "baseline_reward_per_decision": baseline}


def f06_training(rl: dict) -> dict:
    curve = rl["learning_curve"]
    fig, axis = plt.subplots(figsize=(9, 4.6))
    if curve:
        steps = np.array([point["timesteps"] for point in curve])
        values = np.array([point["reward_per_decision"] for point in curve])
        axis.plot(steps, values, color="#D55E00", alpha=0.35, linewidth=1, label="episode")
        window = min(10, len(values))
        if window > 1:
            smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
            axis.plot(steps[window - 1:], smoothed, color="#D55E00", linewidth=2,
                      label=f"mean of {window} episodes")
    rule = rl["best_phase8_arm"]
    baseline = rl["summary"].get(rl["train_variant"], {}).get(rule, {}).get("mean_reward")
    if baseline is not None:
        axis.axhline(baseline, color="#333333", linestyle="--", label=f"{rule}")
    axis.set_xlabel("training timesteps")
    axis.set_ylabel("reward per decision")
    axis.set_title(f"PPO learning curve on {rl['train_variant']} "
                   f"({rl['training']['timesteps']:,} timesteps)")
    axis.legend(fontsize=8)
    simulated_note(axis)
    fig.tight_layout()
    save(fig, "f09-06_rl-training-curves.png")
    return {"episodes": len(curve), "rule_baseline": baseline}


def f07_sample_efficiency(rl: dict) -> dict:
    efficiency = rl["sample_efficiency"]
    curve = rl["learning_curve"]
    fig, axis = plt.subplots(figsize=(9, 4.6))
    if curve:
        steps = np.array([point["timesteps"] for point in curve])
        values = np.array([point["reward_per_decision"] for point in curve])
        window = min(10, len(values))
        smoothed = (np.convolve(values, np.ones(window) / window, mode="valid")
                    if window > 1 else values)
        axis.plot(steps[len(steps) - len(smoothed):], smoothed, color="#D55E00",
                  label="PPO (smoothed)")
    baseline = efficiency.get("baseline")
    if baseline is not None and np.isfinite(baseline):
        axis.axhline(baseline, color="#333333", linestyle="--",
                     label=f"{rl['best_phase8_arm']} baseline")
    if efficiency.get("reached"):
        axis.axvline(efficiency["timesteps"], color="#009E73", linestyle=":",
                     label=f"matched at {efficiency['timesteps']:,} timesteps")
    else:
        axis.text(0.5, 0.08, "PPO did not reach the rule baseline within the budget",
                  transform=axis.transAxes, ha="center", fontsize=9, color="#D55E00")
    axis.set_xlabel("training timesteps (≈ learner interactions)")
    axis.set_ylabel("reward per decision")
    axis.set_title("Sample efficiency — interactions needed to match the best rule")
    axis.legend(fontsize=8)
    simulated_note(axis)
    fig.tight_layout()
    save(fig, "f09-07_rl-sample-efficiency.png")
    return efficiency


def f08_generalisation(rl: dict) -> dict:
    variants = rl["variants"]
    arms = [rl["best_phase8_arm"], "bandit_linucb", "bandit_lin_ts", "rl_ppo"]
    fig, axis = plt.subplots(figsize=(10, 4.6))
    position = np.arange(len(variants))
    width = 0.8 / len(arms)
    values_out = {}
    for index, arm in enumerate(arms):
        heights, lows, highs = [], [], []
        for variant in variants:
            entry = rl["summary"].get(variant, {}).get(arm, {})
            heights.append(entry.get("mean_knowledge_gain", np.nan))
            low, high = entry.get("knowledge_gain_95_ci", [np.nan, np.nan])
            lows.append(low)
            highs.append(high)
        heights = np.array(heights, dtype=float)
        offset = position + (index - (len(arms) - 1) / 2) * width
        axis.bar(offset, heights, width, color=colour(arm), label=arm)
        axis.errorbar(offset, heights,
                      yerr=[heights - np.array(lows, dtype=float),
                            np.array(highs, dtype=float) - heights],
                      fmt="none", ecolor="#333333", capsize=2, linewidth=0.8)
        values_out[arm] = {variant: float(value) for variant, value in zip(variants, heights)}
    axis.axvline(0.5, color="#D55E00", linestyle=":", linewidth=1)
    axis.text(0.5, -0.13, "← PPO trained here | held-out variants →",
              transform=axis.transAxes, fontsize=8, ha="center", color="#D55E00")
    axis.set_xticks(position, [f"{variant}" for variant in variants])
    axis.set_ylabel("mean knowledge gain")
    axis.set_title("Trained on V0, evaluated on every variant — the credibility check")
    axis.legend(fontsize=8, loc="upper right", framealpha=0.9)
    simulated_note(axis)
    fig.tight_layout()
    save(fig, "f09-08_rl-generalisation-across-variants.png")
    return values_out


def f09_action_map(ope: dict, rl: dict | None) -> dict:
    """What each learned arm actually does, over the context it can see."""
    from app.policy import learned

    names = ope["context_features"]
    knowledge = np.linspace(0.05, 0.95, 19)
    progress = np.linspace(0.0, 1.0, 11)
    base = np.zeros(len(names))
    base[names.index("bias")] = 1.0
    for name in ("recent_accuracy", "overall_accuracy"):
        if name in names:
            base[names.index(name)] = 0.5
    if "knowledge_present" in names:
        base[names.index("knowledge_present")] = 1.0

    grid = []
    for row_value in knowledge:
        for column_value in progress:
            vector = base.copy()
            if "knowledge" in names:
                vector[names.index("knowledge")] = row_value
            vector[names.index("progress")] = column_value
            grid.append(vector)
    grid = np.asarray(grid)

    panels: dict[str, np.ndarray] = {}
    for arm in ("bandit_linucb", "bandit_lin_ts"):
        path = MODEL_DIR / f"{arm}-{ope['variant'].lower()}.npz"
        if not path.exists():
            continue
        theta = np.load(path)["theta"]
        chosen = np.argmax(grid @ theta.T, axis=1)
        panels[arm] = np.array([learned.action_from_index(index)["difficulty"]
                                for index in chosen]).reshape(len(knowledge), len(progress))
    ppo_path = MODEL_DIR / "rl" / "policy.zip"
    if rl and ppo_path.exists():
        from stable_baselines3 import PPO

        model = PPO.load(str(ppo_path), device="cpu")
        actions, _ = model.predict(np.clip(grid, -10, 10).astype("float32"), deterministic=True)
        panels["rl_ppo"] = np.array([learned.DIFFICULTIES[row[0]]
                                     for row in np.atleast_2d(actions)]).reshape(
            len(knowledge), len(progress))

    if not panels:
        return {}
    fig, axes = plt.subplots(1, len(panels), figsize=(4.6 * len(panels), 4.2), squeeze=False)
    for axis, (arm, surface) in zip(axes[0], panels.items()):
        image = axis.imshow(surface, aspect="auto", origin="lower", cmap="viridis",
                            vmin=1, vmax=10,
                            extent=[progress[0], progress[-1], knowledge[0], knowledge[-1]])
        axis.set_title(arm)
        axis.set_xlabel("progress through the budget")
        axis.set_ylabel("knowledge estimate (gated)")
        fig.colorbar(image, ax=axis, label="chosen difficulty")
        simulated_note(axis)
    fig.suptitle("What the learned policies do — the gate admitted knowledge only")
    fig.tight_layout()
    save(fig, "f09-09_learned-policy-action-map.png")
    return {arm: {"mean_difficulty": float(surface.mean()),
                  "distinct_difficulties": int(len(np.unique(surface)))}
            for arm, surface in panels.items()}


def f10_final_comparison(rl: dict, seed: int) -> dict:
    """Every Phase 8 arm plus the learned ones, one chart, on the same outcome."""
    variant = rl["train_variant"]
    path = EVAL_DIR / f"policy-per-learner-{variant.lower()}.csv"
    rows: dict[str, dict] = {}
    phase8_n = 0
    if path.exists():
        frame = pd.read_csv(path)
        for arm, group in frame.groupby("policy"):
            phase8_n = max(phase8_n, len(group))
            values = group["knowledge_gain"].to_numpy(dtype=float)
            draws = np.random.default_rng(seed).choice(values, size=(500, len(values)))
            means = draws.mean(axis=1)
            rows[arm] = {"mean": float(values.mean()), "group": "phase 8",
                         "low": float(np.quantile(means, 0.025)),
                         "high": float(np.quantile(means, 0.975))}
    for arm, entry in rl["summary"].get(variant, {}).items():
        if arm in rows and arm not in ARM_COLOUR:
            continue
        low, high = entry.get("knowledge_gain_95_ci", [np.nan, np.nan])
        rows[arm] = {"mean": entry["mean_knowledge_gain"],
                     "group": "study 3" if arm in ARM_COLOUR else "phase 8",
                     "low": low, "high": high}

    order = sorted(rows, key=lambda arm: rows[arm]["mean"])
    fig, axis = plt.subplots(figsize=(9, max(4.0, 0.42 * len(order))))
    position = np.arange(len(order))
    means = np.array([rows[arm]["mean"] for arm in order])
    lows = np.array([rows[arm]["low"] for arm in order], dtype=float)
    highs = np.array([rows[arm]["high"] for arm in order], dtype=float)
    axis.barh(position, means,
              color=["#D55E00" if rows[arm]["group"] == "study 3" else "#0072B2"
                     for arm in order])
    axis.errorbar(means, position, xerr=[means - lows, highs - means], fmt="none",
                  ecolor="#333333", capsize=3, linewidth=0.8)
    study3_n = max((entry.get("learners", 0)
                    for entry in rl["summary"].get(variant, {}).values()), default=0)
    axis.set_yticks(position, order)
    axis.set_xlabel("mean knowledge gain (95 % CI)")
    axis.set_title(f"Every arm on {variant} — Phase 8 in blue, Study 3 in red")
    # The two halves of this chart were not run at the same size, and the CI
    # widths say so loudly. Naming the sample sizes on the figure is cheaper
    # than a reader inferring that the learned arms are noisier by nature.
    axis.text(0.99, -0.16, f"Phase 8 arms n = {phase8_n:,} per arm; "
                           f"Study 3 arms n = {study3_n:,} — CI widths differ for that reason",
              transform=axis.transAxes, ha="right", fontsize=7.5, color="#555555")
    simulated_note(axis)
    fig.tight_layout()
    save(fig, "f09-10_policy-comparison-final.png")
    return {"means": {arm: rows[arm]["mean"] for arm in order},
            "phase8_learners_per_arm": phase8_n, "study3_learners_per_arm": study3_n}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    ope_path = EVAL_DIR / "ope-results.json"
    if not ope_path.exists():
        raise SystemExit("no artifacts/evaluation/ope-results.json — run `make ope` first")
    ope = json.loads(ope_path.read_text(encoding="utf-8"))
    rl_path = EVAL_DIR / "rl-results.json"
    rl = json.loads(rl_path.read_text(encoding="utf-8")) if rl_path.exists() else None
    log = pd.read_parquet(DATA_DIR / "logged-bandit-data.parquet")

    findings = {"f09-01": f01_propensity(log, ope),
                "f09-02": f02_ope_vs_truth(ope),
                "f09-03": f03_estimator_variance(ope),
                "f09-04": f04_cumulative(ope, rl),
                "f09-05": f05_regret(ope, rl)}
    if rl is None:
        print("note: no rl-results.json — f09-06 to f09-10 need `make rl`")
    else:
        findings.update({"f09-06": f06_training(rl),
                         "f09-07": f07_sample_efficiency(rl),
                         "f09-08": f08_generalisation(rl),
                         "f09-10": f10_final_comparison(rl, args.seed)})
    action_map = f09_action_map(ope, rl)
    if action_map:
        findings["f09-09"] = action_map

    FINDINGS.write_text(json.dumps(findings, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote: {FINDINGS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
