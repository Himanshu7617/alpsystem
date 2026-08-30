"""Phase 11 — the six sensitivity analyses, one per panel of f11-05.

    python -m app.research.sensitivity [--seed 20260821] [--skip-tau]

Every headline result in this project rests on a constant somebody chose: the
self-report noise level, the mastery threshold, the item budget, which
simulator variant is "the" simulator, how much the logging policy explored, and
the reward weights. This script varies each one and writes
``artifacts/evaluation/sensitivity.json``.

Five of the six are recomputed from files already on disk and cost seconds. The
sixth — the mastery threshold τ — cannot be: mastery is a property of the
environment's response model at the moment a learner is taught, not a
post-hoc cut on a stored column, so it needs the closed loop run again. That
one obeys the run profile (`PROFILE=dev make stats`) and can be skipped with
``--skip-tau``, which records *why* the panel is missing rather than drawing an
empty one.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.research import runprofile

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "artifacts" / "evaluation"
DATA_DIR = ROOT / "artifacts" / "datasets"
MODEL_DIR = ROOT / "artifacts" / "models"
OUTPUT_PATH = EVAL_DIR / "sensitivity.json"

#: Pre-registration §9.2 fixes τ = 0.80. A threshold rule is only a result if
#: the ordering of the arms survives moving it.
TAU_LEVELS = [0.70, 0.80, 0.90]
TAU_ARMS = ["rule_improved", "model_L0"]

#: ε for the target policy in the OPE sweep. The pre-registered value is 0.15.
EPSILON_LEVELS = [0.0, 0.05, 0.10, 0.15, 0.25, 0.50]

#: Multipliers on the pre-registered reward weights (λ = 0.002, μ = 0.005).
WEIGHT_MULTIPLIERS = [0.25, 0.5, 1.0, 2.0, 4.0]

OUTCOMES = ["knowledge_gain", "items_to_mastery", "mastered"]


# ------------------------------------------------------- 1. probe noise

def probe_noise() -> dict:
    """Read Phase 8's sweep rather than repeating it."""
    path = EVAL_DIR / "probe-noise-sensitivity.json"
    if not path.exists():
        return {"skipped": "probe-noise-sensitivity.json absent — run `make eval-policies`"}
    sweep = json.loads(path.read_text(encoding="utf-8"))
    frame = pd.DataFrame(sweep["rows"])
    spread = (frame.groupby("policy", observed=True)["knowledge_gain"]
              .agg(lambda values: float(values.max() - values.min())))
    return {
        "knob": "probe_noise_sd", "source": "artifacts/evaluation/probe-noise-sensitivity.json",
        "calibrated": sweep["calibrated_level"], "levels": sweep["levels"],
        "learners": sweep["learners"], "rows": sweep["rows"],
        "max_spread_in_knowledge_gain": {arm: float(value) for arm, value in spread.items()},
        "reading": sweep.get("note", ""),
    }


# --------------------------------------------------- 2. mastery threshold

def mastery_threshold(seed: int, learners: int, budget: int, thresholds: dict) -> dict:
    """Rerun the loop at each τ. The only panel that costs compute."""
    from app.research import closed_loop, evaluate_policies

    rows, original = [], closed_loop.MASTERY_TARGET
    started = time.perf_counter()
    try:
        for tau in TAU_LEVELS:
            # `mastered()` reads the module constant at call time, so this
            # reaches the environment, the results and the stopping rule alike.
            closed_loop.MASTERY_TARGET = tau
            for arm in TAU_ARMS:
                cell = evaluate_policies.run_cell(
                    arm=arm, variant="V0", seed=seed, learners=learners, budget=budget,
                    epsilon=evaluate_policies.arms_module.EPSILON, thresholds=thresholds)
                frame = pd.DataFrame(cell["results"])
                rows.append({
                    "policy": arm, "tau": tau,
                    "learners": int(len(frame)),
                    "mastered": float(frame["mastered"].mean()),
                    "items_to_mastery": float(frame["items_to_mastery"].mean()),
                    "knowledge_gain": float(frame["knowledge_gain"].mean()),
                    "final_mastery_fraction": float(frame["final_mastery_fraction"].mean()),
                    "seconds": cell["seconds"],
                })
                print(f"  tau {tau} {arm}: mastered {rows[-1]['mastered']:.3f}, "
                      f"knowledge gain {rows[-1]['knowledge_gain']:+.4f} "
                      f"({cell['seconds']}s)")
    finally:
        closed_loop.MASTERY_TARGET = original

    frame = pd.DataFrame(rows)
    ordering = {}
    for tau, block in frame.groupby("tau"):
        best = block.sort_values("knowledge_gain", ascending=False)["policy"].iloc[0]
        ordering[str(tau)] = best
    return {
        "knob": "mastery_threshold_tau", "preregistered": 0.80,
        "levels": TAU_LEVELS, "arms": TAU_ARMS,
        "learners_per_cell": learners, "budget": budget,
        "rows": rows,
        "best_arm_by_tau": ordering,
        "ordering_stable": len(set(ordering.values())) == 1,
        "elapsed_seconds": round(time.perf_counter() - started, 1),
    }


# -------------------------------------------------------- 3. item budget

def item_budget(learners: pd.DataFrame, budget: int) -> dict:
    """What the comparison would have said at a smaller budget.

    Recomputed exactly, not re-simulated: a learner who mastered at item 84 had
    mastered by any budget ≥ 84, and one who did not master by 200 had not
    mastered by anything smaller. Only the arms' *mastery rates* are recoverable
    this way — knowledge gain at item 100 is not in the file — so that is all
    this panel claims.
    """
    grid = [25, 50, 75, 100, 125, 150, 175, budget]
    rows = []
    for policy, block in learners.groupby("policy", observed=True):
        mastered_at = block.loc[block["mastered"], "items_to_mastery"].to_numpy(dtype=float)
        for cut in grid:
            rows.append({"policy": str(policy), "budget": int(cut),
                         "mastery_rate": float((mastered_at <= cut).sum() / len(block))})
    frame = pd.DataFrame(rows)
    leaders = {str(cut): frame[frame["budget"] == cut]
               .sort_values("mastery_rate", ascending=False)["policy"].iloc[0] for cut in grid}
    return {
        "knob": "item_budget", "preregistered": budget, "levels": grid,
        "rows": rows, "best_arm_by_budget": leaders,
        "ordering_stable": len(set(leaders.values())) == 1,
        "note": "mastery rate only — the stored outcome is a threshold crossing time, "
                "so a smaller budget is a re-cut of it, while knowledge gain at a "
                "smaller budget would need the loop rerun",
    }


# --------------------------------------------------- 4. simulator variant

def simulator_variant(variants: dict[str, pd.DataFrame]) -> dict:
    """Does the ordering of the arms survive a deliberately wrong simulator?"""
    rows, leaders = [], {}
    for variant, frame in sorted(variants.items()):
        summary = frame.groupby("policy", observed=True)[
            ["knowledge_gain", "items_to_mastery", "mastered"]].mean()
        for policy, values in summary.iterrows():
            rows.append({"variant": variant, "policy": str(policy),
                         **{name: float(value) for name, value in values.items()}})
        leaders[variant] = str(summary["knowledge_gain"].idxmax())
    ranks = (pd.DataFrame(rows).pivot(index="policy", columns="variant",
                                      values="knowledge_gain").rank(ascending=False))
    return {
        "knob": "simulator_variant", "preregistered": "V0",
        "levels": sorted(variants), "rows": rows,
        "best_arm_by_variant": leaders,
        "ordering_stable": len(set(leaders.values())) == 1,
        "rank_spread": {policy: float(row.max() - row.min())
                        for policy, row in ranks.iterrows()},
        "note": "V1–V4 are the deliberate mis-specifications from Phase 4; a finding "
                "that only holds on V0 is a property of one generative model",
    }


# ----------------------------------------- 5 & 6. OPE knobs on the logged data

def _logged_frame() -> tuple[pd.DataFrame, list[str]] | tuple[None, str]:
    path = DATA_DIR / "logged-bandit-data.parquet"
    if not path.exists():
        return None, "logged-bandit-data.parquet absent — run `make ope`"
    from app.model import validation_gate
    from app.policy import learned

    frame = pd.read_parquet(path)
    return frame, learned.feature_names(learned.admitted_states(validation_gate.load()))


def _ips_snips(frame: pd.DataFrame, chosen: np.ndarray, rewards: np.ndarray,
               epsilon: float) -> dict:
    from app.research import ope

    raw = ope.importance_weights(frame, chosen, epsilon)
    weights = np.minimum(raw, ope.WEIGHT_CLIP)
    total = float(weights.sum())
    return {
        "ips": float(np.mean(weights * rewards)),
        "snips": float((weights * rewards).sum() / total) if total > 0 else float("nan"),
        "effective_sample_size": ope.effective_sample_size(weights),
        "action_match_rate": float((chosen == frame["action_index"].to_numpy()).mean()),
    }


def _bandit_targets(frame: pd.DataFrame, names: list[str]) -> dict[str, np.ndarray]:
    from app.research import ope

    targets = {}
    for arm in ("bandit_linucb", "bandit_lin_ts"):
        path = MODEL_DIR / f"{arm}-v0.npz"
        if path.exists():
            targets[arm] = ope.frozen_bandit_choices(path, frame, names)
    return targets


def logging_epsilon(frame: pd.DataFrame, names: list[str]) -> dict:
    """How much of the OPE estimate is the assumed target stochasticity?"""
    rewards = frame["reward"].to_numpy()
    targets = _bandit_targets(frame, names)
    if not targets:
        return {"skipped": "no saved bandit θ — run `make ope`"}
    rows = [{"policy": arm, "target_epsilon": epsilon,
             **_ips_snips(frame, chosen, rewards, epsilon)}
            for arm, chosen in targets.items() for epsilon in EPSILON_LEVELS]
    return {
        "knob": "target_epsilon", "preregistered": 0.15,
        "logging_epsilon": float(frame["logging_epsilon"].iloc[0]),
        "levels": EPSILON_LEVELS, "rows": rows,
        "note": "the logging policy is fixed at the ε it actually ran with; what varies "
                "is the exploration assumed of the *target*, which is what the "
                "importance weights are built from. DR is omitted: the reward model is "
                "fitted per run in Phase 9 and is not re-fitted here.",
    }


def reward_weights(frame: pd.DataFrame, names: list[str]) -> dict:
    """Would a different λ or μ have changed which learned arm looks best?"""
    from app.rl import reward as reward_module

    streak = (frame["nxt_incorrect_streak"].to_numpy() * 5).round()
    frustrated = (streak >= reward_module.FRUSTRATION_STREAK).astype(float)
    minutes = frame["seconds"].to_numpy() / 60.0
    mastery = reward_module.MASTERY_SCALE * frame["mastery_delta"].to_numpy()

    baseline = (mastery - reward_module.LAMBDA_TIME * minutes
                - reward_module.MU_FRUSTRATION * frustrated)
    agreement = float(np.mean(np.abs(baseline - frame["reward"].to_numpy()) < 1e-9))
    assert agreement > 0.999, (
        f"the reward cannot be rebuilt from the logged columns ({agreement:.4f} of rows "
        "match); the sweep below would be varying a formula the log was not written with")

    targets = _bandit_targets(frame, names)
    if not targets:
        return {"skipped": "no saved bandit θ — run `make ope`"}
    rows = []
    for multiplier in WEIGHT_MULTIPLIERS:
        for knob in ("lambda_time", "mu_frustration"):
            lam = reward_module.LAMBDA_TIME * (multiplier if knob == "lambda_time" else 1.0)
            mu = reward_module.MU_FRUSTRATION * (multiplier if knob == "mu_frustration" else 1.0)
            rewards = mastery - lam * minutes - mu * frustrated
            for arm, chosen in targets.items():
                rows.append({"policy": arm, "knob": knob, "multiplier": multiplier,
                             "lambda_time": lam, "mu_frustration": mu,
                             "logged_mean_reward": float(rewards.mean()),
                             **_ips_snips(frame, chosen, rewards,
                                          float(frame["logging_epsilon"].iloc[0]))})
    best = {}
    for (knob, multiplier), block in pd.DataFrame(rows).groupby(["knob", "multiplier"]):
        best[f"{knob}x{multiplier}"] = str(block.sort_values("snips",
                                                             ascending=False)["policy"].iloc[0])
    return {
        "knob": "reward_weights", "preregistered": reward_module.SPEC,
        "reconstruction_agreement": agreement,
        "multipliers": WEIGHT_MULTIPLIERS, "rows": rows,
        "best_arm_by_weights": best,
        "ordering_stable": len(set(best.values())) == 1,
        "note": "the reward is recomputed from the logged components and the same logged "
                "decisions are re-weighted; no policy is retrained, so this asks whether "
                "the *evaluation* is weight-sensitive, not whether training would be",
    }


# -------------------------------------------------------------------- main

def main() -> None:
    import sys

    limits = runprofile.load(runprofile.peek(sys.argv))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=limits["profile"], choices=sorted(runprofile.PROFILES))
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--learners", type=int, default=limits["learners"])
    parser.add_argument("--budget", type=int, default=limits["budget"])
    parser.add_argument("--skip-tau", action="store_true",
                        help="skip the one analysis that reruns the closed loop")
    args = parser.parse_args()

    paths = sorted(EVAL_DIR.glob("policy-per-learner-v*.csv"))
    if not paths:
        raise SystemExit("no per-learner outcomes — run `make eval-policies` first")
    variants = {path.stem.split("-")[-1].upper(): pd.read_csv(path) for path in paths}
    results = json.loads((EVAL_DIR / "policy-results-v0.json").read_text(encoding="utf-8"))
    reported_budget = int(results["budget_items"])

    report = {
        "phase": 11, "seed": args.seed, "run_profile": args.profile,
        "why": "every headline number rests on a constant somebody chose; these are "
               "the six, moved",
        "probe_noise": probe_noise(),
        "item_budget": item_budget(variants["V0"], reported_budget),
        "simulator_variant": simulator_variant(variants),
    }

    frame, names = _logged_frame()
    if frame is None:
        report["target_epsilon"] = report["reward_weights"] = {"skipped": names}
    else:
        report["target_epsilon"] = logging_epsilon(frame, names)
        report["reward_weights"] = reward_weights(frame, names)

    if args.skip_tau:
        report["mastery_threshold"] = {
            "skipped": "--skip-tau: mastery is decided inside the environment, so this "
                       "panel needs the closed loop rerun and is the only one that costs "
                       "compute"}
    else:
        from app.research.evaluate_policies import observable_thresholds

        print(f"mastery threshold sweep: {len(TAU_LEVELS)} τ × {len(TAU_ARMS)} arms × "
              f"{args.learners} learners, budget {args.budget}")
        report["mastery_threshold"] = mastery_threshold(
            args.seed, args.learners, args.budget, observable_thresholds())

    stable = {name: block.get("ordering_stable") for name, block in report.items()
              if isinstance(block, dict) and "ordering_stable" in block}
    report["summary"] = {
        "ordering_stable": stable,
        "reading": "a knob whose ordering is not stable is a knob the conclusion depends "
                   "on, and the paper says so rather than reporting the preregistered "
                   "level alone",
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote: {OUTPUT_PATH.relative_to(ROOT)}")
    for name, value in stable.items():
        print(f"  {name}: ordering {'stable' if value else 'MOVES'}")


if __name__ == "__main__":
    main()
