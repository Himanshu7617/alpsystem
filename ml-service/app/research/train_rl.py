"""Phase 9, Study 3 — PPO training and the learned-versus-rule comparison. `make rl`.

    python -m app.research.train_rl --profile smoke --seed 20260821

PPO comes from `stable-baselines3` with its library defaults; nothing here
implements an algorithm. What it does own is the parts that decide whether the
result is believable:

* **Trained on V0, evaluated on V0–V4.** A policy that only works on the
  simulator it was trained on has learned the simulator, and the literature
  review names that as the standard failure of RL-for-education. `f09-08` is the
  figure that would show it.
* **Resumable.** A checkpoint every ``ALP_CHECKPOINT_INTERVAL`` rollouts to
  `artifacts/models/rl/checkpoints/`; a killed run restarts from the newest one
  and trains only the timesteps it still owes. §1.5 rule 2.
* **Compared against the best Phase 8 arm**, on the same cohort, under the same
  budget and action space, with a paired bootstrap CI. **If the learned arms do
  not win, that is the result** — preregistration §10.5 forbids tuning until
  they do.

Outputs: `artifacts/models/rl/{policy.zip, training_config.json, reward_spec.json,
training_metrics.json, training_manifest.json}` and
`artifacts/evaluation/rl-results.json`.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.model import registry
from app.research import runprofile
from app.research.closed_loop import BUDGET
from app.research.evaluate_policies import bootstrap_ci, observable_thresholds, paired_tests
from app.research.ope import log_progress, run_arm
from app.research.simulator import VARIANTS
from app.rl.reward import SPEC as REWARD_SPEC

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "artifacts" / "evaluation"
# Overridable so a test can train into a scratch directory instead of the
# published artifact — a completed run's checkpoint would otherwise decide
# whether a fresh one gets written at all.
RL_DIR = Path(os.environ.get("ALP_RL_DIR") or ROOT / "artifacts" / "models" / "rl")
CHECKPOINT_DIR = RL_DIR / "checkpoints"
POLICY_PATH = RL_DIR / "policy.zip"
RESULTS_PATH = EVAL_DIR / "rl-results.json"

#: Arms Phase 8 already measured that Study 3 compares against. The best of
#: these on the co-primary outcome is "the best rule" in RQ4.
RULE_ARMS = ("rule_legacy", "rule_improved", "mastery_threshold_bkt",
             "irt_cat_maxinfo", "model_L0", "model_L4")

#: PPO's rollout length, so "every N rollouts" is a number of timesteps.
ROLLOUT = 2048


def best_rule_arm() -> str:
    """The strongest Phase 8 arm on the co-primary, read from its own report.

    Chosen from `policy-results-v0.json` rather than typed in, so it tracks the
    experiment instead of a memory of it. Falls back to `rule_improved` when
    Phase 8's artifact is absent.
    """
    path = EVAL_DIR / "policy-results-v0.json"
    if not path.exists():
        return "rule_improved"
    report = json.loads(path.read_text(encoding="utf-8"))
    scored = [(name, summary["mean_knowledge_gain"])
              for name, summary in report["policies"].items() if name in RULE_ARMS]
    return max(scored, key=lambda entry: entry[1])[0] if scored else "rule_improved"


# ------------------------------------------------------------------ training

def newest_checkpoint() -> tuple[Path | None, int]:
    """The latest checkpoint and the timesteps it represents."""
    if not CHECKPOINT_DIR.exists():
        return None, 0
    best, best_steps = None, 0
    for path in CHECKPOINT_DIR.glob("ppo-*-steps.zip"):
        try:
            steps = int(path.stem.split("-")[1])
        except (IndexError, ValueError):
            continue
        if steps >= best_steps:
            best, best_steps = path, steps
    return best, best_steps


def train(timesteps: int, seed: int, variant: str, budget: int,
          checkpoint_interval: int, resume: bool = True) -> dict:
    """Train (or continue training) PPO. Returns the training metrics."""
    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
    from stable_baselines3.common.monitor import Monitor

    from app.rl.env import ALPEnv

    torch.set_num_threads(1)
    environment = Monitor(ALPEnv(variant=variant, seed=seed, budget=budget))
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint, done_already = newest_checkpoint() if resume else (None, 0)
    if checkpoint is not None and done_already >= timesteps:
        print(f"resume: {checkpoint.name} already covers {done_already:,} ≥ {timesteps:,} "
              "timesteps; nothing left to train")
    elif checkpoint is not None:
        print(f"resume: {checkpoint.name} at {done_already:,} timesteps, "
              f"{timesteps - done_already:,} to go")

    if checkpoint is not None:
        model = PPO.load(str(checkpoint), env=environment, device="cpu")
    else:
        model = PPO("MlpPolicy", environment, seed=seed, device="cpu", verbose=0)

    class Curve(BaseCallback):
        """Episode reward against timesteps — `f09-06`'s learning curve."""

        def __init__(self):
            super().__init__()
            self.points: list[dict] = []

        def _on_step(self) -> bool:
            for info in self.locals.get("infos", []):
                episode = info.get("episode")
                if episode is not None:
                    self.points.append({
                        "timesteps": int(self.num_timesteps + done_already),
                        "episode_reward": float(episode["r"]),
                        "episode_length": int(episode["l"]),
                        "reward_per_decision": float(episode["r"]) / max(1, int(episode["l"])),
                    })
            return True

    curve = Curve()
    remaining = max(0, timesteps - done_already)
    started = time.perf_counter()
    if remaining:
        model.learn(
            total_timesteps=remaining,
            reset_num_timesteps=False,
            callback=[curve, CheckpointCallback(
                save_freq=max(1, checkpoint_interval) * ROLLOUT,
                save_path=str(CHECKPOINT_DIR), name_prefix="ppo")],
        )
    elapsed = round(time.perf_counter() - started, 1)
    RL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(POLICY_PATH))
    # A final checkpoint at the full count, so a rerun at the same size resumes
    # instead of retraining, and a rerun at a larger one continues from here.
    model.save(str(CHECKPOINT_DIR / f"ppo-{timesteps}-steps.zip"))
    return {"timesteps": timesteps, "resumed_from": done_already,
            "trained_this_run": remaining, "seconds": elapsed,
            "episodes": len(curve.points), "curve": curve.points}


# ---------------------------------------------------------------- evaluation

def evaluate(arms: list[str], variants: list[str], seed: int, learners: int,
             budget: int, thresholds: dict) -> tuple[pd.DataFrame, list[dict]]:
    """Every arm on every variant, on the identical cohort, paying §10.1 rewards."""
    records, rows = [], []
    for variant in variants:
        for arm in arms:
            run = run_arm(arm, variant, seed, learners, budget,
                          learn=arm.startswith("bandit_"),
                          model_path=POLICY_PATH if arm == "rl_ppo" else None,
                          thresholds=thresholds)
            run.pop("policy")
            records.extend({**record, "variant": variant} for record in run["results"])
            rows.append({"arm": arm, "variant": variant,
                         "mean_reward": run["mean_reward"],
                         "decisions": run["decisions"],
                         "seconds": run["seconds"],
                         "reward_curve": run["reward_curve"]})
            print(f"  {variant} {arm}: mean reward {run['mean_reward']:+.5f} "
                  f"over {run['decisions']:,} decisions ({run['seconds']}s)")
            log_progress({"stage": "rl-eval", "variant": variant, "arm": arm,
                          "mean_reward": run["mean_reward"], "seconds": run["seconds"]})
    return pd.DataFrame(records), rows


def sample_efficiency(curve: list[dict], baseline: float) -> dict:
    """Timesteps of training before PPO's episode reward matches the rule arm.

    `ponytail:` read off the training curve, smoothed over ten episodes, rather
    than by evaluating a frozen policy every N steps. The cheap version answers
    `f09-07`'s question; a proper evaluation sweep is the upgrade if the
    crossing point ever carries weight in the paper.
    """
    if not curve:
        return {"reached": False, "timesteps": None, "baseline": baseline}
    values = np.array([point["reward_per_decision"] for point in curve])
    steps = np.array([point["timesteps"] for point in curve])
    window = min(10, len(values))
    smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
    hit = np.flatnonzero(smoothed >= baseline)
    if hit.size == 0:
        return {"reached": False, "timesteps": None, "baseline": baseline,
                "best_smoothed": float(smoothed.max())}
    return {"reached": True, "timesteps": int(steps[window - 1 + hit[0]]),
            "baseline": baseline, "best_smoothed": float(smoothed.max())}


def main() -> None:
    limits = runprofile.load(runprofile.peek(sys.argv))
    parser = argparse.ArgumentParser(description="Study 3 — PPO and the RQ4 comparison")
    parser.add_argument("--profile", default=limits["profile"], choices=sorted(runprofile.PROFILES))
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--learners", type=int, default=limits["learners"])
    parser.add_argument("--budget", type=int, default=min(limits["budget"], BUDGET))
    parser.add_argument("--timesteps", type=int, default=limits["rl_timesteps"])
    parser.add_argument("--checkpoint-interval", type=int, default=limits["checkpoint_interval"])
    parser.add_argument("--train-variant", default="V0")
    parser.add_argument("--variants", default="V0,V1,V2,V3,V4")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--train-only", action="store_true",
                        help="stop after training; used by the resume test")
    args = parser.parse_args()

    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in variants if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variants {unknown}; known: {sorted(VARIANTS)}")

    print(runprofile.banner({**limits, "profile": args.profile, "learners": args.learners,
                             "budget": args.budget},
                            f"timesteps={args.timesteps} checkpoint_every="
                            f"{args.checkpoint_interval} rollouts"))
    started = time.perf_counter()

    metrics = train(args.timesteps, args.seed, args.train_variant, args.budget,
                    args.checkpoint_interval, resume=not args.no_resume)
    print(f"PPO: {metrics['trained_this_run']:,} timesteps this run "
          f"({metrics['episodes']} episodes, {metrics['seconds']}s), "
          f"wrote {POLICY_PATH.relative_to(ROOT)}")
    log_progress({"stage": "rl-train", **{key: metrics[key] for key in
                                          ("timesteps", "resumed_from", "seconds", "episodes")}})

    if args.train_only:
        return

    rule = best_rule_arm()
    arms = [rule, "bandit_linucb", "bandit_lin_ts", "rl_ppo"]
    thresholds = observable_thresholds()
    print(f"evaluating {', '.join(arms)} on {', '.join(variants)} "
          f"(best Phase 8 arm: {rule})")
    frame, rows = evaluate(arms, variants, args.seed, args.learners, args.budget, thresholds)

    reward_by_arm = {variant: {row["arm"]: row["mean_reward"] for row in rows
                               if row["variant"] == variant} for variant in variants}
    baseline = reward_by_arm.get(args.train_variant, {}).get(rule, float("nan"))

    comparisons = {}
    for variant in variants:
        subset = frame[frame["variant"] == variant]
        comparisons[variant] = {
            "items_to_mastery": [test for test in paired_tests(subset, "items_to_mastery", args.seed)
                                 if rule in (test["policy_a"], test["policy_b"])],
            "knowledge_gain": [test for test in paired_tests(subset, "knowledge_gain", args.seed)
                               if rule in (test["policy_a"], test["policy_b"])],
        }

    summary = {}
    for (variant, arm), group in frame.groupby(["variant", "policy"], observed=True):
        summary.setdefault(variant, {})[arm] = {
            "learners": int(len(group)),
            "mastery_rate": float(group["mastered"].mean()),
            "mean_items_to_mastery": float(group["items_to_mastery"].mean()),
            "mean_knowledge_gain": float(group["knowledge_gain"].mean()),
            "knowledge_gain_95_ci": bootstrap_ci(
                group["knowledge_gain"].to_numpy(dtype=float), args.seed),
            "mean_reward": reward_by_arm.get(variant, {}).get(arm),
            "mean_difficulty": float(group["mean_difficulty"].mean()),
            "interventions_per_learner": {
                name: float(group[f"n_{name}"].mean())
                for name in ("hint", "worked_example", "break_suggestion", "no_intervention")},
        }

    # RQ4, stated in one line rather than left to a reader of the table.
    train_summary = summary.get(args.train_variant, {})
    learned_arms = ["bandit_linucb", "bandit_lin_ts", "rl_ppo"]
    winners = [arm for arm in learned_arms
               if train_summary.get(arm, {}).get("mean_knowledge_gain", -np.inf)
               > train_summary.get(rule, {}).get("mean_knowledge_gain", np.inf)]
    beaten_with_ci = []
    for test in comparisons.get(args.train_variant, {}).get("knowledge_gain", []):
        other = test["policy_b"] if test["policy_a"] == rule else test["policy_a"]
        if other not in learned_arms:
            continue
        # `mean_difference` is policy_a − policy_b, so the sign depends on which
        # side the rule landed on. A CI that excludes zero in the *rule's*
        # favour is not a win for the learned arm, and counting it as one is how
        # a null gets reported as a result.
        sign = 1.0 if test["policy_a"] == other else -1.0
        low, high = sorted(sign * value for value in test["difference_95_ci"])
        if low > 0:
            beaten_with_ci.append(other)

    rule_reward = reward_by_arm.get(args.train_variant, {}).get(rule, np.inf)
    beats_on_reward = [arm for arm in learned_arms
                       if reward_by_arm.get(args.train_variant, {}).get(arm, -np.inf)
                       > rule_reward]

    config = {
        "algorithm": "PPO", "library": "stable-baselines3", "policy": "MlpPolicy",
        "hyperparameters": "library defaults (preregistration §10.4: no sweep)",
        "timesteps": args.timesteps, "train_variant": args.train_variant,
        "budget": args.budget, "seed": args.seed,
        "checkpoint_every_rollouts": args.checkpoint_interval,
        "rollout_timesteps": ROLLOUT,
        "observation": "app.policy.learned.context — gated states and observables only",
        "action_space": "MultiDiscrete([10, 3, 4]) — the Phase 8 shared action space",
    }
    RL_DIR.mkdir(parents=True, exist_ok=True)
    (RL_DIR / "training_config.json").write_text(
        json.dumps(config, indent=2) + "\n", encoding="utf-8")
    (RL_DIR / "reward_spec.json").write_text(
        json.dumps(REWARD_SPEC, indent=2) + "\n", encoding="utf-8")
    (RL_DIR / "training_metrics.json").write_text(
        json.dumps({**metrics, "sample_efficiency": sample_efficiency(metrics["curve"], baseline)},
                   indent=2, default=float) + "\n", encoding="utf-8")
    (RL_DIR / "training_manifest.json").write_text(json.dumps({
        "seed": args.seed, "dataset": f"simulated closed loop, {args.train_variant}",
        "split_strategy": "trained on V0, evaluated on V0–V4 (preregistration §10.4)",
        "hyperparameters": config["hyperparameters"],
        "split_sizes": {"train_timesteps": args.timesteps,
                        "evaluation_learners_per_variant": args.learners},
        "git_sha": registry.git_sha(), "profile": args.profile,
        "reward": REWARD_SPEC,
    }, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {RL_DIR.relative_to(ROOT)}/ (policy.zip + 4 contract files)")

    report = {
        "phase": 9, "study": 3, "run_profile": args.profile, "seed": args.seed,
        "learners_per_variant": args.learners, "budget": args.budget,
        "train_variant": args.train_variant, "variants": variants,
        "best_phase8_arm": rule,
        "reward_spec": REWARD_SPEC,
        "training": {key: metrics[key] for key in
                     ("timesteps", "resumed_from", "trained_this_run", "seconds", "episodes")},
        "learning_curve": metrics["curve"],
        "sample_efficiency": sample_efficiency(metrics["curve"], baseline),
        "mean_reward": reward_by_arm,
        "summary": summary,
        "paired_tests": comparisons,
        "rq4": {
            "question": "does a learned policy beat the best transparent rule?",
            "outcome_metric": "knowledge_gain",
            "beats_on_point_estimate": winners,
            "beats_with_ci_excluding_zero": beaten_with_ci,
            # The outcome and the objective are not the same quantity, and a
            # learned arm can win one while losing the other: buying learning
            # with time that §10.1's λ prices cheaply shows up as a knowledge
            # gain and as a *worse* mean reward. Reporting only the favourable
            # one of the two would be the exact dishonesty §10.5 forbids.
            "beats_on_mean_reward": beats_on_reward,
            "verdict": (
                ("a learned arm beats the best rule on knowledge gain with a CI excluding zero"
                 if beaten_with_ci else
                 "no learned arm beats the best rule on knowledge gain with a CI excluding "
                 "zero — reported as the result (preregistration §10.5)")
                + ("; and on the reward they optimise" if beats_on_reward else
                   "; none beats it on the reward they optimise")),
        },
        "protocol": {
            "cohort": "identical across arms within a variant; comparisons are paired",
            "bandits": "learn online within each variant; PPO is trained on V0 only",
            "simulated": "no human participants",
        },
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote: {RESULTS_PATH.relative_to(ROOT)}")
    print(f"RQ4: {report['rq4']['verdict']}")
    log_progress({"stage": "rl", "elapsed_seconds": round(time.perf_counter() - started, 1),
                  "rq4": report["rq4"]["verdict"]})


if __name__ == "__main__":
    main()
