"""Phase 9, Study 3 — logged data and offline policy evaluation. `make ope`.

    python -m app.research.ope --profile smoke --seed 20260821

Three things, in the order they depend on each other:

1. **A logged dataset with propensities.** `model_L4` at ε = 0.15 (§10.2) runs
   the Phase 8 closed loop; every decision is written to
   `artifacts/datasets/logged-bandit-data.parquet` with the context the policy
   saw, the action, the propensity it was chosen with, the reward, the next
   context and a terminal flag. Plain replay without propensities would mislead:
   under an adaptive policy a hint is requested *because* the learner is
   struggling, so the action and the outcome share a cause.

2. **IPS, SNIPS and Doubly Robust**, each reported with its **effective sample
   size** and the spread of its importance weights. An OPE point estimate
   without an ESS is not interpretable — a value computed from four surviving
   decisions is noise with a decimal point.

3. **The estimates against the truth.** In simulation the on-policy value is
   computable, so every target policy is also *run*, and the gap between
   estimate and truth is the phase's methodological figure (f09-02). §10.3
   pre-commits to DR beating IPS; if it does not, the reward model is broken and
   that is a bug hunt, not a result.

The driver in :func:`run_arm` is shared with `app.research.train_rl` so that a
bandit is trained, evaluated on-policy and logged by the same code path.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from app.model import validation_gate
from app.policy import learned
from app.policy import policies as arms_module
from app.research import runprofile
from app.research.closed_loop import BUDGET, ClosedLoop
from app.research.evaluate_policies import observable_thresholds
from app.rl.reward import SPEC as REWARD_SPEC
from app.rl.reward import reward as step_reward

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "artifacts" / "evaluation"
DATA_DIR = ROOT / "artifacts" / "datasets"
MODEL_DIR = ROOT / "artifacts" / "models"
LOG_PATH = DATA_DIR / "logged-bandit-data.parquet"
RESULTS_PATH = EVAL_DIR / "ope-results.json"
RUN_LOG = EVAL_DIR / "run-log.jsonl"
PPO_PATH = MODEL_DIR / "rl" / "policy.zip"

#: The logging policy and its exploration rate (§10.2). Higher than Phase 8's
#: ε = 0.1 because a target action the logging policy never takes is a target
#: action OPE cannot say anything about.
LOGGING_ARM = "model_L4"
LOGGING_EPSILON = 0.15

#: Ridge penalty of the DR reward model. One knob, fixed, not tuned per run.
REWARD_MODEL_RIDGE = 1.0

#: Exploration of the *target* policies, both in the estimate and in the
#: on-policy ground-truth run (§10.3). A deterministic target over a 120-action
#: space overlaps an ε = 0.15 log almost nowhere — measured ESS was 1.1 to 4.0
#: out of 2,383 decisions, which is an estimate with no information in it. The
#: quantity Study 3 actually cares about is the value of the arm *as it would be
#: deployed*, exploration included, so the target keeps the same ε as the
#: logging policy and the estimator uses the stochastic-target form
#: ``w = π(a|x) / p(a|x)`` rather than an indicator.
TARGET_EPSILON = LOGGING_EPSILON

#: Importance weights are clipped here before any estimate is formed (§10.3).
#: Unclipped, a single exploratory decision whose action the target also picks
#: carries weight 681 — one decision outvoting six hundred — and the estimator's
#: variance is then entirely that one draw. Clipping trades a known downward
#: bias for a usable variance; both the clipped and the raw ESS are reported so
#: a reader can see how much of the estimate rests on how few decisions.
WEIGHT_CLIP = 20.0


# --------------------------------------------------------------- the driver

def run_arm(arm: str, variant: str, seed: int, learners: int, budget: int,
            epsilon: float = 0.0, learn: bool = True, log: bool = False,
            model_path: Path | None = None, thresholds: dict | None = None) -> dict:
    """Run one arm through the closed loop, paying it the §10.1 reward.

    ``learn`` is what separates training a bandit from measuring one: with it
    off the arm's parameters are frozen and the run is an on-policy evaluation,
    which is exactly the ground truth OPE is checked against.
    """
    import torch

    torch.set_num_threads(1)
    gate = validation_gate.load()
    thresholds = thresholds if thresholds is not None else observable_thresholds()
    environment = ClosedLoop(variant, seed, learners, budget=budget)
    if arm in learned.ARM_KEYS:
        policy = learned.build(arm, learners, seed, model_path=model_path, gate=gate)
    else:
        policy = arms_module.build(arm, learners, seed, budget=budget,
                                   thresholds=thresholds, gate=gate)
    policy.epsilon = epsilon
    admitted = learned.admitted_states(gate)

    rows: list[dict | None] = [None] * learners
    transitions: list[dict] = []
    pending: list[dict | None] = [None] * learners
    rewards_seen: list[float] = []
    reward_curve: list[float] = []
    started = time.perf_counter()

    def contexts(views: list[dict | None]) -> list[np.ndarray | None]:
        """The §10.1 context for every live learner, from the arm's own states."""
        estimator = getattr(policy, "estimator", None)
        out: list[np.ndarray | None] = []
        for view in views:
            if view is None:
                out.append(None)
                continue
            index = view["index"]
            states = estimator.states(index) if estimator is not None else {}
            errors = estimator.standard_error(index) if estimator is not None else {}
            out.append(learned.context(view, states, errors, admitted))
        return out

    for _ in range(budget):
        if environment.done:
            break
        views = environment.views(policy.rung, rows)
        vectors = contexts(views)
        actions = policy.act(views)

        before = [np.mean([environment.success_probability(run, concept)
                           for concept in environment.curriculum])
                  for run in environment.runs]
        seconds_before = [run.on_task_seconds for run in environment.runs]

        outcomes = environment.step(actions)
        policy.observe(outcomes)
        rows = getattr(policy, "last_rows", outcomes)

        after = [np.mean([environment.success_probability(run, concept)
                          for concept in environment.curriculum])
                 for run in environment.runs]

        rewards: list[float | None] = [None] * learners
        step_rewards: list[float] = []
        for index, (action, row) in enumerate(zip(actions, outcomes)):
            if action is None or row is None:
                continue
            run = environment.runs[index]
            payoff = step_reward(after[index] - before[index],
                                 run.on_task_seconds - seconds_before[index],
                                 run.incorrect_streak)
            rewards[index] = payoff
            step_rewards.append(payoff)
            rewards_seen.append(payoff)
            if not log:
                continue
            # The transition is completed on the learner's *next* decision,
            # when its next context exists. A learner who finishes here keeps
            # the zero vector and terminal=True instead.
            pending[index] = {
                "learner_id": row["learner_id"], "step": int(row["step"]),
                "concept_id": row["concept_id"], "item_id": row["item_id"],
                "action_index": learned.action_index(action),
                "difficulty": int(action["difficulty"]),
                "concept_move": action["concept_move"],
                "intervention": action["intervention"],
                "propensity": float(action["propensity"]),
                "explored": bool(action.get("explored", False)),
                "correct": int(row["correct"]),
                "reward": float(payoff),
                "mastery_delta": float(after[index] - before[index]),
                "seconds": float(run.on_task_seconds - seconds_before[index]),
                "terminal": bool(run.finished),
                **{f"ctx_{name}": float(value)
                   for name, value in zip(learned.feature_names(admitted), vectors[index])},
            }

        if learn and hasattr(policy, "learn"):
            policy.learn(rewards)
        reward_curve.append(float(np.mean(step_rewards)) if step_rewards else float("nan"))

        if log:
            # Close out each pending transition with the context that follows it.
            next_views = environment.views(policy.rung, rows)
            next_vectors = contexts(next_views)
            names = learned.feature_names(admitted)
            for index, record in enumerate(pending):
                if record is None:
                    continue
                following = next_vectors[index]
                if following is None:
                    following = np.zeros(len(names))
                    record["terminal"] = True
                record.update({f"nxt_{name}": float(value)
                               for name, value in zip(names, following)})
                transitions.append(record)
                pending[index] = None

    return {
        "arm": arm, "variant": variant, "seed": seed,
        "epsilon": epsilon, "learned_online": learn,
        "results": [{**record, "policy": arm} for record in environment.results()],
        "transitions": transitions,
        "mean_reward": float(np.mean(rewards_seen)) if rewards_seen else float("nan"),
        "total_reward": float(np.sum(rewards_seen)),
        "decisions": len(rewards_seen),
        "reward_curve": reward_curve,
        "seconds": round(time.perf_counter() - started, 1),
        "policy": policy,
    }


def log_progress(entry: dict) -> None:
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"script": "ope", **entry}, default=str) + "\n")


# ------------------------------------------------------------- the estimators

def target_probability(frame: pd.DataFrame, chosen: np.ndarray,
                       epsilon: float = TARGET_EPSILON) -> np.ndarray:
    """``π_target(logged action | context)`` for an ε-greedy target."""
    match = (chosen == frame["action_index"].to_numpy()).astype(float)
    return epsilon / learned.N_ACTIONS + (1.0 - epsilon) * match


def importance_weights(frame: pd.DataFrame, chosen: np.ndarray,
                       epsilon: float = TARGET_EPSILON) -> np.ndarray:
    """``π_target(a|x) / p_logging(a|x)``, per decision."""
    return target_probability(frame, chosen, epsilon) / frame["propensity"].to_numpy()


def effective_sample_size(weights: np.ndarray) -> float:
    total = float(weights.sum())
    squared = float((weights ** 2).sum())
    return (total ** 2) / squared if squared > 0 else 0.0


def action_features(actions: np.ndarray) -> np.ndarray:
    """A compact encoding of an action: difficulty, its square, and two one-hots.

    Not 120 disjoint one-hots. Under an adaptive logging policy 85 % of the log
    is the same greedy action, so per-action dummies are identified by a handful
    of exploratory draws each; a difficulty scalar plus move and intervention
    indicators shares strength across neighbouring actions and leaves room for
    the context interactions below.
    """
    actions = np.asarray(actions, dtype=int)
    difficulty = np.array([learned.action_from_index(index)["difficulty"]
                           for index in actions], dtype=float)
    move = np.array([learned.CONCEPT_MOVES.index(
        learned.action_from_index(index)["concept_move"]) for index in actions])
    intervention = np.array([learned.INTERVENTIONS.index(
        learned.action_from_index(index)["intervention"]) for index in actions])
    move_hot = np.eye(len(learned.CONCEPT_MOVES))[move]
    intervention_hot = np.eye(len(learned.INTERVENTIONS))[intervention]
    scaled = (difficulty - 5.5) / 4.5
    return np.column_stack([scaled, scaled ** 2, move_hot, intervention_hot])


#: Mean action encoding over the whole action space. The ε-uniform part of a
#: target policy's value needs it, and it is a constant.
MEAN_ACTION = action_features(np.arange(learned.N_ACTIONS)).mean(axis=0)


def build_design(contexts: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """``[context, action, context ⊗ action]`` — the reward model's features."""
    encoded = action_features(actions)
    interactions = (contexts[:, :, None] * encoded[:, None, :]).reshape(len(contexts), -1)
    return np.hstack([contexts, encoded, interactions])


def uniform_design(contexts: np.ndarray) -> np.ndarray:
    """The design a uniformly-drawn action would produce, in expectation.

    The model is linear, so ``E_a[design(x, a)]`` is the design built from the
    mean action encoding — no 120-fold prediction needed.
    """
    encoded = np.repeat(MEAN_ACTION[None, :], len(contexts), axis=0)
    interactions = (contexts[:, :, None] * encoded[:, None, :]).reshape(len(contexts), -1)
    return np.hstack([contexts, encoded, interactions])


def fit_reward_model(frame: pd.DataFrame, names: list[str]):
    """r̂(context, action) for the DR estimator. Ridge over context × action.

    `ponytail:` a linear model with first-order interactions. It only has to be
    better than nothing for DR to beat IPS; if `f09-02` shows it is not, the
    upgrade is a gradient-boosted r̂, and the evidence for needing one is in the
    figure rather than in a preference.
    """
    from sklearn.linear_model import Ridge

    design = build_design(frame[[f"ctx_{name}" for name in names]].to_numpy(),
                          frame["action_index"].to_numpy())
    model = Ridge(alpha=REWARD_MODEL_RIDGE)
    model.fit(design, frame["reward"].to_numpy())
    return model


def direct_value(model, contexts: np.ndarray, chosen: np.ndarray,
                 epsilon: float = TARGET_EPSILON) -> np.ndarray:
    """``E_{a∼π_target} r̂(x, a)`` — the DR estimator's model term.

    The target is ε-greedy, so its value under the reward model is the greedy
    action's prediction mixed with the uniform one; :func:`uniform_design`
    supplies the second without predicting 120 actions per row.
    """
    greedy = model.predict(build_design(contexts, chosen))
    uniform = model.predict(uniform_design(contexts))
    return epsilon * uniform + (1.0 - epsilon) * greedy


def estimate(frame: pd.DataFrame, chosen: np.ndarray, names: list[str],
             model, seed: int, draws: int = 500) -> dict:
    """IPS, SNIPS and DR for one target policy, with ESS and bootstrap CIs."""
    rewards = frame["reward"].to_numpy()
    raw_weights = importance_weights(frame, chosen)
    weights = np.minimum(raw_weights, WEIGHT_CLIP)
    contexts = frame[[f"ctx_{name}" for name in names]].to_numpy()
    logged_actions = frame["action_index"].to_numpy()

    predicted_logged = model.predict(build_design(contexts, logged_actions))
    direct = direct_value(model, contexts, chosen)
    per_decision_dr = direct + weights * (rewards - predicted_logged)

    rng = np.random.default_rng(seed)
    boots = {"ips": [], "snips": [], "dr": []}
    for _ in range(draws):
        pick = rng.integers(0, len(frame), len(frame))
        weight_draw, reward_draw = weights[pick], rewards[pick]
        boots["ips"].append(float(np.mean(weight_draw * reward_draw)))
        total = weight_draw.sum()
        boots["snips"].append(float((weight_draw * reward_draw).sum() / total)
                              if total > 0 else float("nan"))
        boots["dr"].append(float(np.mean(per_decision_dr[pick])))

    def interval(values: list[float]) -> list[float]:
        clean = np.asarray([value for value in values if np.isfinite(value)])
        if clean.size == 0:
            return [float("nan"), float("nan")]
        return [float(np.quantile(clean, 0.025)), float(np.quantile(clean, 0.975))]

    total = float(weights.sum())
    return {
        "ips": float(np.mean(weights * rewards)),
        "snips": float((weights * rewards).sum() / total) if total > 0 else float("nan"),
        "dr": float(np.mean(per_decision_dr)),
        "ips_95_ci": interval(boots["ips"]),
        "snips_95_ci": interval(boots["snips"]),
        "dr_95_ci": interval(boots["dr"]),
        "direct_method": float(np.mean(direct)),
        "effective_sample_size": effective_sample_size(weights),
        "effective_sample_size_unclipped": effective_sample_size(raw_weights),
        "weight_clip": WEIGHT_CLIP,
        "clipped_share": float((raw_weights > WEIGHT_CLIP).mean()),
        "decisions": int(len(frame)),
        "support_rate": float((weights > 0).mean()),
        "action_match_rate": float((chosen == logged_actions).mean()),
        "target_epsilon": TARGET_EPSILON,
        "weight_max": float(raw_weights.max()),
        "weight_mean": float(raw_weights.mean()),
        "weight_p99": float(np.quantile(raw_weights, 0.99)),
    }


# ---------------------------------------------------------------- targets

def frozen_bandit_choices(path: Path, frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    """What a trained bandit would do at each logged context, greedily."""
    saved = np.load(path)
    theta = saved["theta"]
    contexts = frame[[f"ctx_{name}" for name in names]].to_numpy()
    return np.argmax(contexts @ theta.T, axis=1)


def ppo_choices(path: Path, frame: pd.DataFrame, names: list[str]) -> np.ndarray:
    from stable_baselines3 import PPO

    model = PPO.load(str(path), device="cpu")
    contexts = frame[[f"ctx_{name}" for name in names]].to_numpy().astype("float32")
    actions, _ = model.predict(np.clip(contexts, -10, 10), deterministic=True)
    return np.array([learned.action_index(
        {"difficulty": int(learned.DIFFICULTIES[row[0]]),
         "concept_move": learned.CONCEPT_MOVES[row[1]],
         "intervention": learned.INTERVENTIONS[row[2]]})
        for row in np.atleast_2d(actions)])


def train_bandits(variant: str, seed: int, learners: int, budget: int,
                  thresholds: dict) -> dict:
    """Both bandits, trained online, saved as their θ matrix. Returns the runs."""
    out = {}
    for arm in ("bandit_linucb", "bandit_lin_ts"):
        run = run_arm(arm, variant, seed, learners, budget, learn=True,
                      thresholds=thresholds)
        policy = run.pop("policy")
        path = MODEL_DIR / f"{arm}-{variant.lower()}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, theta=policy.theta, feature_names=np.array(policy.names),
                 updates=policy.updates)
        run["artifact"] = str(path.relative_to(ROOT))
        run["updates"] = int(policy.updates)
        out[arm] = run
        print(f"  trained {arm}: {run['updates']:,} updates, "
              f"mean reward {run['mean_reward']:+.5f} ({run['seconds']}s)")
    return out


def main() -> None:
    profile = runprofile.load(runprofile.peek(__import__("sys").argv))
    parser = argparse.ArgumentParser(description="Study 3 — logged data and OPE")
    parser.add_argument("--profile", default=profile["profile"])
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--learners", type=int, default=profile["learners"])
    parser.add_argument("--budget", type=int, default=min(profile["budget"], BUDGET))
    parser.add_argument("--variant", default="V0")
    parser.add_argument("--epsilon", type=float, default=LOGGING_EPSILON)
    args = parser.parse_args()

    print(runprofile.banner({**profile, "profile": args.profile,
                             "learners": args.learners, "budget": args.budget},
                            f"logging={LOGGING_ARM} epsilon={args.epsilon}"))
    started = time.perf_counter()
    thresholds = observable_thresholds()
    gate = validation_gate.load()
    names = learned.feature_names(learned.admitted_states(gate))

    # 1 — the logged dataset --------------------------------------------------
    logging_run = run_arm(LOGGING_ARM, args.variant, args.seed, args.learners, args.budget,
                          epsilon=args.epsilon, learn=False, log=True, thresholds=thresholds)
    logging_run.pop("policy")
    frame = pd.DataFrame(logging_run["transitions"])
    frame["policy_version"] = LOGGING_ARM
    frame["logging_epsilon"] = args.epsilon
    frame["variant"] = args.variant
    frame["simulator_version"] = args.variant
    frame["seed"] = args.seed
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(LOG_PATH, index=False)
    print(f"wrote: {LOG_PATH.relative_to(ROOT)} ({len(frame):,} logged decisions, "
          f"{frame['action_index'].nunique()} distinct actions)")
    log_progress({"stage": "logging", "decisions": int(len(frame)),
                  "seconds": logging_run["seconds"], "profile": args.profile})

    # 2 — the target policies -------------------------------------------------
    print("training the bandits online (their evaluation is the same run)")
    bandits = train_bandits(args.variant, args.seed, args.learners, args.budget, thresholds)

    targets: dict[str, np.ndarray] = {}
    for arm in ("bandit_linucb", "bandit_lin_ts"):
        targets[arm] = frozen_bandit_choices(
            MODEL_DIR / f"{arm}-{args.variant.lower()}.npz", frame, names)
    if PPO_PATH.exists():
        targets["rl_ppo"] = ppo_choices(PPO_PATH, frame, names)
    else:
        print(f"note: {PPO_PATH.relative_to(ROOT)} absent — run `make rl` to add PPO to the "
              "OPE comparison")

    # 3 — estimate, then measure the truth ------------------------------------
    model = fit_reward_model(frame, names)
    rows = []
    for arm, chosen in targets.items():
        estimated = estimate(frame, chosen, names, model, args.seed)
        truth = run_arm(arm, args.variant, args.seed, args.learners, args.budget,
                        epsilon=TARGET_EPSILON, learn=False, thresholds=thresholds,
                        model_path=PPO_PATH if arm == "rl_ppo" else None)
        truth.pop("policy")
        rows.append({"policy": arm, **estimated,
                     "true_value": truth["mean_reward"],
                     "true_decisions": truth["decisions"],
                     "on_policy_seconds": truth["seconds"]})
        print(f"  {arm}: IPS {estimated['ips']:+.5f}  SNIPS {estimated['snips']:+.5f}  "
              f"DR {estimated['dr']:+.5f}  truth {truth['mean_reward']:+.5f}  "
              f"ESS {estimated['effective_sample_size']:.1f}")

    errors = {name: float(np.mean([abs(row[name] - row["true_value"]) for row in rows]))
              for name in ("ips", "snips", "dr")}
    dr_beats_ips = errors["dr"] < errors["ips"]

    report = {
        "phase": 9, "study": 3, "run_profile": args.profile,
        "variant": args.variant, "seed": args.seed,
        "learners": args.learners, "budget": args.budget,
        "logging_policy": LOGGING_ARM, "logging_epsilon": args.epsilon,
        "target_epsilon": TARGET_EPSILON,
        "logged_decisions": int(len(frame)),
        "logged_mean_reward": logging_run["mean_reward"],
        "action_space": learned.N_ACTIONS,
        "context_features": names,
        "reward_spec": REWARD_SPEC,
        "reward_model": {"kind": "ridge", "alpha": REWARD_MODEL_RIDGE,
                         "features": "context + action encoding + their interactions"},
        "estimates": rows,
        "mean_absolute_error_vs_truth": errors,
        "dr_beats_ips": bool(dr_beats_ips),
        "bandit_training": {arm: {"mean_reward": run["mean_reward"],
                                  "updates": run["updates"], "artifact": run["artifact"],
                                  "reward_curve": run["reward_curve"]}
                            for arm, run in bandits.items()},
        "protocol": {
            "estimator": "per-decision contextual-bandit IPS/SNIPS/DR on the logged context",
            "ground_truth": "the same target policy, at the same epsilon, run on-policy "
                            "on the same cohort",
            "ess": "(sum w)^2 / sum w^2, reported with every estimate (§10.3)",
            "simulated": "no human participants",
        },
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(report, indent=2, default=float) + "\n",
                            encoding="utf-8")
    print(f"wrote: {RESULTS_PATH.relative_to(ROOT)}")
    log_progress({"stage": "ope", "targets": len(rows), "dr_beats_ips": dr_beats_ips,
                  "elapsed_seconds": round(time.perf_counter() - started, 1)})

    if not dr_beats_ips:
        # §10.3 pre-commits to this check. It is a bug hunt, not a finding.
        print(f"WARNING: DR mean |error| {errors['dr']:.5f} is not below IPS "
              f"{errors['ips']:.5f}. Preregistration §10.3: investigate the reward model "
              "before reporting any Study 3 number.")


if __name__ == "__main__":
    main()
