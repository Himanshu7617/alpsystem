"""Fit Simulator v2 against real archival data, then prove the fit by KS test.

    python -m app.research.calibrate_simulator [--target assistments_2012] [--learners 1000]

Fits ability, slip/guess, learning rate, response-time location/scale, session
structure and within-session fatigue from `data/processed/<target>.parquet`,
writes `artifacts/datasets/simulator-params-v2.json`, then **generates from the
fitted model and compares distributions against the real ones**.

Acceptance is a distribution match, not a number: two-sample KS *D* ≤ 0.15 on
per-learner accuracy, log response time, sequence length and within-session
accuracy slope. A larger *D* exits non-zero unless it is registered in
:data:`DOCUMENTED_MISMATCHES` with its reason — BUILD.md Phase 4 step 2.

`assistments_2012` is the default target because it is the archival source whose
mean sequence length (115.7) matches the study's simulated budget of ~120 items
per learner, and it carries timestamps, so sessions and within-session decay are
observable. EdNet has more rows but its RTE semantics are already a recorded
Phase 3 deviation.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import optimize, stats

from dataclasses import replace

from app.research.simulator import N_OPTIONS, PARAMS_PATH, Params, Simulator

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"
KS_PATH = ROOT / "artifacts" / "datasets" / "simulator-calibration-ks.json"

KS_LIMIT = 0.15
SESSION_GAP_MS = 30 * 60 * 1000
MIN_SESSION_FOR_SLOPE = 15

#: A KS statistic above the limit that is a stated limitation rather than a bug.
#: Nothing goes in here without the evidence that justifies it.
DOCUMENTED_MISMATCHES: dict[str, str] = {}


# ------------------------------------------------------------------ real data

def load_real(target: str) -> pd.DataFrame:
    path = PROCESSED_DIR / f"{target}.parquet"
    if not path.exists():
        raise SystemExit(f"{path.relative_to(ROOT)} is missing. Run `make prep-data` first.")
    frame = pd.read_parquet(path, columns=["learner_id", "item_id", "timestamp", "order_index",
                                           "correct", "response_time_ms"])
    frame = frame.sort_values(["learner_id", "order_index"])
    if frame["timestamp"].notna().any():
        gap = frame.groupby("learner_id")["timestamp"].diff()
        boundary = (gap.isna() | (gap > SESSION_GAP_MS)).astype(int)
    else:
        # No clock in this source: fall back to fixed-length blocks so the
        # session-shaped statistics still exist, and say so in the manifest.
        boundary = (frame.groupby("learner_id").cumcount() % 20 == 0).astype(int)
    frame["session"] = boundary.groupby(frame["learner_id"]).cumsum()
    frame["position"] = frame.groupby(["learner_id", "session"]).cumcount()
    return frame


def session_slopes(frame: pd.DataFrame, item_column: str = "item_id") -> np.ndarray:
    """Per-session slope of item-difficulty-residualised correctness on position.

    'Matched difficulty' is done by residualisation rather than by stratifying:
    subtracting each item's mean accuracy removes the difficulty of what was
    served, which is what a within-session decay estimate has to control for.
    """
    residual = frame["correct"] - frame.groupby(item_column)["correct"].transform("mean")
    work = pd.DataFrame({"key": list(zip(frame["learner_id"], frame["session"])),
                         "position": frame["position"].to_numpy(float),
                         "residual": residual.to_numpy(float)})
    sizes = work.groupby("key")["position"].transform("size")
    work = work[sizes >= MIN_SESSION_FOR_SLOPE]
    if work.empty:
        return np.array([])

    def slope(group: pd.DataFrame) -> float:
        x = group["position"].to_numpy()
        y = group["residual"].to_numpy()
        centred = x - x.mean()
        variance = float((centred * centred).sum())
        return float((centred * (y - y.mean())).sum() / variance) if variance else np.nan

    return work.groupby("key")[["position", "residual"]].apply(slope).dropna().to_numpy()


# -------------------------------------------------------------------- fitting

def fit_outcome(frame: pd.DataFrame) -> dict:
    """Guess and slip asymptotes from the empirical accuracy-vs-(ability - difficulty) curve."""
    learner_accuracy = frame.groupby("learner_id")["correct"].mean().clip(0.02, 0.98)
    item_stats = frame.groupby("item_id")["correct"].agg(["mean", "size"])
    item_stats = item_stats[item_stats["size"] >= 20]
    theta = np.log(learner_accuracy / (1 - learner_accuracy))
    difficulty = -np.log(item_stats["mean"].clip(0.02, 0.98) / (1 - item_stats["mean"].clip(0.02, 0.98)))

    subset = frame[frame["item_id"].isin(difficulty.index)]
    distance = (subset["learner_id"].map(theta) - subset["item_id"].map(difficulty)).to_numpy()
    correct = subset["correct"].to_numpy(float)

    bins = np.quantile(distance, np.linspace(0, 1, 26))
    index = np.clip(np.digitize(distance, bins[1:-1]), 0, len(bins) - 2)
    centres = np.array([distance[index == b].mean() for b in range(len(bins) - 1)])
    accuracy = np.array([correct[index == b].mean() for b in range(len(bins) - 1)])

    def curve(x, guess, slip, scale):
        return guess + (1 - guess - slip) / (1 + np.exp(-scale * x))

    (guess, slip, scale), _ = optimize.curve_fit(
        curve, centres, accuracy, p0=[0.2, 0.05, 0.5], bounds=([0.0, 0.0, 0.05], [0.45, 0.35, 3.0])
    )
    return {"guess": float(guess), "slip": float(slip), "link_scale": float(scale),
            "theta": theta, "difficulty": difficulty, "distance": distance}


def fit_response_time(frame: pd.DataFrame, theta: pd.Series, difficulty: pd.Series,
                      fatigue_growth: float, trait_speed_sd: float, stem_beta: float) -> dict:
    """log RT ~ intercept + b_dist * |ability - difficulty| + b_pos * position."""
    subset = frame[frame["item_id"].isin(difficulty.index)].copy()
    subset["distance"] = (subset["learner_id"].map(theta) - subset["item_id"].map(difficulty)).abs()
    log_rt = np.log(subset["response_time_ms"].clip(lower=1)).to_numpy()

    design = np.column_stack([
        np.ones(len(subset)),
        subset["distance"].to_numpy(),
        np.minimum(subset["position"].to_numpy(float) * fatigue_growth, 1.0),  # fatigue proxy, 0-1
    ])
    coefficients, *_ = np.linalg.lstsq(design, log_rt, rcond=None)
    residual = log_rt - design @ coefficients

    # The simulator adds two log-RT terms the archival data cannot identify — a
    # per-learner speed trait and a stem-length item effect. They are carved out
    # of the residual variance so the *total* simulated spread still matches.
    residual_variance = float(residual.var())
    carved = trait_speed_sd ** 2 + stem_beta ** 2
    sigma = float(np.sqrt(max(0.10, residual_variance - carved)))
    return {"rt_intercept": float(coefficients[0]), "rt_beta_distance": float(coefficients[1]),
            "rt_beta_fatigue": float(coefficients[2]), "rt_sigma": sigma,
            "rt_residual_sd": float(np.sqrt(residual_variance))}


def fit_learning_and_fatigue(frame: pd.DataFrame, fatigue_growth: float) -> dict:
    """Learning rate from the practice curve; fatigue from what within-session decay is left over."""
    item_mean = frame.groupby("item_id")["correct"].transform("mean")
    residual = (frame["correct"] - item_mean).to_numpy(float)
    opportunity = frame.groupby("learner_id").cumcount().to_numpy(float)

    # Learning: residualised correctness against practice opportunity, per learner.
    work = pd.DataFrame({"learner": frame["learner_id"].to_numpy(), "x": opportunity, "y": residual})
    sizes = work.groupby("learner")["x"].transform("size")
    work = work[sizes >= 20]
    grouped = work.groupby("learner")
    centred_x = work["x"] - grouped["x"].transform("mean")
    centred_y = work["y"] - grouped["y"].transform("mean")
    numerator = (centred_x * centred_y).groupby(work["learner"]).sum()
    denominator = (centred_x * centred_x).groupby(work["learner"]).sum()
    slopes = (numerator / denominator.replace(0, np.nan)).dropna()

    accuracy = float(frame["correct"].mean())
    variance = accuracy * (1 - accuracy)
    # probability per item -> logits per item -> knowledge gain per item, given
    # the simulator's mastery update knowledge += rate * (3.5 - knowledge) with
    # (3.5 - knowledge) ~ 2.5 on average and a 0.3 weight on incorrect answers.
    logit_gain = slopes.clip(-0.01, 0.02) / variance
    rate = (logit_gain / (2.5 * (accuracy + 0.3 * (1 - accuracy)))).clip(lower=1e-4)
    log_rate = np.log(rate)

    within = session_slopes(frame)
    within_mean = float(np.mean(within)) if len(within) else 0.0
    learning_per_item = float(slopes.mean())
    # Fatigue is what within-session decay is left after removing the learning
    # trend. A non-negative result means this source shows no accuracy fatigue.
    decay = within_mean - learning_per_item
    fatigue_beta = float(max(0.0, -decay / (fatigue_growth * variance)))

    return {
        "learning_rate_log_mean": float(log_rate.mean()),
        "learning_rate_log_sd": float(max(0.15, log_rate.std())),
        "fatigue_accuracy_beta": fatigue_beta,
        "learning_slope_per_item": learning_per_item,
        "within_session_slope_mean": within_mean,
        "within_session_slope_sd": float(np.std(within)) if len(within) else None,
        "residual_decay_after_learning": float(decay),
    }


def fit_ability(frame: pd.DataFrame, guess: float, slip: float, bank_b: np.ndarray,
                seed: int) -> dict:
    """Choose the ability distribution whose simulated accuracy spread matches the real one.

    A closed form would need the item-exposure design; a 2-D search over
    (mean, sd) against the KS statistic optimises the acceptance criterion the
    phase is actually judged on, and costs a second.
    """
    real_accuracy = frame.groupby("learner_id")["correct"].mean().to_numpy()
    lengths = frame.groupby("learner_id").size().to_numpy()
    rng = np.random.default_rng(seed)
    draw_lengths = rng.choice(lengths, size=4000)

    best = (np.inf, 0.9, 1.0)
    for mean in np.linspace(0.0, 2.0, 21):
        for sd in np.linspace(0.4, 2.0, 17):
            ability = rng.normal(mean, sd, len(draw_lengths))
            difficulty = rng.choice(bank_b, len(draw_lengths))
            probability = guess + (1 - guess - slip) / (1 + np.exp(-(ability - difficulty)))
            simulated = rng.binomial(draw_lengths, probability) / draw_lengths
            statistic = float(stats.ks_2samp(real_accuracy, simulated).statistic)
            if statistic < best[0]:
                best = (statistic, float(mean), float(sd))
    return {"ability_mean": best[1], "ability_sd": best[2], "ability_grid_ks": best[0]}


def refine_ability(params: Params, real_accuracy: np.ndarray, seed: int,
                   rounds: int = 6, learners: int = 1500) -> tuple[Params, list[dict]]:
    """Moment-match the ability distribution against the *real* generative process.

    The grid search in :func:`fit_ability` uses a static binomial approximation,
    which ignores learning, per-concept knowledge spread and fatigue. Those shift
    the accuracy distribution enough to fail the KS gate, so the fit is finished
    against the simulator itself: simulate outcomes only (cheap — no paths, no
    extractor), correct the mean in logits and rescale the spread, keep the best.
    """
    history, best, best_statistic = [], params, np.inf
    for round_index in range(rounds):
        simulator = Simulator(params, seed=seed + round_index)
        rows = pd.DataFrame(simulator.run(simulator.learners(learners), observables=False))
        accuracy = rows.groupby("learner_id")["correct"].mean().to_numpy()
        statistic = float(stats.ks_2samp(real_accuracy, accuracy).statistic)
        history.append({"round": round_index, "ability_mean": round(params.ability_mean, 4),
                        "ability_sd": round(params.ability_sd, 4),
                        "sim_accuracy_mean": round(float(accuracy.mean()), 4),
                        "sim_accuracy_sd": round(float(accuracy.std()), 4),
                        "ks_d": round(statistic, 4)})
        if statistic < best_statistic:
            best, best_statistic = params, statistic

        # d(accuracy)/d(ability) for a logistic link is span * p(1-p); 0.2 is the
        # floor that keeps a nearly-saturated round from taking a wild step.
        span = 1 - params.guess - params.slip
        slope = max(0.05, span * float((accuracy * (1 - accuracy)).mean()))
        params = replace(
            params,
            ability_mean=params.ability_mean + (float(real_accuracy.mean()) - float(accuracy.mean())) / slope,
            ability_sd=float(np.clip(params.ability_sd * (float(real_accuracy.std()) / max(1e-6, float(accuracy.std()))), 0.2, 4.0)),
        )
    return best, history


def empirical_quantiles(values: np.ndarray, count: int = 21) -> list[float]:
    return [float(value) for value in np.quantile(values, np.linspace(0, 1, count))]


# ------------------------------------------------------------------ KS gate

def compare(real: pd.DataFrame, simulated: pd.DataFrame) -> dict:
    real_slopes = session_slopes(real)
    simulated = simulated.rename(columns={"session_id": "session"})
    simulated["position"] = simulated.groupby(["learner_id", "session"]).cumcount()
    sim_slopes = session_slopes(simulated)

    checks = {
        "accuracy": (real.groupby("learner_id")["correct"].mean().to_numpy(),
                     simulated.groupby("learner_id")["correct"].mean().to_numpy()),
        "log_response_time": (np.log(real["response_time_ms"].clip(lower=1).to_numpy()),
                              np.log(simulated["response_time_ms"].clip(lower=1).to_numpy())),
        "sequence_length": (real.groupby("learner_id").size().to_numpy(),
                            simulated.groupby("learner_id").size().to_numpy()),
        "session_decay_slope": (real_slopes, sim_slopes),
    }
    results = {}
    for name, (left, right) in checks.items():
        if not len(left) or not len(right):
            results[name] = {"ks_d": None, "note": "one side is empty"}
            continue
        # KS on millions of rows is dominated by the sample size, not the shape;
        # both sides are subsampled to the same 200k cap with a fixed seed.
        rng = np.random.default_rng(20260821)
        left = rng.choice(left, min(len(left), 200_000), replace=False)
        right = rng.choice(right, min(len(right), 200_000), replace=False)
        test = stats.ks_2samp(left, right)
        results[name] = {
            "ks_d": round(float(test.statistic), 4),
            "p_value": float(test.pvalue),
            "real_mean": round(float(np.mean(left)), 4),
            "sim_mean": round(float(np.mean(right)), 4),
            "real_sd": round(float(np.std(left)), 4),
            "sim_sd": round(float(np.std(right)), 4),
            "n_real": int(len(left)),
            "n_sim": int(len(right)),
        }
    return results


def git_sha() -> str:
    """Read HEAD from .git directly: the ml-service image ships no git binary."""
    head = ROOT / ".git" / "HEAD"
    if not head.exists():
        return "unknown"
    reference = head.read_text(encoding="utf-8").strip()
    if reference.startswith("ref: "):
        target = ROOT / ".git" / reference[5:]
        return target.read_text(encoding="utf-8").strip() if target.exists() else "unknown"
    return reference


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="assistments_2012")
    parser.add_argument("--learners", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    print(f"calibrating against data/processed/{args.target}.parquet")
    real = load_real(args.target)

    defaults = Params()
    outcome = fit_outcome(real)
    response_time = fit_response_time(real, outcome["theta"], outcome["difficulty"],
                                      defaults.fatigue_growth, defaults.trait_speed_sd,
                                      defaults.rt_beta_stem)
    dynamics = fit_learning_and_fatigue(real, defaults.fatigue_growth)

    bank_b = pd.read_csv(ROOT / "artifacts" / "datasets" / "item-parameters-v1.csv")["b"].to_numpy()
    ability = fit_ability(real, outcome["guess"], outcome["slip"], bank_b, args.seed)

    session_lengths = real.groupby(["learner_id", "session"]).size().to_numpy()
    learner_totals = real.groupby("learner_id").size().to_numpy()

    params = Params(
        ability_mean=ability["ability_mean"],
        ability_sd=ability["ability_sd"],
        # The lower asymptote of a 4-option multiple-choice item is 1/4 by
        # construction. The archival fit below is reported but not used: both
        # ASSISTments releases are dominated by algebra and fill-in items, where
        # chance really is ~0, and adopting that would give the simulated bank a
        # guessing rate its own item format rules out.
        guess=1.0 / N_OPTIONS,
        slip=outcome["slip"],
        learning_rate_log_mean=dynamics["learning_rate_log_mean"],
        learning_rate_log_sd=dynamics["learning_rate_log_sd"],
        fatigue_accuracy_beta=dynamics["fatigue_accuracy_beta"],
        rt_intercept=response_time["rt_intercept"],
        rt_beta_distance=response_time["rt_beta_distance"],
        rt_beta_fatigue=response_time["rt_beta_fatigue"],
        rt_sigma=response_time["rt_sigma"],
        session_length_quantiles=tuple(empirical_quantiles(session_lengths)),
        learner_total_quantiles=tuple(empirical_quantiles(learner_totals)),
        calibrated_against=args.target,
        fitted={
            "seed": args.seed,
            "git_sha": git_sha(),
            "rows_used": int(len(real)),
            "learners_used": int(real["learner_id"].nunique()),
            "link_scale": outcome["link_scale"],
            "archival_guess_asymptote": round(outcome["guess"], 4),
            "guess_source": f"structural: 1/{N_OPTIONS} for a four-option item, not the archival asymptote",
            "ability_grid_ks": round(ability["ability_grid_ks"], 4),
            "rt_residual_sd": round(response_time["rt_residual_sd"], 4),
            **{key: value for key, value in dynamics.items() if key.startswith(("learning_slope",
                                                                                "within_session",
                                                                                "residual_decay"))},
            "not_identifiable_from_archival": [
                "fatigue_growth (fixed at 0.045/item — nothing observable separates the growth rate "
                "from its accuracy coefficient)",
                "trait_* (no archival source records motor telemetry or device)",
                "probe_noise_sd / probe_bias (no archival source contains self-report probes)",
                "rt_beta_stem (no archival source publishes item text)",
            ],
        },
    )

    print(f"  ability ~ N({params.ability_mean:.2f}, {params.ability_sd:.2f}), "
          f"guess {params.guess:.3f}, slip {params.slip:.3f}")
    print(f"  log RT = {params.rt_intercept:.2f} + {params.rt_beta_distance:.3f}·|θ-b| "
          f"+ {params.rt_beta_fatigue:.3f}·fatigue + N(0, {params.rt_sigma:.2f})")
    print(f"  learning rate ~ logN({params.learning_rate_log_mean:.2f}, "
          f"{params.learning_rate_log_sd:.2f}), fatigue β {params.fatigue_accuracy_beta:.3f}")

    real_accuracy = real.groupby("learner_id")["correct"].mean().to_numpy()
    params, history = refine_ability(params, real_accuracy, args.seed)
    params = replace(params, fitted=params.fitted | {"ability_refinement": history})
    print(f"  ability refined to N({params.ability_mean:.2f}, {params.ability_sd:.2f}) "
          f"over {len(history)} rounds (best accuracy KS {min(h['ks_d'] for h in history):.4f})")

    print(f"generating {args.learners} learners from the fitted model")
    simulator = Simulator(params, seed=args.seed)
    simulated = pd.DataFrame(simulator.run(simulator.learners(args.learners)))

    results = compare(real, simulated)
    report = {
        "target": args.target,
        "seed": args.seed,
        "ks_limit": KS_LIMIT,
        "sim_learners": args.learners,
        "sim_rows": int(len(simulated)),
        "checks": results,
        "documented_mismatches": DOCUMENTED_MISMATCHES,
    }
    KS_PATH.parent.mkdir(parents=True, exist_ok=True)
    KS_PATH.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {KS_PATH.relative_to(ROOT)}")

    failures = []
    for name, result in results.items():
        statistic = result.get("ks_d")
        verdict = "ok" if statistic is not None and statistic <= KS_LIMIT else "FAIL"
        if verdict == "FAIL" and name in DOCUMENTED_MISMATCHES:
            verdict = "documented mismatch"
        print(f"  KS {name:22s} D = {statistic}  {verdict}")
        if verdict == "FAIL":
            failures.append(f"{name} (D = {statistic})")

    params_out = params.to_dict() | {"calibration_ks": {n: r.get("ks_d") for n, r in results.items()}}
    PARAMS_PATH.write_text(json.dumps(params_out, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {PARAMS_PATH.relative_to(ROOT)}")

    if failures:
        raise SystemExit(
            "calibration FAILED the distribution match on: " + ", ".join(failures)
            + f"\nKS D must be <= {KS_LIMIT}. Fix the generative model, or register the mismatch in "
              "DOCUMENTED_MISMATCHES with the evidence and state it as a limitation in docs/simulator.md."
        )


if __name__ == "__main__":
    main()
