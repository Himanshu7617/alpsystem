"""Phase 8 — closed-loop policy evaluation. `make eval-policies`.

    python -m app.research.evaluate_policies --seed 20260821

Ten arms, five simulator variants, three seeds, one action space. Every arm sees
the same cohort of learners under the same generative model and differs only in
the state it conditions on — which is the whole of RQ3.

What this file is *not*: the pre-Phase-0 version of it. That one scored the
"ML" arm with the simulator's own response function (Defect 1), overwrote its
per-policy results inside the sweep loop, and reported near-zero mastery in
every arm because the termination criterion required all thirty-six concepts.
All three are fixed here and each has a test.

Outputs
-------
* ``artifacts/evaluation/policy-results-v{0..4}.json`` — per-arm summaries,
  paired tests, effect sizes and the run's protocol.
* ``artifacts/evaluation/policy-per-learner-v{0..4}.csv`` — one row per learner
  per arm per seed. Every number in a Phase 8 figure comes from these.
* ``artifacts/evaluation/decision-log-v0.parquet`` — every decision of the first
  seed on V0 with its **propensity**. Study 3 cannot be added to a log that did
  not record one.
* ``artifacts/evaluation/decision-divergence-v0.json`` — how often ``model_L4``
  would have chosen differently from ``model_L0`` on ``model_L0``'s own
  trajectory. If the two rarely diverge, no outcome difference is possible, and
  that is itself the finding.
* ``artifacts/evaluation/decision-explanations.json`` — sampled decision traces
  with their rule citations, served by ``GET /v1/decisions/{id}/explanation``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import numpy as np
import pandas as pd

from app.policy import policies as arms_module
from app.research import runprofile
from app.research.closed_loop import BUDGET, ClosedLoop, MASTERY_TARGET, select_curriculum
from app.model import validation_gate
from app.research.simulator import VARIANTS

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "artifacts" / "evaluation"
#: One file per finished (arm, variant, seed) cell. A run that is stopped —
#: because the machine is hot, or because the user hit ctrl-c — resumes from
#: here instead of repaying for the cells it already finished. The key covers
#: the job *and* the source of every module that can change a cell's result, so
#: editing a policy invalidates the cache rather than silently reusing it.
CELL_CACHE = EVAL_DIR / "cells"
RUN_LOG = EVAL_DIR / "run-log.jsonl"

#: Any Cohen's *d* above this halts the phase. The prior work's *d* = 26 was a
#: bug signature; the literature's reference points are +1.25 % AUC and g ≈ 0.3.
EFFECT_SIZE_HALT = 3.0

#: The one documented exemption, and it is arithmetic rather than a result.
#: ``random`` draws its intervention uniformly, so it suggests a break on a
#: quarter of all items — 45.9 breaks per learner against 4.6 under ε-greedy
#: exploration — and a break costs 300 s. Its clock is therefore roughly double
#: every other arm's *by construction*, which shows up as |d| ≈ 3 against each
#: of them on time-to-mastery and on no other outcome. The pair is still
#: measured, still written to the report under ``effect_sizes_explained``, and
#: still visible in `f08-02`; it just does not halt the phase, because the
#: halt exists to catch an implausible *learning* effect.
HALT_EXEMPT = [{"arm": "random", "metric": "time_to_mastery_seconds",
                "reason": "uniform action draw spends ~25 % of items on a 300 s break "
                          "suggestion; the time gap is the action space's arithmetic, "
                          "not an instructional effect"}]


def halt_exemption(test: dict) -> dict | None:
    """The documented reason this large effect is not a defect, if there is one."""
    for entry in HALT_EXEMPT:
        if test["metric"] == entry["metric"] and entry["arm"] in (test["policy_a"],
                                                                 test["policy_b"]):
            return entry
    return None

#: How many learners' decisions are written to the propensity log. Every arm
#: logs the same learners; 200 × 200 items × ten arms is a 400,000-row parquet,
#: which is plenty for Study 3 and does not carry two million dicts through a
#: process boundary.
LOG_LEARNERS = 200

#: How many decisions get their full explanation kept. Every decision records
#: its action and propensity; only a sample keeps the prose-and-citation form,
#: because 200,000 explanation dicts is a 400 MB file nobody reads.
EXPLANATION_SAMPLE = 200

PRIMARY = "items_to_mastery"
#: Registered as a co-primary in preregistration §9.2: the calibrated learning
#: rate censors items-to-mastery for most learners inside a 200-item budget, and
#: knowledge gain is the uncensored quantity that threshold crosses.
CO_PRIMARY = "knowledge_gain"


# ------------------------------------------------------------------ one cell

def observable_thresholds(source: str = "sim-V0", refresh: bool = False) -> dict:
    """Rule thresholds, read off the offline corpus rather than chosen.

    ``hint_on_slow_decision`` and ``break_on_within_session_decay`` both compare
    an observable against a threshold. Both thresholds are the 75th percentile
    of that observable in the logged corpus, so "unusually slow" means unusual
    against real logged behaviour rather than against a number someone liked.
    """
    path = ROOT / "artifacts" / "models" / "policy-thresholds.json"
    if path.exists() and not refresh:
        cached = json.loads(path.read_text(encoding="utf-8"))
        if cached.get("source") == source:
            return cached["thresholds"]
    frame = pd.read_parquet(
        ROOT / "data" / "processed" / f"features-{source}.parquet",
        columns=["decision_latency", "matched_difficulty_speed_slope"])
    thresholds = {name: float(frame[name].quantile(0.75)) for name in frame.columns}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"source": source, "quantile": 0.75,
                                "thresholds": thresholds}, indent=2) + "\n", encoding="utf-8")
    return thresholds


def run_cell(arm: str, variant: str, seed: int, learners: int, budget: int,
             epsilon: float, thresholds: dict, log_decisions: int = 0,
             shadow: str | None = None, explanations: int = 0,
             probe_noise: float | None = None) -> dict:
    """One (arm, variant, seed) rollout. Returns per-learner results and logs."""
    import torch

    torch.set_num_threads(1)
    gate = validation_gate.load()
    parameters = None
    if probe_noise is not None:
        from dataclasses import replace

        from app.research.simulator import Params

        parameters = replace(Params.load(), probe_noise_sd=probe_noise)
    environment = ClosedLoop(variant, seed, learners, budget=budget, params=parameters)
    policy = arms_module.build(arm, learners, seed, budget=budget,
                               thresholds=thresholds, gate=gate)
    policy.epsilon = 0.0 if arm in ("random", "fixed_order") else epsilon
    shadow_policy = (arms_module.build(shadow, learners, seed, budget=budget,
                                       thresholds=thresholds, gate=gate)
                     if shadow else None)
    if shadow_policy is not None:
        shadow_policy.epsilon = 0.0

    rows: list[dict | None] = [None] * learners
    decisions: list[dict] = []
    divergence: list[dict] = []
    traces: list[dict] = []
    started = time.perf_counter()

    for step in range(budget):
        if environment.done:
            break
        views = environment.views(policy.rung, rows)
        actions = policy.act(views)
        if shadow_policy is not None:
            shadow_views = environment.views(shadow_policy.rung, rows)
            shadow_actions = [None if view is None else shadow_policy.greedy(view)[0]
                              for view in shadow_views]
        outcomes = environment.step(actions)
        policy.observe(outcomes)
        if shadow_policy is not None:
            shadow_policy.observe(outcomes)
        rows = getattr(policy, "last_rows", outcomes)

        for index, (action, row) in enumerate(zip(actions, outcomes)):
            if action is None or row is None:
                continue
            if index < log_decisions:
                decisions.append({
                    "policy": arm, "variant": variant, "seed": seed,
                    "learner_id": row["learner_id"], "step": row["step"],
                    "question_number": int(row["question_number"]),
                    "session_index": int(row["session_index"]),
                    "concept_id": row["concept_id"], "item_id": row["item_id"],
                    "requested_difficulty": action["difficulty"],
                    "served_difficulty": int(row["difficulty_score"]),
                    "item_b": float(row["item_b"]),
                    "concept_move": action["concept_move"],
                    "intervention": action["intervention"],
                    "propensity": float(action["propensity"]),
                    "explored": bool(action["explored"]),
                    "correct": int(row["correct"]),
                    "response_time_ms": float(row["response_time_ms"]),
                    "knowledge_estimate": action["explanation"].get("states", {}).get("knowledge"),
                    "knowledge_se": action["explanation"].get(
                        "state_standard_error", {}).get("knowledge"),
                })
            if (shadow_policy is not None and shadow_actions[index] is not None
                    and index < LOG_LEARNERS):
                other = shadow_actions[index]
                divergence.append({
                    "step": row["step"], "learner_id": row["learner_id"],
                    "difficulty_a": action["difficulty"], "difficulty_b": other["difficulty"],
                    "intervention_a": action["intervention"],
                    "intervention_b": other.get("intervention", "no_intervention"),
                    "move_a": action["concept_move"],
                    "move_b": other.get("concept_move", "same_concept"),
                    "knowledge_estimate": action["explanation"].get("states", {}).get("knowledge"),
                })
            if len(traces) < explanations and action["explanation"].get("rules_fired"):
                traces.append({
                    "id": f"{variant.lower()}-{arm}-{row['learner_id']}-step{row['step']:03d}",
                    "policy": arm, "variant": variant, "seed": seed,
                    "learner_id": row["learner_id"], "step": row["step"],
                    "concept_id": row["concept_id"],
                    "action": {key: action[key] for key in
                               ("difficulty", "concept_move", "intervention", "propensity")},
                    "served_difficulty": int(row["difficulty_score"]),
                    "explanation": action["explanation"],
                })

    return {
        "arm": arm, "variant": variant, "seed": seed, "probe_noise": probe_noise,
        "results": [{**record, "policy": arm} for record in environment.results()],
        "decisions": decisions, "divergence": divergence, "traces": traces,
        "seconds": round(time.perf_counter() - started, 1),
        "curriculum": environment.curriculum,
        "screened": environment.screened,
        "mastery_curve": environment.mastery_curve,
        "difficulty_curve": environment.difficulty_curve,
        "active_curve": environment.active_curve,
        "rules_excluded": [dict(entry) for entry in getattr(policy, "rules", None).excluded]
        if getattr(policy, "rules", None) is not None else [],
    }


def _log_progress(entry: dict) -> None:
    """One JSON line per finished unit of work. `make status` reads this."""
    RUN_LOG.parent.mkdir(parents=True, exist_ok=True)
    with RUN_LOG.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"script": "evaluate_policies", **entry}, default=str) + "\n")


def _source_fingerprint() -> str:
    """SHA-256 over every module whose edit changes a cell's result."""
    import hashlib

    base = Path(__file__).resolve().parents[1]
    files = sorted(list((base / "policy").glob("*.py"))
                   + [base / "research" / "closed_loop.py",
                      base / "research" / "simulator.py",
                      base / "research" / "evaluate_policies.py",
                      base / "model" / "validation_gate.py"])
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.read_bytes())
    return digest.hexdigest()[:16]


def _cache_path(job: dict, fingerprint: str) -> Path:
    import hashlib

    key = json.dumps(job, sort_keys=True, default=str) + fingerprint
    stem = f"{job['variant'].lower()}-{job['seed']}-{job['arm']}"
    return CELL_CACHE / f"{stem}-{hashlib.sha256(key.encode()).hexdigest()[:12]}.json.gz"


def _cell(job: dict) -> dict:
    """One cell, resumed from its checkpoint when one exists."""
    import gzip

    fingerprint = job.pop("_fingerprint", None)
    path = _cache_path(job, fingerprint) if fingerprint else None
    if path is not None and path.exists():
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                cell = json.load(handle)
            cell["cached"] = True
            return cell
        except (OSError, ValueError):
            # A checkpoint written by a process that was killed mid-write is
            # worse than no checkpoint. Drop it and recompute the cell.
            path.unlink(missing_ok=True)
    cell = run_cell(**job)
    cell["cached"] = False
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        with gzip.open(temporary, "wt", encoding="utf-8") as handle:
            json.dump(cell, handle, default=float)
        temporary.replace(path)   # atomic: a reader never sees a partial file
    return cell


# ----------------------------------------------------------------- statistics

def bootstrap_ci(values: np.ndarray, seed: int, draws: int = 2000) -> list[float]:
    if len(values) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(draws, len(values)), replace=True).mean(axis=1)
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def paired_tests(frame: pd.DataFrame, metric: str, seed: int) -> list[dict]:
    """Every arm against every other, on the learners they share.

    The cohort is identical across arms for a given (variant, seed) — the
    simulator draws it before any policy acts — so the comparison is paired,
    which is both the right test and the reason the per-learner CSV keeps
    ``learner_index``.
    """
    from scipy import stats

    keys = sorted(frame["policy"].unique())
    comparisons = list(itertools.combinations(keys, 2))
    output = []
    for first, second in comparisons:
        left = frame[frame["policy"] == first].set_index(["seed", "learner_index"])[metric]
        right = frame[frame["policy"] == second].set_index(["seed", "learner_index"])[metric]
        paired = pd.concat([left.rename("a"), right.rename("b")], axis=1, join="inner").dropna()
        difference = (paired["a"] - paired["b"]).to_numpy(dtype=float)
        deviation = float(np.std(difference, ddof=1)) if len(difference) > 1 else 0.0
        cohens_d = float(np.mean(difference) / deviation) if deviation else 0.0
        if len(difference) > 1 and np.any(difference != 0):
            statistic, p_value = stats.wilcoxon(paired["a"], paired["b"])
        else:
            statistic, p_value = float("nan"), 1.0
        output.append({
            "policy_a": first, "policy_b": second, "metric": metric, "n": int(len(difference)),
            "mean_difference": float(np.mean(difference)) if len(difference) else float("nan"),
            "difference_95_ci": bootstrap_ci(difference, seed),
            "test": "wilcoxon_signed_rank", "statistic": float(statistic),
            "p_value": float(p_value),
            "p_value_bonferroni": float(min(1.0, p_value * len(comparisons))),
            "cohens_d": cohens_d,
            "significant_alpha_005": bool(min(1.0, p_value * len(comparisons)) < 0.05),
        })
    return output


def summarise(frame: pd.DataFrame, seed: int) -> dict:
    summary = {}
    for arm, rows in frame.groupby("policy", observed=True):
        summary[arm] = {
            "learners": int(len(rows)),
            "mastery_rate": round(float(rows["mastered"].mean()), 6),
            "censoring_rate": round(float(1 - rows["mastered"].mean()), 6),
            "mean_knowledge_gain": round(float(rows["knowledge_gain"].mean()), 6),
            "knowledge_gain_95_ci": [round(value, 6) for value in
                                     bootstrap_ci(rows["knowledge_gain"].to_numpy(dtype=float), seed)],
            "mean_knowledge_gain_per_item": round(float(rows["knowledge_gain_per_item"].mean()), 8),
            "mean_initial_mastery_fraction": round(float(rows["initial_mastery_fraction"].mean()), 6),
            "mean_items_to_mastery": round(float(rows[PRIMARY].mean()), 4),
            "median_items_to_mastery": float(rows[PRIMARY].median()),
            "items_to_mastery_95_ci": [round(value, 4) for value in
                                       bootstrap_ci(rows[PRIMARY].to_numpy(dtype=float), seed)],
            "mean_time_to_mastery_seconds": round(float(rows["time_to_mastery_seconds"].mean()), 2),
            "median_time_to_mastery_seconds": float(rows["time_to_mastery_seconds"].median()),
            "mean_final_mastery_fraction": round(float(rows["final_mastery_fraction"].mean()), 6),
            "mean_success_probability": round(float(rows["mean_success_probability"].mean()), 6),
            "mean_difficulty": round(float(rows["mean_difficulty"].mean()), 4),
            "accuracy": round(float(rows["accuracy"].mean()), 6),
            "mean_sessions": round(float(rows["sessions"].mean()), 4),
            "disengagement_rate": round(float(rows["disengaged"].mean()), 6),
            "interventions_per_learner": {
                name: round(float(rows[f"n_{name}"].mean()), 4)
                for name in ("hint", "worked_example", "break_suggestion", "no_intervention")},
        }
    return summary


# ---------------------------------------------------------------------- main

def main() -> None:
    import sys

    limits = runprofile.load(runprofile.peek(sys.argv))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default=limits["profile"],
                        choices=sorted(runprofile.PROFILES),
                        help="workload size; ALP_RUN_PROFILE sets it, ALP_MAX_* override it")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--seeds", type=int, default=limits["seeds"],
                        help="how many seeds per cell")
    parser.add_argument("--learners", type=int, default=limits["learners"])
    parser.add_argument("--budget", type=int, default=min(limits["budget"], BUDGET))
    parser.add_argument("--variants", default=limits["variants"])
    parser.add_argument("--arms", default=",".join(arms_module.ARM_KEYS))
    parser.add_argument("--epsilon", type=float, default=arms_module.EPSILON)
    parser.add_argument("--workers", type=int, default=limits["workers"])
    parser.add_argument("--probe-noise-levels", default="0.25,0.55,1.0,2.0",
                        help="self-report noise levels for the f08-14 sensitivity sweep")
    parser.add_argument("--probe-noise-learners", type=int,
                        default=min(300, limits["learners"]))
    parser.add_argument("--no-resume", action="store_true",
                        help="recompute every cell instead of reusing its checkpoint")
    parser.add_argument("--quick", action="store_true",
                        help="print a cohort summary and write nothing (implies --profile smoke)")
    args = parser.parse_args()

    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    arms = [name.strip() for name in args.arms.split(",") if name.strip()]
    seeds = [args.seed + offset for offset in range(args.seeds)]
    if args.quick:
        smoke = runprofile.load("smoke")
        variants, seeds, args.learners = ["V0"], [args.seed], smoke["learners"]
        args.budget = min(args.budget, smoke["budget"])
        args.workers = min(args.workers, smoke["workers"])
    print(runprofile.banner({**limits, "profile": args.profile, "workers": args.workers,
                             "learners": args.learners, "budget": args.budget,
                             "seeds": len(seeds), "variants": ",".join(variants)},
                            f"seed={args.seed} arms={len(arms)}"))

    gate = validation_gate.load()
    print(f"gate: admitted {sorted(gate.admitted) or 'none'}; "
          f"rejected {sorted(gate.report.get('rejected', []))}")
    thresholds = observable_thresholds()
    curriculum = select_curriculum(ClosedLoop("V0", args.seed, 1, budget=1).simulator.items)
    print(f"curriculum ({len(curriculum)}): {', '.join(curriculum)}")

    fingerprint = None if (args.quick or args.no_resume) else _source_fingerprint()
    jobs = []
    for variant, seed, arm in itertools.product(variants, seeds, arms):
        first = seed == seeds[0] and variant == "V0"
        jobs.append({"arm": arm, "variant": variant, "seed": seed, "learners": args.learners,
                     "budget": args.budget, "epsilon": args.epsilon, "thresholds": thresholds,
                     "log_decisions": LOG_LEARNERS if first else 0,
                     "shadow": "model_L4" if (first and arm == "model_L0") else None,
                     "explanations": EXPLANATION_SAMPLE if (first and arm == "model_L4") else 0,
                     "_fingerprint": fingerprint})

    print(f"{len(jobs)} cells: {len(arms)} arms × {len(variants)} variants × {len(seeds)} seeds "
          f"× {args.learners} learners, budget {args.budget}, {args.workers} workers")
    if fingerprint:
        done_already = sum(1 for job in jobs
                           if _cache_path({k: v for k, v in job.items() if k != "_fingerprint"},
                                          fingerprint).exists())
        print(f"resume: {done_already}/{len(jobs)} cells already checkpointed in "
              f"{CELL_CACHE.relative_to(ROOT)}")
    started = time.perf_counter()
    cells = []

    def record(done: int, cell: dict) -> None:
        note = " (cached)" if cell.get("cached") else ""
        print(f"  [{done}/{len(jobs)}] {cell['variant']} seed {cell['seed']} "
              f"{cell['arm']} — {cell['seconds']}s{note} "
              f"[{round(time.perf_counter() - started, 1)}s elapsed]", flush=True)
        _log_progress({"stage": "cell", "profile": args.profile, "done": done,
                       "total": len(jobs), "variant": cell["variant"], "arm": cell["arm"],
                       "seed": cell["seed"], "learners": args.learners,
                       "budget": args.budget, "workers": args.workers,
                       "cell_seconds": cell["seconds"], "cached": cell.get("cached", False),
                       "elapsed_seconds": round(time.perf_counter() - started, 1)})

    if args.workers > 1 and len(jobs) > 1:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.workers) as pool:
            for done, cell in enumerate(pool.imap_unordered(_cell, jobs), start=1):
                cells.append(cell)
                record(done, cell)
    else:
        for done, job in enumerate(jobs, start=1):
            cell = _cell(job)
            cells.append(cell)
            record(done, cell)
    print(f"closed loop finished in {round(time.perf_counter() - started, 1)}s")

    frame = pd.DataFrame([record for cell in cells for record in cell["results"]])
    if args.quick:
        print(frame.groupby("policy")[["mastered", PRIMARY, "mean_success_probability"]].mean())
        raise SystemExit(0)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    excluded = next((cell["rules_excluded"] for cell in cells if cell["rules_excluded"]), [])
    halts = []

    for variant in variants:
        subset = frame[frame["variant"] == variant]
        tests = paired_tests(subset, PRIMARY, args.seed)
        time_tests = paired_tests(subset, "time_to_mastery_seconds", args.seed)
        gain_tests = paired_tests(subset, CO_PRIMARY, args.seed)
        large = [test for test in tests + time_tests + gain_tests
                 if abs(test["cohens_d"]) > EFFECT_SIZE_HALT]
        explained = [{**test, "exemption": halt_exemption(test)}
                     for test in large if halt_exemption(test)]
        over = [test for test in large if not halt_exemption(test)]
        halts.extend(over)
        report = {
            "phase": 8,
            "evaluation_type": "simulated_closed_loop",
            # The profile is recorded because a `dev` run writes to the same
            # paths a `full` run does. A reader must be able to tell which one
            # produced the file they are quoting.
            "run_profile": args.profile,
            "variant": variant,
            "variant_description": VARIANTS[variant],
            "seed": args.seed, "seeds": seeds, "learners_per_seed": args.learners,
            "budget_items": args.budget,
            "epsilon": args.epsilon,
            "action_space": {
                "difficulty": list(arms_module.rules_module.DIFFICULTIES),
                "concept_move": list(arms_module.rules_module.CONCEPT_MOVES),
                "intervention": list(arms_module.rules_module.INTERVENTIONS),
                "size": arms_module.ACTION_SPACE,
            },
            "curriculum": curriculum,
            "mastery_target": MASTERY_TARGET,
            "learners_screened_per_cell": next((cell["screened"] for cell in cells
                                                if cell["variant"] == variant), None),
            "gate": {"admitted": sorted(gate.admitted),
                     "rejected": sorted(gate.report.get("rejected", []))},
            "rules_excluded": excluded,
            "policies": summarise(subset, args.seed),
            "paired_tests_items_to_mastery": tests,
            "paired_tests_time_to_mastery": time_tests,
            "paired_tests_knowledge_gain": gain_tests,
            "effect_size_halt_threshold": EFFECT_SIZE_HALT,
            "effect_sizes_over_threshold": over,
            "effect_sizes_explained": explained,
            "protocol": {
                "cohort": "identical across arms for a given (variant, seed); comparisons paired",
                "censoring": "a learner who does not master by the budget contributes the budget "
                             "and mastered=false; the co-primary knowledge gain is uncensored",
                "inclusion": "learners with the whole curriculum already mastered at their first "
                             "item are screened out (preregistration §9.2)",
                "time": "on-task seconds only — between-session gaps are not learner time",
                "simulated": "no human participants; see docs/preregistration.md §9",
            },
        }
        path = EVAL_DIR / f"policy-results-{variant.lower()}.json"
        path.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
        print(f"wrote: {path.relative_to(ROOT)} ({len(report['policies'])} arms)")

        per_learner = EVAL_DIR / f"policy-per-learner-{variant.lower()}.csv"
        subset.to_csv(per_learner, index=False)
        print(f"wrote: {per_learner.relative_to(ROOT)} ({len(subset):,} rows)")

    curves: dict[str, dict] = {}
    for variant in variants:
        for arm in arms:
            picked = [cell for cell in cells
                      if cell["variant"] == variant and cell["arm"] == arm]
            if not picked:
                continue
            width = max(len(cell["mastery_curve"]) for cell in picked)

            def averaged(key: str) -> list[float]:
                stacked = np.full((len(picked), width), np.nan)
                for index, cell in enumerate(picked):
                    values = cell[key]
                    stacked[index, :len(values)] = values
                    if key == "mastery_curve" and len(values) < width:
                        stacked[index, len(values):] = values[-1] if values else np.nan
                return [round(float(value), 6) for value in np.nanmean(stacked, axis=0)]

            curves.setdefault(variant, {})[arm] = {
                "mastery": averaged("mastery_curve"),
                "difficulty": averaged("difficulty_curve"),
                "active_learners": averaged("active_curve"),
            }
    (EVAL_DIR / "policy-curves.json").write_text(
        json.dumps({"seeds": seeds, "budget": args.budget, "curves": curves},
                   indent=2) + "\n", encoding="utf-8")
    print(f"wrote: artifacts/evaluation/policy-curves.json "
          f"({sum(len(entry) for entry in curves.values())} arm-variant curves)")

    levels = [float(value) for value in args.probe_noise_levels.split(",") if value.strip()]
    ladder_arms = [arm for arm in arms if arm.startswith("model_")]
    if levels and ladder_arms:
        # f08-14. The confidence head is the only consumer of a self-report
        # probe and the gate rejected it, so this sweep is a pre-registered
        # check that the closed loop is *insensitive* to probe noise — a null
        # that has to be measured rather than argued.
        sweep_jobs = [{"arm": arm, "variant": "V0", "seed": seeds[0],
                       "learners": min(args.learners, args.probe_noise_learners),
                       "budget": args.budget, "epsilon": args.epsilon,
                       "thresholds": thresholds, "probe_noise": level}
                      for level in levels for arm in (ladder_arms[0], ladder_arms[-1])]
        print(f"probe-noise sweep: {len(sweep_jobs)} cells")
        if args.workers > 1:
            import multiprocessing as mp

            with mp.get_context("spawn").Pool(args.workers) as pool:
                sweep_cells = list(pool.imap_unordered(_cell, sweep_jobs))
        else:
            sweep_cells = [_cell(job) for job in sweep_jobs]
        sweep = pd.DataFrame([{**record, "probe_noise": cell["probe_noise"]}
                              for cell in sweep_cells for record in cell["results"]])
        summary = (sweep.groupby(["policy", "probe_noise"], observed=True)
                   [[PRIMARY, CO_PRIMARY, "mastered", "mean_success_probability"]]
                   .mean().reset_index())
        (EVAL_DIR / "probe-noise-sensitivity.json").write_text(
            json.dumps({"levels": levels, "calibrated_level": 0.55,
                        "learners": min(args.learners, args.probe_noise_learners),
                        "note": "the gate rejected the confidence head, so no policy reads a "
                                "probe-derived state; a flat curve is the expected result",
                        "rows": summary.to_dict(orient="records")},
                       indent=2, default=float) + "\n", encoding="utf-8")
        print("wrote: artifacts/evaluation/probe-noise-sensitivity.json")

    decisions = [record for cell in cells for record in cell["decisions"]]
    if decisions:
        path = EVAL_DIR / "decision-log-v0.parquet"
        pd.DataFrame(decisions).to_parquet(path, index=False)
        print(f"wrote: {path.relative_to(ROOT)} ({len(decisions):,} decisions with propensities)")

    divergence = [record for cell in cells for record in cell["divergence"]]
    if divergence:
        table = pd.DataFrame(divergence)
        table["same_difficulty"] = table["difficulty_a"] == table["difficulty_b"]
        table["same_intervention"] = table["intervention_a"] == table["intervention_b"]
        table["same_move"] = table["move_a"] == table["move_b"]
        table["identical"] = table[["same_difficulty", "same_intervention", "same_move"]].all(axis=1)
        report = {
            "shadow_of": "model_L0", "shadow_policy": "model_L4", "variant": "V0",
            "decisions": int(len(table)),
            "identical_rate": float(table["identical"].mean()),
            "same_difficulty_rate": float(table["same_difficulty"].mean()),
            "same_intervention_rate": float(table["same_intervention"].mean()),
            "same_concept_move_rate": float(table["same_move"].mean()),
            "mean_absolute_difficulty_gap": float(
                (table["difficulty_a"] - table["difficulty_b"]).abs().mean()),
            "note": "measured on model_L0's own trajectory: what model_L4 would have chosen at "
                    "the same moment. Rarely diverging bounds any possible outcome difference.",
        }
        (EVAL_DIR / "decision-divergence-v0.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
        table.to_parquet(EVAL_DIR / "decision-divergence-v0.parquet", index=False)
        print(f"wrote: artifacts/evaluation/decision-divergence-v0.json "
              f"(identical {report['identical_rate']:.1%})")

    traces = [record for cell in cells for record in cell["traces"]]
    if traces:
        path = EVAL_DIR / "decision-explanations.json"
        path.write_text(json.dumps({"decisions": traces}, indent=2, default=float) + "\n",
                        encoding="utf-8")
        print(f"wrote: {path.relative_to(ROOT)} ({len(traces)} traces)")

    if halts:
        worst = max(halts, key=lambda entry: abs(entry["cohens_d"]))
        raise SystemExit(
            f"HALT: Cohen's d = {worst['cohens_d']:.2f} for {worst['policy_a']} vs "
            f"{worst['policy_b']} on {worst['metric']}, above the {EFFECT_SIZE_HALT} threshold. "
            "BUILD.md Phase 8 acceptance: investigate before reporting.")


if __name__ == "__main__":
    main()
