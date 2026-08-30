"""Analyze pilot session data and compare against synthetic baseline."""
from __future__ import annotations

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy import stats


def load_csv(path: Path) -> dict:
    """Load per-learner outcome CSV, grouped by policy."""
    data = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            policy = row["policy"]
            if policy not in data:
                data[policy] = {"time": [], "mastery": []}
            data[policy]["time"].append(float(row["time_seconds"]))
            data[policy]["mastery"].append(float(row["final_mastery"]))
    return {k: {m: np.array(v) for m, v in vals.items()} for k, vals in data.items()}


def unpaired_cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    n1, n2 = len(a), len(b)
    pooled_std = np.sqrt(
        ((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2)
    )
    return float((np.mean(a) - np.mean(b)) / (pooled_std + 1e-12))


def bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 10000, seed: int = 42) -> tuple:
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx_a = rng.integers(0, len(a), len(a))
        idx_b = rng.integers(0, len(b), len(b))
        diffs.append(np.mean(a[idx_a]) - np.mean(b[idx_b]))
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main():
    parser = argparse.ArgumentParser(description="Analyze pilot data against synthetic baseline")
    parser.add_argument("--pilot-csv", type=Path, required=True, help="Path to pilot outcomes CSV")
    parser.add_argument(
        "--synthetic-baseline",
        type=Path,
        default=Path("artifacts/evaluation/per-learner-outcomes.csv"),
        help="Path to synthetic per-learner outcomes CSV",
    )
    parser.add_argument("--output", type=Path, default=Path("artifacts/evaluation/pilot-analysis.json"))
    args = parser.parse_args()

    print(f"Loading pilot data from {args.pilot_csv}...")
    pilot = load_csv(args.pilot_csv)

    print(f"Loading synthetic baseline from {args.synthetic_baseline}...")
    synthetic = load_csv(args.synthetic_baseline)

    results = {"comparisons": []}

    # Compare each pilot policy against its synthetic counterpart
    for policy in pilot:
        if policy not in synthetic:
            print(f"  Warning: policy '{policy}' not found in synthetic baseline, skipping.")
            continue

        for metric in ["time", "mastery"]:
            pilot_vals = pilot[policy][metric]
            synth_vals = synthetic[policy][metric]

            d = unpaired_cohens_d(pilot_vals, synth_vals)
            ci_lo, ci_hi = bootstrap_ci(pilot_vals, synth_vals)

            # Mann-Whitney U test (unpaired)
            stat, p_val = stats.mannwhitneyu(pilot_vals, synth_vals, alternative="two-sided")

            comparison = {
                "policy": policy,
                "metric": metric,
                "pilot_n": len(pilot_vals),
                "synthetic_n": len(synth_vals),
                "pilot_mean": float(np.mean(pilot_vals)),
                "synthetic_mean": float(np.mean(synth_vals)),
                "unpaired_cohens_d": round(d, 4),
                "bootstrap_ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
                "mann_whitney_p": round(float(p_val), 6),
            }
            results["comparisons"].append(comparison)
            print(f"  {policy}/{metric}: d={d:.3f}, CI=[{ci_lo:.2f}, {ci_hi:.2f}], p={p_val:.4f}")

    # Save results
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved analysis to {args.output}")


if __name__ == "__main__":
    main()
