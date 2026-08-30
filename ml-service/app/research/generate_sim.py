"""Generate the Simulator v2 datasets — the calibrated baseline and its four
deliberate mis-specifications.

    python -m app.research.generate_sim [--learners 1000] [--variants V0,V1,V2,V3,V4]

Writes `artifacts/datasets/sim-v2-<variant>.parquet` and one manifest recording
the seed, the parameter hash, the git SHA and the row count per variant.

Every closed-loop result in Phase 8 and Phase 9 is run against **all five**. A
policy ranking that flips between variants is an artefact of one generative
model, not a finding (BUILD.md Phase 4 step 5).
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from app.research.calibrate_simulator import KS_LIMIT, compare, git_sha, load_real
from app.research.simulator import PARAMS_PATH, VARIANTS, Params, Simulator

ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = ROOT / "artifacts" / "datasets"
MANIFEST_PATH = OUT_DIR / "sim-v2-manifest.json"


def generate(params: Params, variant: str, learners: int, seed: int) -> tuple[pd.DataFrame, dict]:
    variant_params = params.as_variant(variant)
    simulator = Simulator(variant_params, seed=seed)
    frame = pd.DataFrame(simulator.run(simulator.learners(learners)))

    path = OUT_DIR / f"sim-v2-{variant}.parquet"
    frame.to_parquet(path, index=False)
    print(f"wrote: {path.relative_to(ROOT)} ({len(frame):,} attempts, {learners} learners) — {VARIANTS[variant]}")

    payload = json.dumps(variant_params.to_dict(), sort_keys=True, default=str).encode()
    return frame, {
        "variant": variant,
        "description": VARIANTS[variant],
        "file": str(path.relative_to(ROOT)),
        "rows": int(len(frame)),
        "learners": int(frame["learner_id"].nunique()),
        "params_sha256": hashlib.sha256(payload).hexdigest(),
        "accuracy": round(float(frame["correct"].mean()), 4),
        "median_response_time_ms": round(float(frame["response_time_ms"].median()), 1),
        "mean_sequence_length": round(float(len(frame) / frame["learner_id"].nunique()), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--learners", type=int, default=1000)
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    variants = [name.strip() for name in args.variants.split(",") if name.strip()]
    unknown = [name for name in variants if name not in VARIANTS]
    if unknown:
        raise SystemExit(f"unknown variant(s): {', '.join(unknown)}. Known: {', '.join(VARIANTS)}")

    params = Params.load()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    entries, checks = [], {}
    for variant in variants:
        frame, entry = generate(params, variant, args.learners, args.seed)
        entries.append(entry)
        if variant == "V0":
            # The calibration gate ran on its own sample. Re-running the same
            # comparison on the dataset that is actually shipped is what the
            # sim-vs-real figures annotate, so the figure and the file agree.
            checks = compare(load_real(params.calibrated_against), frame)
            for name, result in checks.items():
                print(f"  KS {name:22s} D = {result.get('ks_d')} "
                      f"{'ok' if (result.get('ks_d') or 1) <= KS_LIMIT else 'ABOVE LIMIT'}")

    manifest = {
        "seed": args.seed,
        "git_sha": git_sha(),
        "learners_per_variant": args.learners,
        "calibrated_against": params.calibrated_against,
        "params_file": str(PARAMS_PATH.relative_to(ROOT)),
        "params_sha256": hashlib.sha256(
            json.dumps(params.to_dict(), sort_keys=True, default=str).encode()
        ).hexdigest(),
        "variants": entries,
        "ks_limit": KS_LIMIT,
        "v0_vs_real": checks,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {MANIFEST_PATH.relative_to(ROOT)} ({len(entries)} variants)")


if __name__ == "__main__":
    main()
