"""Phase 12 — the clean-room reproduction, timed per target.

    python3 scripts/reproduce.py [--profile smoke] [--from items] [--only a,b]

Runs the `make all` chain one target at a time in the current checkout, records
the wall clock and the exit status of each, and writes
`artifacts/evaluation/reproduction-timing.json` **as it goes**, so an
interrupted reproduction still reports how far it got. `make all` runs the same
targets; this wrapper exists because the acceptance criterion is a *timing*, and
a single `time make all` cannot say which target the time went to.

Intended use: a fresh checkout with Docker installed.

    make up                     # build the image, start postgres
    python3 scripts/reproduce.py

`fetch-data` verifies the archival downloads against their pinned SHA-256 and
re-downloads only what is missing, so a machine that already holds `data/raw/`
reproduces without pulling 792 MB again — and the run records which it did.

Standard library only: this is the script that runs before anything is built.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "artifacts" / "evaluation" / "reproduction-timing.json"

#: `make all`'s order, which is the dependency order of the phases.
TARGETS = ["items", "telemetry", "fetch-data", "prep-data", "simulate", "features",
           "train-baselines", "train-states", "eval-policies", "integrate", "ope", "rl",
           "bank", "stats", "figures", "report", "paper"]

#: Which phase each target belongs to, for the figure and the report table.
PHASE = {"items": 1, "telemetry": 2, "fetch-data": 3, "prep-data": 3, "simulate": 4,
         "features": 5, "train-baselines": 6, "train-states": 7, "eval-policies": 8,
         "integrate": 8.5, "ope": 9, "rl": 9, "bank": 10, "stats": 11, "figures": 11,
         "report": 12, "paper": 12}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="smoke",
                        help="run profile for every expensive target (BUILD.md §1.5)")
    parser.add_argument("--from", dest="start", default=None,
                        help="resume the chain at this target")
    parser.add_argument("--only", default=None, help="comma-separated subset")
    parser.add_argument("--keep-going", action="store_true",
                        help="record a failure and continue instead of stopping")
    arguments = parser.parse_args()

    targets = TARGETS
    if arguments.only:
        targets = [name.strip() for name in arguments.only.split(",") if name.strip()]
    elif arguments.start:
        targets = targets[targets.index(arguments.start):]

    report = {
        "phase": 12,
        "started": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "profile": arguments.profile,
        "checkout": str(ROOT),
        "targets": [],
        "note": "wall clock per `make` target in one checkout, at the stated run "
                "profile. The published artifacts were produced at the profiles each "
                "phase's PROGRESS entry records; this run measures reproducibility, "
                "not the size of the experiment.",
    }

    def flush() -> None:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    total = 0.0
    for target in targets:
        print(f"\n=== make {target} (PROFILE={arguments.profile})", flush=True)
        started = time.perf_counter()
        result = subprocess.run(["make", target, f"PROFILE={arguments.profile}"],
                                cwd=ROOT, capture_output=True, text=True)
        seconds = time.perf_counter() - started
        total += seconds
        output = (result.stdout + result.stderr).strip().splitlines()
        report["targets"].append({
            "target": target, "phase": PHASE.get(target),
            "seconds": round(seconds, 1), "ok": result.returncode == 0,
            "wrote": [line for line in output if line.startswith("wrote:")][-3:],
            "tail": output[-4:],
        })
        report["total_seconds"] = round(total, 1)
        report["all_ok"] = all(entry["ok"] for entry in report["targets"])
        flush()
        print(f"  {'ok' if result.returncode == 0 else 'FAILED'} in {seconds:.1f}s "
              f"({total / 60:.1f} min elapsed)", flush=True)
        if result.returncode != 0:
            print("\n".join(output[-15:]), flush=True)
            if not arguments.keep_going:
                break

    report["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    flush()
    print(f"\nwrote: {OUTPUT.relative_to(ROOT)} — {len(report['targets'])} targets, "
          f"{total / 60:.1f} min, {'all ok' if report['all_ok'] else 'FAILURES RECORDED'}")
    raise SystemExit(0 if report["all_ok"] else 1)


if __name__ == "__main__":
    main()
