#!/usr/bin/env python3
"""`make status` — where the current or last expensive run got to.

Reads only artifacts the run itself writes, so it is safe to call while a run
is in flight and it works after the run was killed. Standard library only: it
runs on the host, outside the ml-service image.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "artifacts" / "evaluation"
LOG = EVAL / "run-log.jsonl"


def ago(seconds: float) -> str:
    for size, unit in ((86400, "d"), (3600, "h"), (60, "m")):
        if seconds >= size:
            return f"{seconds / size:.1f}{unit} ago"
    return f"{seconds:.0f}s ago"


def main() -> None:
    if not LOG.exists():
        print("no run log yet — nothing expensive has been run from this checkout.")
        return
    entries = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line]
    tail = entries[-1]
    age = time.time() - LOG.stat().st_mtime
    # Phase 8 logs cell progress (done/total); Phase 9 logs named stages with
    # neither. Report the last cell-shaped line for progress and the last line
    # of any shape for what the run is doing now.
    cells = [entry for entry in entries if "done" in entry and "total" in entry]
    last = cells[-1] if cells else {}
    unfinished = last.get("done", 0) < last.get("total", 0)
    state = "running" if age < 300 and (unfinished or not cells) else "idle"

    print(f"{tail.get('script', 'run')}: {state}, last write {ago(age)}")
    if "stage" in tail:
        print(f"  last event {tail['stage']}"
              + (f" {tail['variant']}" if "variant" in tail else "")
              + (f" {tail['arm']}" if "arm" in tail else ""))
    if last:
        print(f"  profile {last['profile']}: {last['workers']} workers, "
              f"{last['learners']} learners, budget {last['budget']}")
        print(f"  progress {last['done']}/{last['total']} cells, "
              f"{last['elapsed_seconds']}s elapsed")
    # Only this run's cells: the log is append-only across runs, and a dev
    # cell's timing would otherwise be averaged into a full run's estimate.
    start = max((index for index, entry in enumerate(cells) if entry["done"] == 1),
                default=0)
    fresh = [entry for entry in cells[start:] if not entry.get("cached")]
    if fresh and unfinished:
        mean = sum(entry["cell_seconds"] for entry in fresh) / len(fresh)
        remaining = (last["total"] - last["done"]) * mean / max(1, last["workers"])
        print(f"  estimate  ~{remaining / 60:.1f} min left at {mean:.0f}s/cell")

    checkpoints = sorted((EVAL / "cells").glob("*.json.gz"))
    print(f"  checkpoints {len(checkpoints)} cells in artifacts/evaluation/cells "
          f"(a stopped run resumes from these)")
    for name in sorted(EVAL.glob("policy-results-*.json")):
        report = json.loads(name.read_text(encoding="utf-8"))
        print(f"  wrote {name.name}: {len(report['policies'])} arms, "
              f"{report['learners_per_seed']}×{len(report['seeds'])} learners, "
              f"{len(report.get('effect_sizes_over_threshold', []))} unexplained halts")


if __name__ == "__main__":
    main()
