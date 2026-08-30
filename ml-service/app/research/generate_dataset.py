"""Command-line generator for versioned ALP synthetic interaction data."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from app.research.simulator_legacy import StudentSimulator


SCHEMA_VERSION = "alp-synthetic-v1"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--learners", type=int, default=200)
    parser.add_argument("--items", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--output", type=Path, default=Path("data/synthetic/interactions.csv"))
    args = parser.parse_args()
    simulator = StudentSimulator(args.seed)
    rows = list(simulator.simulate(simulator.make_profiles(args.learners), args.items))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=[key for key in rows[0] if key != "profile"])
        writer.writeheader()
        writer.writerows([{key: value for key, value in row.items() if key != "profile"} for row in rows])
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {"schema_version": SCHEMA_VERSION, "created_at": datetime.now(timezone.utc).isoformat(), "seed": args.seed, "learners": args.learners, "items_per_learner": args.items, "records": len(rows), "sha256": digest, "generator": "StudentSimulator"}
    args.output.with_suffix(".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
