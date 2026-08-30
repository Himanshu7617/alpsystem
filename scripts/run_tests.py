"""Phase 12 — run every suite and record the result honestly.

    python3 scripts/run_tests.py [--json artifacts/evaluation/test-results.json]

Runs the backend jest suite and the ml-service pytest suite the same way
`make test` does — the container if one is running, the host otherwise — parses
each runner's own summary line and writes one JSON file. A failing suite is
recorded as failing and the script exits non-zero; the report never claims a
green suite that was not green.

Standard library only: this runs on the host, before anything is built.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "evaluation" / "test-results.json"
DB_URL = ("postgresql://adaptive_learning:adaptive_learning@127.0.0.1:54329/"
          "adaptive_learning?schema=public")


def run(command: list[str], cwd: Path, env: dict | None = None) -> tuple[int, str, float]:
    import os

    started = time.perf_counter()
    result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                            env={**os.environ, **(env or {})})
    return result.returncode, result.stdout + result.stderr, time.perf_counter() - started


def container_running() -> bool:
    code, output, _ = run(["docker", "compose", "ps", "--status", "running", "--services"], ROOT)
    return code == 0 and "ml-service" in output.split()


def parse_jest(output: str) -> dict:
    """jest prints `Tests: 1 failed, 12 passed, 13 total`."""
    line = re.search(r"^Tests:\s+(.*)$", output, re.MULTILINE)
    counts = {}
    if line:
        for count, label in re.findall(r"(\d+)\s+(passed|failed|skipped|todo|total)", line.group(1)):
            counts[label] = int(count)
    suites = re.search(r"^Test Suites:\s+(.*)$", output, re.MULTILINE)
    return {"counts": counts, "suites": suites.group(1).strip() if suites else None}


def parse_pytest(output: str) -> dict:
    """pytest -q ends with `129 passed in 22.83s` or `1 failed, 128 passed ...`."""
    line = re.findall(r"^(?:=+\s*)?((?:\d+ (?:passed|failed|error[s]?|skipped|xfailed|"
                      r"xpassed|warning[s]?)(?:, )?)+).*$", output, re.MULTILINE)
    counts = {}
    if line:
        for count, label in re.findall(r"(\d+)\s+(passed|failed|errors?|skipped|xfailed|xpassed)",
                                       line[-1]):
            counts[label.rstrip("s")] = int(count)
    return {"counts": counts}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-backend", action="store_true",
                        help="ml-service only (the backend suite needs postgres)")
    arguments = parser.parse_args()

    suites = []
    if not arguments.skip_backend:
        print("backend: jest")
        code, output, seconds = run(["npm", "test"], ROOT / "backend", {"DATABASE_URL": DB_URL})
        suites.append({"suite": "backend (jest)", "runner": "npx jest",
                       "passed": code == 0, "seconds": round(seconds, 1),
                       **parse_jest(output),
                       "tail": output.strip().splitlines()[-3:]})
        print(f"  {'PASS' if code == 0 else 'FAIL'} in {seconds:.1f}s")

    print("ml-service: pytest")
    if container_running():
        command = ["docker", "compose", "exec", "-T", "-e", "OMP_NUM_THREADS=1",
                   "ml-service", "sh", "-c", "cd ml-service && pytest -q"]
        where = "container"
    else:
        command = ["pytest", "-q"]
        where = "host"
    code, output, seconds = run(command, ROOT if where == "container" else ROOT / "ml-service",
                                {"OMP_NUM_THREADS": "1"})
    suites.append({"suite": "ml-service (pytest)", "runner": f"pytest -q [{where}]",
                   "passed": code == 0, "seconds": round(seconds, 1),
                   **parse_pytest(output),
                   "tail": output.strip().splitlines()[-3:]})
    print(f"  {'PASS' if code == 0 else 'FAIL'} in {seconds:.1f}s")

    report = {
        "phase": 12,
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "all_passed": all(suite["passed"] for suite in suites),
        "total_tests": sum(suite["counts"].get("passed", 0) + suite["counts"].get("failed", 0)
                           for suite in suites),
        "suites": suites,
        "note": "the ml-service suite includes the leakage audit, the gate guardrail, the "
                "policy conformance checks and the Phase 11 statistics guards; the backend "
                "suite drives the live loop end to end against a real database",
    }
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {arguments.json.relative_to(ROOT)} "
          f"({report['total_tests']} tests, "
          f"{'all green' if report['all_passed'] else 'FAILURES RECORDED'})")
    raise SystemExit(0 if report["all_passed"] else 1)


if __name__ == "__main__":
    main()
