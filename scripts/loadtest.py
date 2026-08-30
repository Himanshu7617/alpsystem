#!/usr/bin/env python3
"""Load-test the full adaptive loop and record the per-endpoint latencies.

    python3 scripts/loadtest.py --learners 20 --items 20 --concurrency 8

Every learner runs the real sequence — consent, session, then item after item
through `/v1/next-item`, `/v1/events` and `/v1/attempts` — because the number
Phase 10 needs is the latency of a *decision*, and a decision costs a feature
extraction, a state estimate and a policy call. Load-testing `/predict` on its
own, which is what the previous locustfile did, measured a code path the loop
no longer takes.

Standard library only: it runs on the host against the compose stack, and the
percentiles it writes to `artifacts/evaluation/load-test.json` are what the
Phase 10 figures and `docs/load-test-results.md` read.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "artifacts" / "evaluation" / "load-test.json"

_lock = threading.Lock()
_samples: dict[str, list[float]] = defaultdict(list)
_errors: dict[str, int] = defaultdict(int)


def call(method: str, url: str, label: str, payload: dict | None = None, timeout: float = 30.0):
    """One request, timed. Records the latency under ``label``."""
    body = None if payload is None else json.dumps(payload).encode()
    request = urllib.request.Request(url, data=body, method=method,
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = json.loads(response.read() or b"{}")
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
        with _lock:
            _errors[label] += 1
        return None, str(error)
    elapsed = (time.perf_counter() - started) * 1000.0
    with _lock:
        _samples[label].append(elapsed)
    return data, None


def session(base_url: str, learner: str, items: int) -> str | None:
    """One learner's whole session. Returns an error string, or None."""
    call("POST", f"{base_url}/v1/consent", "consent", {"learner_id": learner})
    created, error = call("POST", f"{base_url}/v1/sessions", "session", {"learner_id": learner})
    if created is None:
        return f"session: {error}"
    session_id = created["session_id"]
    version = created["session_version"]

    for index in range(items):
        served, error = call("GET", f"{base_url}/v1/next-item?session_id={session_id}", "next-item")
        if served is None or "item" not in served:
            return f"next-item[{index}]: {error or served}"
        item_id = served["item"]["item_id"]
        version = served["session_version"]
        presented = time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        events = [
            {"event_id": f"{session_id}-{index}-p", "type": "ITEM_PRESENTED",
             "item_id": item_id, "client_ts": presented, "seq": index * 2},
            {"event_id": f"{session_id}-{index}-s", "type": "OPTION_SELECTED",
             "item_id": item_id, "client_ts": presented, "seq": index * 2 + 1,
             "payload": {"option_index": index % 4}},
        ]
        call("POST", f"{base_url}/v1/events", "events",
             {"session_id": session_id, "learner_id": learner, "events": events})
        attempt, error = call("POST", f"{base_url}/v1/attempts", "attempt", {
            "session_id": session_id, "item_id": item_id, "selected_index": index % 4,
            "presented_at": presented, "response_time_ms": 9000 + index * 100,
            "session_version": version})
        if attempt is None:
            return f"attempt[{index}]: {error}"
        version = attempt["session_version"]

    call("POST", f"{base_url}/v1/sessions/end", "session-end", {"session_id": session_id})
    return None


def percentiles(values: list[float]) -> dict:
    ordered = sorted(values)
    def at(fraction: float) -> float:
        if not ordered:
            return float("nan")
        return round(ordered[min(len(ordered) - 1, int(fraction * len(ordered)))], 1)
    return {"n": len(ordered), "p50": at(0.50), "p95": at(0.95), "p99": at(0.99),
            "max": round(ordered[-1], 1) if ordered else float("nan"),
            "mean": round(statistics.fmean(ordered), 1) if ordered else float("nan")}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:4000")
    parser.add_argument("--ml-url", default="http://127.0.0.1:8000",
                        help="only read once, to record which policy was serving")
    parser.add_argument("--learners", type=int, default=20)
    parser.add_argument("--items", type=int, default=20)
    parser.add_argument("--concurrency", type=int, default=8,
                        help="concurrent learners; keep it below the ml-service slot pool")
    parser.add_argument("--sweep", default="",
                        help="comma-separated concurrency levels, e.g. 1,4,8,16. "
                             "Each level is a separate run; the throughput figure reads them.")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    arguments = parser.parse_args()

    info, _ = call("GET", f"{arguments.ml_url}/model-info", "model-info")
    levels = ([int(value) for value in arguments.sweep.split(",") if value.strip()]
              or [arguments.concurrency])

    runs = []
    for level in levels:
        _samples.clear()
        _errors.clear()
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=level) as pool:
            failures = [outcome for outcome in pool.map(
                lambda index: session(arguments.base_url,
                                      f"load_{int(time.time())}_{level}_{index}", arguments.items),
                range(arguments.learners)) if outcome]
        elapsed = time.perf_counter() - started
        endpoints = {label: percentiles(values) for label, values in sorted(_samples.items())}
        decisions = len(_samples.get("next-item", []))
        runs.append({
            "concurrency": level,
            "elapsed_seconds": round(elapsed, 1),
            "decisions": decisions,
            "decisions_per_second": round(decisions / elapsed, 2) if elapsed else 0.0,
            "endpoints": endpoints,
            "errors": dict(_errors),
            "failures": failures[:5],
        })
        print(f"--- concurrency {level} " + "-" * 40)
        for label, stats in endpoints.items():
            print(f"{label:<12} n={stats['n']:<5} p50 {stats['p50']:>7} ms  "
                  f"p95 {stats['p95']:>7} ms  p99 {stats['p99']:>7} ms")
        print(f"{decisions} decisions in {elapsed:.1f}s "
              f"({runs[-1]['decisions_per_second']}/s), errors: {dict(_errors) or 'none'}")

    headline = max(runs, key=lambda run: run["concurrency"])
    report = {
        "base_url": arguments.base_url,
        "learners": arguments.learners,
        "items_per_learner": arguments.items,
        "serving": (info or {}).get("serving", {}),
        "runs": runs,
        # The figures and the docs quote the busiest run: a latency budget met
        # only at concurrency 1 is not a latency budget.
        "headline": headline,
    }
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {arguments.out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
