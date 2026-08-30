# Load test — the deployed adaptive loop

Regenerate with `make loadtest` (the stack must be up: `make up`). Every number
here is read from `artifacts/evaluation/load-test.json`, which
`scripts/loadtest.py` writes; nothing is transcribed by hand.

## What is measured, and why it changed

The previous version of this document load-tested `POST /predict` — an endpoint
the adaptive loop no longer calls. Phase 10's budget is on a **decision**, and a
decision costs a feature extraction, a state estimate through Phase 7's encoder
and a policy call. So the driver runs the real learner sequence: consent,
session, then `GET /v1/next-item` → `POST /v1/events` → `POST /v1/attempts` per
item, against the compose stack.

The budget is per item, not per session, because the literature this project
builds on notes that session-level detectors resolve too slowly to act on: a
system that decides after the session cannot adapt during it.

## Configuration

| | |
|---|---|
| Target | `http://127.0.0.1:4000` (docker compose: backend, ml-service, postgres) |
| Learners per run | 12 |
| Items per learner | 20 |
| Concurrency levels | 1, 4, 8, 16 |
| Serving policy | `rule_improved` at ε = 0.1, estimator rung L4 |
| Live slot pool | 32 sessions |

## Per-endpoint latency at concurrency 16

| endpoint | requests | p50 (ms) | p95 (ms) | p99 (ms) | max (ms) |
|---|---:|---:|---:|---:|---:|
| `attempt` | 240 | 25.2 | 36.5 | 45.7 | 57.7 |
| `consent` | 12 | 2.5 | 3.8 | 3.8 | 3.8 |
| `events` | 240 | 6.1 | 9.4 | 11.0 | 11.1 |
| `next-item` | 240 | 15.4 | 31.3 | 34.4 | 37.0 |
| `session` | 12 | 20.3 | 47.8 | 47.8 | 47.8 |
| `session-end` | 12 | 4.3 | 6.0 | 6.0 | 6.0 |

**The decision path (`next-item`) is at p95 31.3 ms against a 300 ms budget.**
`f10-03` plots this; `f10-02` places the medians on the request sequence.

## Concurrency sweep

| concurrent learners | decisions | seconds | decisions/s | p50 (ms) | p95 (ms) | p99 (ms) | errors |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 240 | 4.5 | 53.03 | 5.8 | 7.9 | 14.7 | 0 |
| 4 | 240 | 1.3 | 186.31 | 6.5 | 9.4 | 11.7 | 0 |
| 8 | 240 | 1.1 | 220.79 | 9.4 | 13.7 | 16.0 | 0 |
| 16 | 240 | 1.0 | 231.5 | 15.4 | 31.3 | 34.4 | 0 |

`f10-04` plots throughput against decision latency. Throughput saturates near
231.5 decisions/s on this machine while the decision p95 stays inside
the budget, so the limit here is the host, not the model: the ml-service holds
one estimator and one arm in memory and steps a single learner slot per
request.

## What this does not measure

- **A cold slot.** The pool holds 32 live sessions; the 12-learner runs never
  evict one, so no measurement here includes the replay a returning learner
  pays after an eviction or a service restart.
- **Real browsers.** The driver is a Python client on the same machine; no
  network latency, no rendering, no think time.
- **Sustained load.** Each run is seconds long. Nothing here speaks to memory
  growth or to what happens after hours of traffic.
