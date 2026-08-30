# Adaptive Learning Platform (ALP)

A research platform for a single question: **does conditioning adaptive
instructional decisions on a validated multi-state behavioural learner model
(knowledge, engagement, confidence, within-session fatigue) produce better
instructional efficiency than conditioning on correctness and response time
alone, with the policy architecture held constant?**

The platform is the instrument. `BUILD.md` is the execution plan;
`docs/PROGRESS.md` is the honest running state.

> **Results status — all twelve phases are built.** The headline is a **null**:
> in the simulated closed loop no arm differs from the strongest hand-written
> rule on the uncensored co-primary outcome (every |Hedges' g| ≤ 0.012, CIs
> straddling zero, 3,000 learners per arm). The validity gate admitted **one of
> four** candidate learner states. Both findings are reported as findings — see
> [`docs/reports/`](docs/reports/) for the two PDFs and `docs/PROGRESS.md` for
> the per-phase account.
>
> Everything produced **before** the Phase 0 rebuild remains **invalid and must
> not be cited**. Three defects — an oracle "ML" policy, latent ground truth used
> as a model input, and behavioural features that are deterministic functions of
> the latent state — are documented in `artifacts-legacy-invalid/README.md` and
> in `BUILD.md` Section 1. They are fixed in the current pipeline and each has a
> test that fails if it returns.

---

## Architecture

```
docker compose up
  ├── postgres:16          (5432 internal, 54329 host)
  ├── backend  (Node 24 / Express 5 / Prisma 7)   :4000
  │     └── serves files/ as the static frontend
  └── ml-service (Python 3.11 / FastAPI)          :8000
        └── reads artifacts/models/*.joblib
```

1. **Frontend** (`files/`) — vanilla JS, captures per-item interaction
   telemetry. Cursor-trajectory and probe instrumentation lands in Phase 2.
2. **Backend** (`backend/`) — Express, JWT auth, session lifecycle, the
   explainable latent-state rule engine, and a 36-concept prerequisite DAG.
3. **Prisma / PostgreSQL** — learners, sessions, per-session items, roadmap
   state. Migration history lives in `backend/prisma/migrations/`.
4. **ML service** (`ml-service/`) — FastAPI inference plus the research
   pipeline under `app/research/`.

Question content is a static, version-controlled bank
(`backend/src/data/questions/`). Runtime LLM generation was removed in Phase 0:
non-deterministic content is a reproducibility hazard in an experiment.

---

## Quickstart

Requirements: Docker Desktop, and — for host-side work — Node 24 and Python 3.11.

```bash
cp .env.example .env
make up          # builds and starts postgres, backend, ml-service
make seed        # applies migrations and loads the static question bank
```

Then open <http://localhost:4000/index.html>.

Verify the stack:

```bash
curl localhost:4000/health   # {"status":"ok","database":"up"}
curl localhost:8000/health   # {"status":"ok","model_loaded":false,...}
```

`model_loaded: true` is expected: the service loads the saved artifacts from
`artifacts/models/` at start-up and reports exactly which at
`curl localhost:8000/model-info`. Nothing is ever refitted inside a request.

`ALP_POLICY` (`rule` | `bandit` | `rl`) chooses the served arm in **both**
services — it is the only switch, and it selects the same policy object the
experiments were run on.

### Make targets

| Group | Targets |
|---|---|
| Infrastructure | `up` `down` `logs` `seed` `test` `lint` `clean` |
| Data | `fetch-data` `prep-data` `simulate` `features` |
| Models | `train-baselines` `train-states` |
| Evaluation | `eval-policies` `integrate` `ope` `rl` |
| Live system | `bank` `loadtest` `system` |
| Reporting | `stats` `figures` `report` `paper` |
| Verification | `test` `test-report` `reproduce` |
| Run profiles | `smoke` `dev` `full` `status` |

`make help` lists everything. Expensive targets climb a profile ladder —
`smoke` (the default, 40 learners) validates, `dev` (250) reads a real effect,
and `full` (1,000 learners × 3 seeds × 5 variants) refuses to start without a
written justification in `docs/full-run-justification.md`. **`make all` runs at
`smoke` on purpose**: a full experiment is never a side effect of typing a
command.

```bash
make stats && make figures   # the statistics, the tables, all 109 figures + INDEX.md
make test-report             # both suites, counted into artifacts/evaluation/
make report && make paper    # the two PDFs into docs/reports/
python3 scripts/reproduce.py # the whole make chain, timed per target
```

Global seed: **20260821**. Every script takes `--seed` and records it in its
output manifest.

For host-side development without containers, `./scripts/dev.sh` still starts
postgres in Docker and runs the two services on the host.

---

## Documentation

| Document | Path |
|---|---|
| Execution plan (phases, acceptance) | [`BUILD.md`](BUILD.md) |
| Progress and honest state | [`docs/PROGRESS.md`](docs/PROGRESS.md) |
| Architecture and design | [`docs/architecture.md`](docs/architecture.md) |
| API reference | [`docs/api-reference.md`](docs/api-reference.md) |
| System report (PDF) | [`docs/reports/ALP-System-Report.pdf`](docs/reports/ALP-System-Report.pdf) |
| Research paper draft (PDF) | [`docs/reports/ALP-Research-Paper-Draft.pdf`](docs/reports/ALP-Research-Paper-Draft.pdf) |
| Pre-registration | [`docs/preregistration.md`](docs/preregistration.md) |
| Evaluation and statistical protocol | [`docs/evaluation.md`](docs/evaluation.md) |
| Figure index (generated) | [`artifacts/figures/INDEX.md`](artifacts/figures/INDEX.md) |
| Invalidated prior results | [`artifacts-legacy-invalid/README.md`](artifacts-legacy-invalid/README.md) |

Every number in the two PDFs is substituted from a file under `artifacts/` when
the document is built, and every figure records the script and the input files
that produced it. Nothing in either report is transcribed by hand.

One document is deliberately part-quarantined: the top section of
`docs/model-card.md` is the **pre-Phase-0** CatBoost card, kept behind a banner
because BUILD.md forbids deleting an artifact that has not been regenerated. Its
numbers are inflated by the leakage defect and are not citable; the state-head
cards at the end of that file are current.
