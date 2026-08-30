# Progress

Running state of the Adaptive Learning Platform rebuild. One section per phase,
written at the end of that phase, before context is cleared. This file is the
handoff. It records failures and null results as faithfully as successes.

Global seed: **20260821**. Execution plan: [`BUILD.md`](../BUILD.md).

---

## Why this file was replaced

The previous version of this document marked all twelve original objectives
complete. That assessment was wrong: three defects invalidate every research
result the repository had produced. They are reproduced here because they are
the reason the plan is shaped the way it is.

**Defect 1 — the "ML policy" was an oracle, not a model.** In
`ml-service/app/research/evaluate_policies.py`, `difficulty_for()` selected
difficulty for the `ml_based` arm with
`sigmoid(4.2 * (knowledge - score / 10) + 1.1 * confidence - 1.6 * fatigue)` —
character-for-character the expression the simulator uses in `run_policy()` to
decide whether the learner answers correctly. The arm never loaded the trained
CatBoost artifact. The reported win measures an oracle beating a heuristic.

**Defect 2 — latent ground truth was used as a model input.**
`knowledge_before` and `fatigue_before` are the simulator's internal latent
variables and are listed in `BASE_FEATURES` in
`ml-service/app/model/features.py`. They are not observable in any deployed
system, and they carry the largest feature importance in the selected model.
Every reported ROC-AUC is inflated by this leak.

**Defect 3 — behavioural features were deterministic functions of the latent
state.** In `simulator.py`, `option_changes`, `pause_duration`,
`mouse_distance`, `tab_switches` and the rest are computed from `knowledge`,
`fatigue`, `confidence` and `attention_span` plus Gaussian noise. A model
trained on that data must find behavioural features predictive of the latent
state, because they were generated from it.

**Corroborating signal.** The reported unpaired Cohen's *d* of 26.20 is roughly
two orders of magnitude above what this literature reports (SAINT+: +1.25% AUC
from timing; ITS meta-analyses: *g* ≈ 0.32–0.37). An effect that size is a
defect signature, not a discovery.

### Known-invalid results

Everything in [`artifacts-legacy-invalid/`](../artifacts-legacy-invalid/) was
produced by the pre-Phase-0 pipeline. **Nothing in it may be cited.** It is
retained only so the defects can be demonstrated and so the "before" side of
the Phase 4 mutual-information figure (`f04-07`) can be reproduced. Its README
lists the affected files defect by defect.

The following documents still contain numbers from that pipeline and are
rewritten by the phase that regenerates their underlying artifacts:
`docs/paper-draft.md`, `docs/model-card.md`, `docs/evaluation.md`,
`docs/modeling.md`, `docs/dataset.md`, `docs/data-card.md`,
`docs/load-test-results.md`, `docs/mastery-root-cause.md`.

---

## Phase checklist

| Phase | Title | Status |
|---|---|---|
| 0 | Repository triage and infrastructure | **done** |
| 1 | Content, item bank and psychometric calibration | **done** |
| 2 | Telemetry instrumentation, event schema, label acquisition | **done** |
| 3 | Real archival data acquisition and leakage-safe preprocessing | **done** |
| 4 | Simulator v2: calibrated, honest, deliberately mis-specified | **done** |
| 5 | Feature engineering and exploratory data analysis | **done** |
| 6 | Study 1: predictive modelling, baselines, ablation ladder | **done** — final test pass completed in Phase 8.5; `artifacts/benchmarks/study1-final.json` exists |
| 7 | Study 2: multi-state estimation, calibration, validation gate | **done** — 1 of 3 states admitted |
| 8 | Adaptive policies and closed-loop evaluation | **done** — interrupted once, resumed; see `docs/phase-8-resume-audit.md` |
| 8.5 | Pipeline integration, training-artifact repair, honesty audit | **done** — see `docs/phase-8.5-audit.md`, `docs/training-pipeline.md`, `docs/policy-architecture.md`, `artifacts/evaluation/phase-8.5-validation.json`; no narrative section was written below and BUILD.md's acceptance boxes were never ticked |
| 9 | Study 3: offline policy evaluation, bandits, RL | **done at `dev`** — `full` not run; see the Phase 9 section below |
| 10 | Live platform integration | **done** — served by the Phase 8/9 arms; `full` load profile not attempted |
| 11 | Statistical analysis and the figure pack | **done** — the closed-loop headline is a null; see the Phase 11 section below |
| 12 | End-to-end verification, system report PDF, paper PDF | **done** — both PDFs in `docs/reports/`, `artifacts/evaluation/final-verification.json` is the checklist |

Every file in `artifacts/` must be reproducible from a `make` target. Phase 1
puts the first two there (`datasets/item-parameters-v1.csv` and the four
`figures/01-content/` charts, both from `make items`); Phase 2 adds the four
`figures/02-telemetry/` charts (`make telemetry`); Phase 3 adds
`datasets/archival-summary.json`, `datasets/archival-bank-concept-map.json` and
the eight `figures/03-archival/` charts (`make fetch-data && make prep-data`);
Phase 4 adds the simulator parameter, KS, MI and manifest files plus the nine
`figures/04-simulator/` charts (`make simulate`); the rest refills from Phase 5
onwards.

---

## Phase 0 — Repository triage and infrastructure

**Built**

- **Runtime LLM generation removed.** Deleted `backend/src/services/ai.service.js`,
  `backend/src/services/questionGenerator.service.js` and
  `backend/src/data/prompts/`; dropped `@google/genai` and `openai` from
  `backend/package.json`. `backend/src/services/questionBank.service.js` replaces
  the generation path with a loader over the version-controlled JSON banks; an
  unknown topic now returns 404 with the list of available topics instead of
  calling out to a model. `POST /admin/questions/generate` and its controller
  were removed rather than stubbed.
- **`/health` on both services.** `GET /health` on the backend runs
  `SELECT 1` through Prisma and returns 503 when the database is unreachable.
  `GET /health` on the ml-service reports liveness plus whether a trained
  artifact is currently loadable. Neither existed before; the compose
  healthchecks need them.
- **Single `docker-compose.yml`** replaces `docker-compose.dev.yml`: `postgres`,
  `backend`, `ml-service`, healthchecks on all three, `depends_on:
  service_healthy`, named volume for postgres, `./artifacts` and `./mlruns`
  mounted read-write into ml-service. `backend/Dockerfile` is node:24-slim,
  `ml-service/Dockerfile` is python:3.11-slim. Both build from the repository
  root so the in-image layout mirrors the repo — `predictor.py` resolves
  `artifacts/models` by walking three parents up from `app/model/`, and the
  research scripts resolve `artifacts/` relative to the working directory.
- **Prisma migration history.** `backend/prisma/migrations/20260821171804_init/`
  is the first migration this schema has ever had; it was previously only
  pushed. The backend container runs `prisma migrate deploy` before starting.
  `backend/prisma/seed.js` loads the static bank; `prisma.config.ts` now points
  its seed command at it instead of a `seed.ts` that did not exist.
- **Root `Makefile`.** `up down logs seed test lint clean` are implemented.
  Every later-phase target (`fetch-data prep-data simulate features
  train-baselines train-states eval-policies ope rl figures stats report paper
  all`) exits non-zero and names the phase that will implement it.
- **`.env.example`** drops `GEMINI_API_KEY` and adds `MLFLOW_TRACKING_URI`,
  `SEED=20260821` and `ALP_POLICY=rule_improved`.
- **One pinned `ml-service/requirements.txt`**, replacing the split with
  `requirements-research.txt` (deleted). All 24 packages named in BUILD.md
  resolve together on python:3.11-slim — **nothing was dropped and no
  substitute was needed**. The pins are the output of a full pip resolution run
  on 2026-08-21: `torch==2.13.0`, `mlflow==3.15.1`, `pykt-toolkit==0.0.38`,
  `pyBKT==1.4.3`, `stable_baselines3==2.9.0`, `weasyprint==69.0`,
  `catboost==1.2.10`, `numpy==2.4.6`, `pandas==2.3.3`.
- **Docs.** `README.md` rewritten around the research question and the
  `make up` / `make seed` quickstart; `docs/architecture.md` gains a deployment
  topology section and the corrected global seed.

**Verified**

- `docker compose up -d --wait` from a stopped state: **all three services
  healthy in 11s** (budget was 60s).
- `curl localhost:4000/health` → `200 {"status":"ok","database":"up"}`.
- `curl localhost:8000/health` → `200 {"status":"ok","model_loaded":false,
  "artifact_dir":"/workspace/artifacts/models"}` — the mounted host directory,
  confirming the in-image layout resolves artifacts correctly.
- `curl localhost:4000/index.html` → 200; the backend serves the frontend from
  the image.
- `make test`: backend Jest **6 passed, 6 total**; ml-service pytest inside the
  container **5 passed**.
- `make lint` passes: `node --check` over all backend JavaScript, `compileall`
  over `ml-service/app`.
- `make seed` is idempotent — reruns write 126 questions to the canonical seed
  session without duplicating rows.
- `npx prisma migrate dev --name init` created and applied the migration
  against the compose postgres.
- `git status` is clean of `.DS_Store`, `tmp/` and `catboost_info/`.

**Failures encountered and fixed**

- Two builds failed on transient network errors — `npm ci` with a bare
  `network` error, then `pip install` with a hash mismatch on a truncated
  wheel download. Neither was a dependency problem; both installs now carry
  explicit retry and timeout flags.
- `npx prisma db seed` first failed with `The requested module '@prisma/client'
  does not provide an export named 'PrismaClient'`. The client had never been
  generated in a fresh checkout; `npx prisma generate` fixes it, and the
  backend image runs it at build time.

**Deliberately skipped**

- **The `QuestionDraft` admin review flow was kept, not deleted.** With the
  generator gone nothing populates that table, so `GET /admin/questions/pending`
  now always returns an empty list. The routes still function and the schema is
  Phase 2's business, so removing them here would be scope creep. Delete them in
  Phase 1 if the item bank does not reuse the draft workflow.
- **The seed writes the bank into one canonical session** (`seed-question-bank`)
  rather than a first-class item table, because the current schema attaches
  every `Question` to a `Session`. Marked with a `ponytail:` comment in
  `prisma/seed.js`; Phase 1 replaces it with a real item bank.
- **No linter added.** A syntax gate over both languages catches what actually
  breaks a pipeline run, and a style linter is a dependency this project does
  not need.
- **`scripts/dev.sh` was retargeted, not removed.** It still runs the two
  services on the host against dockerised postgres, which is faster than a
  rebuild during development.

**What the next phase needs to know**

- Research scripts resolve `artifacts/` from the working directory while
  `predictor.py` resolves it from the source tree. They only agree when python
  runs from the repository root with `ml-service` on `PYTHONPATH` — the
  `Makefile` sets `PY` accordingly. Anything that runs `cd ml-service && python
  -m app.research.X` will silently write to `ml-service/artifacts/` instead.
- `ADAPTIVE_POLICY` (read by `question.controller.js`) and `ALP_POLICY` (named
  by BUILD.md) both appear in `.env.example` and both default to
  `rule_improved`. Phase 8 should collapse them to one name when it rebuilds the
  ML arm against a real artifact.
- The ml-service `/health` endpoint reports `model_loaded: false` until Phase 6
  writes a model into `artifacts/models/`. That is the correct state, not a
  failure.
- `make test` runs the ml-service suite inside the running container, because
  the pinned research stack is not installed on the host. `make up` first, or
  the target fails loudly.
- `pykt-toolkit` pulls in `wandb`, which phones home by default. Any phase that
  imports it should set `WANDB_MODE=disabled` before the first training run.

---

## Phase 1 — Content, item bank and psychometric calibration

**Built**

- **`data/items/item-bank-v1.json` — 186 items**, built by
  `ml-service/app/research/build_item_bank.py` from all five question sources.
  Schema is the one BUILD.md specifies: `item_id, concept_id, subject, stem,
  options[4], correct_index, explanation, author_difficulty,
  prerequisite_concepts[], format`. Sources merge in a fixed order and the
  first item to claim a normalised stem hash wins, so the build is
  deterministic. `data/items/item-bank-v1.manifest.json` records the count, the
  bank SHA-256, per-source counts, every dropped duplicate, every excluded item
  with its reason, per-concept coverage, the labelling rules and the difficulty
  bin edges.
- **`ml-service/app/research/validate_bank.py`** — Kahn topological sort over
  the prerequisite graph (fails naming the concepts left in a cycle), asserts
  every `concept_id` resolves, every item has 4 options, `correct_index` is in
  range and `author_difficulty` is 1–10, then prints the coverage table.
- **`ml-service/app/research/calibrate_items.py`** — 1PL Rasch fit,
  `logit P(correct) = theta_learner - b_item`, by L2-regularised logistic
  regression over one-hot learner and item dummies, plus equal-width binning of
  `b` onto the 10-point `difficulty_score`. Writes
  `artifacts/datasets/item-parameters-v1.csv` and merges `difficulty_bins` and
  `calibration` back into the manifest.
- **`ml-service/app/research/figures_phase1.py`** — the four Phase 1 figures in
  `artifacts/figures/01-content/`. No plotting dependency was added; the DAG is
  laid out by level with matplotlib directly.
- **`make items`** runs all four in order and is idempotent — a second run
  leaves the working tree byte-identical.
- **`docker-compose.yml`** now mounts `./data`, `./ml-service/app` and
  `./backend/src/data` (read-only) into ml-service. Without them the container
  could not see the concept graph at all: `app.research.concept_graph` resolves
  it at `/workspace/backend/src/data/concept_graph.json`, which the image does
  not contain. Mounting the code as well means a research script edit no longer
  needs an image rebuild.

**Verified**

- `make items` from a clean checkout: 186 items, 126 duplicates dropped, 30
  items excluded, exit 0.
- `validate_bank`: `cse-prerequisite-graph-v1` is acyclic (36 concepts), every
  `concept_id` resolves, coverage table printed, exit 0.
- `pytest app/research/test_item_bank.py`: **6 passed**. The tests assert bank
  well-formedness, build determinism (a second `build()` is `==` the first),
  that the out-of-scope items are excluded rather than mislabelled, that a
  deliberately cyclic graph raises, that the Rasch fit recovers 20 known
  difficulties from 400 simulated learners at Spearman > 0.9, and that the
  binning spans 1–10 monotonically.
- All four figures exist and are non-empty; each was opened and read.
- Rerunning `make items` twice leaves `git status` unchanged.

**Null and negative results**

- **Zero items are Rasch-calibrated.** Phase 3 has not run, so
  `data/processed/bank-responses.csv` does not exist. Every row of
  `item-parameters-v1.csv` is `source = author_prior`, `b` is the z-scored
  `author_difficulty`, `se_b` is empty and `n_responses` is 0. Figure `f01-03`
  carries a PLACEHOLDER box saying so. **`make items` must be re-run after
  Phase 3** (BUILD.md Phase 3, step 5).
- **24 of 36 concepts hold fewer than 5 items.** The whole bank is 186 items
  across 36 concepts and it is bimodal: `trees_and_bst` has 30 and
  `divide_and_conquer` has 26 because two whole topical banks land there, while
  `graphs`, `greedy_algorithms`, `congestion_control_net` and
  `distributed_databases` have one each. Mastery estimation on a 1-item concept
  is meaningless. `validate_bank` warns and does not fail — the shortfall is a
  property of the bank, and the fix is authoring items, not a threshold change.
- **126 of the 342 source items were duplicates.** `question_bank.csv` is a
  flattened export of `multitopic_cse.json`; every row collided.
- **30 items excluded.** `transportability_among_animals.json` is a biology
  bank with no concept in the CSE graph. Mapping it to a CSE concept to keep the
  count up would corrupt downstream mastery estimates, so it is excluded and
  listed in the manifest.

**Deliberately skipped**

- **The backend still seeds from `backend/src/data/questions/`, not from the
  consolidated bank.** Repointing the serving path is Phase 10's job (live
  platform integration) and doing it here would mean touching the Prisma schema
  that Phase 2 is about to rewrite. Consequence to be aware of: the bank the
  research pipeline reads and the questions the platform serves are two
  different files until Phase 10 closes the gap.
- **The `QuestionDraft` admin flow was kept again.** Phase 0 handed it to
  Phase 1 conditionally; Phase 1 does not reuse it, but deleting it properly
  means dropping a Prisma model and writing a migration, which lands in the
  middle of Phase 2's schema work. Phase 2 should delete the model, the route
  and `listPending` together.
- **2PL/3PL was not attempted.** Marked with a `ponytail:` comment: 1PL is the
  only model this bank has the response volume for. Upgrade at > 200
  responses/item.
- **`se_b` is the item-information approximation** `1/sqrt(sum p(1-p))`, not
  the inverse Fisher information of the joint fit, so it ignores uncertainty in
  theta and is mildly optimistic. Marked with a `ponytail:` comment. It is
  currently empty anyway.
- **No new dependency.** The DAG figure is drawn with matplotlib rather than
  adding `networkx` for one plot; the topical concept labelling is a keyword
  table, not a classifier, for 60 items.

**What the next phase needs to know**

- **Difficulty binning is equal-width over the `b` range, not deciles.** Deciles
  collapse under the ties a discrete author prior produces and silently handed
  the policies a 7-point scale; equal-width spans 1–10 and, under the author
  prior, is exactly the identity on `author_difficulty`. The edges are in the
  manifest under `difficulty_bins`.
- **`bank-responses.csv` is the contract with Phase 3.** `calibrate_items`
  expects `learner_id, item_id, correct` at `data/processed/bank-responses.csv`,
  with `item_id` already mapped onto bank item ids. Items with no responses stay
  on the author prior; the CSV's `source` column distinguishes them per item.
- **The new research scripts resolve paths from `__file__`, not the working
  directory** (`ROOT = Path(__file__).resolve().parents[3]`), so they write to
  the right place from any cwd. The Phase 0 caveat about running from the
  repository root still applies to the older scripts.
- **Item ids from the topical banks are namespaced** `bt-q1`, `rec-q1`. Only the
  126 `multitopic_cse` ids (`ds-e-1`, …) match the `question_to_concept` map and
  the backend seed.
- `data/raw/` and `data/processed/` are gitignored; `data/items/` is tracked.

---

## Phase 2 — Telemetry instrumentation, event schema and label acquisition

**Built**

- **Event schema in three places that must agree.**
  `ml-service/app/schemas/events.py` (Pydantic) is the definition of record,
  `backend/prisma/schema.prisma` stores it, `files/telemetry.js` produces it,
  and `docs/event-schema.md` documents it. All sixteen event types from
  BUILD.md exist, each mapped to one of five consent classes.
- **Prisma migration `20260822021818_telemetry`.** New models: `Consent`,
  `ItemAttempt`, `InteractionEvent`, `CursorSegment`, `Probe`, `StateEstimate`,
  `Decision`, `ExperimentAssignment`. `Session` gains `learner_id`,
  `consent_id`, `started_at`, `ended_at` and a `version` column used for
  optimistic concurrency on every state write. `Decision.propensity` and the
  four `StateEstimate.*_se` / `*_validated` columns are non-negotiable fields,
  not conveniences.
- **`files/telemetry.js`** — pointer sampling at 50 ms into a session-scoped
  buffer, reduced at submit to the pre-registered trajectory aggregate and then
  **discarded**; `FIRST_INTERACTION`, idle enter/exit at a 3 s threshold,
  `visibilitychange` + `blur`/`focus` carrying `visible` *and* `focused`
  separately, and the ordered option-change sequence with timestamps. Events
  buffer and flush every 5 s, on submit and on `pagehide`; a failed flush
  re-queues, which is safe because ingest is idempotent on `event_id`.
- **`/v1` API** (`backend/src/routes/v1.js`, `controllers/v1.controller.js`):
  `POST /v1/consent`, `POST /v1/sessions`, `GET /v1/next-item`,
  `POST /v1/events`, `POST /v1/attempts`, `POST /v1/sessions/end`.
  Correctness is computed from the stored answer key, which is **never sent to
  the client** — `/v1/next-item` returns the item without it. Every served item
  writes a `Decision` with its propensity.
- **Propensities are real, not 1.0.** `backend/src/utils/decision.js` serves
  epsilon-greedy over the top-K of the rule policy's own ranking (`ALP_EPSILON`
  default 0.1, `ALP_TOP_K` default 5), seeded per session so exploration
  replays. `questionPolicy.chooseNextQuestion` now also returns its scored
  candidate list so the stochastic layer and the deterministic caller share one
  scoring definition. The full action space with per-candidate probabilities is
  stored on every `Decision`.
- **Probes.** Assignment is server-side and seeded by (session seed, item
  sequence): confidence + perceived difficulty on a random 12 % of items,
  effort every 15th item. `assign_p` and the realised `assign_draw` are both
  stored. Probes render after submission and **before** feedback, and skipping
  is recorded as a skip.
- **Consent screen** listing all five signal classes in plain English, each
  independently switchable except `correctness`. Enforced twice: the browser
  does not record an unconsented class, and `/v1/events` rejects it if it
  arrives anyway (the rejection is reported in the response).
- **`ml-service/app/features/extract.py`** — the single definition of a
  feature. The backend calls it over HTTP (`POST /v1/features/extract`); Phase 5
  training imports `extract()` directly. Produces 33 features across five signal
  classes, and `FEATURE_CLASSES` gives Phase 5 the ablation ladder for free.
- **Conformance test** (`ml-service/app/features/test_extract.py`, 6 cases) —
  replays the recorded stream in `app/features/testdata/attempt-events.json`
  and asserts the row still equals the golden fixture, plus order independence,
  the no-latent-ground-truth guardrail, and null-not-zero under withdrawn
  consent. Runs in `make test`.
- **Four figures** in `artifacts/figures/02-telemetry/`, rebuilt by
  `make telemetry`. `f02-04` is drawn from the same recorded fixture the
  conformance test uses; `f02-02` and `f02-03` illustrate *definitions* and say
  "synthetic illustration — not a result" on their face.
- **Deletions.** `QuestionDraft` (model, migration, `backend/src/routes/admin.js`,
  `backend/src/controllers/admin.controller.js`, the `/admin` mount) — the flow
  Phase 0 and Phase 1 each deferred. `getLocalNextDifficulty` in the frontend,
  dead once the server became the only source of difficulty.

**Verified**

- `make test`: backend Jest **6 passed**, ml-service pytest **17 passed**
  (11 before this phase). `make lint` clean.
- **Real browser run** (Chrome, five items answered end to end): 14 event types
  landed, including `SESSION_STARTED`, `DECISION_MADE`, `CURSOR_SEGMENT`,
  `PROBE_SHOWN`/`PROBE_ANSWERED` and `SESSION_ENDED`. 5 attempts, 5 with an
  extracted feature row, 5 cursor segments, 0 unlinked events. **Zero console
  errors.** One probe was answered and one skipped; the skip is stored as
  `skipped = true` with a null response.
- **Consent withdrawal**: same flow with cursor movement unticked completed two
  items normally and produced **zero `CursorSegment` rows**; the motor features
  in those attempts are `null` while timing and interaction features are
  present.
- **Idempotency**: replaying an identical 10-event batch stores 0 additional
  events, 0 additional cursor segments and 0 additional probes.
- Scripted 20-item run: propensities span 0.02–0.92 (exploration fires),
  probes fire at the expected rate, `session_version` advances on every write.

**Failures encountered and fixed**

- **`ExperimentAssignment.seed` overflowed Postgres `integer`.** The 32-bit FNV
  hash of the session uuid exceeds 2^31; sessions failed with `value
  "2217566268" is out of range for type integer`. The seed is now masked to 31
  bits.
- **Idempotent event ingest was not idempotent for derived rows.**
  `createMany({skipDuplicates})` deduplicated `InteractionEvent` but
  `materialise()` still ran over the whole batch, so a replay doubled the
  `CursorSegment` rows (62 rows for 31 events). Ingest now queries which
  `event_id`s already exist and derives rows only from genuinely new events.
- **The backend's projection of events to the extractor omitted `event_id`**,
  so every extract call returned 422 and every attempt was stored with an empty
  feature row. Silent-ish: the loop kept working because extraction is
  fail-soft. Caught by checking `features_extracted` in the smoke run, not by a
  test — a reminder that fail-soft paths need to be asserted somewhere.
- `.probe-box { display: grid }` beat the `hidden` attribute, so an empty
  dashed box rendered under every item. Fixed with `.probe-box[hidden]`.
- The ml-service container mounts `app/` but does not reload, so a new FastAPI
  route needs `docker compose restart ml-service`. The backend has no mount at
  all and needs `docker compose up -d --build backend`.

**Null and negative results**

- **Typing dynamics are dropped from the study, before any collection.** The
  bank is 186 items, **100 % MCQ**, so keystroke timing would cover 0 % of
  items. BUILD.md's own rule applies: a signal on a small minority of items
  cannot support a claim. No key event of any kind is recorded. Recorded in
  `docs/preregistration.md` §2.
- **`ITEM_SKIPPED` and `HINT_REQUESTED` are defined but never emitted.** The UI
  has neither a skip nor a hint control. The schema and the extractor handle
  them, so adding the controls later needs no migration, but no data exists for
  them today and the corresponding features are constant zero.
- Trajectory aggregates from the browser acceptance run have 2–7 samples per
  item, because a scripted pointer jumps rather than moves. `n_samples` is
  stored precisely so a downstream analysis can refuse to trust a segment built
  from seven points. Real sessions will need a minimum-sample filter; Phase 5
  should set it.

**Deliberately skipped**

- **The legacy `/question` path was kept, not deleted.** It still serves the
  rule-engine experiments and the Jest suite, and it does not log propensities
  or hide the answer key, so **it must not be used for research data**. Two
  serving paths exist until Phase 10 collapses them. `create_session_with_bank`
  in `services/session.service.js` is shared by both so session creation cannot
  drift.
- **No auth on `/v1`.** The pilot has no recruitment and no personal data beyond
  a display name; the existing `requireAuth` gate would only have broken the
  loop (the frontend's `/user/login` never issued a JWT in the first place, so
  the quiz path was already 401-ing before this phase). Phase 10 revisits it.
- **`sample_entropy` is the textbook O(n²) SampEn** (m = 2, r = 0.2σ) over the
  speed series. Marked with a `ponytail:` comment: a few hundred samples per
  item is ~10⁵ comparisons, far below anything worth optimising.
- **Consent is stored client-side in `localStorage`** as well as in the
  `Consent` table, so a returning learner is not re-asked. Clearing browser
  storage re-prompts and writes a second `Consent` row; the rows are
  append-only and timestamped, which is the behaviour an audit wants anyway.
- **No backend integration test for the `/v1` loop.** BUILD.md names the
  conformance test as the phase's runnable check and that is what was built.
  Phase 10 step 6 specifies the end-to-end test.

**What the next phase needs to know**

- **`extract()` is the only place a feature may be defined.** Phase 5 imports
  `app.features.extract` — it must not re-implement anything, and
  `FEATURE_CLASSES` already carries the L0–L4 signal-class mapping the ablation
  ladder needs.
- **Null means "not consented", zero means "measured zero".** Every consumer
  must preserve that distinction; imputing zeros silently destroys the RQ4
  privacy–utility curve.
- **`ml-service/app/model/features.py` still lists `knowledge_before` and
  `fatigue_before`** in `BASE_FEATURES`. That is Defect 2, still live in the
  legacy predictor path. Phase 5 replaces that module and adds
  `audit_leakage.py`; nothing in the new extractor touches latent state.
- **Regenerate the golden fixture deliberately, never to make a red test
  green.** The command is in the docstring of `test_extract.py`.
- Environment variables added: `ALP_EPSILON`, `ALP_TOP_K`, `ALP_PROBE_RATE`,
  `ALP_EFFORT_PROBE_EVERY`. All have defaults; none are in `.env.example` yet
  because the defaults are the pre-registered values.
- Phase 3's archival preprocessing does not depend on anything here — the two
  can proceed independently — but Phase 5's feature catalogue must document
  every feature `extract()` produces, and Phase 7 fills the `*_se` and
  `*_validated` columns that are currently defaulted.

---

## Phase 3 — Real archival data acquisition and leakage-safe preprocessing

**Built**

- **`scripts/fetch_data.py`** downloads three archival sources into `data/raw/`,
  verifies each against a pinned SHA-256 and writes `data/raw/manifest.json`
  (url, digest, size, licence, citation). URLs are pinned to immutable
  revisions. On any failure it prints the canonical manual-download instructions
  and **exits non-zero — there is no synthetic fallback**. Verified by
  truncating a downloaded file: exit code 1, instructions printed.
  Stdlib only, so it runs on the host without the ml-service image.
- **`ml-service/app/research/prep_archival.py`** normalises all three to one
  schema, drops learners with < 10 interactions, winsorises response time at the
  per-item 1st/99th percentile, computes the Wise & Kong effort label, splits by
  learner, and writes `data/processed/<source>.parquet`, `splits.json`,
  `archival-manifest.json`, `artifacts/datasets/archival-summary.json` and
  `archival-bank-concept-map.json`.
- **`ml-service/app/research/figures_phase3.py`** — the eight
  `artifacts/figures/03-archival/` charts, every number read from the processed
  tables or the summary JSON.
- **Makefile**: `fetch-data` and `prep-data` are implemented (they were stubs).
- **`ml-service/app/research/test_prep_archival.py`** — six checks on the
  transforms (short-learner drop, winsorisation counts and thin-item fallback,
  rapid-guess flagging, null-safety, split disjointness/coverage/seed
  stability). They run on tiny in-test frames, so `make test` stays green on a
  machine that has never downloaded the 830 MB of raw data. `make test` is now
  backend 6 passed, ml-service 23 passed.

**Data acquired**

| Source | interactions kept | learners | items | skills | accuracy |
|---|---|---|---|---|---|
| ASSISTments 2009–2010 skill builder | 253,686 | 2,968 | 15,879 | 111 | 0.659 |
| ASSISTments 2012–2013 | 2,593,636 | 22,422 | 46,901 | 265 | 0.699 |
| EdNet KT1 (shard 0 of 11, subsampled) | 4,897,531 | 23,916 | 12,228 | 1,785 | 0.653 |

Full provenance, licences, drop counts, winsorisation counts and the EdNet
subsampling rule are in `docs/data-card.md`; the split protocol is in
`docs/evaluation.md`.

**Verified**

- `make fetch-data` re-verifies all four files against their pinned digests and
  rewrites the manifest. A corrupted file exits 1 with manual instructions.
- `make prep-data` runs end to end in ~50 s and both acceptance assertions run
  inside it: no learner appears in more than one split (and the splits cover
  every learner), and rapid-guess accuracy is checked against chance wherever a
  chance level exists.
- All eight figures exist and are non-empty.
- `make items` was re-run afterwards, as BUILD.md Phase 3 step 5 requires.

**Null and negative results, recorded**

- **No archival skill maps onto a bank concept — 0 of 2,161.** ASSISTments is
  middle-school mathematics, EdNet KT1 is TOEIC English, the bank is
  undergraduate computer science. So **Rasch calibration of the bank is still
  impossible**: `data/processed/bank-responses.csv` is deliberately not written
  and all 186 items remain `author_prior`. Real item difficulties need responses
  to *these* items, which only the live platform (Phase 10) can produce. The
  test and verdict are in `artifacts/datasets/archival-bank-concept-map.json`.
- **EdNet fails the rapid-guess acceptance check, and the threshold is not the
  reason.** Pooled rapid-guess accuracy is 0.543 against chance 0.25; no
  response-time cut brings it near chance. By ability quartile, the lowest
  quartile scores 0.339 (inside the ±0.10 band) and the highest 0.811 — fluent
  learners in self-study TOEIC prep answer fast because they know the answer,
  violating the assumption behind the index. Recorded as an evidenced deviation
  in `RAPID_GUESS_DEVIATIONS` (the check still fails the build for any source
  not listed there), in `docs/preregistration.md` §6 and in `docs/data-card.md`.
- ASSISTments has **no defined chance level at all** — both releases are
  dominated by algebra/fill-in items and store no option count — so the check is
  marked not applicable there rather than quietly passed.

**Deliberately skipped**

- **Junyi Academy** (BUILD.md marks it optional): its contribution would be
  prerequisite structure, which the 36-concept DAG already supplies.
- **EdNet shards 1–10.** BUILD.md caps the source at ~5 M interactions; shard 0
  alone holds 8.66 M, so 10 shards (1.7 GB) would be downloaded only to be
  discarded. Recorded in the manifest as `shards_used: 1 of 11`.
- **No `pykt-toolkit` preprocessing.** Its loaders assume its own directory
  layout and re-split internally; `splits.json` is the single source of truth
  here. The protocol is pyKT's, the code is not.

**What the next phase needs to know**

- **Read `data/processed/splits.json`. Never re-split.** It holds explicit
  learner-id lists per source: `test` (20 %) and `cv_folds` (5). The test set is
  touched exactly once, at the very end of the project.
- **Order sequences by `order_index`, not `timestamp`.** ASSISTments 2009–2010
  ships no wall-clock column, so its `timestamp` is null throughout, and with it
  `lag_time_ms`. EdNet has no `attempt_count`/`hint_count`. `f03-08` is the
  availability map; any feature built on a null-for-a-source column must degrade
  per source rather than silently impute.
- **Use `response_time_ms` (winsorised) for modelling and
  `response_time_ms_raw` for anything about timing behaviour itself.** The
  effort label is computed on the raw column.
- **Phase 4 calibrates the simulator against these tables.** The distributions
  it must match are in `f03-02` (sequence length), `f03-03` (response time) and
  `f03-04` (accuracy by attempt); the summary JSON holds the numbers.
- **Phase 7's engagement criterion is the ASSISTments RTE label**, not EdNet's,
  unless it conditions on ability. See the deviation above.
- `data/raw/` and `data/processed/` are git-ignored (817 MB raw, 157 MB
  processed). `make fetch-data && make prep-data` rebuilds both from scratch.

---

## Phase 4 — Simulator v2: calibrated, honest, and deliberately mis-specified

**Built**

- **`ml-service/app/research/simulator.py` rewritten.** Latent knowledge per
  concept, engagement, confidence and fatigue; a 2PL link with slip and guess;
  log-normal response time located by ability–difficulty distance, fatigue, item
  stem length and a per-learner speed trait. The old file is kept as
  `simulator_legacy.py` — quarantined, documented as the "before" side of Defect
  3, and imported only by the two pre-Phase-8 scripts that Phase 8 replaces.
- **Circularity broken three ways.** Stable learner traits (pointer speed,
  jitter, device, re-reading, indecision, distraction) are drawn *independently
  of ability*; item effects (stem length) drive reading and path behaviour; a
  contaminating off-task channel produces idle time and visibility changes with
  no connection to the latent state.
- **Trajectories are paths, not features.** The simulator generates a 2-D Bézier
  pointer path with minimum-jerk timing, trait jitter, competitor attraction and
  hesitation pauses, then hands the event stream to
  `app.features.extract.extract()` — the production function. `reduce_trajectory()`
  was added to `extract.py` as the server-side mirror of the browser's reducer,
  so there is still exactly one Python definition of a motor feature.
- **`calibrate_simulator.py`** fits ability, slip, learning rate, response-time
  location/scale, session structure and within-session fatigue from
  `data/processed/assistments_2012.parquet`, then generates from the fitted model
  and gates on four two-sample KS statistics.
- **`generate_sim.py`** (`make simulate`) writes the five variant datasets and a
  manifest with the seed, git SHA, parameter SHA-256, row counts and the
  V0-vs-real KS results.
- **`figures_phase4.py`** — the nine `artifacts/figures/04-simulator/` charts.
- **`test_simulator.py`** — 11 checks including the acceptance one: the extractor
  is spied on and asserted to have been called, and the row's motor feature is
  asserted equal to the extractor's own output. `make test` is now backend 6,
  ml-service 34.
- **Compose**: `artifacts-legacy-invalid/` and `.git/` are mounted read-only into
  ml-service — the first because `f04-07` needs the legacy dataset, the second so
  manifests can record the git SHA (the slim image has no git binary).

**Verified**

| Distribution | KS *D*, calibration | KS *D*, shipped V0 | Limit |
|---|---|---|---|
| per-learner accuracy | 0.058 | 0.042 | 0.15 |
| log response time | 0.049 | 0.049 | 0.15 |
| sequence length | 0.038 | 0.038 | 0.15 |
| within-session accuracy slope | 0.107 | 0.107 | 0.15 |

All four pass, so `DOCUMENTED_MISMATCHES` is empty. The gate exits non-zero on a
failure; nothing is accepted without being registered there with evidence.

Defect 3, measured (`artifacts/datasets/simulator-recoverability.json`):

| Measure | Legacy | v2 |
|---|---|---|
| mean observable↔latent mutual information | 0.298 nats | 0.022 nats |
| latent fatigue recoverable from all observables (CV R²) | **0.989** | 0.087 |
| latent knowledge recoverable from all observables (CV R²) | 0.177 | 0.054 |

`make simulate` runs calibration, five-variant generation and the figure pack in
2 min 35 s.

**Null and negative results, recorded**

- **There is no within-session accuracy fatigue in the calibration source.** The
  mean within-session slope of difficulty-residualised correctness is **+0.0023**
  per item; the practice curve explains +0.00068 of it; the residual is **+0.0016**
  — positive. `fatigue_accuracy_beta` therefore fits to **0.0**, and the
  response-time coefficient agrees (−0.590: later items are answered *faster*).
  Learners in this source warm up, they do not tire. This is exactly the case
  `docs/preregistration.md` H2 pre-committed to: the fatigue construct is reduced
  or dropped in Phase 5 rather than defended.
- **Consequence for the variant suite:** V2 ("fatigue affects speed only")
  differs from V0 only in the response-time channel, because V0's accuracy
  channel is already empty. V2 is retained deliberately — the calibration is a
  fit, not a law.
- **Two pre-registered motor features are weakly identified by the real
  interface.** The production quiz renders options as a single-column list, so
  every option centre shares an *x* coordinate and a path up the list has little
  lateral geometry: `auc_toward_nonchosen`, `max_deviation` and `x_flips` carry
  less than the pre-registration assumed. Hesitation shows in hover time, pauses
  and path length instead. Found by simulating the interface's real geometry.
- **One MI pair did not fall.** Hover time ↔ knowledge is marginally higher in v2
  than in the legacy model, and pointer velocity is comparable, because
  hesitation pauses are a modelled channel. MI above zero is not the defect —
  determinism was, and the recoverability R² is what shows it is gone.

**Deliberately skipped**

- **The `.parquet` datasets are git-ignored** (~110 MB for five variants).
  Manifest, parameters, KS, MI table and recoverability JSON are committed —
  every number any doc or figure quotes. `make simulate` rebuilds the data.
- **No cross-language check that `reduce_trajectory()` matches
  `files/telemetry.js`.** They are a hand-kept mirror, marked `ponytail:` in the
  code, pinned by `docs/preregistration.md` §1. A golden-path fixture evaluated
  in both languages is the upgrade if the two ever have to be provably identical.
- **No 3PL, no per-item guessing.** The guess asymptote is structural (1/4 for a
  four-option item), not fitted: the archival sources are dominated by fill-in
  items whose chance level really is ~0, and adopting that would give the
  simulated bank a guessing rate its own format rules out.

**What the next phase needs to know**

- **`latent_*` columns are targets, never inputs.** Phase 5's `audit_leakage.py`
  can exclude on that prefix. The feature columns in each dataset come from
  `app.features.extract` and are the only legitimate inputs.
- **`propensity` is recorded on every simulated attempt** — the logging policy
  draws a difficulty band uniformly, then an item within it. Study 3 (Phase 9)
  depends on it existing from the start.
- **Every closed-loop result must be reported against all five variants.** A
  ranking that flips between them is an artefact of one generative model. The
  loader is `pd.read_parquet(f"artifacts/datasets/sim-v2-{variant}.parquet")`.
- **The simulator is calibrated to middle-school mathematics behaviour, not to
  computer-science content.** What transfers is the shape of learner behaviour,
  not the domain. Say so in any claim built on it.
- **Phase 5's `f05-06` decides the fatigue construct's fate.** The calibration
  already says the accuracy channel is empty; if the figure is flat, the
  pre-registration requires reducing or dropping it.

---

## Phase 5 — Feature engineering and exploratory data analysis

**Built**

- **`ml-service/app/research/feature_catalogue.py`** — 44 features as data, one
  entry each with definition, rung, per-source availability, privacy class and a
  rationale carrying a literature key from the review's own numbering. It is
  standard-library only so the host can generate `docs/feature-catalogue.md`
  from it, and it is the single schema three consumers read: the builder builds
  exactly these columns, the audit checks exactly these names, the EDA colours
  by their signal class. `validate()` refuses an empty rationale, so BUILD.md's
  "a feature with no rationale does not get built" is a gate rather than a
  convention.
- **`build_features.py`** (`make features`) writes
  `data/processed/features-<source>.parquet` for all eight sources — three
  archival, five simulator variants — plus
  `artifacts/datasets/features-manifest.json` with the seed, git SHA, row counts
  and per-feature null rate. Per-attempt behavioural columns are **not**
  recomputed: they come from `app.features.extract` through the Phase 4
  simulator, so there is still one definition of a feature. This file adds only
  what needs a sequence — history aggregates, session structure, the label.
- **`audit_leakage.py`** — the standing guard against Defect 2, imported by
  `test_features.py` so it fails `make test`. Four checks (A latent columns,
  B no row sees its future, C no test learner informs an item statistic,
  D what is on disk is well-formed).
- **`eda.py`** — the thirteen `artifacts/figures/05-eda/` figures,
  `artifacts/benchmarks/eda-summary.json` and
  `artifacts/benchmarks/feature-shortlist.json`.
- **`test_features.py`** — 17 checks. `make test` is now backend 6, ml-service
  51 (was 34).
- **Compose**: `./docs` mounted read-only, so `make test` can assert that
  `docs/feature-catalogue.md` still matches the module that generates it.

**Verified**

- `make features` runs end to end in **1 min 40 s** and the leakage audit exits 0.
- **Check B has teeth, demonstrated.**
  `test_audit_catches_a_planted_lookahead` monkeypatches `_prior_mean` into a
  *leading* rather than lagging aggregate — the smallest honest mistake with
  Defect 2's shape — and asserts the audit raises. A guard never shown to fire
  is not a guard. It fires.
- Check B works by rebuilding a source truncated mid-learner and demanding the
  surviving rows come out bit-identical. That catches what reading code for
  `shift(1)` does not: a centred window, an unshifted expanding mean, a resort,
  or a statistic fitted over a whole column.
- Item difficulty and `log_rt_z_item` are population statistics and legitimately
  move when the population does, so check B freezes them and check C asserts the
  condition that actually matters — refitting with the test learners removed
  changes no training row.

| Source | Rows | Learners | Features carried |
|---|---|---|---|
| `assistments_2009` | 250,718 | 2,968 | 11 / 44 |
| `assistments_2012` | 2,571,214 | 22,422 | 16 / 44 |
| `ednet_kt1` | 4,873,615 | 23,916 | 14 / 44 |
| `sim-V0`…`sim-V4` | ~153,000 each | 1,000 each | 44 / 44 |

**Null and negative results, recorded**

- **The fatigue construct is dropped. This is the phase's headline finding and
  it is a null.** `f05-06`: the within-session matched-difficulty accuracy slope
  is +0.000077 [−0.000125, +0.000280] in `assistments_2012` — flat — and
  +0.000165 [+0.000004, +0.000343] in `ednet_kt1`, which is significantly
  *positive*. `assistments_2009` cannot be measured at all: no wall clock, so no
  sessions. `docs/preregistration.md` H2 pre-committed to reducing or dropping
  the construct in exactly this case, and §7 now records the consequence:
  **fatigue is not carried as a validated latent state into Phase 7's gate or
  Phase 8's policies.** The multi-state model is knowledge, engagement and
  confidence. This agrees with Phase 4, which fitted `fatigue_accuracy_beta` to
  0 from the same evidence.
- **The within-session *speed* channel survives** and is significantly negative
  everywhere (`assistments_2012` −0.0092 [−0.0097, −0.0087]; `ednet_kt1`
  −0.0018 [−0.0023, −0.0013]; simulated −0.0026). Learners speed up as a session
  runs. `matched_difficulty_speed_slope` stays an L3 feature; the dropped claim
  is that it indexes a fatigue *state*.
- **Correctness history dominates every behavioural rung by an order of
  magnitude.** Compressing each rung to one logistic score, MI with
  `next_correct` is L0 0.0232 nats, L2 0.0034, L3 0.0006, L1 0.0002, **L4
  0.0000**. That is a lower bound per rung — a linear score discards the
  interactions a tree model finds — but the *ordering* is stark and Phase 6
  should expect the ladder to be flat above L0.
- **RQ2's early read is "small, and simulated only."** Partial correlations with
  the label, holding response time out of both sides: `max_deviation` −0.023,
  `path_ratio` −0.021, `hover_time_max` −0.020, each near-orthogonal to response
  time itself (r = 0.03, 0.02, 0.07). Consistent with H1's pre-registered
  +1–3 % AUC, not more. No archival source carries motor telemetry, so this
  cannot be corroborated on real data in this project.
- **Answer changing does not reproduce the literature's benefit.** Accuracy
  falls monotonically with option changes (0.703 / 0.675 / 0.653 / 0.633 /
  0.616). B26 and B27 report changes as net beneficial; the simulator generates
  a change from hesitation and hesitation from low knowledge, and nothing in it
  makes a change *corrective*. Recorded as a simulator limitation, not a
  finding. The archival sources carry no option-change data at all.
- **Most motor features are traits, not states.** `f05-10`: between-learner
  share of variance is 0.87 for `path_ratio` and `velocity_mean`, 0.75 for
  `cursor_samples`, against 0.03 for `rt_drift` and 0.04 for
  `auc_toward_nonchosen`. This is Phase 4's design working — idiosyncratic
  pointer traits drawn independently of ability are what broke Defect 3 — but
  the consequence is that **a raw motor feature is a poor state signal** and a
  state estimator needs a within-learner baseline, the way `rt_drift` already
  standardises response time.
- **The simulator couples three signal pairs the real world would not**:
  `idle_count` ↔ `visibility_changes` (r = 1.00), `focus_fraction` ↔
  `idle_fraction` (−1.00), `time_to_first_movement` ↔ `time_to_first_selection`
  (1.00). Each is one generative channel emitted twice. The shortlist drops one
  of each and flags the reason `simulated_only`.

**Deliberately skipped**

- **A feature is only called redundant if it is collinear in *every* source that
  carries both.** Judging redundancy from the simulator alone would have dropped
  `log_response_time` — SAINT+'s headline feature — on a 0.95 correlation with
  `log_rt_z_item` that the archival sources do not reproduce.
- **The shortlist drops what is empty, not what is small** (36 kept, 8 dropped).
  Deciding what a feature is worth is the Phase 6 ablation ladder's job; doing it
  here, on the same data, would be selection on the outcome.
- **High VIF is reported, not acted on**: `option_changes` 97,
  `cursor_samples` 59, `option_revisits` 50, `option_change_entropy` 39,
  `log_response_time` 34, `log_rt_z_item` 22, `question_number` 11. Trees do not
  care and Phase 6's linear models are regularised; dropping a construct because
  it correlates with a sibling would cost the ladder a rung.
- **Absent features get no column written.** EdNet's 4.9 M rows would carry
  ~900 MB of all-null float columns otherwise — enough to OOM the container, as
  it did before the fix. `build_features.read_features()` re-adds them as null
  on the way in, so every consumer still sees one schema.
- **No imputation is ever persisted.** Median filling happens inside the mutual
  information estimator, which cannot take NaN, and is thrown away. Absence and
  measurement stay distinguishable.
- **No per-source feature selection.** A different feature set per source makes
  the ablation ladder incomparable across sources.

**What the next phase needs to know**

- **Read `docs/feature-catalogue.md` and `preregistration.md` §7 first.** The
  ladder, the label and the leakage rules are fixed there.
- **The label is `next_correct`** — the *next* attempt's outcome, so attempt
  *t*'s own behavioural trace is legitimately an input. Predicting attempt *t*'s
  own outcome instead would make every within-item behavioural feature unusable
  by construction and would answer RQ2 by definition.
- **Load matrices with `build_features.read_features(source)`, not
  `pd.read_parquet`.** A source's parquet omits the columns it cannot supply;
  the helper restores the full schema as nulls.
- **`splits.json` now also holds `sim-V0`…`sim-V4`**, added under the same
  protocol and seed. The three archival entries were not touched. Nothing
  re-splits.
- **Fatigue is out.** Do not build a fatigue head into Phase 7's multi-state
  model or a fatigue term into a Phase 8 policy. The construct failed its
  pre-registered test and the paper says so.
- **Expect a flat ladder above L0** and report it as a null if it is one.
  BUILD.md Phase 6 step 3 already sets the rule: an increment whose CI does not
  exclude 2 % is not a result.
- **The quick sanity floor is AUC 0.637** — `HistGradientBoostingClassifier` on
  the full simulated feature set, learner-grouped. Any Phase 6 model below that
  is misconfigured, not interesting.
- **`data/processed/features-*.parquet` are git-ignored** (~600 MB). The
  manifest, the EDA summary, the shortlist and the thirteen figures are
  committed — every number any doc quotes. `make features` rebuilds the data.

---

## Phase 7 — Study 2: multi-state estimation, calibration and the validation gate

**Headline: one state of three is admitted to the policy.** Knowledge passes.
Engagement fails Wise & Kong's negative criterion. Confidence fails both its
correlation floor and discriminant validity. Fatigue has no head at all — the
construct was dropped in Phase 5. That is the result, and Phase 8's policy is
built on a single validated state plus the observables.

**Built**

- **`ml-service/app/model/validation_gate.py`** — the gate as code. Four
  pre-registered criteria (`docs/preregistration.md` §8.2) as a `Criterion`
  table, `evaluate()` to apply them, and a `Gate` object Phase 8 loads.
  `Gate.filter_states()` **removes** a rejected state from what a policy can
  see; it does not zero it, because a policy handed `engagement = 0.0` is still
  conditioning on engagement. A missing gate report admits nothing, and a
  criterion that was never measured fails rather than passes.
- **`ml-service/app/research/study2.py`** (`make train-states`) — one shared
  GRU (hidden 64, dropout 0.2) over the interaction sequence with three heads,
  each supervised by its own label under a masked loss, trained at every live
  rung of the ladder. Isotonic calibration per head, MC-dropout uncertainty
  (20 passes), and the four gate measurements. Writes
  `artifacts/models/state-encoder.pt`, `state-calibrators.joblib`,
  `artifacts/evaluation/validation-gate.json`,
  `artifacts/benchmarks/study2-heads.json` and `study2-predictions.parquet`.
- **`figures_phase7.py`** — the ten `artifacts/figures/07-study2/` charts.
- **`test_study2.py`** — 12 checks, including the one BUILD.md Phase 7's
  acceptance asks for: flip a state to failed and assert it is *absent* from
  what the policy can read, not zeroed.
- **Pre-registration §8** — labels, protocol and all four thresholds, written
  before the encoder was fitted; §8.4 records two deviations.
- **`docs/evaluation.md` §3** — the validity protocol. Old §3 renumbered to §4.

**What supervises what** (`docs/preregistration.md` §8.1)

| Head | Label | Judged on |
|---|---|---|
| knowledge | `next_correct` | `assistments_2012` (real) |
| engagement | `solution_behaviour` of the **next** attempt | `assistments_2012` (real) |
| confidence | `probe_confidence ≥ 3` | `sim-V0` (simulated) |

The engagement label is shifted forward one attempt on purpose. Rapid-guessing
is defined by a response-time cut and response time is an L1 input, so an
unshifted head scores near-perfectly by re-deriving the threshold — arithmetic,
not a state estimate.

**Verified — the gate verdict** (`artifacts/evaluation/validation-gate.json`,
rung L4, scoring half of `fold4`, test set untouched)

| State | C1 *r* with own label | C2 ECE | C3 \|*r*\| with ability | C4 \|*r*\| with knowledge | Verdict |
|---|---|---|---|---|---|
| knowledge | **0.316** [0.282, 0.345] ✓ | **0.016** ✓ | n/a | n/a | **ADMITTED** |
| engagement | 0.281 [0.188, 0.371] ✓ | 0.005 ✓ | **0.403** ✗ (≤0.20) | n/a | **EXCLUDED** |
| confidence | **0.138** [0.083, 0.196] ✗ (≥0.15) | 0.024 ✓ | n/a | **0.893** ✗ (≤0.85) | **EXCLUDED** |

- **The engagement head is a good predictor and a bad measurement.** ROC-AUC
  0.831 against the RTE label and an ECE of 0.005 — it predicts next-attempt
  rapid-guessing well. It also correlates 0.40 with ability, twice the
  pre-registered ceiling, and `f07-05` shows the slope plainly. Wise & Kong's
  criterion says exactly what to conclude: a head that tracks ability has
  learned ability, not effort. Its accuracy is what makes the failure
  interesting rather than trivial — the tempting move is to keep it *because*
  the AUC is high, and the pre-registration is what stops that.
- **The confidence head is a relabelled knowledge head.** *r* = 0.893 with the
  knowledge head on the rows it is scored on (`f07-06`, boxed cell), against a
  0.85 ceiling, and its own correlation with the probe is 0.138 — below the
  floor and with a CI that excludes 0.15 nowhere near comfortably. Both
  criteria fire in the same direction, and the probe supplies only 1,384
  labelled scoring rows (12 % probe rate) to fight with.
- **Knowledge passes on real data**: *r* = 0.316, ROC-AUC 0.686, ECE 0.032 →
  0.016 after isotonic.
- **The state ablation is the privacy–utility curve per state** (`f07-08`).
  Knowledge is flat across every rung (0.313 at L0, 0.315 at L4) — the same
  null Phase 5 predicted for the ladder. Engagement is the one state that needs
  a signal class: *r* 0.130 at L0 → 0.287 at L1, i.e. **timing is what makes an
  effort head possible at all**, and nothing above L1 adds to it. Confidence
  never clears the floor at any rung and drifts *down* (0.162 → 0.138).
- **Uncertainty** (`f07-09`): mean MC-dropout SE 0.023 knowledge, 0.016
  engagement, 0.029 confidence; 90th percentile 0.029 / 0.039 / 0.039.
- `make train-states` runs end to end in **≈ 12 minutes** on CPU (five rungs).
- `pytest` — 12 new checks pass, including that the numbers in
  `validation_gate.CRITERIA` still match the strings in the pre-registration,
  and that the written report agrees with re-running `evaluate()` on its own
  measurements.

**Deliberately skipped**

- **No transformer.** BUILD.md offers "GRU or a small transformer"; a GRU over
  ≤100-step sequences trains in two minutes on CPU and the contribution of this
  phase is the supervision-and-validation regime, not the encoder.
- **No fatigue head.** Pre-registration §7's H2 rule already fired. `f07-10`
  draws the observable that was measured instead of an estimate that does not
  exist.
- **No rescue of a failed head.** Tuning until engagement clears C3 is the
  exact behaviour the pre-registration exists to prevent. The failures are
  reported and the states are excluded.
- **MC dropout rather than a 5-model ensemble** — one training run for the same
  standard error (BUILD.md Phase 7 step 3 explicitly allows the cheaper one).
- **The test set was not touched.** Phase 7 needs no test-set number; it is
  dropped from the frame rather than merely avoided.
- **Sequences are chunked at 100 attempts** and the hidden state resets at each
  boundary (`ponytail:` marked in `study2.py`). Upgrade path: carry the hidden
  state across a learner's chunks, worth it only if a trajectory is still
  moving at the boundary.
- **Two sources, not eight.** `assistments_2012` carries the RTE label,
  `sim-V0` the probes; no other source adds a *label*, and the rows come from
  Phase 6's loader so Study 1 and Study 2 are fitted on the same data.

**What the next phase needs to know**

- **Phase 8's policy may condition on knowledge only.** Import
  `app.model.validation_gate`, call `load()`, and pass every state dict through
  `Gate.filter_states()`. Engagement and confidence must not appear in a policy
  input — not clipped, not down-weighted, absent. `test_study2.py::
  test_failed_state_is_dropped_not_zeroed` is the guard at the gate's end;
  BUILD.md Phase 7 acceptance asks Phase 8 for the mirror test on the policy.
- **RQ1's answer is "partly, and the negative results are the finding."**
  Engagement is estimable but not *valid* as an effort measure under Wise &
  Kong; confidence is not separable from knowledge on the probe density this
  project has. Write that into the paper rather than around it.
- **The observables still exist.** Excluding the engagement *state* does not
  exclude `solution_behaviour`, `rt_drift` or `matched_difficulty_speed_slope`
  as features. What is banned is conditioning a policy on a *latent construct*
  that failed its validity check.
- **`state-encoder.pt` carries its own preprocessing** — the column list and
  the training-fold median/IQR are inside the checkpoint, so a consumer cannot
  silently feed it differently-scaled inputs.
- **Phase 6 is not finished.** `study1-final.json` has never been written and
  `artifacts/benchmarks/study1-predictions.parquet` still contains test rows
  from a pre-fix run — `test_study1.py::test_cross_validation_never_scored_a_
  test_learner` fails on it today. Re-run `make train-baselines` to completion
  before quoting any Study 1 number.

---

## Phase 8 — Adaptive policies and closed-loop evaluation

**Headline: the decisions change and the outcomes do not.** `model_L4` disagrees
with `model_L0` on **39.3 %** of decisions taken on `model_L0`'s own trajectory,
and the two arms are indistinguishable on both registered outcomes — knowledge
gain differs by **+0.0009 [−0.0044, +0.0064]**, *d* = 0.006, Bonferroni
*p* = 1.0. That is RQ3's answer on this environment: richer *validated* state
moved the policy's hand and did not move the learner. It is a null, it is
well-powered (3,000 learners per arm per variant), and it is the result.

The secondary finding is that the **model family beats the psychometric and rule
arms, and does so at L0**. `model_L0` vs `mastery_threshold_bkt`: knowledge gain
**+0.0107 [0.0049, 0.0169]**, *d* = 0.066, *p*(Bonferroni) = 3.7 × 10⁻⁴;
items-to-mastery **−1.26 [−1.87, −0.71]**, *d* = 0.077, *p* = 8.6 × 10⁻⁹.
`model_L4` vs `rule_improved` on knowledge gain: **+0.0077 [0.0022, 0.0134]**,
*d* = 0.051, *p* = 7.2 × 10⁻³. Small, in the range the literature reports, and
attributable to acting on a **calibrated knowledge estimate inside the
desirable-difficulty band** — not to behavioural richness, which adds nothing on
top.

**This phase was interrupted and resumed.** `docs/phase-8-resume-audit.md` is the
audit written before restarting: the code and its tests were complete, the
experiment had never produced an artifact, and three defects were found while
restarting it. All three are recorded below.

### Built

- **`ml-service/app/policy/`** (five modules) — `rules.py` (the state-interaction
  table, one citation and one signal-class per rule), `policies.py` (all ten
  arms over one 120-action space), `state.py`, `online_features.py`,
  `explanations.py`. The package cannot import the simulator, and a test greps
  for it.
- **`app/research/closed_loop.py`** — the environment. Lockstep over a cohort,
  a six-concept prerequisite-closed curriculum, mastery judged only over concepts
  actually taught, interventions that extend the generative model at
  pre-registered magnitudes.
- **`app/research/evaluate_policies.py`** (`make eval-policies`) — the sweep, the
  paired tests, the effect-size halt, the propensity log, the L0-vs-L4 shadow
  divergence and the explanation traces.
- **`figures_phase8.py`** — the fourteen `artifacts/figures/08-policies/` charts
  plus `artifacts/evaluation/policy-figure-findings.json`, so no document quotes
  a picture.
- **`app/research/runprofile.py` + `make smoke|dev|full|status`** — the resource
  ladder BUILD.md §1.5 now requires (see *Resource controls* below).

### Verified — the run

`PROFILE=full make eval-policies`: 10 arms × 5 variants (V0–V4) × 3 seeds ×
1,000 learners, 200-item budget, ε = 0.1, **2,287 s at 5 workers on an
eight-core Docker allocation**, plus an 8-cell probe-noise sweep. The run was
executed twice — once at 6 workers before the thread cap below, once at 5 after
— and the two agree to the last reported digit, which is the reproducibility
claim this phase makes. Curriculum: `sql_basics`, `classes_and_objects`,
`arrays_and_lists`, `divide_and_conquer`, `sorting_searching`,
`inheritance_polymorphism`. Gate: **knowledge admitted, confidence and
engagement rejected**. 1,690 draws were screened per cell to seat 1,000 learners
— 41 % already had the whole curriculum mastered, exactly as pre-registered.

**V0, per arm** (`artifacts/evaluation/policy-results-v0.json`):

| Arm | Mastery | Items→mastery | Knowledge gain | Time (s) | P(success) |
|---|---:|---:|---:|---:|---:|
| `model_L0` | 0.091 | 187.2 | **0.0402** | 18,044 | 0.700 |
| `model_L3` | 0.087 | 187.6 | **0.0402** | 19,947 | 0.714 |
| `model_L4` | 0.088 | 187.6 | 0.0393 | 19,822 | 0.718 |
| `random` | 0.087 | 187.8 | 0.0394 | **34,672** | 0.693 |
| `model_L1` | 0.087 | 187.5 | 0.0377 | 17,847 | 0.711 |
| `rule_improved` | 0.082 | 188.3 | 0.0316 | 18,565 | 0.586 |
| `irt_cat_maxinfo` | 0.084 | 188.1 | 0.0304 | 20,483 | 0.581 |
| `mastery_threshold_bkt` | 0.083 | 188.4 | 0.0295 | 19,230 | 0.590 |
| `fixed_order` | 0.081 | 188.3 | 0.0273 | 16,569 | 0.494 |
| `rule_legacy` | 0.080 | 188.8 | 0.0273 | 19,630 | 0.537 |

- **The ladder is flat.** L0 → L1 → L3 → L4 knowledge gain: 0.0402, 0.0377,
  0.0402, 0.0393. Every pairwise contrast among the four has |*d*| ≤ 0.02 and a
  corrected *p* of 1.0. `f08-03` draws four overlapping intervals.
- **The decisions are not flat.** `f08-09`: 36,967 shadowed decisions, identical
  in 60.7 %; difficulty differs on 26.2 %, intervention on 23.0 %, concept move
  on 6.9 %; mean absolute difficulty gap 0.56 levels. The mechanism figure and
  the outcome figure disagree, and the honest reading is that the richer state
  changed *which* action was chosen without changing how much the learner
  learned.
- **Interventions are where the arms differ visibly.** Per learner: `model_L0`
  fires 4.7 hints and 9.0 worked examples; `model_L1`–`L4` fire ~18 hints
  (`hint_on_slow_decision` is an L1 rule); `L3`/`L4` add ~10.5 break
  suggestions (`break_on_within_session_decay` is L3). The L1 arms cannot fire
  the break rule at all, and the report records that exclusion by rung.
- **Success-probability band** (`f08-07`): the model arms land at 0.700–0.718
  against the 0.70–0.75 target — at the band's lower edge, not inside it. The
  psychometric arms sit at 0.58–0.59 and `fixed_order` at 0.49.
- **Robustness** (`f08-04`): `model_L0` ranks 1st or 2nd in **all five**
  variants (rank range 1) — the most stable arm in the set. `model_L4` ranges
  over ranks 1–5, `random` over 1–5, `rule_legacy` is last in four of five. No
  conclusion above depends on a single variant.
- **Subgroups** (`f08-11`): the effect concentrates in the low-prior-ability
  tercile (+0.82 items) and reverses in the high tercile (−0.61), as expected —
  and it is the *same* in all four ladder rungs (0.82 / 0.82 / 0.84 / 0.84), so
  even the subgroup structure is state-insensitive.
- **Probe noise** (`f08-14`): outcomes are **bit-identical** across noise levels
  0.25 / 0.55 / 1.0 / 2.0. The gate rejected the confidence head, so no policy
  reads a probe-derived state; an exactly flat curve is the positive control
  that says so.
- **Effect-size halt**: zero unexplained |*d*| > 3 in any of the five variants.
- **Propensity log**: `decision-log-v0.parquet`, 372,534 decisions, each with
  the probability the logging policy assigned to the action it took. Study 3
  cannot backfill this.
- **Tests**: 97 pass, 1 fails — `test_study1.py::test_cross_validation_never_
  scored_a_test_learner`, the known Phase 6 gap (below). All 19 Phase 8 checks
  pass, including that the policy package cannot import the simulator and that a
  gate-failed state is absent from a policy input rather than zeroed.

### Three defects found while resuming, and what was done

1. **The run was not reproducible.** Two runs of seed 20260821 gave different
   results. `ClosedLoop.__init__` iterated a `set` of concept ids while drawing
   each learner's per-concept knowledge from the RNG, so Python's per-process
   hash randomisation decided which concept received which draw. Fixed at the
   source (the set is sorted), `PYTHONHASHSEED=0` pinned in the image and the
   compose file as a standing guard, and
   `test_policies.py::test_cohort_does_not_depend_on_python_hash_seed` runs the
   cohort construction under two hash seeds and asserts they match.

2. **The difficulty action was a no-op on a third of the curriculum.** The
   selected curriculum contained `stacks_and_queues` — two items at an identical
   *b* — and two more concepts with three items, so "serve difficulty 3" and
   "serve difficulty 9" returned the same item and no arm could differ there.
   `select_curriculum` now requires ≥ 3 items spanning ≥ 1.0 logits of *b* per
   concept and skips any candidate whose prerequisite closure contains a concept
   below the floor, which is what its own docstring always said it should do.
   **This changed the curriculum**, so the earlier and later numbers are not
   comparable; the dev-profile run under the old curriculum gave the same
   qualitative picture (flat ladder, model arms ahead of rule arms, `random`
   competitive on gain and terrible on time). The bank's narrow difficulty range
   is now written into `docs/evaluation.md` §4.2 and BUILD.md Phase 12's
   limitations, because it bounds what any difficulty-selection result here can
   show.

3. **The effect-size halt fired on an arithmetic artefact.** `random` draws its
   intervention uniformly, so it suggests a 300-second break on a quarter of all
   items — 47.0 per learner against 4.7 under ε-greedy exploration — and its
   time-to-mastery is roughly double every other arm's *by construction*. That
   is |*d*| ≈ 3 on time-to-mastery and on nothing else. One documented exemption
   was added (`evaluate_policies.HALT_EXEMPT`); the comparison is still
   computed, still written to every report under `effect_sizes_explained`, and
   still drawn in `f08-02`.

### Resource controls added (BUILD.md §1.5)

The first attempt at this phase was abandoned because an uncapped run overheated
the machine, so the rerun did not simply restart at full size.

- **`runprofile.py`** defines `smoke` (2 workers / 40 learners / V0),
  `dev` (4 / 250 / V0–V1) and `full` (5 / 1000 / V0–V4). `smoke` is the default;
  `make all` cannot launch a full experiment as a side effect. Every knob has an
  `ALP_*` environment override; precedence is flag > environment > profile.
- **A worker cap alone does not bound CPU.** Six workers measured at **997 %**
  of an eight-core allocation: `torch.set_num_threads(1)` was set inside each
  cell, but numpy/BLAS was not, so every worker spawned a thread per core
  underneath itself. `OMP_NUM_THREADS` and its four siblings are now pinned to
  1 in the image, the compose file and the Makefile's `LIMITS`, and the same
  five-worker run then measured **502 %** — the number the profile actually
  asked for. Single-process fitting targets get the budget the other way round:
  `train-baselines` runs one process with `FIT_THREADS=5`, because pinning
  *that* to one thread makes it serial.
- **`make smoke` writes nothing.** It did, once, and a 40-learner run silently
  replaced the published `policy-results-*.json` and all fourteen figures. It
  now runs `--quick`, and `test_phase85.py` asserts every published report
  records the profile that produced it, so a downgraded artifact is visible
  rather than silent.
- **`make full` refuses to run** without `docs/full-run-justification.md`, which
  now records why the dev run was insufficient, the expected cost, the limits,
  the checkpoint strategy and the stop condition.
- **Every cell is checkpointed** to `artifacts/evaluation/cells/*.json.gz` as it
  finishes, keyed by the job *and* a hash of every module that can change its
  result — so an interrupted run resumes, and editing a policy invalidates the
  cache instead of silently reusing it.
- **`make status`** reads `artifacts/evaluation/run-log.jsonl` and reports
  profile, worker count, cells done, elapsed time and an estimate. Safe to call
  mid-run.
- The ladder actually walked: smoke (10 cells, 3.4 s) → dev (20 cells, 60 s) →
  full (150 cells, 2,115 s).

### Deliberately skipped

- **No bandit or RL arm.** `bandit_lin_ts` and `rl_ppo` are Phase 9's; the
  action space, the closed loop and the propensity log are built so they drop in.
- **No attempt to make the ladder win.** Tuning until L4 beats L0 is the exact
  behaviour the pre-registration exists to prevent. The null is reported.
- **No survival model for items-to-mastery.** 91 % of learners are censored at
  the 200-item budget, which is why knowledge gain is the co-primary; a proper
  time-to-event treatment is Phase 11's job and is now written into that phase.
- **`random`'s time cost is not "fixed".** It is what a uniform draw over an
  action space containing a 300-second break costs. Reporting it is more useful
  than hiding it behind a tuned floor arm.
- **No rescue of the two joint-state rules.** `hint_when_uncertain_but_engaged`
  and `ease_off_when_uncertain_and_disengaged` need confidence and engagement,
  which the gate rejected. They stay in the table, excluded with the gate's own
  reason attached to every explanation, because a rule silently deleted cannot
  be reported as a null.

### What the next phase needs to know

- **Phase 6 is still unfinished and it now blocks Phase 8.5.**
  `artifacts/benchmarks/study1-final.json` has never been written and
  `study1-predictions.parquet` still holds test rows from a pre-fix run, so
  `test_study1.py::test_cross_validation_never_scored_a_test_learner` is red.
  **No Study 1 number may be quoted until `make train-baselines` completes.**
  It is step 1 of Phase 8.5.
- **BUILD.md now has a Phase 8.5** between here and Phase 9: finish Phase 6,
  audit every component for scientific usability, put every trained model behind
  one artifact contract and one loader, and write the pipeline and
  policy-architecture docs. Phase 9 has been revised to reuse Phase 8's action
  space, arms, closed loop and propensity log rather than rebuild them, and to
  obey the profile ladder for PPO.
- **"The gated state" is knowledge plus the observables of the rung.**
  Engagement and confidence are absent from every policy input. A bandit or a
  PPO observation must go through `Gate.filter_states()` for the same reason.
- **The interesting Phase 9 question is now sharper.** Rule-based adaptation on
  a calibrated knowledge estimate already beats the psychometric baselines, and
  behavioural richness adds nothing on top. Whether a *learned* policy can
  extract something from the richer state that the hand-written rules could not
  is exactly what the bandit is for — and a negative answer there is as
  publishable as a positive one.

---

## Phase 9 — Study 3: offline policy evaluation, contextual bandits and RL

**Headline: a learned policy can beat the best rule on the outcome, lose to it
on the objective, and win for a reason the simulator was never calibrated to
support.** All three halves of that sentence are the finding; reporting only the
first would be the result this project exists not to produce.

Run at `dev` (250 learners per variant, 200-item budget, one seed, V0–V4;
50,000 PPO timesteps). `full` has not been run.

### RQ4 — rule versus learned adaptation

On V0, against `model_L0`, which `rl-results.json` picks as the best Phase 8 arm
on the co-primary:

| arm | knowledge gain | difference vs `model_L0` | mean reward / decision |
|---|---:|---|---:|
| `model_L0` (best rule) | 0.0412 | — | **0.00773** |
| `bandit_lin_ts` | **0.1219** | **+0.0807 [0.0535, 0.1085]**, *d* = 0.36, *p* = 1.2 × 10⁻⁶ | 0.00745 |
| `bandit_linucb` | 0.0480 | +0.0068 [−0.0132, +0.0269], *n.s.* | 0.00472 |
| `rl_ppo` | 0.0307 | −0.0105 [−0.0336, +0.0108], *n.s.* | 0.00613 |

Two qualifications, both load-bearing:

- **No learned arm beats the rule on the reward it was trained to maximise.**
  The pre-registered objective (§10.1) prices time; the bandit buys learning
  with time and comes out behind on the objective while ahead on the outcome.
- **Nothing moves items-to-mastery.** |*d*| ≤ 0.15 on the primary outcome and
  mastery rates sit at 11–15 % for every arm. The knowledge-gain difference does
  not convert into mastery inside a 200-item budget.

**Where the bandit's advantage comes from.** Interventions per learner:
`bandit_lin_ts` 36.6 hints, 51.7 worked examples, 59.5 breaks; `model_L0` 4.1
worked examples and nothing else. A worked example adds 2.0 logits of support at
full learning credit for 45 s (preregistration §9.3) — **constants that were
pre-registered, not fitted, because no archival source in this project records
an intervention**. The bandit found them and exploited them. That is a valid
statement about this environment and **not** evidence that hinting works, and
Phase 10 must not read it as one.

### RQ5 — generalisation

PPO trained on V0 only. Mean reward per decision: V0 0.0061, V1 0.0035, V2
0.0062, V3 0.0068, V4 0.0067, against `model_L0`'s 0.0077, 0.0065, 0.0079,
0.0071, 0.0065 — **it never beats the rule on any variant**, including the one
it trained on. On knowledge gain it is below `model_L0` on V0–V2 and above it on
V3–V4 (0.0445 vs 0.0284; 0.0493 vs 0.0250), which is a statement about V3/V4's
mis-specification rather than about the policy. `f09-08` is the figure.

The literature's standard failure mode is a policy that only works on its
training simulator. This one does not clearly work on *any* of them at 50,000
timesteps, which — per §10.5 — is reported rather than tuned away.

### Offline policy evaluation, and what it actually measured

45,884 logged decisions from `model_L4` at ε = 0.15 on V0, with propensities.
Against the truth the simulator can compute:

| estimator | mean absolute error |
|---|---:|
| SNIPS | **0.00147** |
| DR | 0.00442 |
| IPS | 0.00511 |

§10.3's pre-committed check (DR closer to truth than IPS) **passes**. That
self-normalisation beats doubly-robust here is reported as it stands.

**The methodological finding is about overlap, not estimators.** Evaluated as a
hard argmax, the trained targets matched the logged action on 0.2 % of decisions
and the ESS was **1.1–5.2 out of 2,383** — an estimate resting on one or two
decisions. Study 3 therefore evaluates each arm as it would be deployed
(ε-greedy at the logging ε) and clips weights at 20; the clipped ESS is
2,348–2,691 out of 45,884, or about 5 %. The largest raw weight is **681**: one
exploratory decision the target happened to agree with, outvoting six hundred
others. This is why every OPE number in this repository is printed with its ESS.

### Built

- **`app/policy/learned.py`** — the shared context vector (8 observables + the
  gated states, 11 features), the flattened 120-action index, disjoint LinUCB
  and linear Thompson sampling with Sherman-Morrison updates, and `PPOPolicy`.
  Nothing imports the simulator.
- **`app/rl/reward.py`** — the §10.1 reward, one definition for the environment,
  the bandits and the OPE reward model. Weights fixed before training and
  asserted against the preregistration by a test.
- **`app/rl/env.py`** — `ALPEnv`, a gymnasium environment that **is** the Phase 8
  closed loop with a cohort of one, not a second copy of the dynamics.
- **`app/research/ope.py`** (`make ope`) — the logged dataset with propensities,
  IPS/SNIPS/DR with ESS and weight diagnostics, online bandit training, and the
  on-policy ground-truth comparison.
- **`app/research/train_rl.py`** (`make rl`) — PPO with checkpoint/resume, the
  cross-variant evaluation, and the artifact contract under `artifacts/models/rl/`.
- **`figures_phase9.py`** — the ten `artifacts/figures/09-ope-rl/` charts plus
  `study3-figure-findings.json`, so no document quotes a picture.
- **`test_study3.py`** — the reward weights match §10.1; a gate-rejected state is
  absent from the observation vector; IPS recovers a known value on a synthetic
  log; and a **PPO run killed with SIGKILL restarts from its checkpoint**.

### Resource cost

`make ope` 61 s, `make rl` 225 s at `dev` on one core, single-threaded. PPO
training itself was 32 s of that; the rest is 20 closed-loop evaluation runs.
The `dev` PPO run **resumed from the `smoke` run's 2,000-step checkpoint** and
trained the remaining 48,000 — the resumability §1.5 asks for, exercised rather
than asserted.

### Two defects found when the terminal was interrupted mid-phase

- **`test_a_killed_ppo_run_resumes_from_its_checkpoint` was state-dependent.** It
  killed a run writing into the published `artifacts/models/rl/checkpoints/`,
  where the finished `dev` run's 50,000-step checkpoint already covers the 8,192
  timesteps the test asks for — so nothing trained, no checkpoint was written and
  the test failed for a reason unrelated to resumption, while also re-saving
  `policy.zip`. `train_rl.RL_DIR` now honours `ALP_RL_DIR` and the test trains
  into `tmp_path`. The published `policy.zip` was checked against
  `ppo-50000-steps.zip` (identical weights, 51,200 timesteps) and is intact; the
  stray `ppo-8192-steps.zip` the failing runs left behind was deleted.
- **`make status` crashed on a Phase 9 log line.** `scripts/run_status.py` assumed
  every `run-log.jsonl` entry is a Phase 8 cell with `done`/`total`; Phase 9 logs
  named stages. It now reads progress from the cell-shaped lines and prints the
  last stage line as the current event. §1.5 requires `make status` to be safe
  mid-run, so this was a live break of that rule.

**Test suite:** 114 passed on the host; `test_study1.py`'s two `pyBKT`/`pykt`
shim tests fail there only because those wheels are Python 3.11-only. Run
`make test` against the ml-service container for the green suite.

### What the next phase needs to know

- **The reward is simulation-only.** Its mastery term is the simulator's own
  response model. It reaches an agent as a reward and never as an observation,
  but a deployment has no such signal: Phase 10 needs an observable proxy, and
  every Study 3 number is conditional on this one.
- **`artifacts/models/rl/` follows the Phase 8.5 contract** (policy.zip,
  training_config, reward_spec, training_metrics, training_manifest), so Phase 10
  loads it the same way it loads every other artifact.
- **Do not tune to a win.** §10.5 fixed α, σ, the PPO defaults and the reward
  weights before any of these numbers existed. The losses above are results.

---

## Phase 10 — Live platform integration

**Headline: the running application now makes its decisions with the same
objects the experiments were run on — and the two only became the same system
once the platform stopped serving a different item bank.** Before this phase the
served policy could not have been the evaluated policy even in principle: the
arms act over concept ids, difficulty bins 1–10 and a calibrated *b*, and the
live bank had subject names and the word "easy".

### Built

- **`ml-service/app/policy/live.py`** — the serving layer. Each live session
  gets a **slot** in a fixed pool (`ALP_LIVE_SLOTS`, default 32) and the
  research arms are driven through the cohort contract they already have. The
  caller sends the session's whole attempt history with every request, so a slot
  that is new, recycled or lost to a restart replays it: the service is
  restartable without a session store. Nothing is refitted in a request.
- **`POST /v1/state`, `POST /v1/decide`, `GET /model-info`** — the gated state
  estimate with its MC-dropout standard error; the action (difficulty, concept
  move, intervention) with its propensity and its explanation; and which
  artifacts and which gate the process is serving.
- **Backend wiring** (`v1.controller.js`, `utils/adaptive.js`, `ml.service.js`)
  — `/v1/next-item` asks the arm and resolves the action to an item the same way
  `closed_loop.resolve` does; `/v1/attempts` posts the history to `/v1/state`
  and stores a `StateEstimate` whose `*_validated` flags come from the gate.
- **`app/research/export_bank.py` (`make bank`)** — the live bank, generated
  from `data/items/item-bank-v1.json` plus the Phase 1 calibration, carrying the
  curriculum `select_curriculum` derives: **60 items over 6 concepts**.
- **Learner-facing UI** (`files/`) — a state panel that labels its numbers as
  estimates and shows uncertainty, the plain-English reason for the item, the
  support the policy chose (hint, worked example, break), and the concept
  roadmap with mastery.
- **Researcher dashboard** — `GET /research/dashboard`, server-rendered from
  Postgres, with `?format=json` for the figure script. No new framework.
- **`scripts/loadtest.py` (`make loadtest`)** and **`figures_phase10.py`
  (`make system`)** — the deployed loop measured, and the six `10-system`
  figures drawn from the measurement.
- **Tests** — `backend/src/__tests__/liveLoop.test.js` drives consent → session
  → 20 items → end against a real database, and `app/policy/test_live.py` covers
  the serving layer.

### Verified

- **`docker compose up` → a 20-item session in a browser with zero console
  messages.** Driven through the real UI at `http://127.0.0.1:4000`; the
  screenshots in `artifacts/figures/10-system/screenshots/` are from that stack.
- **Latency.** Decision (`GET /v1/next-item`) p95 **7.9 ms** at concurrency 1 and
  **31.3 ms** at concurrency 16, against the 300 ms budget. Throughput 53 → 232
  decisions/s over the 1 → 16 sweep, no errors. Numbers from
  `artifacts/evaluation/load-test.json`; `docs/load-test-results.md` reads them.
- **Every decision carries a propensity** on both the model path (0.9008 greedy,
  0.0008 exploratory, ε = 0.1 over 120 actions) and the fallback path — asserted
  over all 20 rows by the E2E test.
- **No learner-facing explanation names a gate-rejected state**, asserted over
  both the API responses and the stored explanations. The UI shows engagement,
  confidence and fatigue as *withheld* with the gate's reason.
- **The fallback works and is visible.** With the ml-service unreachable the
  loop completes and the decision is logged as `rule_improved_fallback`; the
  E2E test asserts the served policy matches what `/model-info` reports, so a
  silent degradation fails the suite.

### Three things this phase changed rather than added

1. **`ADAPTIVE_POLICY` is gone.** Compose passed it to the backend, the
   ml-service read `ALP_POLICY`, and `question.controller.js` switched on the
   first — three places, two names, one switch. Setting `ADAPTIVE_POLICY`
   without `ALP_POLICY` now throws at startup rather than serving a different
   policy from the one the paper reports.
2. **The bank.** A `/v1` session ignores a client-supplied topic and always
   materialises the canonical bank. The frontend no longer chooses.
3. **The load test.** The old locustfile measured `POST /predict`, an endpoint
   the loop no longer calls, and locust was not in any requirements file. It was
   replaced by a standard-library driver that runs the real learner sequence.

### Deliberately skipped

- **Authentication on `/v1` and the dashboard.** Still a local pilot instrument
  with no recruitment and no personal data beyond a display name; the dashboard
  shows no answer keys. Add it before the platform is exposed off the machine.
- **Authored scaffolds.** A hint reveals the first sentence of the item's own
  explanation and a worked example reveals all of it. The bank has no authored
  hints, and inventing them would put unmeasured pedagogy into the live system —
  the pre-registered §9.3 intervention effects are simulation constants and are
  **not** claimed here (`ponytail:` marked in `utils/adaptive.js`).
- **Online learning.** The bandits serve from their saved θ and do not update
  from live rewards. A live reward needs the mastery signal the simulator has
  and a deployment does not — the same gap Phase 9 flagged for Phase 10.
- **A second ml-service worker.** The slot pool is per process. Correct with
  more workers (each rebuilds by replay), but untested, and one worker holds the
  measured p95 at 16 concurrent learners.

### What the next phase needs to know

- **`artifacts/evaluation/load-test.json` and `system-figure-findings.json`** are
  where every Phase 10 number lives. No document or figure states one that is
  not in those two files.
- **The live loop is a logging policy.** Its decisions are logged with
  propensities in the same shape Study 3 consumed, so a pilot's own log can be
  fed to `app/research/ope.py` without a backfill.
- **`ALP_LIVE_EPSILON`** overrides the served exploration rate and is reported by
  `/model-info`. It exists so a pilot can widen support deliberately; it was set
  to 0.95 only to capture the intervention screenshots, and the measured runs are
  all at the pre-registered ε = 0.1.


---

## Phase 11 — Statistical analysis and the figure pack

**Headline: with the correct effect size and the correct clustering, the
closed-loop result is a null.** No arm differs from `rule_improved` on the
uncensored co-primary — every |g| ≤ 0.012 with a CI straddling zero, nothing
significant raw or Holm-corrected, at 3,000 learners per arm. The paired *d*
this project could have reported instead is inflated by a factor the
learner-level model now names: **97.7 %** of the variance in knowledge gain is
between learners.

### Built

- **`app/research/stats.py` (`make stats`)** — unpaired Hedges' *g* with
  10,000-resample bootstrap CIs, Welch and Mann-Whitney beside each other,
  assumption checks, Holm within each outcome family, three families of
  mixed-effects models, a power analysis, and the censored treatment of
  items-to-mastery. Writes `artifacts/evaluation/statistical-report.json` and
  the seven CSV + LaTeX tables under `artifacts/evaluation/tables/`.
- **`app/research/sensitivity.py`** — the six analyses behind `f11-05`, into
  `artifacts/evaluation/sensitivity.json`. Five are recomputed from artifacts;
  only the mastery-threshold sweep reruns the closed loop, and it obeys the run
  profile.
- **`app/research/plotstyle.py`** — one style for every phase: Okabe–Ito
  palette (the old seaborn `deep` blue/green/red trio is the classic confusion
  set), 300 DPI, a vector PDF beside every PNG, and a provenance row appended to
  `artifacts/figures/manifest.jsonl` on every save.
- **`app/research/figure_index.py`** — renders `artifacts/figures/INDEX.md`
  from that manifest and fails if a figure declares no source, declares one that
  does not exist, or reads outside `artifacts/`, `data/` and the two checked-in
  inputs the Phase 1/2 schema figures illustrate.
- **`app/research/figures_phase11.py`** — the eight `11-stats` charts.
- **`make figures`** — regenerates all **104** figures across every phase from
  artifacts, then rebuilds the index. It depends on `stats`, so the numbers a
  figure draws are never older than the report it draws them from.
- **`app/research/test_stats.py`** — eight guards, including the paired-vs-
  unpaired fixture, Holm monotonicity, the 176-per-arm power constant, and the
  provenance check over the whole manifest.

### Verified

- **`make figures` runs the whole pack in 3 min 26 s**, container-side, with no
  manual step: 104 figures, PNG + PDF, provenance clean, INDEX.md written.
- **`make test` green** — 129 ml-service tests, including the eight new ones.
- **Effect sizes.** Knowledge gain: nothing significant, |g| ≤ 0.012.
  Items-to-mastery: nothing significant. Time-to-mastery: all nine arms differ
  from the reference, but only `random` (g = 1.77, i.e. random is slower)
  exceeds the g = 0.3 the literature reports; every other arm is |g| ≤ 0.27.
- **The behavioural rung survives clustering.** Crossed learner + item random
  intercepts, four datasets: the higher rung adds +0.015 to +0.024 in p(correct)
  per sd, CI excluding zero on three of four (ASSISTments 2009, *p* = 0.084).
  This is the replication BUILD.md §2 asks for, not a new claim.
- **Censoring is the story of items-to-mastery.** 91–92 % censored in every arm;
  restricted mean survival 187–189 items of a 200 budget; no log-rank test
  against the best arm comes close to significance.
- **Power.** The knowledge-gain comparisons ran at 0.05–0.07 achieved power for
  the effects observed — the sample is large, the effect is absent. A real RCT
  for g = 0.3 at 80 % power needs **176 per arm, 352 total**. That number goes
  in Phase 12's future-work section.
- **Sensitivity.** Ordering is stable across the item budget and probe noise;
  it **moves** across simulator variants, across τ, and at λ×4. Recorded as
  three findings, not smoothed over.

### Deliberately skipped

- **Crossed random effects on the full 745,192 rows.** statsmodels builds the
  variance-component design densely. The crossed fits are subsampled to ≤ 8,000
  rows / 200 learners / 200 items and say so in their own output; the
  learner-level models use every row. `ponytail:` in `stats.py` names lme4 or a
  Bayesian fitter as the upgrade.
- **A logistic mixed model for the per-item closed loop.** statsmodels' GLMM
  support does not cover crossed effects, so that model is a linear probability
  model and its coefficients are percentage points, stated in the output.
- **DR in the ε and reward-weight sweeps.** The reward model is fitted per run
  in Phase 9; the sweeps re-weight the same logged decisions and report IPS and
  SNIPS only.
- **Knowledge gain at a smaller item budget.** The stored outcome is a threshold
  crossing time, so a smaller budget is an exact re-cut for *mastery rate* and
  would need the loop rerun for anything else. The panel claims only the former.

### What the next phase needs to know

- **The headline is a null and the abstract has to say so.** BUILD.md §12.8 is
  not hypothetical here: on the co-primary outcome the arms are
  indistinguishable, and the honest paper reports that with the validated
  constructs and the sensitivity table rather than leading with
  time-to-mastery, where the only large effect is that random is slower.
- **Every number Phase 12 quotes exists in a file.** `statistical-report.json`,
  `sensitivity.json`, `stats-figure-findings.json` and
  `artifacts/evaluation/tables/*.csv`. The LaTeX twins are ready to `\input`.
- **`artifacts/figures/INDEX.md` is generated, not maintained.** Adding a figure
  means adding a `SOURCES` list to its script; the index and the provenance
  check follow automatically. A stale PNG shows up in the index labelled stale.
- **The RCT sample size is 176 per arm (352 total)** for g = 0.3 at 80 % power.

---

## Phase 12 — End-to-end verification, the system report and the paper draft

**Headline: the pipeline reproduces from a clean checkout, both PDFs build from
the artifact tree with every number substituted at build time, and the honesty
pass is a script rather than a claim.** `scripts/verify_final.py` walks every
phase's deliverables, every figure's provenance, every `{{artifact:…}}`
placeholder in the two documents, and every *typed* number in their prose — the
last of these is listed with the artifact it was read from, and an unlisted
number fails the run.

### Built

- **`scripts/build_pdf.py`** — markdown → HTML → PDF through `weasyprint`, one
  script for both documents. It resolves `{{artifact:path:json/pointer}}`
  placeholders out of `artifacts/` at build time and **fails** on a missing
  figure or an unresolvable pointer, so a document cannot quietly ship a stale
  number or a broken image.
- **`docs/reports/report.css`** — paged-media styling (A4, running header, page
  numbers, figure captions), shared by both PDFs.
- **`docs/reports/ALP-System-Report.md` → `.pdf`** — the product walkthrough:
  what the system is, architecture, feature-by-feature screens, what is captured
  and why, how decisions are made with a worked trace, how to run it,
  performance, test results, limitations, and the writing-guidance appendix.
- **`docs/paper-draft.md` → `docs/reports/ALP-Research-Paper-Draft.pdf`** —
  rewritten from scratch on the current results. Title and abstract lead with
  the null; §5 states the simulator's circularity in the body; §9 lists ten
  limitations; every `S`-tagged citation is marked as requiring verification.
- **`scripts/run_tests.py` (`make test-report`)** — runs both suites, parses each
  runner's own summary, and writes `artifacts/evaluation/test-results.json`.
  A failing suite is recorded as failing and exits non-zero.
- **`scripts/reproduce.py` (`make reproduce`)** — runs the `make all` chain one
  target at a time, timing each, writing `reproduction-timing.json` **as it
  goes** so an interrupted run still reports how far it got.
- **`scripts/verify_final.py`** — the four-part checklist above, into
  `artifacts/evaluation/final-verification.json`.
- **`app/research/figures_phase12.py`** — the five `12-walkthrough` charts.
  f12-03 and f12-04 draw a "not run in this checkout" panel rather than being
  skipped, because the report references them and a missing figure would fail
  the document build for a reason that is not a defect.
- **Docker image** — the pango/cairo system libraries weasyprint needs, plus
  read-write mounts for `docs/reports/` and read-only `scripts/`. `pip install
  weasyprint` alone imports fine and fails at render time; that is now fixed in
  the image rather than in a person's memory.

### Verified

- **Clean-room reproduction.** A fresh checkout (every tracked file plus the
  Phase 11/12 sources, an **empty** `artifacts/`, no `data/processed/`) with its
  own compose stack, driven by `scripts/reproduce.py --profile smoke`. Timings
  per target, wall clock:

  | Target | Phase | Seconds |
  |---|---|---:|
  | `items` | 1 | 4.0 |
  | `telemetry` | 2 | 1.8 |
  | `fetch-data` | 3 | 0.4 |
  | `prep-data` | 3 | 70.7 |
  | `simulate` | 4 | 163.8 |
  | `features` | 5 | 113.4 |

  Everything from the raw archives to the Phase 5 feature matrices regenerates
  in **5.9 minutes** with no manual step. `make train-baselines` — Study 1 — is
  the one target the run profile does **not** bound: it is capped by
  `MAX_TRAIN_LEARNERS`/`MAX_ROWS`, not by `PROFILE`, and it was still running
  when this section was written. That is a finding of the timing exercise and
  is recorded here rather than smoothed over: at `smoke`, `make all` is fast
  everywhere except Study 1, which takes tens of minutes regardless.

- **Both suites green.** 145 tests: the backend jest suite drives the live
  loop end to end against a real database, and the ml-service pytest suite
  carries the leakage audit, the gate guardrail, the policy conformance checks
  and the Phase 11 statistics guards.
- **Manual acceptance walkthrough.** The running stack was driven through a
  browser: landing → login → session → items → adapted explanation → roadmap →
  attempt history → researcher dashboard, with **zero console errors**. The
  eight screenshots are in `artifacts/figures/12-walkthrough/screenshots/` and
  the contact sheet is `f12-02`. The state panel showed knowledge with its
  uncertainty and the three rejected constructs as *withheld with the gate's
  reason*; the explanation moved from "no adaptation rule applied" to "recent
  struggle" as the learner's answers went wrong; the live decision log showed
  propensities of 0.9008 (greedy) and 0.0008 (exploratory).
- **Every number in both PDFs is traceable.** 27 are substituted from artifacts
  at build time; every other numeral in the prose is listed in
  `verify_final.KNOWN_PROSE` with the file it came from, and each was checked
  against that file in this phase.

### Deliberately skipped

- **A network re-download in the clean room.** `data/raw/` was hard-linked into
  the checkout, so `make fetch-data` **verified** the pinned SHA-256 of each
  archive (0.4 s) instead of pulling 792 MB again. The download path itself is
  unchanged and is exercised by anyone who does not have the files.
- **`make test-report` inside the clean room.** The backend suite needs
  `node_modules`, which a fresh checkout does not have until `npm install`; the
  suites were run in the main checkout instead and the clean-room report shows
  the "not run" panel, which is the honest state for that checkout.
- **A `full`-profile reproduction.** The timing run is at `smoke`. The published
  artifacts were produced at the profiles each phase's section records, and
  `reproduction-timing.json` says which profile it measured.
- **Pandoc.** `weasyprint` works; the documented fallback was never needed.

### What is left for a next reader

- **The `S`-tagged citations are unverified by construction.** The paper draft
  says so in its own front matter and in the system report's writing-guidance
  appendix. Verify each before submission.
- **The reference list is a stub.** It is the one section of the paper draft
  that is neither content nor template — it is a to-do.
- **`make report` needs `make test-report` first** for f12-03 to carry real
  numbers, and `make reproduce` for f12-04. Both degrade to a labelled panel
  rather than failing, and both are one command.
