# Adaptive Learning Platform: Repository Review

## Scope and review method

This review covers every tracked implementation source file in the frontend (`files/`), Node API (`backend/`), Python ML service (`ml-service/`), operational script, question-bank fixtures, and the embedded research instructions in `things/`. It is the required baseline before implementation work begins.

## High-level architecture

```text
Browser (static HTML/CSS/JS)
  ├─ POST /user/login ───────────────> Express API
  └─ POST /question/submit (intended) ─┘
                                        ├─ PostgreSQL via Prisma
                                        ├─ Gemini question generation
                                        └─ FastAPI ML service /predict
```

The intended API-session path is separate from the delivered browser path:

```text
Intended: topic -> generated question bank -> Session + Question rows ->
          behavioural event -> rule engine -> latent state -> next question

Delivered UI: static CSE question bank -> basic answer/time values ->
              incompatible API call -> local fallback difficulty
```

## Low-level design

| Component | Current responsibility | Status |
| --- | --- | --- |
| `files/` | Login, local CSE MCQ quiz, local statistics and review list | Functional as a demo, not connected to the ALP session model |
| `backend/src/controllers/question.controller.js` | Creates a generated question session, returns first question, updates state and returns next question | Partially implemented; API contract and question selection have defects |
| `backend/src/utils/ruleEngine.js` | Normalizes behavioural data, estimates five latent variables, selects difficulty | Executable and unit-tested, but heuristic and not research calibrated |
| `backend/prisma/schema.prisma` | Stores session summary and generated questions | Insufficient audit trail and no user, event, experiment, model, or concept entities |
| `ml-service/` | Returns next difficulty from three fields | Placeholder only; no trained model, persistence, metrics, or integration |
| `things/` | Research workflow and minimal templates | Useful requirements source, not a complete executable skill package |
| `backend/src/data/questions/` | Sample generated question-bank JSON | Fixtures only; not exposed as a reproducible dataset |

## Database schema

`Session` stores one topic-level session and the latest scalar values for mastery, confidence, engagement, cognitive load, fatigue, score counts, `asked_questions`, and JSON history. `Question` stores generated question content keyed by UUID and linked to a session.

The model does not persist learners, individual interaction events, answers, consent/privacy metadata, concept prerequisites, interventions, model version, experiment assignment, dataset version, or outcomes beyond the volatile session aggregate. JSON history prevents efficient analysis and reproducible feature reconstruction.

## API documentation: current contract

| Route | Intended request | Current response | Notes |
| --- | --- | --- | --- |
| `POST /user/login` | `{ username }` | display-name echo | No identity or persistence |
| `POST /question` | `{ topic }` | `{ session_id }` | Calls Gemini and stores generated questions; no schema validation or JSON recovery |
| `GET /question/start-session?session_id=...` | session ID | first easy question | Marks it asked |
| `GET /question/submit?session_id=...` | body containing `question_id` and features | state plus next question | A GET body is non-portable; UI instead issues `POST` without session or question identifiers |
| `POST /predict` (FastAPI) | `isCorrect`, `timeTaken`, `attempts`, `pastAccuracy` | `nextDifficulty` | Stateless hand-coded conditions, not ML |

## Current adaptive logic

The rule engine receives response time, reading/idle time, correctness, difficulty, attempts, skip, option changes, mouse and keyboard signals, question number, session duration, accuracy decay, and tab-switch data. It updates:

- knowledge from correctness and difficulty;
- confidence from response-time ratio and answer-changing patterns;
- engagement from tab switching, pointer activity and idleness;
- cognitive load from time, correctness, retries and typing friction; and
- fatigue from duration, question count, accuracy decay and response-time drift.

It then maps the state to easy, medium, or hard. The formulas have no documented provenance, calibration procedure, uncertainty estimate, concept-level mastery, action taxonomy, or causal safeguards. Correctness parsing is also unsafe for strings: a value such as `"false"` is truthy.

## Incomplete modules and technical debt

1. The frontend/API contract is broken (`POST` versus `GET`; different field names; missing `session_id` and `question_id`), so the demonstrated UI does not use the rule engine or PostgreSQL sessions.
2. `getQuestionsFromSession` filters with `session_id`, but `pickQuestion` uses the non-unique source `id`; repeated generated IDs across sessions are possible and questions are selected without deterministic ordering.
3. Submitted answers are not verified against `Question.correctAnswer`; the API trusts client-provided correctness.
4. The ML service has empty feature engineering, a three-condition pseudo-model, no training pipeline, no artifact versioning, and is not actually called by the Node session controller.
5. Empty controllers and service modules, unused imports, duplicate/deprecated difficulty logic, and a package test script that intentionally fails indicate incomplete migration.
6. There are no Prisma migrations, seed workflow, environment example, request validation, authentication, rate limiting, error taxonomy, API tests, or integration tests.
7. Question generation interpolates prompt text with a literal `"{TOPIC}"`, assumes model output is valid JSON, uses an unverified model name, and has no fallback to bundled fixtures.
8. Browser behavioural instrumentation does not collect the feature set described in the research prompt. It records only total response time and local answer state.
9. The frontend uses `innerHTML` to render question-derived attempt text, which is unsafe if question content comes from a remote generator.
10. The dev script is improved but the frontend README is stale and claims an old ML integration path.

## Scalability and reliability concerns

State is read, modified, and written without an optimistic version or event append, so concurrent submits can overwrite session state. Entire question banks and history are stored in relational/JSON hybrid records with no pagination. Synchronous LLM generation is a latency and cost bottleneck. There is no queue, cache, observability, data-retention policy, or tenant isolation. Behavioural telemetry needs explicit consent, minimization, and retention controls before any real-user study.

## Research limitations

The present code cannot answer the stated research question. It has no defined population or outcome, no pre-registered hypotheses, no experimental assignment, no static-curriculum baseline, no versioned synthetic data, no train/validation/test split, no leakage controls, no trained ML model, no model-comparison harness, no statistical testing, and no paper. Results generated solely by a simulator can establish internal benchmark behaviour, not real-world learning efficacy.

## Assumptions and design decisions

- Initial research development uses synthetic learners because no approved human-subject dataset is present.
- The first evaluation target is concept-level time-to-mastery under controlled simulated cohorts; it must be labelled as simulated evidence.
- Client telemetry is optional and privacy-minimized. Raw pointer trajectories and keystrokes are not persisted; only consented aggregates are accepted.
- The improved rule engine remains a standalone, deterministic baseline. This keeps it explainable and permits a fair ML comparison.
- API compatibility is preserved where safe through additive versioned routes; the broken legacy submit route should be deprecated rather than silently reinterpreted.

## Verification performed

- `node backend/src/utils/testRuleEngine.js`: passed all existing diagnostic assertions.
- Node syntax checks for the question controller and rule engine: passed.
- Python compilation of `ml-service/app`: passed.
- JSON validation of bundled question fixtures: passed.
