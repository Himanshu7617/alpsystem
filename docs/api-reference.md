# Adaptive Learning Platform: API Reference

Base URL: `http://localhost:4000`  
ML Service URL: `http://localhost:8000`

---

## 1. User Endpoints

### `POST /user/login`
Authenticates or registers a learner session username.

**Request Body:**
```json
{
  "username": "student_01"
}
```

**Response (200 OK):**
```json
{
  "message": "Login successful",
  "user": {
    "username": "student_01"
  }
}
```

---

## 2. Research API (`/v1`) — Phase 2

The instrumented loop. Correctness is computed server-side from the stored
answer key, which is never sent to the client, and every served item is a logged
`Decision` carrying its propensity. Event and payload definitions:
[`event-schema.md`](event-schema.md).

These endpoints are unauthenticated: this is a pilot instrument with no
participant recruitment and no personal data beyond a display name. Phase 10
revisits that when the platform becomes the live system.

### `POST /v1/consent`
Records the learner's per-signal-class choices. Call before the first session.

```json
{ "learner_id": "student_01", "classes": { "timing": true, "interaction": true, "motor": false, "probes": true } }
```

**201:**
```json
{ "consent_id": "978f50ff-…", "classes": { "correctness": true, "timing": true, "interaction": true, "motor": false, "probes": true } }
```

`correctness` is always `true`; the field is ignored on input.

### `POST /v1/sessions`
Creates a session, materialises the canonical item bank into it, and writes the
`ExperimentAssignment` (arm, policy, seed).

The bank is the server's choice. A `topic` in the request body is recorded and
ignored: the served policy acts over the canonical bank's concept ids and
difficulty bins (`data/items/item-bank-v1.json`, exported to the backend by
`make bank`), and a session built from any other bank would log decisions the
experiments cannot interpret.

```json
{ "learner_id": "student_01", "topic": "multi-topic cse", "consent_id": "978f50ff-…" }
```

**201:** `session_id`, `seed`, `arm`, `policy_name`, `policy_version`, `consent`, `session_version`.

### `GET /v1/next-item?session_id=…`
Selects the next item, writes a `Decision` row, and returns the item **without
the answer key**.

**200:**
```json
{
  "item": { "item_id": "db-e-1", "stem": "…", "options": ["…"], "difficulty": "easy", "difficulty_score": 3, "concepts": ["sql_basics"], "format": "mcq" },
  "seq": 3,
  "decision_id": "…",
  "propensity": 0.9008,
  "policy": "rule_improved",
  "explanation": "Serving a difficulty-3 item on sql basics because recent struggle.",
  "intervention": { "kind": "hint", "title": "A hint before you answer", "body": "…" },
  "target_concept": "sql_basics",
  "roadmap": { "concepts": [{ "concept_id": "sql_basics", "name": "SQL Fundamentals", "status": "in_progress", "mastery": 0.29 }] },
  "fallback": null,
  "probes": [{ "kind": "confidence", "scale_points": 4, "assign_p": 0.12, "assign_draw": 0.07 }],
  "consent": { "…": true },
  "session_version": 5,
  "total_items": 60,
  "items_asked": 3
}
```

**Since Phase 10 the item is chosen by the served research arm**, not by the
backend's own ranking. The backend posts the session's decision context to the
ml-service's `POST /v1/decide`, receives an action — difficulty bin, concept
move, intervention — with its propensity and its explanation, and resolves that
action to an item the same way the closed-loop environment does: move to the
target concept, skip concepts already mastered, take the unasked item whose
difficulty bin is closest. Which arm serves is `ALP_POLICY`
(`rule` | `bandit` | `rl`, or any Phase 8 arm key), and `propensity` is that
arm's own probability for the action it took.

`intervention` is `null` unless the action carried one. `fallback` is `null` on
the normal path and names the reason when the ml-service could not be reached:
the loop then falls back to the Phase 2 rule engine, logs the decision under
`rule_improved_fallback`, and says so in the response. **A learner is never
blocked on a model.**

**409** if the session changed concurrently — re-read and retry.

### `POST /v1/events`
Batch ingest. Idempotent on `event_id`, including the derived `CursorSegment`
and `Probe` rows.

```json
{ "session_id": "…", "learner_id": "student_01", "events": [ { "event_id": "…", "type": "OPTION_CHANGED", "item_id": "ds-e-1", "client_ts": "2026-08-22T10:00:15.000Z", "seq": 8, "payload": { "from_index": 0, "to_index": 1, "change_number": 1 } } ] }
```

**202:**
```json
{ "received": 12, "stored": 12, "rejected": [], "session_version": 5 }
```

Events of a signal class the learner declined appear in `rejected` with the
reason and are not stored.

### `POST /v1/attempts`
Submits an attempt. Computes correctness server-side, links this item's events
to the new attempt, calls the ML service's extractor, stores the feature row,
the rule-engine `StateEstimate`, and advances the session under optimistic
concurrency.

```json
{ "session_id": "…", "item_id": "ds-e-1", "selected_index": 1, "presented_at": "2026-08-22T10:00:00.000Z", "response_time_ms": 18500, "session_version": 5 }
```

**201:**
```json
{ "attempt_id": "…", "seq": 3, "correct": true, "correct_index": 1, "explanation": "…", "state": { "knowledge": 0.31, "confidence": 0.55, "engagement": 0.79, "cognitive_load": 0.22, "fatigue": 0.08 }, "recommendation": { "difficulty": "medium", "action": "…", "reasons": ["…"] }, "features_extracted": true, "session_version": 6 }
```

**409** on a stale `session_version`. If the ML service is unreachable the
attempt still succeeds with `features_extracted: false` — a learner is never
blocked on the feature pipeline.

### `POST /v1/sessions/end`
`{ "session_id": "…" }` → `{ "session_id": "…", "ended": true }`.

---

## 3. Legacy question endpoints (`/question`)

Pre-Phase-2 path, retained for the existing Jest suite and the rule-engine
experiments. It requires a JWT, returns the answer key to the client, and logs
no propensities, so **it is not used for research data**. Phase 10 collapses it
into `/v1`.

### `POST /question`
Initializes a new adaptive learning session and generates/stores the topic question bank.

**Request Body:**
```json
{
  "topic": "multi-topic cse"
}
```

**Response (201 Created):**
```json
{
  "message": "Questions fetched and session created successfully",
  "session_id": "c7a8b9f0-1234-4567-89ab-cdef01234567"
}
```

---

### `GET /question/start-session`
Retrieves the first diagnostic question (easy level) for an initialized session.

**Query Parameters:**
- `session_id` (string, required): The UUID of the session.

**Response (200 OK):**
```json
{
  "question": {
    "question_id": "q-uuid-101",
    "id": "ds-easy-1",
    "difficulty": "easy",
    "questionType": "mcq",
    "question": "Which data structure follows LIFO?",
    "options": ["Queue", "Stack", "Graph", "Hash table"],
    "correctAnswer": "Stack",
    "explanation": "A stack removes the most recently inserted item first.",
    "estimatedTimeSeconds": 30,
    "concepts": ["Data Structures", "Stack"]
  }
}
```

---

### `POST /question/submit`
Submits an answer along with behavioral telemetry. Updates the 5-dimensional latent state and returns the next adaptive question and roadmap recommendation.

**Request Body:**
```json
{
  "session_id": "c7a8b9f0-1234-4567-89ab-cdef01234567",
  "question_id": "ds-easy-1",
  "selected_answer": "Stack",
  "timeTaken": 14.5,
  "readingTime": 3.2,
  "timeAfterLastInteraction": 1.1,
  "attempts": 1,
  "option_changes": 0,
  "mouse_distance": 450.2,
  "mouse_speed": 85.0,
  "hover_time": 8.0,
  "typing_speed": 40.0,
  "backspaces": 0,
  "delete_frequency": 0,
  "pause_duration": 0.5,
  "questionNumber": 1,
  "sessionDuration": 14.5,
  "tab_switches": 0
}
```

**Response (200 OK):**
```json
{
  "next_question": {
    "question_id": "q-uuid-102",
    "id": "ds-medium-1",
    "difficulty": "medium",
    "question": "What is average search complexity in a Hash Table?",
    "options": ["O(1)", "O(log n)", "O(n)", "O(n log n)"],
    "correctAnswer": "O(1)"
  },
  "topic_mastered": false,
  "mastery": 0.15,
  "is_correct": true,
  "current_difficulty": "medium",
  "student_state": {
    "knowledge": 0.15,
    "confidence": 0.60,
    "engagement": 0.83,
    "cognitive_load": 0.20,
    "fatigue": 0.01
  },
  "recommendation": {
    "policy_version": "rule-v1.2",
    "action": "advance",
    "difficulty": "medium",
    "reasons": ["knowledge_band"]
  },
  "learning_roadmap": {
    "roadmap_version": "concept-roadmap-v1",
    "target_concept": "Data Structures",
    "action": "practice_target_concept"
  }
}
```

---

## 4. ML Service Endpoints

### `POST /v1/features/extract` (FastAPI Service on port 8000)

Raw events → the per-attempt feature row. The single definition of every
feature (`ml-service/app/features/extract.py`); offline training imports the
same module. Request: `{ consent, item, attempt, events[] }` (see
`app/schemas/events.py`). Response:

```json
{ "feature_version": "v1-phase2", "features": { "correct": 1, "total_response_time": 18.5, "max_deviation": 61.3, "…": null }, "available_classes": ["correctness", "session", "timing", "interaction", "motor"] }
```

A feature whose signal class the learner declined is `null`, never `0`.

### `POST /v1/state` (FastAPI Service on port 8000)

The session's attempts so far → the **gated** multi-state estimate with its
uncertainty. The caller sends the whole history; the service replays whatever
its slot has not seen, so an estimate survives a restart of the service without
a session store.

```json
{ "session_id": "…", "attempts": [ { "session_id": "…", "item_id": "db-e-1", "concept_id": "sql_basics",
  "correct": true, "difficulty_score": 2, "response_time_ms": 12000, "item_b": -1.16,
  "timestamp": 1730000000000, "position": 0, "question_number": 1, "features": { "…": 0 } } ] }
```

**200:**
```json
{
  "states": { "knowledge": 0.628 },
  "standard_error": { "knowledge": 0.010 },
  "rung": "L4",
  "steps": 1,
  "gate": { "admitted": ["knowledge"],
            "rejected": { "engagement": "engagement failed C3_effort_independent_of_ability",
                          "confidence": "confidence failed C1_label_correlation, C4_discriminant_validity",
                          "fatigue": "construct dropped before modelling (preregistration §7)" } }
}
```

A state Phase 7's gate rejected is **absent** from `states` — dropped, not
zeroed — and the reason travels with the response so the platform can say why
rather than showing a blank.

### `POST /v1/decide` (FastAPI Service on port 8000)

Decision context → the action, its propensity and its explanation. The action is
the shared Phase 8 action space, not an item id: which item realises a
difficulty in a concept is the platform's bank lookup, exactly as it is the
environment's in the closed loop.

```json
{ "session_id": "…",
  "view": { "position": 3, "session_position": 3, "active_concept": "sql_basics",
            "concept_attempts": 2, "outcomes": [1, 0, 0], "incorrect_streak": 2,
            "curriculum": ["sql_basics", "…"] },
  "attempts": [ "… optional: warms a cold slot in one round trip" ] }
```

**200:**
```json
{ "difficulty": 1, "concept_move": "prerequisite_concept", "intervention": "hint",
  "propensity": 0.0008, "policy": "rule_improved", "explored": true, "rung": "L1",
  "action_index": 45, "action_space": 120,
  "explanation": { "rules_fired": [{ "rule": "recent_struggle", "citation": "rule-v1.2" }],
                   "excluded_by_gate": { "low_engagement": "engagement head failed C3 …" } } }
```

`excluded_by_gate` names the rules the validation gate disabled. They are
reported rather than dropped silently, because a system that quietly stops
applying a rule cannot be audited for having done so.

### `GET /model-info` (FastAPI Service on port 8000)

Which artifacts this process is serving, and what the gate allows: the arm, its
rung, ε, the estimator rung, the admitted and rejected states, and the live slot
pool's occupancy. `GET /health` remains the liveness probe.

### `POST /predict` (FastAPI Service on port 8000)
Scores item difficulty candidates against a fitted classifier artifact.

**Request Body:**
```json
{
  "isCorrect": true,
  "timeTaken": 14.5,
  "attempts": 1,
  "pastAccuracy": 0.80,
  "difficulty_score": 2,
  "knowledge_before": 0.10,
  "fatigue_before": 0.01,
  "total_response_time": 14.5,
  "reading_time": 3.2,
  "time_after_last_interaction": 1.1,
  "skip": false,
  "option_changes": 0,
  "mouse_distance": 450.2,
  "mouse_speed": 85.0,
  "hover_time": 8.0,
  "typing_speed": 40.0,
  "backspaces": 0,
  "delete_frequency": 0,
  "pause_duration": 0.5,
  "question_number": 1,
  "session_duration": 14.5,
  "tab_switches": 0
}
```

**Response (200 OK):**
```json
{
  "nextDifficulty": "medium",
  "predictedSuccess": 0.732,
  "policy": "ml",
  "model": "catboost"
}
```
