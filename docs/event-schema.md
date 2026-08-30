# Event schema

The telemetry contract between the browser, the backend and the feature
extractor. Three artefacts implement it and must stay in step:

| Artefact | Role |
|---|---|
| `ml-service/app/schemas/events.py` | Pydantic models; the definition of record |
| `backend/prisma/schema.prisma` | storage |
| `files/telemetry.js` | the producer in the browser |

Figure [`f02-01`](../artifacts/figures/02-telemetry/f02-01_event-schema.png) draws the
storage side; [`f02-04`](../artifacts/figures/02-telemetry/f02-04_telemetry-timeline.png)
shows one attempt as a timeline.

## Envelope

Every event carries the same envelope:

| Field | Type | Notes |
|---|---|---|
| `event_id` | uuid | generated in the browser; the **idempotency key** for batch ingest |
| `session_id` | string | |
| `learner_id` | string \| null | display name; no personal data is collected |
| `item_id` | string \| null | bank id, e.g. `ds-e-1` |
| `attempt_id` | string \| null | back-filled server-side once the attempt exists |
| `type` | enum | see below |
| `client_ts` | ISO-8601 | browser clock |
| `server_ts` | ISO-8601 | set on ingest; the clocks are recorded separately rather than reconciled |
| `seq` | int | monotonic per session, assigned in the browser |
| `payload` | jsonb | type-specific, documented below |

## Types and signal classes

Each type belongs to one **signal class**. A class the learner has not consented
to is dropped at ingestion (`POST /v1/events` rejects it and says so in the
response) and is never recorded in the browser in the first place. `core` cannot
be switched off — without it there is no adaptive loop to consent to.

| Type | Class | Payload |
|---|---|---|
| `SESSION_STARTED` | core | `consent`, `user_agent` |
| `SESSION_ENDED` | core | — |
| `ITEM_PRESENTED` | core | `presented_at` |
| `ANSWER_SUBMITTED` | core | `selected_index`, `response_time_ms`, `option_change_sequence` |
| `ITEM_SKIPPED` | core | `response_time_ms` |
| `DECISION_MADE` | core | `decision_id`, `propensity`, `difficulty` |
| `FIRST_INTERACTION` | timing | `kind`, `ms_since_presented` |
| `OPTION_SELECTED` | interaction | `to_index`, `ms_since_presented` |
| `OPTION_CHANGED` | interaction | `from_index`, `to_index`, `change_number`, `ms_since_presented` |
| `HINT_REQUESTED` | interaction | `ms_since_presented` |
| `IDLE_ENTERED` | interaction | `threshold_ms` |
| `IDLE_EXITED` | interaction | `ms_since_presented` |
| `VISIBILITY_CHANGED` | interaction | `visible`, `focused`, `ms_since_presented` |
| `CURSOR_SEGMENT` | motor | the 13 aggregate trajectory fields (below) |
| `PROBE_SHOWN` | probes | `probe_id`, `kind`, `scale_points`, `assign_p`, `assign_draw` |
| `PROBE_ANSWERED` | probes | `probe_id`, `response`, `skipped` |

`VISIBILITY_CHANGED` records `visible` **and** `focused` separately so
idle-with-focus (thinking) and idle-without-focus (gone) stay distinguishable.

Option changes are recorded as an ordered sequence with timestamps, not a
count: the count is derivable, the order is not.

## CURSOR_SEGMENT payload

One per item attempt, emitted at submit. Raw pointer samples never leave the
browser — they live in a session-scoped buffer, are reduced to these numbers,
and are discarded. The definitions are pre-registered in
[`preregistration.md`](preregistration.md) and illustrated in
[`f02-03`](../artifacts/figures/02-telemetry/f02-03_trajectory-feature-definitions.png).

`auc_toward_nonchosen`, `max_deviation`, `x_flips`, `sample_entropy`,
`velocity_peak`, `velocity_mean`, `pause_count`, `path_ratio`,
`time_to_first_movement_ms`, `time_to_first_selection_ms`, `hover_time_ms[]`,
`n_samples`, `sample_interval_ms`.

## Storage

`InteractionEvent` is the append-only record. Two typed tables are **derived**
from it at ingest so the analysis has something to join against, and both can be
rebuilt from the event stream at any time:

- `CURSOR_SEGMENT` → `CursorSegment`
- `PROBE_SHOWN` / `PROBE_ANSWERED` → `Probe` (created, then updated)

`ItemAttempt` is the unit of analysis. Its `features` column holds the extractor
output verbatim, with `feature_version` alongside.

`Decision` records what the policy did: `policy_name`, `policy_version`,
`action`, the full `action_space` with each candidate's probability, and
`propensity`. **Propensity is not optional** — Study 3 is impossible without it
and it cannot be backfilled.

`StateEstimate` records the four states with a standard error and a
`*_validated` flag each. Nothing is marked validated before Phase 7 defines the
gate.

## Ingestion guarantees

- **Idempotent.** `POST /v1/events` accepts a batch and ignores any `event_id`
  already stored, including the derived `CursorSegment` and `Probe` rows. A
  client retry after a dropped response is safe.
- **Ordered by `seq`, not by arrival.** Events are buffered in the browser and
  flushed every 5 s, on submit, and on `pagehide`.
- **Consent-filtered at both ends.** The browser does not record a class it has
  no consent for; the server drops it again if it arrives anyway.

See [`api-reference.md`](api-reference.md) for the endpoints.
