# Data Privacy Policy — Adaptive Learning Platform (ALP)

**Effective Date**: August 12, 2026  
**Version**: 1.1 (Draft for IRB / Legal Review — §2 rewritten for the Phase 2 instrument)  
**Study Target Cohort**: n = 30–50 undergraduate Computer Science students  
**Data Retention Period**: 12 months after study completion  

---

## 1. Overview and Purpose

This Data Privacy Policy outlines the practices governing the collection, storage, processing, access, and deletion of behavioral telemetry data within the Adaptive Learning Platform (ALP). The platform is designed to evaluate AI-driven and rule-based adaptive learning policies in computer science education while adhering strictly to participant privacy, data minimization, and self-hosted infrastructure security principles.

---

## 2. Telemetry and Data Collected

Rewritten for the Phase 2 instrument. The previous version of this section
listed nineteen features from the pre-Phase-0 pipeline, several of which are no
longer collected; the authoritative definitions now live in
[`event-schema.md`](event-schema.md) and [`preregistration.md`](preregistration.md).

### 2.1 Signal classes and consent

Collection is organised into five classes. **Each is independently switchable by
the learner on the consent screen, and the platform runs correctly with any
subset disabled.** That is not a courtesy: measuring outcomes under each
configuration is how research question RQ4 (privacy vs utility) is answered.

| Class | What is recorded | Optional |
|---|---|---|
| `correctness` | the chosen option, whether it was right, item difficulty, position in session | **No** — without it there is no adaptive practice |
| `timing` | response time, time to first interaction, decision latency, idle time before submit | Yes |
| `interaction` | option-change sequence, hint requests, idle spells, tab visibility and window focus | Yes |
| `motor` | aggregate cursor-trajectory descriptors (curvature, reversals, pauses, hover time per option) | Yes |
| `probes` | occasional self-reports of confidence, perceived difficulty and effort | Yes |

Choices are stored in the `Consent` table with a timestamp, and are enforced
twice: the browser does not record a class it has no consent for, and the server
rejects any event of that class if it arrives anyway.

### 2.2 Data minimisation

- **No raw pointer coordinates leave the browser.** Pointer positions are
  sampled at 50 ms into a session-scoped in-memory buffer, reduced at submission
  to the thirteen aggregate numbers listed in `event-schema.md`, and then
  discarded. No path, no replay, no screen recording is stored or transmitted.
- **No keystroke content, and no keystroke timing at all.** The item bank is
  100 % multiple-choice, so typing dynamics were dropped before any collection
  (see `preregistration.md` §2). No key event of any kind is recorded.
- **No personal data.** A learner is identified by a self-chosen display name.
  No email, institutional identifier, IP address or device fingerprint is
  stored; the user-agent string is recorded once per consent record so browser
  differences in pointer sampling can be accounted for.
- **Probes are skippable**, and a skip is stored as a skip rather than silently
  dropped.

### 2.3 What is derived and stored

`InteractionEvent` (the raw event log), `ItemAttempt` (one row per item, with
the extracted feature row), `CursorSegment` (the trajectory aggregate),
`Probe`, `StateEstimate` (model estimates of knowledge, engagement, confidence
and fatigue, each with an uncertainty and a validation flag), and `Decision`
(what the system did and with what probability).

**State estimates are estimates.** They are never presented to a learner as a
diagnosis, and until the Phase 7 validation gate passes they are not shown at
all.

---

## 3. Data Storage and Infrastructure Security

- **Self-Hosted Infrastructure**: All data collection, API routing, ML policy execution, and data storage occur exclusively on **on-premises / self-hosted infrastructure**. No telemetry or user data is transmitted to third-party cloud analytics services or external API providers.
- **Database Architecture**: Data is stored in a isolated PostgreSQL database.
- **Access Control**: Database access is strictly restricted to authenticated backend microservices via encrypted connections. Direct database access is restricted exclusively to primary research team personnel.

---

## 4. Retention Period and Data Lifecycle

- **Retention Window**: All telemetry records, interaction logs, and participant session data will be stored for exactly **12 months following study completion**.
- **Automated Deletion**: Upon reaching the 12-month expiration mark, all database entries and associated study logs will be permanently deleted and purged from backend backup storage.

---

## 5. Right to Withdraw and Data Deletion Process

Participants retain full autonomy over their data before, during, and after study completion:
- **Data Deletion Request**: Participants may request complete erasure of their stored telemetry and account records at any time.
- **Submission Process**: Requests can be submitted via email to the principal investigator / research team (`[PI Name and Email — placeholder]`).
- **Purge SLA**: Upon receiving a deletion request, all associated participant records in the PostgreSQL database will be completely purged within **30 days**.

---

## 6. Third-Party Sharing & Commercialization Policy

- **No Selling or Renting**: Collected data will **never** be sold, rented, leased, or commercialized.
- **No Third-Party Analytics**: Data is never shared with third-party advertising, analytics, or external AI model vendors.
- **Academic Reporting**: Dissemination of research findings (e.g., conference papers, journal submissions) will include only anonymized aggregate statistics without individual identifiers.

---

## 7. Contact Information

For questions, concerns, or data deletion requests regarding this policy:

- **Principal Investigator**: `[PI Name — Placeholder]`  
- **Email**: `[PI Email — placeholder]`  
- **Institution**: `[Department of Computer Science / University — Placeholder]`  
