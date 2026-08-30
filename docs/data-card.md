# Data Card

## Item bank `item-bank-v1`

- **Bank**: `data/items/item-bank-v1.json`
- **Manifest**: `data/items/item-bank-v1.manifest.json`
- **Item parameters**: `artifacts/datasets/item-parameters-v1.csv`
- **Item count**: 186
- **Concept graph**: `cse-prerequisite-graph-v1` (36 concepts, asserted acyclic)
- **SHA-256 of the bank**: `c95bec4ec987584b7a9ae649dd26596eac6a7dfe1ab5914c60054ebac39ad451`
- **Seed**: `20260821`

Rebuild with `make items`. The target is idempotent: it rewrites the bank, the
manifest, `item-parameters-v1.csv` and the four Phase 1 figures from the source
question files, and a second run produces byte-identical output.

### Per-item schema

| Field | Type | Description |
|---|---|---|
| `item_id` | String | Unique; topical banks are namespaced (`bt-`, `rec-`) because they all number their items `q1..q30` |
| `concept_id` | String | A concept in `cse-prerequisite-graph-v1` |
| `subject` | String | Inherited from the concept |
| `stem` | String | Question text |
| `options` | String[4] | Answer options, order as authored |
| `correct_index` | Integer | Index into `options` |
| `explanation` | String | Shown after a response |
| `author_difficulty` | Integer | Author-asserted, 1–10. A *prior*, not an estimate |
| `prerequisite_concepts` | String[] | Copied from the concept's graph prerequisites |
| `format` | String | `mcq` for every item currently in the bank |

### Provenance and merge rule

Sources are merged in a fixed order and the first item to claim a normalised
stem hash (lowercase, punctuation stripped, whitespace collapsed, SHA-256)
wins, so the merge is deterministic.

| Source | Items kept |
|---|---|
| `multitopic_cse.json` | 126 |
| `question_bank.csv` | 0 |
| `binary_trees.json` | 30 |
| `recursion.json` | 30 |
| `transportability_among_animals.json` | 0 |

- **126 items dropped as duplicate stems.** `question_bank.csv` is a
  flattened export of `multitopic_cse.json`; every one of its rows collides with
  the richer JSON record, which is listed first and therefore wins.
- **30 items excluded**, all from
  `transportability_among_animals.json`. It is a biology bank and no concept in
  the CSE prerequisite graph fits it. Excluding it is deliberate: labelling
  those items with a CSE concept to keep the count up would corrupt every
  downstream mastery estimate. The file is retained under
  `backend/src/data/questions/` and the excluded ids are listed in the manifest.

### Concept labelling

126 items carry an explicit `question_to_concept` mapping in
`backend/src/data/concept_graph.json`. The topical banks label items with free
text instead, so they are resolved by a small keyword table, then by a
per-source topic default (`binary_trees.json` → `trees_and_bst`,
`recursion.json` → `divide_and_conquer`).

| Keyword in `concepts`/`tags` | Concept |
|---|---|
| `memoization` | `dynamic_programming` |
| `dynamic programming` | `dynamic_programming` |
| `backtracking` | `advanced_algorithms` |
| `quick sort` | `sorting_searching` |
| `dfs` | `graph_algorithms` |
| `graph algorithms` | `graph_algorithms` |
| `time complexity` | `complexity_analysis` |
| `space complexity` | `complexity_analysis` |

### Known limitation — concept coverage

24 of 36 concepts hold fewer than 5 items. `make items` prints the full
coverage table and warns; `validate_bank` does **not** fail on it, because the
shortfall is a property of the bank rather than a bug. Mastery estimates on
these concepts are not trustworthy and any per-concept result must say so.

| Concept | Subject | Items |
|---|---|---|
| `congestion_control_net` | Computer Networks | 1 |
| `distributed_databases` | DBMS | 1 |
| `graphs` | Data Structures | 1 |
| `greedy_algorithms` | Algorithms | 1 |
| `application_protocols` | Computer Networks | 2 |
| `deadlocks` | Operating Systems | 2 |
| `normalization` | DBMS | 2 |
| `stacks_and_queues` | Data Structures | 2 |
| `transactions_acid` | DBMS | 2 |
| `advanced_algorithms` | Algorithms | 3 |
| `arrays_and_lists` | Data Structures | 3 |
| `design_patterns` | OOP | 3 |
| `dynamic_programming` | Algorithms | 3 |
| `hash_tables` | Data Structures | 3 |
| `heaps` | Data Structures | 3 |
| `indexing_optimization` | DBMS | 3 |
| `ip_addressing` | Computer Networks | 3 |
| `network_security` | Computer Networks | 3 |
| `routing` | Computer Networks | 3 |
| `scheduling` | Operating Systems | 3 |
| `synchronization` | Operating Systems | 3 |
| `transport_protocols` | Computer Networks | 3 |
| `concurrency_control` | DBMS | 4 |
| `solid_principles` | OOP | 4 |

### Difficulty calibration — currently a placeholder

`artifacts/datasets/item-parameters-v1.csv` holds `item_id, b, se_b,
n_responses, source, difficulty_bin`.

**0 of 186 items are Rasch-calibrated, and Phase 3 did not change that.**
Phase 3 has now run and re-ran `make items` afterwards (BUILD.md Phase 3, step
5). No archival skill names a bank concept — ASSISTments is middle-school
mathematics, EdNet KT1 is TOEIC English, and the bank is undergraduate computer
science — so `data/processed/bank-responses.csv` was deliberately not written
and every item stays `source = author_prior`: `b` is the z-scored
`author_difficulty` and `se_b` is empty. The overlap test and its verdict are in
`artifacts/datasets/archival-bank-concept-map.json`.

Real calibration therefore waits on responses to *these* items, which only the
live platform (Phase 10) or a future deployment can produce. The placeholder is
labelled as one on figure `f01-03`, and any per-item difficulty claim must say
that it rests on an author prior.

The 1PL model is `logit P(correct) = theta_learner - b_item`, fitted by
L2-regularised logistic regression over one-hot learner and item dummies. The
recovery check in `ml-service/app/research/test_item_bank.py` fits 400
simulated learners against 20 known difficulties and asserts Spearman > 0.9.

`b` maps to the 10-point `difficulty_score` the policies consume by equal-width
bins over the bank's `b` range. Edges (from the manifest): -1.1926, -0.8745, -0.5564, -0.2383, 0.0798, 0.3979, 0.7160, 1.0341, 1.3522. Under the
author prior this mapping is exactly the identity on `author_difficulty`.

---

## Archival learner data (Phase 3)

Every offline claim in this project is anchored on real learner data. Three
sources are fetched by `make fetch-data`, verified against pinned SHA-256
digests, and normalised by `make prep-data`. Nothing below is hand-typed: the
numbers come from `artifacts/datasets/archival-summary.json`, and the raw
digests from `data/raw/manifest.json`.

### Provenance and licence

| Source | File | Licence | Citation |
|---|---|---|---|
| ASSISTments 2009–2010 skill builder | `data/raw/assistments-2009-2010-skill-builder.csv` (83.2 MB) | Released for research use by the ASSISTments project (WPI) | Feng, Heffernan & Koedinger (2009) |
| ASSISTments 2012–2013 with affect predictions | `data/raw/assistments-2012-2013.zip` (576.4 MB) | Same; mirror published under MIT | Feng, Heffernan & Koedinger (2009), 2012–13 release |
| EdNet KT1 (shard 0 of 11) | `data/raw/ednet-kt1-shard0.parquet` (170.7 MB) | CC BY-NC 4.0 (Riiid Labs) | Choi et al. (2020) |
| EdNet question metadata | `data/raw/ednet-questions.parquet` (0.3 MB) | CC BY-NC 4.0 (Riiid Labs) | Choi et al. (2020) |

The canonical hosts (a Google Sites page for ASSISTments, a Google Drive folder
for EdNet) serve no stable scriptable URL, so `scripts/fetch_data.py` pins an
immutable mirror revision per source and prints the canonical manual-download
instructions on any failure. **It never falls back to synthetic data**, and a
SHA-256 mismatch exits non-zero.

**Junyi Academy is not fetched.** BUILD.md lists it as optional; its
contribution would be prerequisite structure, which the bank's own 36-concept
DAG already supplies. Recorded here as a deliberate omission, not an oversight.

### Normalised schema

`data/processed/<source>.parquet`, one row per interaction:

| Field | Notes |
|---|---|
| `learner_id`, `item_id`, `skill_id` | Namespaced per source (`a09_`, `a12_`, `ednet_`) so tables can be concatenated without id collisions |
| `timestamp` | Epoch ms. **Null for ASSISTments 2009–2010**, which ships no wall-clock column at all |
| `order_index` | Position within the learner's sequence — the ordering every source does have |
| `correct` | 0/1 |
| `response_time_ms` | Winsorised at the per-item 1st/99th percentile |
| `response_time_ms_raw` | Pre-winsorisation, kept so the effort label and figures read the untouched value |
| `lag_time_ms` | Gap since the previous interaction ended. Null for 2009–2010 |
| `attempt_count`, `hint_count` | Null for EdNet KT1, which records neither |
| `solution_behaviour`, `learner_rte` | Wise & Kong effort label, below |
| `source`, `skill_name` | Provenance and the human-readable skill label |

### Per-source result

| | ASSISTments 2009 | ASSISTments 2012 | EdNet KT1 |
|---|---|---|---|
| rows read | 525,534 | 6,123,270 | 8,662,979 |
| interactions kept | 253,686 | 2,593,636 | 4,897,531 |
| learners | 2,968 | 22,422 | 23,916 |
| items | 15,879 | 46,901 | 12,228 |
| skills | 111 | 265 | 1,785 |
| mean sequence length | 85.5 | 115.7 | 204.8 |
| accuracy | 0.659 | 0.699 | 0.653 |
| response time present | 100 % | 100 % | 100 % |
| lag time present | 0 % | 89.5 % | 99.5 % |
| timestamp present | 0 % | 100 % | 100 % |

### What was dropped, and why

| Rule | 2009 | 2012 | EdNet |
|---|---|---|---|
| multi-skill duplicate rows (one row per skill for the same response) | 178,674 | — | — |
| scaffolding rows (`original == 0`) | 71,402 | 303,533 | — |
| rows with no skill label | 16,059 | 3,196,113 | — |
| unparseable start time | — | 0 | — |
| negative response time set to null | 5 | 0 | 0 |
| learners with < 10 interactions | 5,713 rows / 1,195 learners | 29,988 rows / 6,576 learners | 102,290 rows / 18,498 learners |

The 3.2 M rows dropped from 2012 for a missing skill label are a property of
that release, not a filtering bug: over half its problem logs carry no skill.

**EdNet subsampling rule.** BUILD.md caps this source at ~5 M interactions. Only
shard 0 of 11 is downloaded (8.66 M interactions of the ~131 M released), then
whole learners are drawn without replacement in a seed-20260821 permutation
until the next learner would cross 5,000,000 — 42,414 of 72,351 learners,
4,999,821 interactions, before the < 10-interaction filter. **Sequences are
never truncated**: the cap is applied at learner granularity so within-learner
history stays intact, which is what the leakage-safe split protocol requires.

### Winsorisation

Response time is clipped to the 1st/99th percentile **of its own item**. Items
with fewer than 20 responses fall back to the source-level percentiles — a 1st
percentile estimated from five observations is noise.

| | 2009 | 2012 | EdNet |
|---|---|---|---|
| rows clipped | 9,195 (3.63 %) | 83,048 (3.20 %) | 92,832 (1.90 %) |
| items using source-level bounds | 11,612 | 19,287 | 492 |
| source-level bounds (ms) | 1,469 – 364,091 | 2,932 – 383,053 | 1,000 – 102,000 |

### Response-time effort (Wise & Kong)

Per-item normative threshold: `clip(0.10 × item mean response time, 1 s, 10 s)`.
A response faster than its item's threshold is rapid-guessing behaviour; a
learner's RTE is the proportion of their responses that are solution behaviour.
Figure `f03-05` draws the distribution with the 0.90 disengagement line.

| | 2009 | 2012 | EdNet |
|---|---|---|---|
| median item threshold | 3.59 s | 4.63 s | 2.39 s |
| rapid-guess rate | 4.24 % | 2.17 % | 1.83 % |
| learners below RTE 0.90 | 434 | 1,012 | 1,363 |
| rapid-guess accuracy | 0.210 | 0.327 | 0.543 |
| solution-behaviour accuracy | 0.678 | 0.708 | 0.655 |

**The chance-level check applies to EdNet only.** It is the one source whose
release records an option count (4). ASSISTments is dominated by algebra and
fill-in items and stores no options at all — its `answer_text` column is empty
for `choose_1` items — so guessing has no defined chance level there. Its
rapid-guess accuracy (0.21 and 0.33 against solution behaviour's 0.68 and 0.71)
is reported as descriptive evidence, not as a pass.

**Recorded deviation — EdNet fails the chance-level acceptance check.** Pooled
rapid-guess accuracy is 0.543 against a chance level of 0.25. No response-time
cut fixes it: accuracy by response-time bucket bottoms out near 0.50 and rises
again below 1 s. Split by learner ability quartile the picture resolves — the
lowest quartile's rapid responses score **0.339**, inside the ±0.10 band, while
the highest quartile's score **0.811**. Rapid responses in self-study TOEIC
preparation are a mixture: guessing by weak learners, and fluent learners
answering fast *because they know the answer*. That is the assumption Wise &
Kong's index makes and this population violates. `prep_archival.py` records the
deviation with its evidence (`RAPID_GUESS_DEVIATIONS`) rather than deleting the
check, `rte_valid_pooled` is `false` for this source in the manifest, and
figure `f03-06` shows both panels. **Phase 7 must not use EdNet RTE as an
engagement criterion without conditioning on ability; ASSISTments carries that
criterion.**

### Mapping to the item bank — no overlap

`artifacts/datasets/archival-bank-concept-map.json` tests every archival skill
name against the 36 bank concept names (case- and punctuation-normalised exact
match). **Zero of 2,161 archival skills match.** The subject domains do not
overlap, so no archival response can calibrate a bank item, and
`data/processed/bank-responses.csv` is deliberately not written. See the
difficulty-calibration section above.

### Shortfalls, stated plainly

- ASSISTments 2009–2010 has **no timestamps**, so no lag time, no session
  segmentation and no within-session fatigue analysis on that source.
- EdNet KT1 has **no attempt or hint counts**, so the attempt-number curve
  (`f03-04`) is an ASSISTments-only figure.
- EdNet RTE fails the pooled chance-level check (above).
- No archival source carries motor telemetry. RQ2 is answerable only on data
  the platform's own instrument collects (Phase 2, Phase 10).

---

## Simulated data `sim-v2` (Phase 4)

Generated by `make simulate` from `ml-service/app/research/simulator.py` under the
parameters in `artifacts/datasets/simulator-params-v2.json`. The generative
model, its calibration and its limitations are in
[`docs/simulator.md`](simulator.md); this section is the dataset description.

**Legacy `alp-synthetic-v1` is superseded**, not merely deprecated: the section
below it in this file describes the pre-Phase-0 dataset whose behavioural columns
were deterministic functions of the latent state. It stays quarantined in
`artifacts-legacy-invalid/` and is read by exactly one thing — figure `f04-07`,
the before/after of that defect.

### Files

| File | Contents |
|---|---|
| `artifacts/datasets/sim-v2-V0.parquet` | calibrated baseline, 153,956 attempts, 1,000 learners |
| `artifacts/datasets/sim-v2-V1.parquet` | 2PL link (varying item discrimination) |
| `artifacts/datasets/sim-v2-V2.parquet` | fatigue affects speed only |
| `artifacts/datasets/sim-v2-V3.parquet` | behavioural signals half noise |
| `artifacts/datasets/sim-v2-V4.parquet` | non-stationary learning rate |
| `artifacts/datasets/sim-v2-manifest.json` | seed, git SHA, parameter SHA-256, row counts, V0-vs-real KS |
| `artifacts/datasets/simulator-params-v2.json` | every fitted parameter and what was not identifiable |
| `artifacts/datasets/simulator-calibration-ks.json` | the acceptance gate's own KS run |
| `artifacts/datasets/simulator-mi-comparison.csv`, `simulator-recoverability.json` | the Defect 3 evidence |

The `.parquet` files are **git-ignored** (~110 MB). Everything else in the table
is committed, and `make simulate` rebuilds the lot in about two and a half
minutes. That is the rule for this repository: an artifact must be reproducible
from a `make` target; it does not have to be in git.

### Per-attempt schema

| Group | Fields |
|---|---|
| Identity | `learner_id`, `session_id`, `position`, `timestamp`, `archetype`, `variant` |
| Item | `item_id`, `concept_id`, `difficulty_score`, `item_b` |
| Outcome | `correct`, `response_time_ms`, `probability_correct`, `selected_index` |
| Logging policy | `propensity` — the probability with which the logging policy served this item. **Recorded at generation time because Study 3 cannot be added to logs that lack it** |
| Probes | `probe_confidence` (4-point, 12 % of items), `probe_effort` (5-point, every 15th) |
| Contamination | `distracted` |
| Latent ground truth | `latent_knowledge` (this item's concept), `latent_knowledge_mean` (over all concepts), `latent_engagement`, `latent_confidence`, `latent_fatigue` |
| Features | every column in `app.features.extract.FEATURE_CLASSES`, produced by the production extractor |

**`latent_*` columns are evaluation-only targets and must never be model
inputs.** That is Defect 2, and the prefix is what Phase 5's leakage audit
excludes on. The feature columns are the *only* legitimate inputs, and they are
computed by the same function the backend serves — the simulator has no feature
definitions of its own.

### Archetypes are labels, not generators

`archetype` (`steady`, `rapid`, `careful`, `fatigable`) is assigned **after** a
learner is drawn, from the learning-rate × fatigue-susceptibility quadrant. The
legacy simulator did the opposite: hand-typed archetype presets generated the
learners. Nothing in the generative model reads the label; it exists so figures
like `f04-06` can show four contrasting learners.


---

## Legacy: synthetic ALP interaction dataset `alp-synthetic-v1` (INVALID)

**Not citable.** The dataset this section describes was produced by the
pre-Phase-0 pipeline and now lives in `artifacts-legacy-invalid/`. It is
affected by Defects 2 and 3 (see `docs/PROGRESS.md`): `knowledge_before` and
`fatigue_before` are latent simulator state used as model inputs, and every
behavioural column is a deterministic function of that latent state. Phase 4
replaces it. The schema is kept below only so the legacy files can be read.

## Learner Archetype Distribution

The 300 simulated learners are sampled across four distinct behavioral archetypes:

| Archetype | Cohort Size | Learning Rate | Initial Knowledge | Confidence | Guessing Prob | Fatigue Growth | Persistence |
|---|---|---|---|---|---|---|---|
| **Steady** | 75 learners | 0.075 | Gaussian(0.35, 0.12) | 0.62 | 0.12 | 0.025 | 0.76 |
| **Rapid** | 75 learners | 0.115 | Gaussian(0.48, 0.12) | 0.72 | 0.10 | 0.020 | 0.72 |
| **Careful** | 75 learners | 0.060 | Gaussian(0.35, 0.12) | 0.48 | 0.06 | 0.030 | 0.88 |
| **Fatigable** | 75 learners | 0.75 learners | Gaussian(0.35, 0.12) | 0.57 | 0.13 | 0.060 | 0.58 |

---

## Schema & Feature Definitions

| Field Name | Type | Description |
|---|---|---|
| `learner_id` | String | Unique learner identifier (e.g. `sim-00001`) |
| `archetype` | String | Learner behavioral archetype (`steady`, `rapid`, `careful`, `fatigable`) |
| `event_index` | Integer | Chronological sequence index of the question item (1 to 30) |
| `question_id` | String | Item identifier |
| `difficulty` | String | Discrete item difficulty (`easy`, `medium`, `hard`) |
| `difficulty_score` | Integer | Numerical difficulty rating (1 to 10) |
| `knowledge_before` | Float | Inferred student knowledge prior to answering |
| `fatigue_before` | Float | Simulated student fatigue prior to answering |
| `correct` | Boolean | True if learner answered correctly |
| `guessed` | Boolean | True if item was answered via random guessing |
| `total_response_time` | Float | Total elapsed time on question in seconds |
| `reading_time` | Float | Initial reading time prior to interaction in seconds |
| `time_after_last_interaction` | Float | Idle time prior to submission in seconds |
| `attempts` | Integer | Number of attempt retries |
| `skip` | Boolean | True if learner skipped the item |
| `option_changes` | Integer | Number of times option selection was toggled |
| `mouse_distance` | Float | Total mouse cursor travel distance in pixels |
| `mouse_speed` | Float | Average mouse cursor movement speed in px/sec |
| `hover_time` | Float | Total hover time over option buttons in seconds |
| `typing_speed` | Float | Typing speed in words per minute |
| `backspaces` | Integer | Keystroke backspace count |
| `delete_frequency` | Integer | Keystroke delete count |
| `pause_duration` | Float | Long typing pause duration in seconds |
| `question_number` | Integer | Session item index |
| `session_duration` | Float | Cumulative session duration in seconds |
| `tab_switches` | Integer | Window blur / tab switch count |
