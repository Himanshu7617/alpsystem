# Phase 8 resume audit

Phase 8 was interrupted part-way: a long uncapped run overheated the machine and
was stopped. This file records what survived that interruption, what was
resumed, and what had to be rerun — written **before** any Phase 8 work was
restarted, per BUILD.md §0.1.

Audited on 2026-08-22 against the repository at `master` + working tree.

## What was found

Phase 8's **code was complete and its tests passed**; its **experiment had never
produced an artifact**. `ml-service/app/policy/` (five modules, 1,321 lines),
`app/research/closed_loop.py`, `figures_phase8.py` and `test_policies.py` were
all present and untracked; `artifacts/evaluation/` contained only Phase 7's
`validation-gate.json`, and `artifacts/figures/` had no `08-policies/`
directory. The Phase 8 docs (`docs/rule-policy.md` §Phase 8,
`docs/evaluation.md`, `docs/preregistration.md` §9, `docs/model-card.md`) were
already written.

So the interruption happened during the first `make eval-policies`, before any
result was written. Nothing partial had to be salvaged, and nothing valid had to
be discarded.

## Task-by-task

| Phase 8 task | Status found | Existing artifact | Resumed from | Rerun required? | Reason |
|---|---|---|---|---|---|
| Step 1 — delete the oracle | complete | `test_policies.py` greps `app/policy` for a simulator import | — | no | test passes; guard is code, not a result |
| Step 1 — fix the termination criterion | complete | `closed_loop.py` (fixed six-concept curriculum, mastery over taught concepts) | — | no | verified by test and by non-zero mastery in every arm |
| Step 1 — fix the sweep loop overwrite | complete | per-cell results kept in `evaluate_policies.py` | — | no | — |
| Step 2 — shared action space | complete | `app/policy/rules.py` (`DIFFICULTIES × CONCEPT_MOVES × INTERVENTIONS`) | — | no | one definition, reported in every results JSON |
| Step 3 — ten policy arms | complete | `app/policy/policies.py` | — | no | all ten build and act |
| Step 4 — state-interaction rules with citations | complete | `app/policy/rules.py`, `docs/rule-policy.md` | — | no | two joint-state rules correctly excluded by the gate |
| Step 5 — the closed-loop run | **never produced output** | none | from zero | **yes** | the interrupted step |
| Step 6 — explainability endpoint | complete | `app/policy/explanations.py`, `main.py` `GET /v1/decisions/{id}/explanation` | — | partial | the endpoint reads `decision-explanations.json`, which step 5 writes |
| Figures `f08-01`…`f08-14` | code complete, never run | none | after step 5 | **yes** | depend on step 5's artifacts |
| `docs/PROGRESS.md` Phase 8 section | missing | — | — | **yes** | written after the run |

## What was done differently on resume

Three problems were found while restarting step 5. All three are fixes to Phase
8 itself, not new work:

1. **The run was not reproducible.** Two runs of the same seed gave different
   results. `ClosedLoop.__init__` iterated a `set` of concept ids while drawing
   each learner's per-concept knowledge from the RNG, so Python's per-process
   hash randomisation decided which concept got which draw. Fixed at the source
   (the set is sorted) with `PYTHONHASHSEED=0` pinned as a standing guard, and
   a regression test that runs the cohort construction under two different hash
   seeds and asserts the cohorts match.

2. **The difficulty action was a no-op on a third of the curriculum.** The
   selected curriculum contained `stacks_and_queues` (two items at an identical
   *b*) and two more concepts with three items, so "serve difficulty 3" and
   "serve difficulty 9" resolved to the same item and no policy could differ
   there. `select_curriculum` now requires ≥ 3 items spanning ≥ 1.0 logits of
   *b* per concept and skips any candidate whose prerequisite closure contains
   a concept that fails the floor — which is what its own docstring had always
   said it should do. The curriculum changed accordingly; both curricula's
   results are reported in `docs/PROGRESS.md`.

3. **The effect-size halt fired on an arithmetic artefact.** `random` draws its
   intervention uniformly, so it suggests a 300-second break on a quarter of all
   items — 45.9 per learner against 4.6 under ε-greedy exploration — and its
   time-to-mastery is roughly double every other arm's *by construction*. That
   is |d| ≈ 3 on time-to-mastery and on nothing else. One documented exemption
   was added (`evaluate_policies.HALT_EXEMPT`); the comparison is still measured
   and still written to every report under `effect_sizes_explained`.

## Resource controls added before rerunning

The interruption was caused by an uncapped run, so step 5 was not simply
restarted at full size. `ml-service/app/research/runprofile.py` defines
`smoke` / `dev` / `full`; `make full` refuses to run without
`docs/full-run-justification.md`; every finished cell is checkpointed to
`artifacts/evaluation/cells/` so an interrupted run resumes rather than
restarts; `make status` reports progress from `artifacts/evaluation/run-log.jsonl`
while a run is in flight. BUILD.md §1.5 is the written rule.

The rerun climbed the ladder: `smoke` (10 cells, 3.4 s) → `dev` (20 cells,
60 s) → `full` (150 cells) at 6 of 10 cores.

## Out of scope for this resume, and still outstanding

**Phase 6 is not finished.** `artifacts/benchmarks/study1-final.json` has never
been written and `study1-predictions.parquet` still holds test rows from a
pre-fix run, so `test_study1.py::test_cross_validation_never_scored_a_test_learner`
fails against it. Phase 8 does not depend on it — the policy arms load the
Phase 7 encoder and the BKT/IRT parameters, not a Study 1 model — but **no
Study 1 number may be quoted until `make train-baselines` completes**. It is
step 1 of Phase 8.5.
