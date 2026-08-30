# Phase 8.5 audit — what is scientifically usable

Written before any Phase 8.5 code was added, per BUILD.md Phase 8.5 step 2. The
question in every row is **not** "does it run" but "can a claim rest on it".

Audited on 2026-08-23 against the working tree at `master`.

## The table

| Component | Current implementation | Working? | Scientifically usable? | Action |
|---|---|:--:|:--:|---|
| **Feature extraction** | `app/features/` + `app/research/build_features.py`, 44 features across five signal classes, one catalogue (`feature_catalogue.py`) generating both the doc and the column lists | yes | **yes** | none — `audit_leakage.py` is the standing guard and runs in `make test` |
| **Logistic regression baseline** | `study1.py`, scikit-learn, learner-grouped CV, tuned on folds only | yes | **not yet** | **the final test pass never ran**; `study1-final.json` and the model artifact do not exist. `make train-baselines` — Phase 8.5 step 1 |
| **Other predictive models** | XGBoost / LightGBM / CatBoost / MLP + pykt DKT/SAKT/AKT/SAINT via `baselines.py` | yes | **not yet** | same blocker: CV numbers exist, test numbers do not |
| **State validation gate** | `app/model/validation_gate.py`, four pre-registered criteria, `Gate.filter_states()` removes rather than zeroes | yes | **yes** | none — thresholds in `preregistration.md` §8, verdict in `validation-gate.json`, asserted by `test_study2.py` |
| **Rule policy** | `app/policy/rules.py`, eight rules, each with a signal class, its latent states and a citation | yes | **yes** | none — the two joint-state rules are excluded by the gate and reported, not deleted |
| **Model-assisted policy arms** | `app/policy/policies.py`, ten arms over one 120-action space, loading the Phase 7 encoder, BKT and Rasch parameters | yes | **yes** | none — the arms load trained artifacts; a grep test forbids importing the simulator |
| **Simulator (V0–V4)** | `app/research/simulator.py`, calibrated against archival data (KS gate), four documented mis-specifications | yes | **yes** | none — assumptions are documented in `docs/simulator.md` and in each variant's description string |
| **Closed-loop evaluator** | `app/research/closed_loop.py` + `evaluate_policies.py`, paired tests, bootstrap CIs, effect-size halt | yes | **yes**, after three fixes | fixed during the Phase 8 resume: hash-order non-determinism, a degenerate difficulty action on a third of the curriculum, one documented halt exemption. See `docs/phase-8-resume-audit.md` |
| **Telemetry** | Phase 2 event schema, `/v1/features/extract`, the same extractor offline and online | yes | **yes** | none |
| **Action logging** | every arm emits `difficulty`, `concept_move`, `intervention` from one shared space | yes | **yes** | none — `test_phase85.py` asserts the reported space equals the emitted one |
| **Propensity logging** | `decision-log-v0.parquet`, one propensity per decision, ε-greedy at ε = 0.1 | yes | **yes** | none — `test_phase85.py` asserts every propensity is present and in (0, 1] |
| **Existing figures/results** | 05–08 packs regenerate from artifacts; `artifacts-legacy-invalid/` is quarantined and mounted read-only | yes | **mixed** | Phase 6's figures are drawn from CV numbers only; nothing in the paper may quote a Study 1 test number until step 1 completes |
| **Live ml-service** | `app/main.py`, `/predict` via `predictor.py` | **partly** | **no** | **`predictor.py` loaded `artifacts/models/best-next-correct.joblib`, which the pipeline stopped producing before Phase 0.** `load_model()` returned `None`, `/predict` fell back to a rule, and nothing said so. Now routed through `registry.load()`, and `/health` reports which artifact and from where. Full wiring is Phase 10 |

## What was marked invalid and is not used downstream

- **`artifacts-legacy-invalid/`** — the pre-Phase-0 results (the *d* = 26 arm, the
  oracle policy, the 0 % mastery criterion). Kept read-only because figure
  `f04-07` needs the old dataset as the "before" side of a comparison. **No
  number from it appears in any current document.**
- **`docs/paper-draft.md`** — built entirely on those results (19.33 % mastery,
  CatBoost AUC 0.8637, *d* = 0.546). It is stale and BUILD.md Phase 12 already
  says it is rewritten from scratch. Nothing may cite it in the meantime.
- **`artifacts/benchmarks/study1-predictions.parquet`** — contains test rows
  from a pre-fix run. `test_study1.py::test_cross_validation_never_scored_a_
  test_learner` fails against it and will keep failing until
  `make train-baselines` rewrites the file.

## What Phase 8.5 added

1. **`app/model/registry.py`** — the artifact contract (`model.joblib`,
   `feature_schema.json`, `metrics.json`, `calibration.json`,
   `training_manifest.json`) and the single loader. `save()` refuses a manifest
   without seed, dataset, split strategy, hyperparameters, split sizes and git
   sha. `LoadedModel.predict_proba()` selects the model's own columns in the
   model's own order and raises on a missing one, so column order cannot change
   a prediction and a wrong-shaped frame is an error rather than a guess. The
   run identifier is a hash of the inputs, not a timestamp, so the same seed on
   the same commit writes the same manifest.
2. **`study1.py` writes through the registry** instead of a loose joblib.
3. **`predictor.py` reads through the registry**, with the legacy paths kept
   only as a fallback for an old checkout, and refuses to feed the estimator a
   payload missing its features — it falls back to the rule and *says which
   features were missing* rather than returning a confident wrong number.
4. **`test_phase85.py`** — the standing guards: no latent variable at any rung
   of the catalogue, the policy package cannot import the simulator, tuning is
   learner-grouped and reads a fixed split file, the final pass exists and
   scored real rows, every logged decision carries a usable propensity, the
   reported action space is the emitted one, and every published report records
   the run profile that produced it.
5. **`validate_phase85.py`** (`make integrate`) — the end-to-end chain at the
   smoke size, written to `artifacts/evaluation/phase-8.5-validation.json` pass
   or fail. A check whose input is missing is recorded as **blocked** with the
   `make` target that produces it, never as a pass.
6. **`docs/training-pipeline.md`** and **`docs/policy-architecture.md`** — the
   path from raw data to a loadable artifact, and the three-layer separation
   (estimation ≠ rule adaptation ≠ learned adaptation) that the pre-Phase-0
   repository collapsed.
7. **Four figures** in `artifacts/figures/08.5-integration/`, each annotated
   with counts read from the artifact tree at render time, so a stage that
   stops producing its output shows up red rather than as a stale picture.
