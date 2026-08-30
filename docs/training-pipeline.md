# Training pipeline

One path from raw data to a loadable artifact, and one loader that reads it.
Written in Phase 8.5, after the audit found that "it trained once" and "another
process can load it and get the number the paper reports" had come apart.

## The path

```
data/raw/                      fetched, SHA-256 pinned            make fetch-data
      │
      ▼
data/processed/*.parquet       one schema, learner-grouped splits make prep-data
      │                        (data/processed/splits.json)
      ▼
features-<source>.parquet      the catalogue, one row per attempt make features
      │
      ├─► audit_leakage.py     exits non-zero on a leak; also in `make test`
      ▼
cross-validation              5 learner-grouped folds; tuning only make train-baselines
      │
      ▼
final test pass               the held-out learners, touched once
      │
      ▼
artifacts/models/<name>/      the artifact contract (below)
      │
      ▼
app/model/registry.load()     the only loader — policy, service and figures
```

Every step is a `make` target, every target takes `--seed`, and every output
records the seed it was produced with. `make all` runs the whole path at the
`smoke` profile; the sizes come from `runprofile.py` (BUILD.md §1.5).

## The artifact contract

A trained model is a **directory**, never a loose file:

```
artifacts/models/<name>/
    model.joblib            the fitted estimator or pipeline
    feature_schema.json     exact column list and dtypes it expects
    metrics.json            held-out numbers, and which split they came from
    calibration.json        the calibrator and its diagnostics, or why none
    training_manifest.json  seed, dataset, split strategy, features, target,
                            hyperparameters, split sizes, git sha, run id
```

`registry.save()` refuses to write a directory whose manifest is missing any of
`seed`, `dataset`, `split_strategy`, `hyperparameters`, `split_sizes`,
`git_sha` — a model that cannot say how it was trained is not evidence.

Nothing writes a wall-clock timestamp. Two runs of the same seed on the same
commit produce the same manifest, so the run identifier is a hash of the inputs
(`registry.run_id`) rather than the clock.

## The loader

`app/model/registry.load(name)` returns a `LoadedModel` carrying the estimator,
its feature list, its metrics, its calibration record and its manifest.
`LoadedModel.predict_proba(frame)` selects the model's own columns **in the
model's own order** and raises `KeyError` on a missing one. Column order
therefore cannot change a prediction, and a differently-shaped frame is an
error rather than a guess.

An incomplete artifact directory raises `FileNotFoundError` naming the missing
files. Hand-assembling one is not supported: train it with the `make` target
that owns it.

## What supervises what

| Artifact | Target | Data | Split | Owner |
|---|---|---|---|---|
| `study1-<source>-<model>/` | `next_correct` | archival + `sim-V0` | learner-grouped, 5 CV folds + disjoint test | Phase 6 (`make train-baselines`) |
| `state-encoder.pt`, `state-calibrators.joblib` | per-head state labels | `assistments_2012`, `sim-V0` | fold4 half-and-half; test untouched | Phase 7 (`make train-states`) |
| `bkt-params.json`, `item-response-stats.json` | — (fitted parameters) | `sim-V0` | training folds | Phase 8 (`make eval-policies`) |
| `policy-thresholds.json` | — (corpus quantiles) | `sim-V0` | offline corpus | Phase 8 |
| `artifacts/models/rl/` | reward | simulator V0 | on-policy | Phase 9 |

## Rules that hold across the whole path

1. **Learner-grouped splits.** The same learner never appears in both training
   and evaluation. `splits.json` is fixed once and every script reads it.
2. **The test set is touched once**, at the end, by the final pass only.
   `--no-test` runs cross-validation without it.
3. **No latent simulator variable is ever a feature.** `audit_leakage.py` is
   the standing guard and runs inside `make test`.
4. **Tuning happens on the CV folds.** Hyperparameters are recorded in the
   manifest so the reported number and the fitted model cannot diverge.
5. **Calibration is fitted on training data only**, and both the calibrated and
   uncalibrated numbers are written.
6. **A model that is not saved through `registry.save()` is not a result.** If
   a figure or a document quotes a metric, it comes from `metrics.json`.
