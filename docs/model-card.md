# Model Card: Next-Response Performance Predictor (CatBoost Benchmark Leader)

> **The card below this banner describes the pre-Phase-0 pipeline and is not
> citable.** Its features included `knowledge_before` and `fatigue_before` —
> the simulator's latent ground truth (Defect 2 in `docs/PROGRESS.md`) — so
> every metric in it is inflated by leakage. It is kept because BUILD.md
> forbids deleting an artifact that has not been regenerated; Phase 6 replaces
> it. The **state-head cards at the end of this file are current.**

## Model Details
- **Model Name**: CatBoost Next-Correct Classifier (`artifacts/models/best-next-correct.joblib`)
- **Model Version**: `1.0.0`
- **Model Architecture**: Gradient-Boosted Decision Trees (CatBoostClassifier, `iterations=120, depth=5, learning_rate=0.06`)
- **Developer**: Adaptive Learning Research Architect Team
- **Date**: July 2026

## Intended Use
- **Primary Use Case**: Predict the probability of a learner correctly answering their *next* question given behavioral telemetry from the prior interaction.
- **Decision Loop Integration**: Used by the FastAPI service to score item difficulty candidates (`easy`, `medium`, `hard`) and select the difficulty targeting the optimal desirable difficulty zone ($\approx 0.72$ target success probability).
- **Out of Scope**: High-stakes student assessment or grading without human oversight.

## Benchmark Model Comparison

All models trained on 6,525 interaction records (225 training learners) and evaluated on 2,175 held-out interaction records (75 testing learners). Learner-grouped split prevents data leakage.

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Inference (ms/row) |
|---|---|---|---|---|---|---|
| **CatBoost (Selected)** | **0.7977** | **0.7425** | **0.6492** | **0.6927** | **0.8637** | **0.0030** |
| Random Forest | 0.7954 | 0.7465 | 0.6322 | 0.6846 | 0.8580 | 0.0154 |
| XGBoost | 0.8000 | 0.7408 | 0.6623 | 0.6994 | 0.8570 | 0.0037 |
| LightGBM | 0.7963 | 0.7357 | 0.6558 | 0.6934 | 0.8558 | 0.0061 |
| MLP Neural Net | 0.7839 | 0.7149 | 0.6401 | 0.6754 | 0.8405 | 0.0022 |
| Logistic Regression | 0.7724 | 0.7066 | 0.6021 | 0.6502 | 0.8235 | 0.0039 |
| Decision Tree | 0.7738 | 0.7222 | 0.5785 | 0.6424 | 0.8180 | 0.0020 |

---

## Top Feature Importances (CatBoost)

1. `difficulty_score` (32.26%): Numerical difficulty weighting of the item.
2. `knowledge_before` (25.65%): Rule-inferred student mastery level.
3. `total_response_time` (11.89%): Total elapsed seconds on the preceding item.
4. `hover_time` (9.93%): Total hover duration over option buttons.
5. `reading_time` (4.35%): Estimated time spent initial-reading the question text.

---

### Sequence Models with Extended Histories (80 items)

When evaluating on extended history sequences (80 items per learner), neural sequence architectures failed to close the performance gap compared to tabular gradient boosting methods. The ROC-AUC gap vs CatBoost (0.863) stayed the same (remained very wide).
- **CatBoost ROC-AUC**: 0.863
- **LSTM (80 items) ROC-AUC**: 0.698
- **Transformer (80 items) ROC-AUC**: 0.738

---

## Ethical & Privacy Considerations
- **Synthetic Data**: Trained entirely on seeded synthetic learner profiles (`StudentSimulator`).
- **Privacy Minimization**: Operates on aggregated time and count features; does not store raw keystroke content or raw mouse coordinate logs.


---

# Model Card: Multi-State Learner Encoder (Phase 7, Study 2)

## Model details

- **Artifact**: `artifacts/models/state-encoder.pt` + `state-calibrators.joblib`
- **Architecture**: one shared GRU (input = 44 features × 2 channels, hidden 64,
  dropout 0.2) over the interaction sequence, with three linear heads. Standard
  composition — no architectural novelty is claimed. The contribution is the
  supervision-and-validation regime.
- **Trained by**: `make train-states` → `ml-service/app/research/study2.py`,
  seed `20260821`. Sources: `assistments_2012` (real) and `sim-V0` (simulated),
  pooled with per-head masked losses.
- **Inputs**: every feature enters as a (value, present) pair — absence stays
  distinguishable from measurement. Column list and the training-fold
  median/IQR travel inside the checkpoint.
- **Uncertainty**: MC dropout, 20 passes, standard deviation per row.
- **Splits**: `fold0`–`fold3` train; `fold4` is halved by learner into a
  calibration half and a scoring half. **The test set was not read.**

## Intended use

To supply a Phase 8 policy with **validated** state estimates. A state that did
not pass the gate is not an input to any policy: `validation_gate.Gate.
filter_states()` removes it. Not for grading, placement, or any decision about
an individual learner without human oversight.

## Per-head cards

Every number below is from `artifacts/evaluation/validation-gate.json` and
`artifacts/benchmarks/study2-heads.json`. Thresholds are pre-registered in
`docs/preregistration.md` §8.2.

### Knowledge head — **ADMITTED**

- **Label**: `next_correct`, the next attempt's outcome. **Judged on**:
  `assistments_2012`, 15,564 rows / 169 learners, base rate 0.673.
- **C1** *r* = 0.316 [0.282, 0.345] ✓ (≥ 0.15) · **C2** ECE 0.032 → **0.016**
  after isotonic ✓ (≤ 0.05) · ROC-AUC 0.686 · mean SE 0.023.
- **Ablation**: flat across the ladder (0.313 at L0 → 0.315 at L4). Correctness
  history is what this head runs on; behavioural signal classes add nothing
  measurable to it.
- **Use**: admitted to the Phase 8 policy.

### Engagement head — **EXCLUDED (fails C3)**

- **Label**: `solution_behaviour` of the **next** attempt — the response-time
  effort classification of `docs/preregistration.md` §6. **Judged on**:
  `assistments_2012`, 15,395 rows / 169 learners, base rate 0.983.
- **C1** *r* = 0.281 [0.188, 0.371] ✓ · **C2** ECE 0.005 ✓ · **C3**
  |*r*| with ability = **0.403** ✗ (ceiling 0.20) · ROC-AUC 0.831 · mean SE 0.016.
- **Why it is excluded even though it predicts well.** Wise & Kong's negative
  criterion: an effort index must be independent of ability. This one tracks
  ability at twice the pre-registered ceiling (`f07-05`), so what it has learned
  is competence wearing an effort label. High AUC is what makes the failure
  worth reporting — it is exactly the head someone would ship without the check.
- **Ablation**: the only head that needs a signal class. *r* 0.130 at L0 → 0.287
  at L1: **timing is what makes an effort head possible**, and L2–L4 add nothing.
- **Use**: **none.** No Phase 8 policy may read it. The underlying observables
  (`solution_behaviour`, `rt_drift`) remain available as features; what is
  banned is conditioning on the latent construct.

### Confidence head — **EXCLUDED (fails C1 and C4)**

- **Label**: `probe_confidence ≥ 3` — the post-submission self-report,
  dichotomised at *fairly sure*. **Judged on**: `sim-V0` (simulated probes;
  no archival source carries one), 1,384 rows / 81 learners, base rate 0.764.
- **C1** *r* = **0.138** [0.083, 0.196] ✗ (floor 0.15) · **C2** ECE 0.032 →
  0.024 ✓ · **C4** |*r*| with the knowledge head = **0.893** ✗ (ceiling 0.85) ·
  ROC-AUC 0.584 · mean SE 0.029.
- **Why it is excluded**: it is close to a relabelled knowledge head, and it
  does not track its own label well enough to argue otherwise. Both failures
  point the same way.
- **Use**: **none.** A real-probe replication is the only thing that would
  change this verdict; the platform collects real probes the moment humans use
  it (`docs/preregistration.md` §3).

### Fatigue — **no head exists**

Dropped in Phase 5 by the pre-registered H2 rule: no archival source shows a
within-session accuracy decline (`assistments_2012` +0.000077 [−0.000125,
+0.000280]; `ednet_kt1` significantly *positive*). `f07-10` draws the
observable that was measured rather than an estimate that does not exist. The
within-session *speed* channel survives as a feature, without the claim that it
indexes a fatigue state.

## Limitations

- **Two of three states failed.** The paper reports the failures; the policy
  runs on knowledge plus observables.
- **Confidence validity rests on simulated probes.** It is a bound on what the
  probe protocol can support, not evidence about human self-report.
- **169 scoring learners** for the real-data heads — Phase 6's learner cap,
  kept so both studies are fitted on the same rows. Every interval is
  learner-clustered so the cost is visible rather than hidden.
- **C3's ability term shares rows with the estimate** (`docs/preregistration.md`
  §8.4). The splits are by learner, so no earlier data exists for a scoring
  learner; the overlap makes the criterion easier to fail, which is the
  conservative direction.
- **The hidden state resets every 100 attempts.**

## Ethical and privacy considerations

The signal classes map to the consent classes of `docs/preregistration.md` §4,
and the ablation says what each state costs in telemetry: knowledge needs
correctness only, engagement needs timing, and no state in this project needed
motor telemetry to reach its best measured validity. A learner who declines
everything above `correctness` still gets the one state the gate admitted.
