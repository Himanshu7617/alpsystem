# Pre-registration

Definitions, hypotheses and thresholds are written here **before** the data they
concern is collected or generated. Standing guardrail 7 in
[`BUILD.md`](../BUILD.md): deviations are recorded as deviations, not edited into
the original text.

| Phase | Section added | Date |
|---|---|---|
| 2 | Trajectory feature definitions, probe protocol, consent classes | 2026-08-22 |
| 3 | Archival effort label, split protocol, recorded deviation | 2026-08-22 |
| 5 | Ablation ladder, label definition, leakage rules; H2 outcome | 2026-08-22 |
| 7 | State-head labels, validity gate criteria and thresholds | 2026-08-22 |

Later phases append their own sections: hypotheses and the ablation ladder
(Phase 5–6), validity gate thresholds (Phase 7), RL reward weights (Phase 9).

---

## 1. Motor / trajectory features (Phase 2)

Computed in the browser (`files/telemetry.js`) from pointer samples taken every
**50 ms**, reduced at submit, then the raw samples are discarded. The aggregate
is the only thing stored. Coordinates are viewport pixels; time is milliseconds
since the item was presented.

Let `P = [(t_i, x_i, y_i)]` be the samples, `S` the first sample, `T` the centre
of the option that was ultimately chosen, and `L` the straight line `S → T`.

| Feature | Definition |
|---|---|
| `auc_toward_nonchosen` | `Σ_i sign · d_i · Δt_i / 1000`, where `d_i` is the signed perpendicular distance of sample `i` from `L`, and `sign` is +1 on the side of the **competing option** — the non-chosen option with the greatest hover time. Units: pixel-seconds. Positive means the path bowed toward the alternative. |
| `max_deviation` | `max_i |d_i|`, pixels. |
| `x_flips` | number of sign changes in `Δx` between consecutive samples (zero steps ignored). |
| `sample_entropy` | SampEn of the per-step speed series with `m = 2`, tolerance `r = 0.2 σ`, natural log. Zero when undefined (fewer than `m + 2` samples, or zero variance). |
| `velocity_peak` | `max_i (Δdistance_i / Δt_i)`, reported in px/s. |
| `velocity_mean` | mean of the same series, px/s. |
| `pause_count` | number of maximal runs where speed < **0.05 px/ms** lasting > **300 ms**. |
| `path_ratio` | total path length ÷ `|T − S|`. 1.0 is a perfectly straight path. |
| `time_to_first_movement_ms` | first sampled pointer movement after presentation. |
| `time_to_first_selection_ms` | first `OPTION_SELECTED` after presentation. |
| `hover_time_ms[k]` | time the pointer spent inside option `k`'s bounding rectangle, accumulated in 50 ms units. Index-aligned with the item's options. |
| `n_samples`, `sample_interval_ms` | provenance for the four above; a short attempt yields few samples and every derived value must be read against them. |

Two derived features are produced by the extractor rather than the browser:
`hover_time_max` and `hover_time_nonchosen` (total hover on options other than
the one submitted).

**Deviations from BUILD.md's wording, recorded here rather than silently:**

- BUILD.md lists `path_length / straight_line_length`; the stored name is
  `path_ratio`. Same quantity.
- BUILD.md lists `hover_time_per_option[4]`; the stored name is
  `hover_time_ms[]`, length = number of options.

## 2. Typing dynamics — dropped before collection

The item bank is **186 items, 100 % MCQ** (`data/items/item-bank-v1.manifest.json`).
BUILD.md Phase 2 step 3 requires keystroke timing only on `format != "mcq"`
items, and instructs that a signal available on a small minority of items be
dropped with its coverage cited. Coverage here is **zero**, so typing dynamics
(`typing_speed`, `backspaces`, `delete_frequency`) are **not collected and not
studied**. They remain in the legacy `BASE_FEATURES` list of the pre-Phase-0
pipeline, which Phase 5 replaces.

## 3. Probe protocol

| Probe | Trigger | Scale |
|---|---|---|
| Confidence | random **12 %** of items | 4-point: Guessing / Not sure / Fairly sure / Certain |
| Perceived difficulty | the same items as the confidence probe | 4-point: Very easy / Easy / Hard / Very hard |
| Effort | every **15th** item | 5-point: None … Everything |

- Probes appear **after submission and before feedback**, so the outcome cannot
  colour the self-report.
- Assignment is server-side, seeded by `(session seed, item sequence)`, and both
  the probability `assign_p` and the realised draw `assign_draw` are stored, so
  assignment is analysable rather than merely happening.
- **Every probe is skippable, and a skip is recorded** (`skipped = true`) rather
  than dropped. Non-response is data: a learner who stops answering probes is
  informative about engagement.

**Threats, stated up front.** Probes are self-report and reactive: asking "how
sure were you?" can change the behaviour being measured, and it interrupts the
session. Human observation labels (BROMP) would avoid this but require trained
observers and participants; there are none in this project. The confidence
probe is therefore treated as a *noisy criterion for convergent validity* in
Phase 7, never as ground truth.

## 4. Consent classes and the privacy–utility curve

Five classes, each independently switchable, matching the ablation ladder:

| Class | Contains | Off-switch |
|---|---|---|
| `correctness` | outcome, selected option, item difficulty | **not optional** |
| `timing` | response time, reading time, decision latency | yes |
| `interaction` | option changes, idle, visibility/focus, hints | yes |
| `motor` | the trajectory aggregate above | yes |
| `probes` | the self-reports above | yes |

A class that is off produces **null**, never zero, in the feature row — absence
and measurement must stay distinguishable (`test_withdrawn_consent_yields_null_not_zero`
in `ml-service/app/features/test_extract.py`). The RQ4 curve in Phase 6 is read
off exactly these configurations.

## 5. Pre-registered expectations

Recorded now so a later result cannot be reinterpreted as a prediction.
Reference points from the literature review: SAINT+ reports **+1.25 % AUC** from
timing features; ITS meta-analyses report **g ≈ 0.32–0.37**; contextual bandits
report **+15.2 %** skill gain over non-contextual baselines.

- **H1.** Trajectory features carry information beyond response time and
  correctness — expected effect small, of the order of +1–3 % AUC, not more.
- **H2.** A within-session accuracy decline at matched difficulty exists but is
  modest. **If figure `f05-06` is flat, the fatigue construct is reduced or
  dropped rather than defended.**
- **H3.** Any closed-loop advantage from the richer state is of the order of
  the ITS meta-analytic effect. **A Cohen's *d* above 3 halts the analysis for a
  bug hunt** (standing guardrail 5).
- **A null result is reported as a null result.** No tuning until something wins.

---

## 6. Archival effort label and split protocol (Phase 3)

Fixed before any model is fitted on these tables.

**Response-time effort.** A response is *rapid-guessing behaviour* when its
response time falls below its item's normative threshold
`clip(0.10 × item mean response time, 1 s, 10 s)`; otherwise it is *solution
behaviour*. A learner's RTE is the proportion of their responses that are
solution behaviour, and RTE < **0.90** is the disengagement flag. Items with a
missing response time get a null label, never a zero.

**Split protocol.** By learner: a 20 % held-out test set plus 5 grouped CV folds
over the rest, seeded 20260821, written once to `data/processed/splits.json`.
Sequence order is `order_index`. The test set is read exactly once, at the end.

**Deviation, recorded rather than edited away.** BUILD.md Phase 3's acceptance
check requires rapid-guess accuracy within ±0.10 of chance for MCQ sources, on
the reasoning that a threshold failing it is wrong. EdNet KT1 fails it pooled
(0.543 against a chance level of 0.25) and no response-time cut fixes it. The
ability breakdown shows why: the lowest ability quartile's rapid responses score
0.339 — inside the band — and the highest quartile's score 0.811. Fluent
learners answer fast because they know the answer, violating the rapid-guessing
assumption rather than the threshold. Consequences fixed now, before Phase 7
sees a result:

- EdNet RTE is **not** an engagement criterion for RQ1 unless conditioned on
  ability. ASSISTments carries that criterion.
- The deviation lives in code (`RAPID_GUESS_DEVIATIONS` in
  `prep_archival.py`), so the check still runs and still fails for any source
  not listed there with its evidence.

**ASSISTments has no defined chance level.** Both releases are dominated by
algebra and fill-in items and record no option count (`answer_text` is empty for
`choose_1` rows), so the check is marked not applicable there rather than
quietly passed.

---

## 7. The ablation ladder and Phase 6's protocol (Phase 5)

Fixed now, before any model is fitted on the feature matrices Phase 5 built.

**The label is `next_correct`** — the outcome of the learner's *next* attempt.
Features describe attempt *t*; the decision they inform is the selection of
attempt *t+1*. This is the decision-relevant framing and it is what makes a
present-tense behavioural trace (response time, option changes, the pointer
path for the item just answered) legitimately available at decision time. The
alternative — predicting attempt *t*'s own outcome — makes every within-item
behavioural feature unusable by construction, which would answer RQ2 by
definition rather than by measurement.

**The rungs are the signal classes of `docs/feature-catalogue.md`**, and they
are cumulative:

| Rung | Adds | Features |
|---|---|---|
| L0 | correctness, attempts, prior accuracy, item difficulty | 8 |
| L1 | response time, latency, lag, personal speed drift | 8 |
| L2 | option changes, hints, idle, visibility | 9 |
| L3 | session position, elapsed time, matched-difficulty slopes | 6 |
| L4 | the pre-registered trajectory aggregate | 13 |

Every model is run at every rung. ΔAUC is reported with bootstrap 95 % CIs
clustered **by learner**, and Holm-corrected across the ladder. Per BUILD.md
Phase 6 step 3: **an increment whose CI does not exclude 2 % is not a result**,
because pyKT attributes 1–2 % to protocol variation alone. A null is reported
as a null.

**Leakage rules, enforced rather than asserted** (`audit_leakage.py`, run by
`make test`):

- No `latent_*`, `probe_*` or `next_*` column is ever a model input.
- A feature row is unchanged by removing every row after it — verified by
  rebuilding the source truncated and comparing, not by reading the code.
- Item difficulty and the per-item response-time standardisation are fitted on
  training learners only, verified by refitting without the test learners.

**Splits are `data/processed/splits.json` and are never recomputed.** Phase 5
appended the five simulator variants under the same protocol and seed; the
three archival entries were not touched.

### Outcome of H2, recorded rather than edited into the hypothesis

H2 pre-committed: *if `f05-06` is flat, the fatigue construct is reduced or
dropped rather than defended.* It is flat.

| Source | Within-session matched-difficulty accuracy slope | 95 % CI | Verdict |
|---|---|---|---|
| `assistments_2012` | +0.000077 | [−0.000125, +0.000280] | flat |
| `ednet_kt1` | +0.000165 | [+0.000004, +0.000343] | **positive** |
| `assistments_2009` | not measurable | — | no wall clock, so no sessions |

No real source declines within a session and one significantly *improves*.
Consequences, fixed now rather than after Phase 7 sees a result:

- **`fatigue` is not carried as a validated latent state.** Phase 7's
  validation gate does not attempt to certify it, and no policy in Phase 8
  conditions on it. The multi-state model is knowledge, engagement and
  confidence.
- **The within-session *speed* channel survives and is retained as a feature.**
  It is significant and negative in every source (`assistments_2012`
  −0.0092 [−0.0097, −0.0087], `ednet_kt1` −0.0018 [−0.0023, −0.0013]) —
  learners speed up as a session runs. `matched_difficulty_speed_slope` stays
  in L3 as an observable; what is dropped is the claim that it indexes a
  fatigue *state*.
- **The paper says this.** Not "fatigue was modelled" but "a within-session
  accuracy decline could not be detected in any of the three archival sources,
  so the construct was dropped before modelling."

This agrees with Phase 4, which found the calibration source's residual
within-session accuracy slope to be +0.0016 and fitted
`fatigue_accuracy_beta` to 0.

---

## 8. Validity gate thresholds and the state-head protocol (Phase 7)

Fixed **before** the multi-state encoder is fitted. Nothing below was chosen
after seeing a head's score; a threshold that a head then misses is reported as
a miss (BUILD.md Phase 7 step 4: *"a failing state is reported in the paper and
excluded from the policy. That outcome is a result, not a setback."*).

**The states are knowledge, engagement and confidence.** Fatigue is not
certified and not modelled — §7's H2 outcome dropped the construct before this
phase started.

### 8.1 What supervises each head

Each head is trained against **its own label**, never against correctness. A
head with no label of its own is not a measurement.

| Head | Label | Column | Sources carrying it |
|---|---|---|---|
| Knowledge | the next attempt's outcome | `next_correct` | `assistments_2012`, `sim-V0` |
| Engagement | the **next** attempt's response-time-effort classification (§6): solution behaviour = 1, rapid-guessing = 0 | `solution_behaviour`, shifted one attempt forward within a learner | `assistments_2012` |
| Confidence | the post-submission confidence probe (§3), dichotomised at *Fairly sure* (`probe_confidence >= 3`) | `probe_confidence` | `sim-V0` |

Three consequences are fixed here rather than argued later:

- **The engagement label is shifted forward by one attempt.** Rapid-guessing is
  *defined* by a response-time cut, and response time at attempt *t* is an L1
  model input at attempt *t*. A head predicting the label of the attempt whose
  response time it can see would score near-perfectly by reconstructing the
  threshold, which measures the arithmetic, not the state. Shifting the label
  makes the head predict a *future* disengagement from a past trace, which is
  also the only version a policy can act on.
- **The confidence label is not shifted.** The probe is a self-report about the
  attempt just submitted, collected before feedback, and it is never a model
  input (`probe_*` is in `FORBIDDEN_PREFIXES`). Predicting it from the trace
  that preceded it is a readout of the state at that moment, which is what the
  policy reads at the next decision point.
- **Confidence is supervised on simulated probes only, and engagement on real
  archival data only.** No archival source carries a confidence probe, and the
  simulator carries no response-time-effort label. Each head's validity is
  therefore reported on the source that carries its label, and the source is
  named next to every number.

The 4-point probe is dichotomised rather than modelled as an ordinal scale
because the policy in Phase 8 consumes a probability, and a calibrated
probability with a reported ECE is the object the gate is written against.

### 8.2 The gate criteria and their thresholds

A head is **admitted to the policy only if it passes every criterion that
applies to it**. All numbers are measured on the held-out validation fold
(`fold4`), which no head trains on, and `fold4` is itself split by learner into
a **calibration half** (where the isotonic map is fitted) and a **scoring
half** (where every number below is measured), so no criterion is evaluated on
the data that shaped it. **The test set is not touched in this phase.**

| # | Criterion | Applies to | Pass condition | Why this number |
|---|---|---|---|---|
| C1 | Correlation with its own label | all heads | point-biserial *r* ≥ **0.15** **and** the learner-clustered 95 % CI excludes 0 | *r* = 0.15 is the conventional small-effect floor; below it a state cannot move a threshold-based decision by a useful amount. The CI condition stops a large-*n* point estimate from passing on precision alone. |
| C2 | Calibration | all heads | ECE ≤ **0.05** after isotonic calibration | A policy that acts at a 0.7 threshold needs 0.7 to mean 0.7. 5 % is one bin's width of slack in the 15-bin estimator used throughout this project. |
| C3 | Wise & Kong negative criterion | engagement only | \|*r*(engagement estimate, learner ability)\| ≤ **0.20** | Wise & Kong require an effort index to be **independent of ability**: a measure correlating with ability has learned ability, not effort. Ability is the learner's mean correctness over their solution-behaviour responses in the training folds. 0.20 is the small-effect ceiling — the mirror image of C1's floor. |
| C4 | Discriminant validity | confidence vs knowledge | \|*r*(confidence estimate, knowledge estimate)\| ≤ **0.85** | The heads share an encoder and both track competence, so a high correlation is expected; the question is whether confidence is a *relabelled* knowledge head. Above 0.85 the two orderings are interchangeable for any threshold rule and the claim of a distinct state is not supported. |

C3 is directional and pre-registered as such: a head can fail by correlating
with ability **in either sign**.

### 8.3 Protocol

- **Architecture.** One shared GRU encoder over the interaction sequence, three
  linear heads on the hidden state. Standard composition; the contribution is
  the supervision-and-validation regime, not the architecture.
- **Masked losses.** Sources pooled in one model; each head's loss is masked to
  the rows whose label exists. A row never trains a head it has no label for.
- **Missingness.** Every feature enters as a (value, present) pair, so absence
  stays distinguishable from measurement inside the model as well as in the
  parquet (§4). No imputation is persisted.
- **Splits are `data/processed/splits.json`**, unchanged. `fold0`–`fold3`
  train; `fold4` is halved by learner into a calibration half and a scoring
  half; `test` is untouched.
- **Uncertainty** is MC dropout, 20 stochastic passes, reported as the standard
  deviation of the per-row estimate. Chosen over a 5-model ensemble because it
  is one training run rather than five for the same purpose.
- **Ablation.** The full head set is retrained at every live rung of the §7
  ladder, producing the privacy–utility curve at the state level.
- **Pre-registered expectation.** The engagement head faces a label whose base
  rate is ~2 % rapid-guessing; a head that discriminates but sits at a poor ECE
  is the expected failure mode, and C2 is where it would be caught.

### 8.4 Deviations, recorded rather than edited away

- **C3's ability term is computed on the scoring rows, not on the training
  folds.** §8.2 as first written said "the learner's mean correctness over
  their solution-behaviour responses in the training folds". The splits are
  **by learner** (§6), so a learner in the scoring half has no rows in the
  training folds at all and the quantity does not exist. Ability is therefore
  the same average taken over that learner's scoring-half responses. The
  estimate and the ability then share rows, but one is a model output and the
  other an observed label average, and C3 is a *negative* criterion — shared
  rows can only make the correlation easier to find, which is the conservative
  direction for a criterion that fails on a high correlation.
- **Study 2 is fitted on Study 1's rows.** The learner cap and seeded sample
  come from `study1.load`, so a difference between the two studies is a
  difference in method rather than in data. The cost is a smaller scoring set
  than the full archive would give; every gate number is reported with a
  learner-clustered CI so the reader can see it.

---

## 9. The closed loop and Phase 8's protocol (Phase 8)

Fixed before the first closed-loop rollout was generated. Phase 8 is the RQ3
test: hold the policy architecture constant, vary only the state
representation, and see whether the decisions and the outcomes change.

### 9.1 What is being compared

Ten arms, one action space:
`{difficulty 1..10} × {same_concept | next_concept | prerequisite_concept} ×
{no_intervention | hint | worked_example | break_suggestion}` — 120 actions,
identical for every arm, each emitted with the probability the arm chose it.

| Arm | State it conditions on |
|---|---|
| `random` | none — uniform over the 120 actions |
| `fixed_order` | none — a static difficulty ramp on a fixed concept clock |
| `mastery_threshold_bkt` | BKT posterior (Corbett & Anderson 1994), mastery at p(L) ≥ 0.95 |
| `irt_cat_maxinfo` | Rasch θ by EAP, maximum-information item selection |
| `rule_legacy` | the deployed engine's knowledge variable, correctness only |
| `rule_improved` | the deployed 5-variable engine **restricted to gated states** |
| `model_L0`, `model_L1`, `model_L3`, `model_L4` | Phase 7's encoder at that rung, gated |

**`model_L0` vs `model_L4` is the pre-registered headline contrast.** Same
selection rule, same action space, same target band, same rule table; the only
difference is the rung the knowledge estimate was trained at and, consequently,
which rules have their inputs available.

`rule_improved` is restricted because three of the five variables the deployed
engine uses — cognitive load, fatigue, engagement — are constructs this project
has either dropped (§7's H2) or rejected at the validation gate (§8). Running
the engine unrestricted would condition a policy on constructs the same paper
reports as invalid. The excluded terms are recorded in every decision
explanation rather than deleted.

### 9.2 Outcomes

- **Primary: items to mastery.** Right-censored at a **200-item budget**: a
  learner who does not master contributes 200 and `mastered = false`. Censoring
  is reported alongside every mean, because a mean over censored data is a
  lower bound on the difference, not the difference.
- **Primary: time to mastery**, in **on-task seconds** — response time plus
  inter-item gaps plus the time an intervention costs. Between-session gaps are
  wall clock, not learner effort, and are excluded.
- **Co-primary: latent knowledge gain over the taught curriculum**, total and
  per item. Registered as a co-primary *before the first rollout* for a reason
  that is a property of the Phase 4 calibration, not of any result: the fitted
  learning-rate distribution (log-mean −7.26, log-sd 2.13, floored at 1e-4)
  implies a median learner needs on the order of a thousand items to move one
  concept across the mastery threshold. A 200-item budget therefore censors
  most learners, and a censored primary outcome would report "no difference"
  whatever the policies did. Knowledge gain is uncensored and is the quantity
  items-to-mastery is a threshold crossing of.
- Secondary: mastery rate, final mastery fraction, mean realised success
  probability (the check on the desirable-difficulty band), sessions,
  intervention counts, and a disengagement proxy (latent engagement below 0.3
  at any point — an evaluation-only quantity, never a policy input).

**Inclusion criterion.** A learner enters the cohort only if at least one
curriculum concept is unmastered at their first item. Drawn from the calibrated
ability distribution, 41 % of learners already satisfy the mastery criterion
before any instruction; keeping them would impose the same ceiling on every arm
and hide whatever difference exists among the learners who need teaching. The
screening rate is recorded in every results file.

**Mastery is defined over the taught curriculum, not the whole graph.** Every
learner is taught the same six-concept prerequisite-closed curriculum, derived
from the item bank's coverage. A concept is mastered when the response model
puts the learner's probability of answering that concept's median-difficulty
item correctly at **≥ 0.80**; a run ends when all six are mastered. The
pre-Phase-0 criterion required all thirty-six concepts while advancing one
concept per item, which is why every arm reported ~0 % mastery; that was a
broken criterion, not a policy result.

### 9.3 Instructional action effects — constants, not fitted

No archival source in this project records a hint, a worked example or a break,
so their effects cannot be fitted and are **fixed here before any rollout**,
identically for every arm:

| Action | Effect on the response model | Effect on learning | Time cost |
|---|---|---|---|
| `hint` | +1.0 logit of support | gain × 0.7 | +15 s |
| `worked_example` | +2.0 logits of support | gain × 1.0 | +45 s |
| `break_suggestion` | none | none | +300 s, within-session fatigue reset |

Directions, not magnitudes, are what the literature fixes: hints raise immediate
success and lower what a success is worth (Aleven et al. 2016; Baker et al.
2008 on help abuse), worked examples raise both at a larger time cost (Sweller &
Cooper 1985). Because the constants are shared by every arm, they cannot create
the L0-versus-L4 contrast; they do bound how large any intervention effect can
be, and the paper says so. Sensitivity to them is not claimed to be measured.

### 9.4 Exploration and propensities

Every adaptive arm is ε-greedy with **ε = 0.1** over the full action space;
`random` is already uniform and `fixed_order` is static, so both keep ε = 0.
The propensity of the chosen action — `ε/120 + (1−ε)·1[greedy]` — is recorded
on every decision. Study 3 cannot be added to a log that did not record one,
and ε is equal across arms so exploration cannot advantage one of them.

### 9.5 Hypotheses, and what would falsify them

- **H3 (RQ3, the thesis).** `model_L4` reaches mastery in fewer items than
  `model_L0`. Reported as a paired difference over the shared cohort with a
  bootstrap 95 % CI. **A CI that includes zero is a null and is reported as a
  null.** No arm is re-tuned after seeing this number.
- **H3-mechanism, pre-committed.** If `model_L4` would have chosen a different
  action from `model_L0` on **fewer than 5 %** of `model_L0`'s own decisions,
  then no outcome difference is attributable to the state representation
  whatever the p-value says, and the phase reports that instead. The divergence
  is measured by running `model_L4` in shadow mode on `model_L0`'s trajectory.
- **H4 (RQ4).** The privacy–utility curve L0 → L1 → L3 → L4 is monotone
  non-decreasing in outcome. Phase 5 and Phase 7 both found the ladder flat for
  knowledge; a flat curve here is the expected result and is reported as one.
- **H5 (robustness).** The arm ranking is stable across V0–V4. Any rank flip is
  annotated in the figure rather than averaged away; a result that only holds
  under V0 is a result about V0.
- **Effect-size halt.** Any Cohen's *d* above **3.0** halts the phase for a bug
  hunt before anything is reported. The literature's reference points are
  +1.25 % AUC and *g* ≈ 0.3; the pre-Phase-0 pipeline's *d* = 26 was a defect
  signature.

### 9.6 Scale

1,000 learners × 3 seeds × 5 simulator variants × 10 arms. The cohort is drawn
before any policy acts and is **identical across arms** within a
(variant, seed) cell, so every comparison is paired on the same learners.

## 10. Study 3 — offline evaluation, bandits and RL (Phase 9)

Registered before any bandit or PPO result existed. `app.research.test_study3`
asserts that the numbers below still equal the constants in `app/rl/reward.py`;
if they diverge, the test fails rather than the document quietly winning.

### 10.1 The reward — fixed before training

```
reward = 100.0 · Δp(curriculum success) − 0.002 · minutes on task − 0.005 · frustrated
```

| term | weight | why this number |
|---|---|---|
| mastery | `MASTERY_SCALE = 100.0` | puts the environment's Δ success probability in probability points |
| time | `LAMBDA_TIME = 0.002` per minute | a measured step gains 0.00008 (= 0.008 scaled) and costs 103 s, so time is ≈ 40 % of the learning it buys |
| frustration | `MU_FRUSTRATION = 0.005` per attempt | half an item's typical gain, applied when the incorrect streak is ≥ 3 |

The scale numbers are descriptive statistics of `rule_improved` on V0 (20
learners, 60 items), read before any learned arm was run. **The mastery term is
deliberately the larger one**: a dominant time penalty rewards ending the
episode rather than teaching.

The mastery term is the simulator's own response model. It reaches an agent as
a *reward* and never as an observation — the observation vector is built by
`app.policy.learned.context`, which admits only gated states and observables. No
deployment has this signal, so Phase 10 needs an observable proxy; that is a
limitation of Study 3, recorded here rather than discovered later.

**Any change to these weights after seeing a result is a deviation** and is
reported in §10.5 with the before-and-after numbers, not edited in place.

### 10.2 The logging policy

`model_L4` at **ε = 0.15**, on V0, one seed. Higher than Phase 8's ε = 0.1
because Study 3 needs support for actions the target policies choose and the
logging arm does not; the price is variance, which §10.3 requires be reported
rather than hidden. Every logged decision carries `ε/120 + (1−ε)·1[greedy]` as
its propensity, the context the target policy would have seen, the action, the
reward, the next context and a terminal flag.

### 10.3 Offline policy evaluation

IPS, SNIPS and Doubly Robust, each with **effective sample size**
`(Σw)² / Σw²` and the weight distribution reported alongside it. An OPE point
estimate without its ESS is not reportable in this project. The reward model
behind DR is a ridge regression on (context, action); it is fitted on the logged
data only.

Because this is a simulator the true on-policy value is computable, so every
OPE estimate is plotted against the ground truth obtained by running the same
target policy on-policy on the same cohort. **Pre-committed check:** DR closer
to truth than IPS. If it is not, the reward model is broken and the phase stops
for a bug hunt before any bandit or RL number is reported.

### 10.4 Learned arms

- `bandit_linucb` (disjoint LinUCB, α = 1.0) and `bandit_lin_ts` (linear
  Thompson sampling, σ = 0.1), both over the same gated context, both with the
  same ridge prior λ = 1.0, both online over the shared action space.
- `rl_ppo`: `stable-baselines3` PPO with library defaults, trained on **V0
  only** and evaluated on V0–V4. Timesteps come from the run profile
  (`smoke` 2 000, `dev` 50 000, `full` 300 000); no hyperparameter sweep is run.

### 10.5 Hypotheses, and what would falsify them

- **H6 (RQ4).** A learned policy reaches mastery in fewer items than the best
  transparent rule arm from Phase 8, on V0, under the same budget and action
  space. Reported as a paired difference with a bootstrap 95 % CI. **A CI that
  includes zero is a null and is reported as one.**
- **H7 (generalisation).** A PPO policy trained on V0 keeps its advantage on
  V1–V4. The literature's standard failure mode is that it does not; an
  advantage that exists only on the training variant is reported as a result
  about V0 and about the simulator, not about teaching.
- **No tuning to a win.** Reward weights, α, σ and the PPO defaults are fixed
  above. If the learned arms lose to the rules, that is the finding. Any tuning
  performed afterwards is reported as exploratory, in a separate table, and
  never replaces the pre-registered comparison.
