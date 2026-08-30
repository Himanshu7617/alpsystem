# Evaluation protocol

## 1. Splitting the archival data (Phase 3)

Written by `ml-service/app/research/prep_archival.py`, read by every downstream
script from `data/processed/splits.json`. **No other script re-splits.** The
file records the seed (`20260821`) and explicit learner-id lists per source.

The protocol follows pyKT's, and the paper says so:

- **Split by learner, never by row.** A learner's interactions are correlated;
  a row-level split leaks the same person into train and test and inflates
  every metric.
- **A held-out test set of 20 % of learners**, plus **5 grouped CV folds** over
  the remaining 80 %. The test set is touched exactly once, at the very end.
- **Within a sequence, a model predicting interaction *t* sees only
  interactions strictly before *t*.** Sequence order is `order_index`, which
  every source has, rather than `timestamp`, which ASSISTments 2009–2010 does
  not.

| Source | learners | test | train | folds |
|---|---|---|---|---|
| `assistments_2009` | 2,968 | 594 | 2,374 | 475 / 475 / 475 / 475 / 474 |
| `assistments_2012` | 22,422 | 4,484 | 17,938 | 3,588 / 3,588 / 3,588 / 3,587 / 3,587 |
| `ednet_kt1` | 23,916 | 4,783 | 19,133 | 3,827 / 3,827 / 3,827 / 3,826 / 3,826 |

Two checks run inside `make prep-data` and fail the build, not a report:

1. **Disjointness.** No learner appears in more than one split, and the splits
   cover every learner.
2. **Rapid-guess accuracy near chance** for sources with a defined chance
   level. EdNet fails this pooled and the failure is recorded as an evidenced
   deviation rather than suppressed — see `docs/data-card.md`.

Figure `f03-07` shows accuracy and sequence-length distributions per split, so
an adversarial split would be visible rather than assumed away.

## 2. Study 1 — the offline predictive protocol (Phase 6)

Run by `make train-baselines`. Written by `ml-service/app/research/study1.py`,
read by the figure pack and by `docs/modeling.md`. Every number lives in
`artifacts/benchmarks/study1-cv.json` or `study1-final.json`.

**The label is `next_correct`** — the outcome of the learner's *next* attempt
(`docs/preregistration.md` §7). Features describe attempt *t* and the decision
they inform is the selection of attempt *t+1*, which is the decision an
adaptive system actually makes. Predicting attempt *t*'s own outcome would make
every within-item behavioural feature unusable by construction and so would
answer RQ2 by definition rather than by measurement.

**Cross-validation** is over the five learner-grouped folds in `splits.json`.
Every training row receives a prediction from a model that never saw its
learner; those out-of-fold predictions are written to
`artifacts/benchmarks/study1-predictions.parquet` and are what every AUC, ΔAUC
and calibration number is computed from. The figures read that file rather than
refitting, so a figure and the JSON cannot disagree.

**Hyperparameters are searched once per source, at the top rung, then held
fixed across every rung.** Searching per rung would confound the ladder: the
increment from adding motor features has to be that, and not a luckier random
search. `RandomizedSearchCV`, budget fixed at 12 draws, on the validation fold
only. (`optuna` is not in the pinned dependency set; BUILD.md Phase 6 step 5
specifies this fallback.)

**A rung that adds no feature for a source is not fitted.** EdNet has no
attempt or hint counters, so its L2 is its L1; no archival source carries motor
telemetry, so none has an L4. Those rungs are recorded as `skipped_rungs` with
the reason. "We tried L2 and it did not help" and "L2 was empty" are different
statements and the output must not blur them.

**Every interval is bootstrapped by learner, not by row**, and every ΔAUC is
*paired*: the same resampled learners score both models. Attempts within a
learner are correlated, so a row-level bootstrap reports an interval several
times narrower than the data supports — which is precisely how a null gets
published as a finding. Ladder p-values are Holm-corrected across the rungs of
one ladder.

**How the ladder is read.** pyKT (K12) attributes 1–2 % AUC to protocol
variation alone, so an increment whose 95 % CI does not exclude **+2 %** is
reported as a null, however significant it is. `clears_protocol_band` in the
JSON is that verdict and `f06-02` draws the band. SAINT+ (K8) reports +1.25 %
AUC from timing features and is drawn as the reference line: a behavioural
increment far above it should be read as a bug before a discovery.

**Calibration is reported ahead of AUC** (BUILD.md Phase 6 step 4). The output
drives a decision — a policy targeting a 0.7 success probability needs 0.7 to
mean 0.7 — and AUC is invariant to exactly the monotone distortion that breaks
that. ECE and reliability curves are given raw and after isotonic calibration
fitted on the training folds only.

**Subgroups are always reported** by prior-ability tercile. The RL-tutor
literature (M76) found the benefit fell on lower performers specifically, so an
aggregate null can hide a real subgroup effect.

**The test set is touched once**, at the end, into
`artifacts/benchmarks/study1-final.json`. That file is not regenerated during
tuning, and `study1.py --no-test` exists so a development run cannot touch it
by accident.

### Sampling caps, and why they cost nothing

Six models at five rungs over EdNet's 4.9 M rows is hours of CPU to move an AUC
in the third decimal. Each source is capped at **5,000 learners / 250,000
rows**, sampled by learner so sequences stay whole, seeded, and recorded in
every output file. The justification is `f06-10`: the learning curve on
`assistments_2009` is flat from roughly a thousand learners (AUC 0.7112 at 949,
0.7132 at 1,899). The figure is the evidence for the cap rather than an
assertion about it.

The deep sequence models carry a heavier caveat, stated in every result they
produce: **one CV fold, at most 3,000 training learners, 3 epochs, CPU, no
hyperparameter search.** They are a bounded comparison, not a tuned one. Also
recorded: pyKT's SAINT consumes exercise, concept and response but **not**
elapsed time, so what runs here is SAINT, not SAINT+. The timing channel
reaches the tabular models through the L1 rung instead, which is what the
ablation ladder is for.

### Recorded deviations

- **MLflow uses a SQLite file, not the file store.** BUILD.md §3 chose "file
  backend `./mlruns`, no tracking server". MLflow 3.15 refuses to open a file
  store at all — it is in maintenance mode and raises unless
  `MLFLOW_ALLOW_FILE_STORE` is set. `sqlite:///mlruns/mlflow.db` keeps the
  actual decision (one file under `./mlruns`, no server, `mlflow ui` works)
  on a backend that still receives fixes. Browse it with
  `mlflow ui --backend-store-uri sqlite:///mlruns/mlflow.db`.
- **Two upstream libraries needed shims** to run at all, both quarantined at the
  top of `baselines.py` and both covered by a test that fails if the upstream
  bug is fixed, so the workaround cannot outlive its cause. pykt-toolkit 0.0.38
  has a stray `from turtle import forward` that makes every model unimportable
  without a GUI toolkit; pyBKT 1.4.3 breaks at import against scikit-learn 1.9
  and at fit time against NumPy 2.
- **The majority-class floor reports AUC 0.5 by definition.** It predicts a
  constant within each fold, but the folds' training base rates differ slightly,
  so *pooling* its out-of-fold predictions turns a constant model into a weak
  fold-identity detector. The pooled value is kept beside it, labelled as the
  artefact it is.

---

## 3. Study 2 — the state-validity protocol (Phase 7)

Run by `make train-states`. Written by `ml-service/app/research/study2.py`,
judged by `ml-service/app/model/validation_gate.py`. Every number lives in
`artifacts/benchmarks/study2-heads.json` or
`artifacts/evaluation/validation-gate.json`; the thresholds live in
`docs/preregistration.md` §8 and were fixed before the encoder was fitted.

**The problem this protocol exists to solve.** A state head is created by adding
an output layer and a name. Whether it *measures* the construct its name claims
is a separate question, and the literature review's complaint about this
territory is that the question usually goes unasked. So each head here is
supervised by its own label — never by correctness — and no head reaches a
policy until it has passed pre-registered criteria.

| Head | Label | Judged on |
|---|---|---|
| knowledge | `next_correct` | `assistments_2012` (real) |
| engagement | `solution_behaviour` of the **next** attempt (RTE, §1) | `assistments_2012` (real) |
| confidence | `probe_confidence ≥ 3`, post-submission self-report | `sim-V0` (simulated) |

**Fatigue has no head.** The pre-registered H2 rule fired in Phase 5: no
archival source shows a within-session accuracy decline, so the construct was
dropped before modelling rather than defended. `validation_gate.DROPPED_STATES`
carries the reason into the gate report and figure `f07-10` draws the
observable that was measured instead of an estimate that does not exist.

**The engagement label is shifted forward one attempt.** Rapid-guessing is
defined by a response-time cut, and response time at attempt *t* is an L1 model
input at attempt *t*. A head predicting the label of the attempt whose response
time it can see scores near-perfectly by re-deriving the threshold — that is
arithmetic, not a state estimate. The shift also makes the head answer the
question a policy actually asks: is this learner about to disengage?

**Folds.** `fold0`–`fold3` train. `fold4` is halved by learner: the isotonic
calibrator is fitted on one half and every reported number is measured on the
other, so no criterion is scored on the data that shaped it. **The test set is
not read in this phase** — it is dropped from the frame rather than merely
avoided.

**Calibration is a gate criterion, not a footnote.** An adaptive policy acting
at a 0.7 threshold needs 0.7 to mean 0.7, and AUC is invariant to exactly the
monotone distortion that breaks that. ECE is reported before and after isotonic
for every head.

**Uncertainty** is MC dropout over 20 stochastic passes, reported as a standard
error per row so Phase 8 can decline to act on an unreliable state. It was
chosen over a five-model ensemble because it is one training run for the same
purpose.

**The gate is code, not prose.** `validation_gate.evaluate()` applies four
pre-registered criteria — correlation with the head's own label with a
learner-clustered CI, calibration, Wise & Kong's negative criterion that an
effort index must not track ability, and discriminant validity of confidence
against knowledge. `Gate.filter_states()` **removes** a rejected state from what
a policy can see rather than zeroing it: a policy handed `engagement = 0.0` is
still conditioning on engagement. A missing gate report admits nothing.

**Every rung of the ladder is retrained**, which gives the privacy–utility curve
at the level of the individual state (`f07-08`) rather than only at the level of
the predictive model.

## 4. Study 4 — the closed-loop protocol (Phase 8)

The results this section used to describe were produced by the pre-Phase-0
pipeline and are **not citable**: `evaluate_policies.py` selected difficulty for
the `ml_based` arm with the simulator's own response function (Defect 1), so the
reported win measured an oracle beating a heuristic. That file has been
rewritten; what follows is what it does now.

### 4.1 What varies, and what is held constant

Ten arms share one action space — `{difficulty 1..10} × {same | next |
prerequisite concept} × {no intervention | hint | worked example | break}`, 120
actions — one budget, one curriculum and one generative model. The arms differ
only in the state they condition on. Four of them (`model_L0`, `model_L1`,
`model_L3`, `model_L4`) differ from each other in *nothing but the rung the
knowledge estimate was trained at*, which is the RQ3 contrast.

The selection rule for every model arm is the same inversion: the calibrated
knowledge head gives P(correct on the next item); with the mean difficulty of
the recent items as the reference, θ̂ = b̄ + logit(p̂), and the item served is the
one whose difficulty puts predicted success in the 0.70–0.75 band. An estimate
with a large MC-dropout standard error is shrunk toward the concept's median
item, so an arm can act on knowing that it does not know.

### 4.2 Termination, censoring and inclusion

- **Mastery** is per concept: the response model puts the learner's probability
  of answering that concept's median-difficulty item correctly at ≥ 0.80. A run
  ends when all six curriculum concepts clear it.
- **The curriculum is six prerequisite-closed concepts**, derived from item-bank
  coverage rather than listed by hand. The old criterion required all
  thirty-six concepts while advancing one concept per item, which is why every
  arm reported ~0 % mastery — a broken criterion, not a policy result.
- **A curriculum concept must be one the difficulty action can act on.** Each
  needs ≥ 3 items spanning ≥ 1.0 logits of *b* (`closed_loop.MIN_ITEMS_PER_CONCEPT`,
  `MIN_DIFFICULTY_SPREAD`), and a candidate whose prerequisite closure contains
  a concept that fails the floor is skipped rather than dragging it in. Without
  the floor the selected curriculum included `stacks_and_queues` — two items at
  an identical *b* — where "serve difficulty 3" and "serve difficulty 9" return
  the same item, so a sixth of every rollout was a segment on which no policy
  could possibly differ from `random`. This is a property of the item bank and
  it bounds what any difficulty-selection result in this project can show; it
  belongs in the limitations, not in a footnote.
- **Right-censoring at 200 items** is reported next to every mean.
- **Inclusion:** learners who have already mastered the whole curriculum before
  their first item are screened out (`docs/preregistration.md` §9.2). They are
  41 % of draws from the calibrated ability distribution and they put the same
  ceiling on every arm.
- **Time is on-task seconds** — response time, inter-item gaps and the time an
  intervention costs. Between-session gaps are wall clock, not learner effort.

### 4.3 Why knowledge gain is a co-primary

Phase 4 fitted the learning-rate distribution to the archival practice curves
and got log-mean −7.26 with a floor at 1e-4: the median simulated learner gains
about 0.001 logits per item, so crossing a mastery threshold takes on the order
of a thousand items. Inside a 200-item budget most learners are censored, and a
censored primary outcome reports "no difference" whichever policy ran. Latent
knowledge gain over the curriculum is the uncensored quantity that
items-to-mastery is a threshold crossing of, and it was registered as a
co-primary before the first rollout — not after seeing the censoring.

### 4.4 What this design cannot answer

**The calibrated generative model has no desirable-difficulty effect.** Learning
gain in the simulator is `rate × (1 if correct else 0.3) × (3.5 − knowledge)`,
which is monotone increasing in the probability of a correct answer. A policy
that targets a 0.72 success band is therefore *penalised* relative to one that
serves the easiest available item, and no variant V0–V4 changes that: V2 removes
the fatigue-accuracy channel, V1 varies discrimination, V3 dilutes the
behavioural signal and V4 makes the learning rate non-stationary. None of them
makes intermediate difficulty optimal.

Two consequences, and both are limitations of the model rather than results:

1. Comparisons **between** the band-targeting arms and the baselines are
   partly a comparison of how easy the items each arm serves are.
2. The `model_L0` versus `model_L4` contrast is unaffected, because both target
   the identical band with the identical rule table. That is the contrast the
   phase is designed around, and it is the one to read.

A generative model with a genuine desirable-difficulty optimum would be a sixth
variant and is named in the future-work section rather than quietly added here.

### 4.5 Statistics

The cohort is drawn before any policy acts and is identical across arms within
a (variant, seed) cell, so every comparison is **paired on the same learners**:
Wilcoxon signed-rank on the paired differences, Bonferroni-corrected across all
arm pairs, with a bootstrap 95 % CI on the mean difference. Any Cohen's *d*
above 3.0 exits the run non-zero before anything is written — the literature's
reference points are +1.25 % AUC and *g* ≈ 0.3, and the pre-Phase-0 pipeline's
*d* = 26 was a defect signature.

Whatever Phase 8 reports, no closed-loop number is a claim about human learning
gain. There are no participants in this project.


---

## 5. Study 3 — the offline-evaluation protocol (Phase 9)

Study 4 asked whether a *rule* over a better state teaches better. Study 3 asks
whether **learning the decision** beats writing it down, and it has to answer
that from logged data before it is allowed to answer it by running anything.

### 5.1 The logged dataset

`model_L4` at ε = 0.15 (preregistration §10.2) runs the same closed loop and
writes `artifacts/datasets/logged-bandit-data.parquet`: one row per decision
with the context the policy saw (`ctx_*`), the action, **the propensity it was
chosen with**, the reward, the next context (`nxt_*`), a terminal flag, and the
policy, simulator and seed that produced it.

The propensity is the whole point. Under an adaptive policy a hint is given
*because* the learner is struggling, so action and outcome share a cause and a
plain replay of the log measures that shared cause. ε is raised from Phase 8's
0.1 because an action the logging policy never takes is an action offline
evaluation can say nothing about.

### 5.2 The estimators, and the number that decides whether to believe them

IPS, SNIPS and Doubly Robust, each reported with its **effective sample size**
`(Σw)²/Σw²` and the spread of its weights. DR's reward model is a ridge
regression on context × action interactions, fitted on the logged data only.

Two facts about this action space make the ESS the headline rather than a
footnote:

- **A deterministic target barely overlaps the log.** Evaluated as a hard
  argmax, the trained bandits matched the logged action on 0.2 % of decisions
  and the ESS was 1.1–5.2 out of 2,383. The estimators are therefore applied to
  the arm *as it would be deployed* — ε-greedy at the same ε — using the
  stochastic-target form `w = π(a|x)/p(a|x)`.
- **Weights are clipped at 20** (`ope.WEIGHT_CLIP`). Unclipped, one exploratory
  decision the target happens to agree with carries weight 681 — a single draw
  outvoting six hundred. Clipping buys a usable variance with a known downward
  bias, and both the clipped and the raw ESS are reported so the reader can see
  how much of the estimate rests on how few decisions.

### 5.3 Validating the estimators against the truth

This is a simulator, so the on-policy value is computable: every target policy
is also *run*, on the same cohort, at the same ε, and `f09-02` plots the
estimate against the truth. §10.3 pre-commits to DR beating IPS in mean
absolute error; if it does not, the run prints a warning naming the reward model
and the phase stops for a bug hunt rather than reporting a Study 3 number.

This check is only possible in simulation. It is the methodological point of
the phase, and it is the reason the OPE machinery is trusted at all before it is
ever pointed at data from a deployment.

### 5.4 The learned arms

`bandit_linucb` and `bandit_lin_ts` learn online over the same gated context the
model arms condition on; `rl_ppo` is `stable-baselines3` PPO trained on V0 and
evaluated on V0–V4. Reward: preregistration §10.1, fixed before training. The
mastery term is the simulator's response model and reaches an agent **only as a
reward** — the observation vector is `app.policy.learned.context`, which admits
gated states and observables and nothing else. A deployment has no such signal,
which is Phase 10's problem and is recorded as a limitation rather than
discovered there.

Comparisons against the best Phase 8 arm are paired on the same learners with a
bootstrap 95 % CI, exactly as in §4.5. A learned arm that does not beat the rule
is reported as not beating the rule.

---

## 6. The statistical protocol (Phase 11)

Everything below is computed by `make stats` into
`artifacts/evaluation/statistical-report.json` and drawn by `make figures` into
`artifacts/figures/11-stats/`. No number in this section is typed by hand.

### 6.1 Effect sizes are unpaired, and that is a correction

Every arm sees the same cohort, so a *paired* comparison is available and §4.5
uses it as the significance test — correctly, because the pairing is real. It is
the wrong **effect size**: dividing by the standard deviation of the
within-learner difference removes the between-learner variance a reader assumes
is in the denominator. The learner-level mixed models put a number on how much
that is: **97.7 %** of the variance in knowledge gain is between learners. That
ratio is how the pre-Phase-0 version of this repository reported *d* = 26.

Phase 11 therefore headlines **Hedges' *g* on independent samples**, with a
percentile CI from **10,000 bootstrap resamples**, alongside Welch's *t* and
Mann-Whitney *U*. `app/research/test_stats.py` holds a fixture with a large
learner effect and a small treatment effect and asserts the two statistics do
not agree.

### 6.2 Mixed-effects models

| Model | Data | Random effects | Fitted on |
|---|---|---|---|
| Correctness | out-of-fold Study 1 predictions, per dataset | learner + item, crossed | ≤ 8,000 rows, ≤ 200 learners, ≤ 200 items |
| Closed-loop per item | `decision-log-v0.parquet` | learner + item, crossed | ≤ 8,000 rows |
| Learner outcomes | `policy-per-learner-v0.csv` | learner (repeated across arms) | all 30,000 rows |

statsmodels has no crossed-effects syntax; the documented workaround — one
artificial group over every row with both factors as variance components — is
what `stats.crossed_mixedlm` does, and it builds that design densely, which is
why the crossed fits are subsampled and say so in their output (`ponytail:` in
`stats.py`). The learner-level models are not subsampled.

The correctness model enters the lowest rung's out-of-fold log-odds and the
*increment* the highest live rung adds to it, both standardised. That increment
is the mixed-model form of the ablation ladder's ΔAUC: **+0.015 to +0.024** in
probability of a correct answer per standard deviation, positive on all four
datasets, its CI excluding zero on three of them (ASSISTments 2009 is
*p* = 0.084). Per-item variance shares: learner 2–6 %, item 5–9 %, residual
86–95 % — the clustering is small per attempt and decisive in aggregate, which
is exactly the case where treating attempts as independent manufactures
significance.

### 6.3 Multiple comparisons

Holm within each outcome family — nine arms against `rule_improved`, one family
per outcome. Raw and corrected *p* are both reported, per comparison and in
`f11-06`. Nothing changes status under correction in the current run: knowledge
gain and items-to-mastery have no significant comparison before or after, and
time-to-mastery has all nine either way.

### 6.4 Censoring

The 200-item budget censors **91–92 %** of learners in every arm, so the mean of
`items_to_mastery` is a lower bound on a difference, not a difference. Phase 11
reports it as a time-to-event outcome: Kaplan-Meier per arm, restricted mean
survival to the budget (187–189 items across all ten arms), and a log-rank test
against the arm with the highest mastery rate. The uncensored co-primary,
`knowledge_gain`, is reported beside it everywhere.

### 6.5 Power

Achieved power is computed for what was observed, not assumed. At *n* = 3,000
per arm the knowledge-gain comparisons run at **0.05–0.07** power — that is what
"no detectable difference" means when the observed *g* is 0.01: the sample is
large and the effect is not there.

The number the future-work section needs: a real RCT for **g = 0.3** at 80 %
power and α = 0.05 needs **176 learners per arm, 352 in total**.

### 6.6 Sensitivity

Six analyses, `make stats` → `artifacts/evaluation/sensitivity.json`, one panel
each in `f11-05`. Five are recomputed from files already on disk; the mastery
threshold τ is the one that reruns the closed loop, because mastery is decided
inside the environment rather than stored as a cut point, and it obeys the run
profile like every other expensive target.

| Knob | Varied over | Ordering of the arms |
|---|---|---|
| Probe noise sd | 0.25 – 2.0 | stable (flat — the gate rejected the probe-derived head) |
| Mastery threshold τ | 0.70 / 0.80 / 0.90 | **moves** |
| Item budget | 25 – 200 | stable |
| Simulator variant | V0 – V4 | **moves** |
| Target-policy ε | 0.0 – 0.5 | SNIPS stable, IPS scales with ε |
| Reward weights λ, μ | ×0.25 – ×4 | **moves** at λ×4 |

A knob whose ordering moves is a knob the conclusion depends on, and the paper
says so rather than quoting the pre-registered level alone. The variant row is
the important one: it is the Defect-3 answer — a closed-loop finding that only
holds on V0 is a property of one generative model.

### 6.7 The figure and table pack

`ml-service/app/research/plotstyle.py` holds the one style: the Okabe–Ito
colourblind-safe palette, 300 DPI, and a PDF beside every PNG. Every `save()`
appends the figure's caption, its script and the files that script reads to
`artifacts/figures/manifest.jsonl`, and `make figures` renders
`artifacts/figures/INDEX.md` from it — 104 figures, each with a provenance line
recorded by the code that drew it. The same pass asserts that no figure reads a
file outside `artifacts/`, `data/`, or the two checked-in inputs a schema
illustration legitimately shows.

Tables: `artifacts/evaluation/tables/*.csv` with a LaTeX twin of each —
model comparison, ablation ladder, state validity, policy comparison, OPE,
statistical tests, censoring.
