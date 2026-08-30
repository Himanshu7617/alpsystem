# Simulator v2

The generative learner model behind Study 4. It exists for one reason: to answer
the closed-loop question archival data cannot — *what happens when the policy
changes what is served next* — and it is built so that answer is not an artefact
of its own assumptions.

- Code: `ml-service/app/research/simulator.py`
- Calibration: `ml-service/app/research/calibrate_simulator.py`
- Generation: `ml-service/app/research/generate_sim.py` (`make simulate`)
- Parameters: `artifacts/datasets/simulator-params-v2.json`
- Fit evidence: `artifacts/datasets/simulator-calibration-ks.json`,
  `artifacts/figures/04-simulator/`

**No human participants are involved anywhere in this document.** Every number
here describes generated data.

---

## 1. Generative model

Figure `f04-01` is the DAG. In symbols, for learner *l*, concept *c*, item *i*,
position *t* within a session:

**Learner, drawn once.** Ability `θ_l ~ N(μ_θ, σ_θ)`; learning rate
`η_l ~ logN(μ_η, σ_η)`; fatigue susceptibility `φ_l ~ logN(0, 0.4)`; total
interactions drawn from the empirical per-learner distribution.

**Traits, drawn once and independently of ability** — the whole Defect 3 fix:
per-learner log-RT offset (`speed`), pointer travel speed (`move_speed`, px/ms),
cursor jitter (px), re-reading tendency, baseline indecision, per-item
distraction probability, and device (mouse or trackpad).

**Latent state.** `K_{l,c}` starts at `θ_l + N(0, 0.5)` per concept and updates
`K ← K + η_l·g·(3.5 − K)` with `g = 1` after a correct answer and `0.3`
otherwise; between sessions `K ← 0.96·K`. Fatigue `F ← F + 0.045·φ_l` per item
inside a session and resets at the session boundary. Confidence is an EWMA of
correctness; engagement decays with fatigue and drops on off-task events.

**Outcome.** `P(correct) = γ + (1 − γ − s)·σ(a_i·(K_{l,c} − b_i − β_F·F))`, with
guess `γ`, slip `s`, item difficulty `b_i` from the calibrated item bank, and
discrimination `a_i` (fixed at 1 outside variant V1).

**Response time.** `log RT = ρ₀ + ρ_d·|K − b_i| + ρ_F·F + ρ_L·stem_z_i +
speed_l + N(0, σ_RT)` — log-normal, located by ability–difficulty distance,
fatigue, item stem length and the learner's own speed trait.

**Behaviour.** Every behavioural signal is a **sum of four independent sources**:
a state term (uncertainty `1 − |2p − 1|`, fatigue), a trait term, an item term
(long stems are re-read), and a contaminating off-task channel (a distraction
event produces idle time and a visibility change, and is drawn from a trait, not
from the state).

**Cursor path.** A quadratic Bézier from the pointer's resting position to the
chosen option, with the control point pulled toward the competing option in
proportion to uncertainty, minimum-jerk timing, per-sample Gaussian jitter from
the learner's trait, and a hesitation pause. Sampled at 50 ms — **only while the
pointer moves**, because that is when a browser fires `pointermove`. Travel time
is `path length / move_speed`: a motor trait, not a function of how long the
learner thought. Deriving the sample count from response time instead would
rebuild Defect 3 inside the motor channel.

**Probes.** Confidence probes on a random 12 % of items, effort probes every 15,
mapped from the latent through a bias and Gaussian noise onto a 4- or 5-point
scale — self-report is a noisy criterion, never the latent value
(`docs/preregistration.md` §3).

**Feature extraction.** The simulator emits an event stream
(`ITEM_PRESENTED … CURSOR_SEGMENT … ANSWER_SUBMITTED`) and calls
`app.features.extract.extract()` — the same function the backend calls in
production. It never writes a feature itself. `test_simulator.py` asserts this
by spying on the extractor.

---

## 2. Calibration

`make simulate` fits against `data/processed/assistments_2012.parquet`
(2,593,636 interactions, 22,422 learners). That source was chosen because its
mean sequence length (115.7) matches the simulated budget of ~120 items per
learner and it carries timestamps, so sessions and within-session decay are
observable at all.

| Parameter | Fitted | How |
|---|---|---|
| `ability_mean`, `ability_sd` | 1.46, 1.83 | grid search on the accuracy KS, then six moment-matching rounds against the simulator itself |
| `guess` | 0.25 | **structural**, 1/4 for a four-option item |
| `slip` | 0.101 | upper asymptote of the empirical accuracy-vs-(θ̂ − b̂) curve |
| `learning_rate_log_mean/sd` | −7.26, 2.13 | per-learner practice-curve slope, converted to knowledge gain per item |
| `rt_intercept` | 10.95 | least squares on log RT |
| `rt_beta_distance` | −0.227 | per logit of \|θ̂ − b̂\| |
| `rt_beta_fatigue` | −0.590 | per unit of the within-session fatigue proxy |
| `rt_sigma` | 0.911 | residual SD, minus the variance carved out for the speed trait and stem effect |
| `fatigue_accuracy_beta` | **0.0** | see §4 |
| session lengths, learner totals | 21 empirical quantiles each | directly from the source |

Not identifiable from archival data, and recorded as such in the params file:
`fatigue_growth` (fixed at 0.045/item — nothing observable separates the growth
rate from its coefficient), every `trait_*` (no archival source records motor
telemetry or device), the probe noise and bias (no archival source contains
self-report), and `rt_beta_stem` (no archival source publishes item text).

### The acceptance gate is a distribution match

| Distribution | KS *D* (calibration sample) | KS *D* (shipped V0 dataset) | Limit |
|---|---|---|---|
| per-learner accuracy | 0.058 | 0.042 | 0.15 |
| log response time | 0.049 | 0.049 | 0.15 |
| sequence length | 0.038 | 0.038 | 0.15 |
| within-session accuracy slope | 0.107 | 0.107 | 0.15 |

All four pass, so `DOCUMENTED_MISMATCHES` in `calibrate_simulator.py` is empty.
A *D* above the limit exits the calibration non-zero; it can only be accepted by
registering it there with its evidence, which also puts it in the manifest.
Figures `f04-02`–`f04-05` overlay each pair with its *D* annotated.

---

## 3. Breaking the circularity (Defect 3)

The pre-Phase-0 simulator computed `option_changes`, `pause_duration`,
`mouse_distance`, `mouse_speed` and `tab_switches` directly from `knowledge`,
`fatigue`, `confidence` and `attention_span` plus Gaussian noise. Any model
trained on that data *must* find behaviour predictive of the latent state,
because the behaviour was the state re-encoded.

Figure `f04-07` and `artifacts/datasets/simulator-recoverability.json`:

| Measure | Legacy simulator | Simulator v2 |
|---|---|---|
| mean mutual information, observable ↔ latent | 0.298 nats | 0.022 nats |
| latent **fatigue** recoverable from all observables (CV R²) | **0.989** | 0.087 |
| latent **knowledge** recoverable from all observables (CV R²) | 0.177 | 0.054 |

The fatigue row is the defect in one number: in the legacy model the latent was
recoverable almost perfectly, which is what made "behavioural analytics helps"
true by construction.

**MI above zero is not the defect.** The research premise is that behaviour
carries state information; the defect was that it carried *only* state
information. One pair (hover time ↔ knowledge) is marginally higher in v2 than
in the legacy model, and pointer velocity is comparable, because hesitation
pauses are a modelled channel. What changed is that the latent can no longer be
inverted out of the behaviour vector, because traits, item effects and off-task
events now carry variance the state does not explain.

---

## 4. What the real data said, and what it cost the model

**There is no within-session accuracy fatigue in the calibration source.** The
mean within-session slope of difficulty-residualised correctness is **+0.0023**
per item, and the per-learner practice curve accounts for +0.00068 of it, so the
residual after removing learning is **+0.0016** — *positive*. Fitting
`fatigue_accuracy_beta` to that gives **0.0**: in these data, learners do not get
less accurate as a session goes on, they get slightly better (a warm-up effect).
The fitted response-time coefficient points the same way: `rt_beta_fatigue` is
**−0.590**, so later items are answered *faster*, not slower.

Consequences, stated before Phase 5 sees a result:

- The fatigue construct's effect on **accuracy** is zero in V0 by fit, not by
  choice. `docs/preregistration.md` H2 anticipated exactly this and pre-committed
  to reducing or dropping the construct rather than defending it. Phase 5's
  figure `f05-06` is the decision point.
- Variant **V2** ("fatigue affects speed only") therefore differs from V0 only in
  the response-time channel. It is retained because the calibration is a fit, not
  a law: a different population could show accuracy fatigue, and V2 is where that
  assumption is varied.
- Any Phase 8 policy that acts on estimated fatigue must be judged on this: in
  the calibrated world there is little accuracy for it to protect.

**The option layout limits two motor features.** The production quiz renders
options as a single-column list, so all option centres share an *x* coordinate.
Perpendicular deviation from the start→target line (`auc_toward_nonchosen`,
`max_deviation`) and horizontal reversals (`x_flips`) are consequently weakly
identified — a straight-line path up a list has little lateral geometry.
Hesitation shows up instead in hover time, pauses and path length. This is a
property of the real interface, discovered by simulating its real geometry, and
it is a limitation for RQ2 that Phase 5 must respect rather than a simulator bug.

---

## 5. Mis-specification variants

Five sibling generative models. **Every closed-loop result in Phases 8–9 is
reported against all five**; a policy ranking that flips between them is an
artefact of one generative model, not a finding.

| Variant | Mis-specification | Rationale |
|---|---|---|
| `V0` | the calibrated baseline | — |
| `V1` | 2PL link: item discrimination `a_i ~ logN(0, 0.35)` instead of 1 | the Rasch assumption of equal discrimination is the most common IRT simplification |
| `V2` | fatigue affects speed only, never accuracy (`β_F = 0`, `ρ_F × 1.5`) | tests whether policies that act on fatigue survive when fatigue does not touch accuracy |
| `V3` | behavioural signals are half noise: state weight × 0.5, trait weight × 1.5 | tests whether a behavioural policy survives when behaviour mostly reflects who the learner is, not what they know |
| `V4` | non-stationary learning rate, drift `~ N(0, 0.5)` per learner | tests the stationarity assumption every knowledge-tracing model makes |

Generation defaults to 1,000 learners per variant. Sequence lengths are drawn
from the calibrated per-learner distribution (mean ≈ 120, matching BUILD.md's
"1,000 learners × 120 items"); a **fixed** 120 items per learner would have
failed the sequence-length KS check by construction, so the budget is matched in
expectation rather than exactly.

`artifacts/datasets/sim-v2-manifest.json` records the seed, the git SHA of the
checkout that generated the data, the SHA-256 of the parameter set, the row count
per variant, and the V0-vs-real KS results. The `.parquet` files themselves are
git-ignored: they are ~110 MB and `make simulate` rebuilds them in about
two and a half minutes.

---

## 6. Limitations

1. **The calibration target is not the target domain.** ASSISTments 2012–2013 is
   middle-school mathematics; the item bank is undergraduate computer science.
   What transfers is the *shape* of learner behaviour — accuracy spread, response
   time distribution, session structure, within-session dynamics — not the
   content. No archival source in this project covers the bank's domain
   (`artifacts/datasets/archival-bank-concept-map.json`: 0 of 2,161 skills map).
2. **Item difficulties are author priors.** Phase 1's Rasch calibration has no
   real responses to the bank's items, so `b_i` is a z-scored author judgement.
   The simulator inherits that uncertainty.
3. **Motor, trait and probe parameters are assumed, not fitted.** They are
   plausible, they are pre-registered, and they are varied in V3 — but no
   archival dataset could constrain them.
4. **Fatigue saturates.** At 0.045 per item it reaches its 1.0 ceiling after
   roughly 22 items, so in the long tail of the session-length distribution the
   fatigue latent is pinned. With the fitted accuracy coefficient at zero its
   only remaining effect there is on response time. The growth rate is one of
   the parameters archival data cannot identify.
5. **Distraction is a single lumped channel.** One Bernoulli per item producing
   idle time and a visibility change. Real off-task behaviour has structure
   (bursts, time-of-day, notification-driven) this does not model.
6. **A simulated closed-loop result is not a randomised controlled trial.** It
   establishes that a validated state *changes decisions* in a direction a
   calibrated model predicts to be beneficial, and bounds that effect under
   mis-specification. Claims about real learning gain require the RCT in the
   future-work section.
