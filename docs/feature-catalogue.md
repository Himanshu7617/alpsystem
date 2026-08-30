# Feature catalogue

**Generated** — `python -m app.research.feature_catalogue --write-doc`. The source of
truth is `ml-service/app/research/feature_catalogue.py`; edit that, not this file.

One row per modelling input. A feature with an empty rationale cannot be built:
`validate()` refuses it and `make features` calls `validate()` before writing a row
(BUILD.md Phase 5 step 1).

The **rung** column is the ablation ladder of Phase 6 and, read the other way, the
consent classes of [`preregistration.md`](preregistration.md) §4 — so it is also the
privacy–utility curve of RQ4. Citation keys (`K8`, `B16`, `P4`, …) are the
[literature review](requirements/Literature-Review-Adaptive-Learning-Behavioural-Analytics.pdf)'s own.

Latent state (`latent_*`), self-report probes (`probe_*`) and the label (`next_correct`)
are **not in this table** and never enter a model. They are carried beside the features
for evaluation only, and `audit_leakage.py` fails the build if one crosses over.

## L0 — correctness (8 features)

| Feature | Definition | Source | Privacy | Rationale |
|---|---|---|---|---|
| `correct` | 1 if the attempt was scored correct, else 0. | real + simulated | low | The outcome every knowledge-tracing model since K1 (Corbett & Anderson 1994) conditions on. Present-tense: it describes attempt t, and the label is at t+1. |
| `attempt_count` | Number of attempts the learner made on this item before it was scored. | real + simulated | low | Repeated attempts on one item are the classic wheel-spinning signal (B3, B4); PFA (K14's logistic family) counts successes and failures separately for this reason. |
| `prior_accuracy` | Mean of `correct` over the learner's attempts strictly before this one. | real + simulated · history, `shift(1)`-guarded | low | The ability term of every logistic knowledge-tracing model; K13 and K14 both find that a well-specified logistic ability feature is hard for deep models to beat at this data size. |
| `prior_attempts` | Count of the learner's attempts strictly before this one. | real + simulated · history, `shift(1)`-guarded | low | The denominator `prior_accuracy` must be read against: 1.0 from one attempt and 1.0 from two hundred are different states. Also the practice-count term of PFA. |
| `skill_prior_accuracy` | Mean of `correct` over the learner's earlier attempts on the same skill / concept. | real + simulated · history, `shift(1)`-guarded | low | Per-skill mastery is what BKT (K1, K4) tracks and what an adaptive policy actually selects on. A learner strong overall can be weak on the concept in front of them. |
| `skill_prior_attempts` | Count of the learner's earlier attempts on the same skill / concept. | real + simulated · history, `shift(1)`-guarded | low | Opportunity count. The x-axis of a learning curve, and the exposure term PFA needs to separate a low mastery estimate from a thin one. |
| `item_difficulty` | Item difficulty on the logit scale. Archival: `-logit(p)` from the item's accuracy among **training** learners, shrunk toward the source mean with a 20-response prior. Simulated: the generative `item_b`. | real + simulated | low | The Rasch b parameter. P1 and the CAT literature select items by ability-difficulty distance; without it, response time is uninterpretable because a slow response to an easy item and a slow response to a hard one mean opposite things. |
| `n_options` | Number of answer options presented. | simulated only | low | Sets the chance level, which is what a rapid-guess accuracy check is read against (P7, P8). Phase 3 recorded that neither ASSISTments release stores it. |

## L1 — timing (8 features)

| Feature | Definition | Source | Privacy | Rationale |
|---|---|---|---|---|
| `log_response_time` | log1p of the winsorised response time in seconds. | real + simulated | low | SAINT+ (K8) reports +1.25 % AUC from elapsed time and is the honest reference point for this whole project. Logged because raw response time is heavy-tailed. |
| `log_rt_z_item` | `log_response_time` standardised by the item's mean and sd among **training** learners. | real + simulated | low | The lit-review's 'include — core' list names *log response time standardised per item* explicitly (§7.5). Raw speed confounds a fast learner with an easy item; P1's joint speed-accuracy model separates them the same way. |
| `rt_drift` | `log_response_time` minus the mean `log_response_time` over the learner's earlier attempts. | real + simulated · history, `shift(1)`-guarded | low | Speed relative to the learner's own baseline rather than the population's. P3 relaxes van der Linden's constant-speed assumption for exactly this reason; a fatigue-aware model has to break it (P1's stated limitation). |
| `log_lag_time` | log1p of the seconds since the learner's previous attempt. | real + simulated | low | SAINT+'s second temporal feature (K8): the spacing between attempts carries forgetting and session structure. Needs a wall clock. |
| `reading_time` | Seconds from item presentation to the first interaction with it. | simulated only | medium | Cocea & Weibelzahl (B12) build disengagement detectors on reading time against stem length: too short to have read the question is the cleanest off-task evidence a log holds. |
| `decision_latency` | Seconds from the first option selection to submission. | simulated only | medium | Named in the lit-review's core include list (§7.5) and the RQ2 question. B17 links latency-to-commit to drift-diffusion decision uncertainty. |
| `time_after_last_interaction` | Seconds between the last recorded interaction and submission. | simulated only | medium | A long silent tail before submit is hesitation or absence, and it separates the two when read with `idle_time`. Off-task time features are B2's original ITS signal. |
| `response_time_vs_estimate` | Response time divided by the item's authored time estimate. | simulated only | low | A normative rather than empirical speed reference — the same shape as the Wise & Kong threshold (P7), which is a fraction of an item's normative time, not of the learner's. |

## L2 — interaction (9 features)

| Feature | Definition | Source | Privacy | Rationale |
|---|---|---|---|---|
| `option_changes` | Number of `OPTION_CHANGED` events before submission. | simulated only | medium | Answer-changing is a measured metacognitive signal with a known asymmetry: B26 and B27 find wrong-to-right changes dominate and that who benefits depends on ability. The lit-review lists option-change sequences *conditioned on ability* as experimental-include. |
| `option_change_entropy` | Shannon entropy (bits) of the distribution of options selected during the attempt. | simulated only | medium | Separates one decisive correction from cycling through every option. Two changes with the same count carry different information; B20's catalogue of cursor measures makes the same distinction for paths. |
| `option_revisits` | Times an already-selected option was selected again. | simulated only | medium | The revisit term of the hierarchical speed-accuracy-revisits model (P4), which shows revisiting is not noise on top of response time but its own behavioural channel. |
| `first_selection_changed` | 1 if the first selected option was not the submitted one. | simulated only | medium | The binary form of the B26/B27 answer-change literature, kept alongside the count because the first change is the one those papers analyse. |
| `hint_count` | Hints requested on this item. | real + simulated | low | Help abuse is half of Baker's gaming-the-system detector (B1), and the intervention built on it reportedly halved gaming — one of the few closed loops in this literature. |
| `skipped` | 1 if the item was submitted with no answer. | simulated only | low | Non-response is data, not absence (`docs/preregistration.md` §3). A learner who starts skipping is the disengagement signal B13 detects from session logs. |
| `idle_count` | Number of idle spells (no input for the client's idle threshold). | simulated only | medium | The lit-review's core include list names *idle gaps with browser focus/visibility state*. Count and duration separate one long absence from constant micro-interruption. |
| `idle_time` | Total seconds spent idle during the attempt. | simulated only | medium | The duration half of the above. It is also what makes `log_response_time` honest: time on an item is not time on task. |
| `visibility_changes` | Times the tab lost or regained visibility during the attempt. | simulated only | medium | Direct off-task evidence in a browser, the modern equivalent of B2's off-task detector, and cheap: the Page Visibility API needs no extra permission. |

## L3 — session (6 features)

| Feature | Definition | Source | Privacy | Rationale |
|---|---|---|---|---|
| `question_number` | 1-based position of the attempt within its session. | real + simulated | low | The x-axis of every within-session decay curve, and the exposure term a fatigue claim is measured against (B32's state-space fatigue model). |
| `session_duration` | Seconds elapsed in the session at the moment of submission. | real + simulated | low | Time-on-task, the second fatigue axis. B32 models fatigue against elapsed time rather than item count; both are kept because they disagree when items differ in length. |
| `matched_difficulty_accuracy_slope` | OLS slope of difficulty-residualised correctness on within-session position, over the learner's earlier attempts in this session. Null before 4 attempts. | real + simulated · history, `shift(1)`-guarded | low | The lit-review's experimental-include list names *within-session matched-difficulty accuracy decay* — accuracy alone falls when a policy hands out harder items, so the item's own base rate is removed first. This is the fatigue construct's only observable. |
| `matched_difficulty_speed_slope` | The same slope computed on item-residualised `log_response_time`. | real + simulated · history, `shift(1)`-guarded | low | Its companion in the same list. P1's constant-speed-across-a-test assumption is exactly what this measures, and Phase 4 found the real slope is negative — learners speed up. |
| `idle_fraction` | `idle_time` divided by the attempt's total response time. | simulated only | medium | Scale-free off-task share, comparable across a 5-second and a 5-minute item, which the raw duration is not. |
| `focus_fraction` | Share of the attempt during which the tab was visible. | simulated only | medium | The visibility counterpart of `idle_fraction`; the pair distinguishes present-but-stuck from absent, which B2 and B12 treat as different states with different interventions. |

## L4 — motor (13 features)

| Feature | Definition | Source | Privacy | Rationale |
|---|---|---|---|---|
| `auc_toward_nonchosen` | Signed area between the pointer path and the straight line to the chosen option, positive toward the most-hovered non-chosen option (pixel-seconds). | simulated only | high | The attraction measure of the mouse-tracking literature (B16, B20): a path that bows toward the alternative is the canonical index of competing response activation. RQ2's central feature. |
| `max_deviation` | Greatest perpendicular distance of the path from that line (px). | simulated only | high | The second standard attraction statistic in B16's methodological guidance; reported alongside AUC because the two dissociate when a path deviates late. |
| `x_flips` | Sign changes in horizontal pointer direction between samples. | simulated only | high | The zigzag / direction-change measure named in the lit-review's experimental include list and catalogued by B20 as a distinct construct from deviation magnitude. |
| `sample_entropy` | SampEn of the per-step speed series (m = 2, r = 0.2 sigma). | simulated only | high | Movement regularity. B18 predicts respondent difficulty in web surveys from exactly this family of mouse features; irregular movement marks effortful processing. |
| `velocity_peak` | Maximum pointer speed during the attempt (px/s). | simulated only | high | The velocity-profile measure of the lit-review's include list; B17 ties peak velocity to drift-diffusion decision parameters. |
| `velocity_mean` | Mean pointer speed during the attempt (px/s). | simulated only | high | Kept with the peak because their ratio separates one decisive dash from sustained movement — B19's reproducibility guidance asks for both rather than a single summary. |
| `pause_count` | Runs of speed < 0.05 px/ms lasting > 300 ms. | simulated only | high | Pause count is named in the lit-review's include list. Hesitation is demoted there to *a feature feeding confidence*, not a state of its own (§7.5) — which is how Phase 7 uses it. |
| `path_ratio` | Total path length divided by the straight-line distance. 1.0 is straight. | simulated only | high | The oldest and most robust cursor measure in B20's catalogue, and the least sensitive to sampling rate — the sanity check against which the fancier statistics are read. |
| `time_to_first_movement` | Seconds from presentation to the first pointer movement. | simulated only | high | Initiation time. B16 separates it from movement time because a long still period before moving is deliberation, whereas a slow path is competition during the movement itself. |
| `time_to_first_selection` | Seconds from presentation to the first option selection. | simulated only | high | The motor-channel companion of `decision_latency`, retained because the two disagree when the learner selects early and then deliberates about changing. |
| `hover_time_max` | Longest time spent hovering over any single option (s). | simulated only | high | Dwell on an option is the attraction measure that survives a one-column layout, where lateral geometry is weak — the interface limitation Phase 4 recorded for AUC and x-flips. |
| `hover_time_nonchosen` | Total hover time on options other than the submitted one (s). | simulated only | high | Direct evidence of a considered-and-rejected alternative, which is the mouse-tracking construct (B16) stated in a form that does not depend on where the options are drawn. |
| `cursor_samples` | Number of 50 ms pointer samples in the attempt. | simulated only | high | Provenance, and pre-registered as such: every trajectory statistic above must be read against how many samples produced it (`docs/preregistration.md` §1). |

## Per-source availability

A source that cannot supply a feature yields **null**, never zero — absence and
measurement stay distinguishable (`preregistration.md` §4).

| Feature | Source | Why it is null |
|---|---|---|
| `attempt_count` | ednet_kt1 | KT1 stores one row per response, no attempt or hint counters |
| `n_options` | assistments_2009 | option count not recorded |
| `n_options` | assistments_2012 | option count not recorded |
| `n_options` | ednet_kt1 | not carried through KT1's response table |
| `log_lag_time` | assistments_2009 | no wall-clock column in the 2009-2010 release |
| `reading_time` | assistments_2009 | no interaction telemetry in an archival log |
| `reading_time` | assistments_2012 | no interaction telemetry in an archival log |
| `reading_time` | ednet_kt1 | no interaction telemetry in an archival log |
| `decision_latency` | assistments_2009 | no interaction telemetry in an archival log |
| `decision_latency` | assistments_2012 | no interaction telemetry in an archival log |
| `decision_latency` | ednet_kt1 | no interaction telemetry in an archival log |
| `time_after_last_interaction` | assistments_2009 | no interaction telemetry in an archival log |
| `time_after_last_interaction` | assistments_2012 | no interaction telemetry in an archival log |
| `time_after_last_interaction` | ednet_kt1 | no interaction telemetry in an archival log |
| `response_time_vs_estimate` | assistments_2009 | no interaction telemetry in an archival log |
| `response_time_vs_estimate` | assistments_2012 | no interaction telemetry in an archival log |
| `response_time_vs_estimate` | ednet_kt1 | no interaction telemetry in an archival log |
| `option_changes` | assistments_2009 | no interaction telemetry in an archival log |
| `option_changes` | assistments_2012 | no interaction telemetry in an archival log |
| `option_changes` | ednet_kt1 | no interaction telemetry in an archival log |
| `option_change_entropy` | assistments_2009 | no interaction telemetry in an archival log |
| `option_change_entropy` | assistments_2012 | no interaction telemetry in an archival log |
| `option_change_entropy` | ednet_kt1 | no interaction telemetry in an archival log |
| `option_revisits` | assistments_2009 | no interaction telemetry in an archival log |
| `option_revisits` | assistments_2012 | no interaction telemetry in an archival log |
| `option_revisits` | ednet_kt1 | no interaction telemetry in an archival log |
| `first_selection_changed` | assistments_2009 | no interaction telemetry in an archival log |
| `first_selection_changed` | assistments_2012 | no interaction telemetry in an archival log |
| `first_selection_changed` | ednet_kt1 | no interaction telemetry in an archival log |
| `hint_count` | ednet_kt1 | KT1 stores one row per response, no attempt or hint counters |
| `skipped` | assistments_2009 | no interaction telemetry in an archival log |
| `skipped` | assistments_2012 | no interaction telemetry in an archival log |
| `skipped` | ednet_kt1 | no interaction telemetry in an archival log |
| `idle_count` | assistments_2009 | no interaction telemetry in an archival log |
| `idle_count` | assistments_2012 | no interaction telemetry in an archival log |
| `idle_count` | ednet_kt1 | no interaction telemetry in an archival log |
| `idle_time` | assistments_2009 | no interaction telemetry in an archival log |
| `idle_time` | assistments_2012 | no interaction telemetry in an archival log |
| `idle_time` | ednet_kt1 | no interaction telemetry in an archival log |
| `visibility_changes` | assistments_2009 | no interaction telemetry in an archival log |
| `visibility_changes` | assistments_2012 | no interaction telemetry in an archival log |
| `visibility_changes` | ednet_kt1 | no interaction telemetry in an archival log |
| `question_number` | assistments_2009 | no wall-clock column in the 2009-2010 release |
| `session_duration` | assistments_2009 | no wall-clock column in the 2009-2010 release |
| `matched_difficulty_accuracy_slope` | assistments_2009 | no wall-clock column in the 2009-2010 release |
| `matched_difficulty_speed_slope` | assistments_2009 | no wall-clock column in the 2009-2010 release |
| `idle_fraction` | assistments_2009 | no interaction telemetry in an archival log |
| `idle_fraction` | assistments_2012 | no interaction telemetry in an archival log |
| `idle_fraction` | ednet_kt1 | no interaction telemetry in an archival log |
| `focus_fraction` | assistments_2009 | no interaction telemetry in an archival log |
| `focus_fraction` | assistments_2012 | no interaction telemetry in an archival log |
| `focus_fraction` | ednet_kt1 | no interaction telemetry in an archival log |
| `auc_toward_nonchosen` | assistments_2009 | no interaction telemetry in an archival log |
| `auc_toward_nonchosen` | assistments_2012 | no interaction telemetry in an archival log |
| `auc_toward_nonchosen` | ednet_kt1 | no interaction telemetry in an archival log |
| `max_deviation` | assistments_2009 | no interaction telemetry in an archival log |
| `max_deviation` | assistments_2012 | no interaction telemetry in an archival log |
| `max_deviation` | ednet_kt1 | no interaction telemetry in an archival log |
| `x_flips` | assistments_2009 | no interaction telemetry in an archival log |
| `x_flips` | assistments_2012 | no interaction telemetry in an archival log |
| `x_flips` | ednet_kt1 | no interaction telemetry in an archival log |
| `sample_entropy` | assistments_2009 | no interaction telemetry in an archival log |
| `sample_entropy` | assistments_2012 | no interaction telemetry in an archival log |
| `sample_entropy` | ednet_kt1 | no interaction telemetry in an archival log |
| `velocity_peak` | assistments_2009 | no interaction telemetry in an archival log |
| `velocity_peak` | assistments_2012 | no interaction telemetry in an archival log |
| `velocity_peak` | ednet_kt1 | no interaction telemetry in an archival log |
| `velocity_mean` | assistments_2009 | no interaction telemetry in an archival log |
| `velocity_mean` | assistments_2012 | no interaction telemetry in an archival log |
| `velocity_mean` | ednet_kt1 | no interaction telemetry in an archival log |
| `pause_count` | assistments_2009 | no interaction telemetry in an archival log |
| `pause_count` | assistments_2012 | no interaction telemetry in an archival log |
| `pause_count` | ednet_kt1 | no interaction telemetry in an archival log |
| `path_ratio` | assistments_2009 | no interaction telemetry in an archival log |
| `path_ratio` | assistments_2012 | no interaction telemetry in an archival log |
| `path_ratio` | ednet_kt1 | no interaction telemetry in an archival log |
| `time_to_first_movement` | assistments_2009 | no interaction telemetry in an archival log |
| `time_to_first_movement` | assistments_2012 | no interaction telemetry in an archival log |
| `time_to_first_movement` | ednet_kt1 | no interaction telemetry in an archival log |
| `time_to_first_selection` | assistments_2009 | no interaction telemetry in an archival log |
| `time_to_first_selection` | assistments_2012 | no interaction telemetry in an archival log |
| `time_to_first_selection` | ednet_kt1 | no interaction telemetry in an archival log |
| `hover_time_max` | assistments_2009 | no interaction telemetry in an archival log |
| `hover_time_max` | assistments_2012 | no interaction telemetry in an archival log |
| `hover_time_max` | ednet_kt1 | no interaction telemetry in an archival log |
| `hover_time_nonchosen` | assistments_2009 | no interaction telemetry in an archival log |
| `hover_time_nonchosen` | assistments_2012 | no interaction telemetry in an archival log |
| `hover_time_nonchosen` | ednet_kt1 | no interaction telemetry in an archival log |
| `cursor_samples` | assistments_2009 | no interaction telemetry in an archival log |
| `cursor_samples` | assistments_2012 | no interaction telemetry in an archival log |
| `cursor_samples` | ednet_kt1 | no interaction telemetry in an archival log |

## Carried, never modelled

`learner_id`, `source`, `variant`, `item_id`, `skill_id`, `order_index`, `session_id`, `split`, `next_correct`, `next_item_id`, `next_skill_id`, `learner_rte`, `solution_behaviour`, `archetype`, `latent_knowledge`, `latent_engagement`, `latent_confidence`, `latent_fatigue`, `probe_confidence`, `probe_effort`, `propensity`, `difficulty_score`, `distracted`

