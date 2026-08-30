# Rule Policy v1.2

`backend/src/utils/ruleEngine.js` provides the deterministic **improved rule baseline**. It is deliberately separate from the forthcoming learned policy.

## Inputs and transformations

All signals are aggregate, per-question values. The preprocessor accepts snake_case and camelCase aliases, coerces finite numeric values, bounds count/time features to non-negative values, and parses booleans explicitly. In particular, the strings `"false"`, `"0"`, and `"no"` are false; they are not treated as JavaScript truthy values.

The API derives correctness from the stored answer key whenever a selected answer is present. A browser cannot elevate mastery by supplying `isCorrect: true`.

## State estimates

| State | Evidence used | Intended interpretation |
| --- | --- | --- |
| Knowledge | correctness, difficulty, attempts, short-term accuracy trend | concept-agnostic proficiency proxy |
| Confidence | correctness, relative response time, answer changes, delayed interaction | certainty proxy, not self-report |
| Engagement | tab switches, aggregate pointer movement/speed, idle time | on-task activity proxy |
| Cognitive load | relative response time, correctness, retries, editing and pauses | momentary task-demand proxy |
| Fatigue | session duration, item count, performance and speed drift, idleness | session-level depletion proxy |

Each state is constrained to `[0, 1]`. These values are hypotheses to calibrate against outcomes; they are not validated psychological measurements.

## Difficulty selection

Difficulty is no longer a direct reaction to the latest answer. The policy first assigns a knowledge-band target, then evaluates the latest five interactions for rolling accuracy, response-time ratio, and accuracy trend. It reduces difficulty for sustained struggle, slowing responses, high cognitive load, high fatigue, or low engagement. It promotes difficulty only after at least three accurate and efficient responses with adequate knowledge and confidence. A transition is limited to one level per question to avoid noisy jumps.

## Actions and explanation

The selector preserves the legacy difficulty response and additionally returns a policy version, action, and machine-readable reason codes.

| Trigger | Action | Reason code |
| --- | --- | --- |
| Cognitive load > 0.70 | scaffold | `high_cognitive_load` |
| Fatigue > 0.70 | offer a break | `high_fatigue` |
| Engagement < 0.40 | re-engage | `low_engagement` |
| Knowledge < 0.40 | foundation practice | `low_knowledge` |
| Knowledge and confidence > 0.75, no support trigger | advance hard | `ready_for_challenge` |
| Otherwise | progressive practice | `maintain_progressive_practice` |

Actions are a serving contract and a research logging field. The current UI may only render difficulty; future roadmap work will use the action to choose a concept, scaffold, review item, or break.

---

# Phase 8: the state-interaction rules

`ml-service/app/policy/rules.py` is the research policy's rule table. It is
separate from the v1.2 engine above and answers a different question: **given a
*validated* learner state, what should the system do next?** Every rule declares
three things, and the code refuses to run one that is missing any of them:

- the **signal class** its inputs live in (`L0` correctness … `L4` motor), so an
  arm restricted to a rung cannot fire a rule it has no inputs for;
- the **latent states** it conditions on, which the Phase 7 validation gate is
  allowed to veto;
- a **citation**, because a rule with no literature behind it is a preference.

## The table

| Rule | Rung | States | Fires when | Action | Citation |
| --- | --- | --- | --- | --- | --- |
| `target_desirable_difficulty_band` | L0 | knowledge | always (it *is* the selection rule) | difficulty whose predicted success lands in 0.70–0.75 | Metcalfe & Kornell (2005); Wilson et al. (2019) |
| `productive_confusion` | L0 | — | recent accuracy ≤ 0.5 **and** rising | **no intervention** — and it locks out later ones | VanLehn et al. (2003); D'Mello et al. (2014) |
| `prerequisite_on_wheel_spinning` | L0 | — | ≥ 8 attempts on the concept, accuracy ≤ 0.5, flat or falling | move to the prerequisite concept | Beck & Gong (2013); Käser et al. (2014) |
| `worked_example_on_repeated_failure` | L0 | — | 3 consecutive incorrect | worked example | Sweller & Cooper (1985); Salden et al. (2010) |
| `hint_on_slow_decision` | L1 | — | decision latency above the corpus 75th percentile after an incorrect | hint | Aleven et al. (2016); Beck et al. (2008) |
| `break_on_within_session_decay` | L3 | — | matched-difficulty speed slope above the corpus 75th percentile, ≥ 8 items into the session | suggest a break | Wise & Kong (2005); preregistration §7 |
| `hint_when_uncertain_but_engaged` | L1 | confidence, engagement | low confidence, high engagement | hint | Aleven et al. (2016); Baker et al. (2008) |
| `ease_off_when_uncertain_and_disengaged` | L1 | confidence, engagement | low confidence, low engagement | difficulty − 2 | Baker et al. (2008); Wise & Kong (2005) |

## The two rules that do not run, and why that is the point

The last two rows are the joint-state rules this project set out to test — the
ones where "a rule that depends on two states jointly" is the systems
contribution. **Neither fires.** Phase 7's validation gate rejected both states
they need: engagement failed Wise & Kong's negative criterion (|*r*| with
ability 0.403 against a 0.20 ceiling) and confidence failed both its correlation
floor and discriminant validity against the knowledge head (*r* = 0.893).

`rules.ruleset()` therefore returns them in `excluded` with the gate's own
reason attached, and every decision explanation carries that list. They are not
deleted from the table, because a rule that is silently removed cannot be
reported as a null — and this null is a finding about what the state estimates
could and could not support, not a gap in the implementation.

The same gate restricts the `rule_improved` arm: three of the deployed engine's
five variables (cognitive load, fatigue, engagement) are constructs this project
dropped or rejected, so that arm runs the two that survive and records the rest
as excluded.

## Interaction between rules

Rules run in table order and accumulate modifiers over the band-selected
difficulty. `productive_confusion` sets a lock: once it fires, no later rule may
add an intervention, so "do not interrupt a learner who is struggling but
improving" beats "offer a hint". That precedence is the only ordering the table
depends on.

## What the environment does with an action

The closed loop implements a hint as +1.0 logit of support with the learning
gain scaled to 0.7, a worked example as +2.0 logits at full gain, and a break as
a fatigue reset — all at a time cost, all identical for every arm, all fixed
before the first rollout (`docs/preregistration.md` §9.3). None of them is
fitted: no archival source in this project records an intervention, so the
directions come from the literature and the magnitudes are declared rather than
estimated.
