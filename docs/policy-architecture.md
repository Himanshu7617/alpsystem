# Policy architecture

Three things this project keeps apart, because collapsing them is the most
common way an "adaptive learning" system claims more than it has.

```
              observed behaviour (features)
                          │
                          ▼
        ┌──────────── ESTIMATION ────────────┐
        │  logistic regression, the state    │   "P(next correct) = 0.63"
        │  encoder, BKT, Rasch θ             │   "P(struggle) = 0.81"
        └────────────────┬───────────────────┘
                         │  estimates + uncertainty
                         ▼
                 VALIDATION GATE  (Phase 7)
        a state that failed its criteria is REMOVED,
        not zeroed — a policy handed engagement = 0.0
        is still conditioning on engagement
                         │  admitted states only
             ┌───────────┴───────────┐
             ▼                       ▼
    RULE-BASED ADAPTATION     LEARNED ADAPTATION
    explicit, auditable,      contextual bandit, then
    one citation per rule     PPO — action selection is fitted
             │                       │
             └───────────┬───────────┘
                         ▼
              THE SHARED ACTION SPACE
     difficulty 1..10 × {same | next | prerequisite}
     × {none | hint | worked_example | break_suggestion}
                         ▼
            action + propensity + explanation
```

## 1. Estimation is not a policy

A logistic regression that outputs `P(next_correct) = 0.63` has estimated
something. It has not decided anything. It becomes part of a policy only when
something maps its output onto an action in the shared space — in this project,
the desirable-difficulty rule that picks the difficulty whose predicted success
lands in 0.70–0.75.

**Never call a predictive model "the adaptive policy" unless it emits the
action.** The pre-Phase-0 version of this repository did, and the arm it called
"ML" was reading the simulator's own response function (Defect 1).

## 2. Rule-based adaptation

`ml-service/app/policy/rules.py`. Every rule declares three things and the code
refuses to run one that is missing any of them:

- the **signal class** its inputs live in (`L0` correctness … `L4` motor), so an
  arm restricted to a rung cannot fire a rule it has no inputs for;
- the **latent states** it conditions on, which the validation gate may veto;
- a **citation**, because a rule with no literature behind it is a preference.

The full table, with the two joint-state rules the gate excluded and why, is in
`docs/rule-policy.md`.

## 3. Learned adaptation

Same context, same action space, fitted action selection:

```
context ─► bandit / PPO ─► action ─► environment ─► reward ─► update
```

Phase 9 builds these. They observe exactly what the rule policy observes: the
output of `Gate.filter_states()` plus the observable features of their rung.
**Knowledge is the only admitted state** — engagement and confidence failed
Phase 7's criteria and are absent from every policy input.

## What every policy must emit

| Field | Why |
|---|---|
| `difficulty`, `concept_move`, `intervention` | the shared 120-action space; comparisons are only fair if every arm chooses from the same set |
| `propensity` | P(action \| this policy). Study 3's offline evaluation dies without it and it cannot be backfilled |
| `explanation` | named states, their standard errors, the rules that fired **and the rules the gate excluded** |

An explanation may never cite a state that failed the gate.
`test_policies.py` asserts it, and `test_study2.py` asserts the gate drops a
failed state rather than zeroing it.

## The separation that is enforced in code

`app/policy/` **cannot import the simulator.** The environment
(`app/research/closed_loop.py`) owns the learner; the policy owns the decision;
they exchange an action and an attempt record and nothing else. A grep-based
test keeps it that way. It is a guardrail, not a proof, and that is the right
size for it.
