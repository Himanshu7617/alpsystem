# Full-run justification

`make full` is the only command in this repository that is allowed to occupy
the machine for tens of minutes. BUILD.md §0.3 requires a written reason before
each such run. One section per full run, newest last.

---

## Phase 8 — closed-loop policy evaluation (2026-08-22)

**Why the smaller run was insufficient.** The `dev` profile (250 learners, one
seed, V0–V1) establishes that the pipeline is correct — every arm runs, the
figures draw, no effect size is implausible — but it cannot answer Phase 8's
question. Two reasons, both measured rather than assumed:

1. **Precision.** The pre-registered contrast is `model_L0` vs `model_L4` on
   items-to-mastery and knowledge gain. At 250 learners the bootstrap CI on
   mean knowledge gain is roughly ±0.010 against arm differences of ~0.005, so
   the dev run cannot distinguish "no effect" from "an effect the size the
   literature reports" (g ≈ 0.3). 3,000 learners per arm per variant narrows the
   interval by a factor of √12.
2. **Robustness is the credibility claim.** `f08-04` ranks the arms under V0–V4
   and the phase's honesty rests on whether the ranking survives
   mis-specification. V2–V4 are not run at all in the dev profile.

**The exact question the larger run answers.** Does holding the policy
architecture and action space constant while enriching the state vector
(L0 → L1 → L3 → L4) change instructional decisions *and* learner outcomes, and
does that conclusion survive four deliberate mis-specifications of the
generative model? RQ3 and RQ4 in BUILD.md §2.

**Expected compute cost.** 150 cells (10 arms × 5 variants × 3 seeds) at
1,000 learners and a 200-item budget, plus an 8-cell probe-noise sweep.
Measured at the dev size: ~10 s per 250-learner cell, so ~40 s per full cell,
≈ 20 minutes wall clock at 6 workers. CPU-bound, ~1 GB per worker.

**Configured limits.** `ALP_RUN_PROFILE=full` sets 6 workers on a 10-core
machine — four cores stay free, which is the thermal control. No sweep, no
parallel full runs, no retry loop.

**Checkpoint strategy.** Every (arm, variant, seed) cell is written to
`artifacts/evaluation/cells/*.json.gz` as it finishes, keyed by the job *and* a
hash of every module that can change its result. A run stopped part-way —
because the machine is hot, or the user interrupted it — resumes from those
checkpoints; editing a policy invalidates them rather than silently reusing
them. `make status` reports how far the current or last run got.

**Stop condition.** The run halts itself if any Cohen's *d* above 3 appears on
a comparison without a documented exemption (`evaluate_policies.HALT_EXEMPT`) —
the prior work's *d* = 26 was a defect signature. Otherwise it stops when the
150 cells are done. If a repeat is needed and the machine is hot, the correct
response is to keep the dev-profile results and record the computational
limitation, not to force the run.
