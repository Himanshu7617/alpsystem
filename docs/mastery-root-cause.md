# Mastery Root Cause Analysis

## Hypothesis (a): At which tau/max_items does mastery_rate first become >0?
Based on the threshold sweep, mastery rates first become >0 when we either significantly increase `max_items` or lower `tau`. The earliest non-zero mastery rates appear at **tau=0.6, max_items=250**.

Here is an excerpt of the relevant threshold sweep data:

| tau | max_items | policy | mastery_rate | mean_time | mean_final_mastery |
|-----|-----------|--------|--------------|-----------|--------------------|
| 0.60| 150       | legacy_rule | 0.000000 | 7078.17 | 0.498164 |
| 0.60| 250       | legacy_rule | 0.016667 | 12531.73 | 0.450501 |
| 0.60| 250       | ml_based    | 0.086667 | 11319.11 | 0.554209 |
| 0.80| 400       | legacy_rule | 0.000000 | 20972.79 | 0.364712 |
| 0.80| 400       | ml_based    | 0.076667 | 18795.41 | 0.534237 |

## Hypothesis (b): What max true_knowledge do learners reach?
The maximum `true_knowledge` reached by learners is **1.0** (the ceiling).
Because true knowledge reaches the maximum value easily, the archetypes are **not** too conservative. The simulated learners are successfully learning the material; it is merely that the simulation's rule engine fails to recognize it.

## Hypothesis (c): Does tracked_K systematically diverge from true_knowledge? By how much?
Yes, `tracked_K` systematically diverges from `true_knowledge` by a massive margin. The mean divergence across all trace steps is approximately **-0.18**, with the maximum divergence reaching up to **0.38**. While `true_knowledge` rapidly climbs towards 1.0, `tracked_K` never exceeds ~0.71 (even after 400 items).

## Conclusion: Primary Root Cause
The primary root cause of the 0% mastery rate is that **the rule engine calculates `tracked_K` as the mean of 36 individual concept mastery scores, but only updates ONE concept per step.** 

While the learner's `true_knowledge` increases globally on every correct response, `tracked_K` dilutes every learning gain by a factor of 36. This mathematical mismatch causes `tracked_K` to grow extremely slowly, making it impossible for the mean mastery across all concepts to reach the 0.80 threshold within the `max_items` limit, even though the simulated learner has already mastered the material (true knowledge = 1.0).
