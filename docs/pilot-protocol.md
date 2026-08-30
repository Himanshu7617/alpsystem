# Pilot Study Protocol — Adaptive Learning Platform Evaluation

**Document Title**: Pilot Study Protocol — Adaptive Learning Platform Evaluation  
**Version**: 1.0 (Draft for IRB & Research Team Review)  
**Date**: August 12, 2026  
**Target Participant Cohort**: n = 30–50 undergraduate Computer Science students  
**Data Retention Window**: 12 months after study completion  

---

## 1. Executive Summary & Purpose

This protocol outlines the experimental design and evaluation procedures for the pilot deployment of the Adaptive Learning Platform (ALP). The primary goal is to empirically evaluate whether an AI/ML-driven adaptive question sequencing policy (`ml_based`) outperforms an expert-crafted rule-based heuristic policy (`improved_rule`) in accelerating student mastery and reducing learning latency without escalating learner cognitive load or frustration.

---

## 2. Participant Recruitment & Target Sample

- **Sample Size**: Target sample size of **n = 30–50** participants.
- **Eligibility Criteria**:
  - Currently enrolled undergraduate Computer Science or Software Engineering students.
  - Completed or currently enrolled in introductory programming (CS1/CS2) or data structures coursework.
  - Age 18 or older.
- **Recruitment Strategy**: Voluntary participation recruited through course announcement channels, email listservs, and departmental bulletin boards.
- **Incentives**: Certificate of participation / extra credit opportunity (subject to course instructor approval).

---

## 3. Experimental Design

The study employs a **within-subjects, counterbalanced design**. Each participant completes **TWO** separate learning sessions:

1. **Policy Condition A (`improved_rule`)**: Rule-based adaptive policy using deterministic pedagogical heuristics (e.g., dynamic difficulty stepping based on accuracy trends and engagement signals).
2. **Policy Condition B (`ml_based`)**: Machine Learning-based adaptive policy driven by real-time learner state inference (Knowledge, Confidence, Engagement, Latency, Frustration) and trained policy recommendation.

### Counterbalancing & Randomization
To control for practice, order, and carryover effects, participants will be randomly assigned 1:1 to one of two condition orderings upon enrollment:
- **Group 1**: Session A (`improved_rule`) $\rightarrow$ Session B (`ml_based`)
- **Group 2**: Session B (`ml_based`) $\rightarrow$ Session A (`improved_rule`)

---

## 4. Session Structure & Materials

- **Session Duration**: ~25–30 minutes per session (total participant commitment: ~60 minutes across both sessions with a recommended 24-hour inter-session break).
- **Question Item Bank**: Drawn from the validated **126-question Computer Science Education (CSE) item bank** covering core domain concepts (variables, conditionals, loops, functions, array structures, basic algorithms).
- **Item Delivery**: Questions are served dynamically via the web platform interface. For each item, real-time telemetry is recorded and feedback/explanations are rendered post-submission.
- **Mastery Criteria**: Session terminates when the participant achieves the predefined mastery threshold across target concept clusters or reaches the maximum session item cap (30 items).

---

## 5. Primary and Secondary Outcome Measures

### 5.1 Primary Outcome
- **Time-to-Mastery**: Total cumulative active response time (in seconds) required for a student to reach the threshold mastery criterion across target domain concepts.

### 5.2 Secondary Outcomes
- **State Trajectories**: Real-time trajectory dynamics across five core cognitive states:
  - **Knowledge (K)**: Estimated concept mastery.
  - **Confidence (C)**: Inferred decision certainty.
  - **Engagement (E)**: Attentional focus and interaction persistence.
  - **Latency (L)**: Normalized item processing speed.
  - **Frustration (F)**: Detected friction or struggle indicators.
- **Final Mastery Score**: Proportion of concept nodes mastered upon session completion.
- **User Satisfaction Survey**: Post-session survey measuring perceived adaptivity, usability, and task workload using a 5-point Likert scale (1 = Strongly Disagree to 5 = Strongly Agree).

---

## 6. Automated Telemetry & Data Collection

The platform automatically logs exactly nineteen (19) interaction features for each item response:

1. `total_response_time` (Float, seconds)
2. `reading_time` (Float, seconds)
3. `time_after_last_interaction` (Float, seconds)
4. `correct` (Boolean)
5. `attempts` (Integer)
6. `skip` (Boolean)
7. `option_changes` (Integer)
8. `mouse_distance` (Float, pixels)
9. `mouse_speed` (Float, px/sec)
10. `hover_time` (Float, seconds)
11. `typing_speed` (Float, WPM)
12. `backspaces` (Integer)
13. `delete_frequency` (Integer)
14. `pause_duration` (Float, seconds)
15. `question_number` (Integer)
16. `session_duration` (Float, seconds)
17. `accuracy_decay` (Float)
18. `tab_switches` (Integer)
19. `timeRatio` (Float, relative response time ratio)

### Data Handling and Privacy Summary:
- Data is stored in an on-premises / self-hosted PostgreSQL database.
- No raw keystrokes or pointer coordinates are retained.
- Data will be stored for **12 months following study completion**, after which all records will be deleted.

---

## 7. Statistical Analysis Plan

All statistical evaluations will be executed on anonymized participant metrics:

- **Paired Comparisons**: Two-tailed **Wilcoxon signed-rank test** for within-subject comparisons of time-to-mastery, final mastery scores, and Likert survey scores between `improved_rule` and `ml_based` policies.
- **Effect Size Estimation**: Calculation of paired/unpaired **Cohen's d** (and Rank-Biserial correlation for non-parametric rank tests).
- **Confidence Intervals**: Computation of non-parametric **Bootstrap 95% Confidence Intervals** (1,000 iterations) for mean differences in time-to-mastery and state progression rates.

---

## 8. Ethical Considerations & Participant Safeguards

- **IRB Approval**: Formal Institutional Review Board (IRB) ethical approval must be secured prior to participant recruitment.
- **Informed Consent**: Every participant must review and sign the plain-language Informed Consent Form (`docs/pilot-consent-template.md`) before accessing study sessions.
- **Right to Withdraw**: Participants may withdraw at any point without penalty. Data deletion requests will be fulfilled and purged from storage within 30 days.

---

## 9. Study Timeline

| Phase | Activity | Duration |
|---|---|---|
| **Phase 1** | IRB submission, platform staging verification, participant recruitment | 2 weeks |
| **Phase 2** | Participant onboarding, counterbalanced session execution, data collection | 2 weeks |
| **Phase 3** | Data cleaning, statistical modeling, Wilcoxon & Bootstrap analysis, final report | 2 weeks |

**Total Estimated Study Timeline**: 6 weeks  
