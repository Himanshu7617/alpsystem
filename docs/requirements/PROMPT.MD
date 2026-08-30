# Adaptive Learning System Using Deep Behavioural Analytics

## 1. Project Overview

This project aims to design and develop an **Adaptive Learning System (ALS)** that personalizes the learning experience by analyzing not only a learner's answers but also their **behaviour while learning and answering questions**.

Traditional adaptive learning systems often adapt content primarily using:

* Correctness of answers
* Accuracy
* Historical performance
* Question difficulty
* Sometimes response time

This project proposes a deeper approach.

The system will collect and analyze multiple **behavioural, performance, and interaction signals** to build a more comprehensive model of the learner.

The objective is to move from:

> "Did the learner answer correctly?"

to:

> "What does the learner's complete behavioural pattern suggest about their knowledge, confidence, engagement, fatigue, uncertainty, and learning state?"

These inferred learner states will then be used to determine the most appropriate **next learning action**.

---

# 2. Core Research Concept

The central concept of this project is:

```text
Learner Interaction
        ↓
Behavioural Data Collection
        ↓
Behavioural Feature Analysis
        ↓
Learner State Estimation
        ↓
Adaptive Decision Engine
        ↓
Personalized Next Learning Action
```

The system should continuously observe learner interactions and maintain an evolving representation of the learner.

The learner model should not be limited to a single metric such as "score" or "mastery percentage".

Instead, the system should attempt to model multiple learner dimensions.

---

# 3. Learner State Model

The proposed system should investigate estimating the following learner states.

## 3.1 Knowledge / Mastery

Represents how well the learner understands a concept or topic.

Possible inputs:

* Answer correctness
* Accuracy
* Historical performance
* Performance on different difficulty levels
* Number of attempts
* Previous correctness
* Performance trends

Example:

```text
Topic: Algebra

Knowledge Score: 0.82

Interpretation:
The learner demonstrates relatively strong mastery
of Algebra concepts.
```

---

## 3.2 Confidence

Represents how certain or uncertain the learner appears while answering.

Possible signals:

* Time before selecting an answer
* Number of option changes
* Number of answer changes
* Hesitation
* Consistency of responses
* Response time relative to question difficulty
* Difference between fast incorrect answers and slow uncertain answers

Example:

```text
Correct Answer
+
Very Long Response Time
+
Multiple Option Changes
```

Possible interpretation:

```text
The learner may understand the answer but have
low confidence.
```

Confidence must not automatically be treated as knowledge.

A learner may:

* Know the answer but lack confidence
* Be highly confident but incorrect

The system should therefore attempt to distinguish between these cases.

---

## 3.3 Engagement

Represents how actively and consistently the learner is interacting with the learning environment.

Possible signals:

* Interaction frequency
* Mouse activity
* Click patterns
* Time-on-task
* Idle periods
* Question skipping
* Session consistency
* Sudden reduction in interaction

Example:

```text
Normal interaction
        ↓
Gradually increasing idle time
        ↓
More skipped questions
        ↓
Lower interaction frequency
```

Possible interpretation:

```text
Reduced engagement.
```

---

## 3.4 Fatigue

Represents the possibility that learner performance or interaction quality is declining because of prolonged cognitive effort.

Possible signals:

* Long continuous session duration
* Gradually increasing response time
* Declining accuracy
* Increasing answer changes
* Increasing skips
* Reduced interaction activity
* Performance deterioration over time

Example:

```text
Beginning of session:
High accuracy
Fast responses

After prolonged interaction:
Lower accuracy
Slower responses
More hesitation
```

Possible interpretation:

```text
Possible fatigue.
```

Fatigue should be treated as an inferred state rather than directly observable ground truth.

---

## 3.5 Cognitive Difficulty / Cognitive Load

The system should investigate whether learner interactions can indicate when a question or learning activity is cognitively difficult.

Possible signals:

* Excessively long response time
* Multiple attempts
* Repeated answer changes
* Incorrect responses
* Skipping
* Performance degradation on specific concepts
* Interaction patterns

The objective is to distinguish between:

```text
The learner does not know the concept
```

and:

```text
The learner may understand the concept but is
currently struggling with the complexity of the task.
```

---

## 3.6 Uncertainty / Hesitation

The system should attempt to detect behavioural patterns that may indicate uncertainty.

Possible signals:

* Multiple option changes
* Long delay before answering
* Repeated interaction with different options
* Frequent answer revisions
* Unusual response patterns

This state may overlap with confidence but should be investigated separately if supported by literature.

---

# 4. Behavioural Data Collection

The system should collect learner interaction data at the question and session level.

## 4.1 Question-Level Data

Each learner interaction with a question may include:

```text
user_id
session_id
question_id
topic_id
difficulty_level

question_start_time
answer_submission_time
total_time_spent

selected_answer
is_correct

number_of_attempts
number_of_option_changes

was_question_skipped

previous_question_result
previous_question_difficulty
```

Additional fields may be introduced if justified by research.

---

## 4.2 Interaction-Level Data

Where practical and ethically appropriate, the system may capture:

### Mouse Behaviour

Examples:

* Mouse movement
* Mouse inactivity
* Hover duration
* Movement frequency
* Interaction speed
* Click patterns

### Typing Behaviour

Where typing input exists:

* Typing speed
* Pause duration
* Backspace frequency
* Typing bursts
* Time between keystrokes

These signals should only be included if they provide meaningful value and can be justified through literature.

The project should avoid collecting behavioural data merely because it is technically possible.

Every collected signal should have a defined purpose.

---

## 4.3 Session-Level Data

The system should track:

```text
session_id
session_start_time
session_end_time
total_session_duration

questions_attempted
questions_correct
questions_incorrect
questions_skipped

average_response_time
response_time_trend

accuracy_trend
interaction_trend
```

The purpose is to detect changes in learner behaviour over time.

---

# 5. Core System Components

The system should conceptually contain the following major components.

## 5.1 Learning Content / Question Engine

Responsible for:

* Managing topics
* Managing concepts
* Managing questions
* Assigning difficulty levels
* Serving questions to learners

Questions should ideally contain metadata such as:

```text
question_id
topic
concept
difficulty
question_type
prerequisite_concepts
```

---

## 5.2 Behavioural Data Collection Layer

Responsible for collecting raw learner interaction events.

Examples:

```text
QUESTION_OPENED

OPTION_SELECTED

OPTION_CHANGED

ANSWER_SUBMITTED

QUESTION_SKIPPED

MOUSE_ACTIVITY

USER_IDLE

SESSION_STARTED

SESSION_ENDED
```

The system should use an event-based approach where appropriate.

Raw events may later be converted into meaningful behavioural features.

---

## 5.3 Feature Engineering / Behavioural Analytics Layer

This component converts raw interaction data into useful features.

Example:

```text
Raw Events

OPTION A SELECTED
        ↓
OPTION B SELECTED
        ↓
OPTION C SELECTED
        ↓
ANSWER SUBMITTED
```

Possible derived features:

```text
option_change_count = 2

time_before_first_selection = 15 seconds

total_response_time = 42 seconds
```

The feature layer may generate:

* Response time features
* Accuracy features
* Performance trends
* Interaction frequency
* Option change counts
* Skip frequency
* Session trends
* Difficulty-relative performance

Features should be evaluated for usefulness rather than assumed to be valuable.

---

# 6. Learner State Estimation Engine

The Learner State Estimation Engine is one of the central components of the proposed system.

Its responsibility is to transform behavioural and performance features into estimates of learner state.

Conceptually:

```text
Behavioural Features
        +
Performance Features
        ↓
Learner State Estimation Model
        ↓
Knowledge Score
Confidence Score
Engagement Score
Fatigue Score
Cognitive Difficulty Score
```

The exact modelling technique is not fixed yet.

Potential approaches to investigate include:

* Rule-based models
* Statistical models
* Bayesian approaches
* Machine learning
* Deep learning
* Knowledge tracing
* Sequential models
* Hybrid approaches

The final approach should be selected based on:

* Literature review
* Dataset availability
* Explainability
* Computational complexity
* Ability to operate in near real time
* Experimental validity

Deep learning should not be used merely because the project title contains "Deep Behavioural Analytics".

The methodology must be academically justified.

---

# 7. Knowledge / Mastery Modelling

The system should maintain a representation of learner knowledge over time.

Possible concept:

```text
Learner
    │
    ├── Algebra → 82% mastery
    ├── Geometry → 45% mastery
    ├── Calculus → 67% mastery
    └── Probability → 30% mastery
```

The knowledge model should consider sequential learner interactions.

The following approaches may be investigated:

* Bayesian Knowledge Tracing
* Item Response Theory
* Deep Knowledge Tracing
* Transformer-based Knowledge Tracing
* Other modern learner modelling approaches

The final selection should be based on the literature review and research methodology.

---

# 8. Adaptive Decision Engine

The Adaptive Decision Engine determines what the system should do next.

The decision should not depend only on correctness.

It should consider the estimated learner state.

Possible inputs:

```text
Knowledge
Confidence
Engagement
Fatigue
Cognitive Difficulty
Recent Performance
Question Difficulty
Learning History
```

Possible outputs:

```text
NEXT_ACTION
```

Where NEXT_ACTION may be:

* Increase difficulty
* Maintain difficulty
* Decrease difficulty
* Repeat a concept
* Present a similar question
* Present a prerequisite concept
* Provide a hint
* Provide an explanation
* Recommend revision material
* Recommend a break
* Change learning activity

Example:

### Scenario 1

```text
Incorrect Answer
+
Very Fast Response
+
High Confidence Pattern
```

Possible interpretation:

```text
Possible misconception or careless response.
```

Potential action:

```text
Provide explanation or diagnostic question.
```

### Scenario 2

```text
Incorrect Answer
+
Long Response Time
+
Multiple Option Changes
```

Possible interpretation:

```text
Possible uncertainty or cognitive difficulty.
```

Potential action:

```text
Provide similar question, hint, or simpler explanation.
```

### Scenario 3

```text
Declining Accuracy
+
Increasing Response Time
+
Long Session Duration
```

Possible interpretation:

```text
Possible fatigue.
```

Potential action:

```text
Recommend a break or reduce cognitive difficulty.
```

These examples are hypotheses and must be validated through literature and experimentation.

---

# 9. Adaptation Policy

The project should investigate how learner state should influence adaptation.

A simplified conceptual model:

```text
Current Learner State
        ↓
Decision Policy
        ↓
Select Best Next Learning Action
        ↓
Observe Learner Response
        ↓
Update Learner State
        ↓
Repeat
```

This creates a continuous feedback loop.

The adaptation policy may eventually use:

* Rules
* Machine learning
* Reinforcement learning
* Contextual bandits
* Hybrid approaches

However, the choice should be justified experimentally.

---

# 10. Deep Behavioural Analytics

"Deep Behavioural Analytics" in this project refers to the analysis of learner behaviour beyond simple performance metrics.

The system should attempt to answer questions such as:

```text
Why might the learner have answered incorrectly?
```

Instead of only recording:

```text
Incorrect
```

The system should investigate contextual behavioural evidence.

For example:

```text
Incorrect
+
Long Response Time
+
Multiple Answer Changes
+
Previously Strong Performance
```

may indicate something different from:

```text
Incorrect
+
Very Fast Response
+
Repeated Poor Performance
+
Low Performance on Prerequisite Concepts
```

The objective is to investigate whether behavioural context can improve learner modelling and adaptive decisions.

---

# 11. Data Model Requirements

The system will likely require entities such as:

```text
User

Learning Session

Topic

Concept

Question

Question Attempt

Interaction Event

Behavioural Feature

Learner State

Adaptation Decision

Learning Recommendation
```

Conceptual relationship:

```text
User
 │
 └── Learning Session
         │
         ├── Question Attempt
         │       │
         │       └── Interaction Events
         │
         ├── Behavioural Features
         │
         ├── Learner State
         │
         └── Adaptation Decisions
```

The exact database schema can evolve during implementation.

---

# 12. Real-Time / Near Real-Time Requirements

The system should attempt to update the learner model dynamically.

The intended flow is:

```text
Learner Answers Question
        ↓
Interaction Data Captured
        ↓
Features Updated
        ↓
Learner State Recalculated
        ↓
Adaptation Decision Generated
        ↓
Next Learning Action Selected
```

The system does not necessarily require complex real-time streaming infrastructure unless justified.

The implementation should prioritize research validity over unnecessary architectural complexity.

---

# 13. Explainability Requirements

The system should attempt to explain adaptation decisions where possible.

For example:

Instead of:

```text
Next Question Difficulty: Easy
```

The system could provide an explanation such as:

```text
The difficulty was reduced because recent responses
showed increased response time, multiple answer
changes, and lower performance.
```

Explainability is important because:

* Adaptive decisions should be understandable
* Learners may benefit from feedback
* Researchers need to analyze system behaviour
* It helps evaluate whether the system is making reasonable decisions

However, explanations should not falsely imply certainty about hidden psychological states.

---

# 14. Research Questions

The final research questions will be refined after the literature review.

Potential directions include:

### RQ1

Can behavioural interaction data improve learner state modelling compared with performance-only adaptive learning systems?

### RQ2

Which behavioural signals provide the most useful information for predicting learner state?

### RQ3

Does combining behavioural signals with traditional performance data improve adaptive learning decisions?

### RQ4

Can a multidimensional learner model improve learning outcomes compared with correctness-based adaptation?

### RQ5

Which behavioural signals contribute most significantly to knowledge, confidence, engagement, or fatigue estimation?

These are preliminary and must not be treated as final until supported by the literature review.

---

# 15. Experimental Requirements

The final system should be evaluated scientifically.

At minimum, experiments should investigate the difference between:

## Baseline System

Adaptive learning based primarily on:

```text
Correctness
+
Historical Performance
+
Difficulty
```

## Proposed Behaviour-Aware System

Adaptive learning based on:

```text
Correctness
+
Performance History
+
Response Time
+
Attempts
+
Option Changes
+
Skipping Behaviour
+
Session Behaviour
+
Additional Validated Behavioural Features
```

Possible evaluation areas include:

* Prediction accuracy
* Knowledge/mastery estimation accuracy
* Learning improvement
* Adaptation quality
* Engagement
* Completion rate
* Response efficiency
* User experience

The evaluation methodology should include appropriate statistical analysis.

---

# 16. Ablation Study Requirement

The system should support investigation of the contribution of individual behavioural features.

Example:

```text
Model A
Correctness only
```

vs.

```text
Model B
Correctness
+
Response Time
```

vs.

```text
Model C
Correctness
+
Response Time
+
Attempts
+
Option Changes
```

vs.

```text
Model D
All Validated Behavioural Features
```

This will help answer:

> Does behavioural data actually improve the system, and which signals matter?

The research should avoid claiming that more features automatically result in a better system.

---

# 17. Privacy and Ethical Requirements

Behavioural analytics can involve sensitive learner data.

The system should therefore consider:

* Data minimization
* User consent
* Transparency about collected interaction data
* Secure storage
* Anonymization or pseudonymization where possible
* Avoiding unnecessary surveillance
* Avoiding unsupported psychological conclusions

Mouse and typing data should only be collected when:

1. There is a clear research purpose.
2. The learner is informed.
3. The data is necessary for the experiment.

The system must not claim to diagnose psychological or medical conditions.

States such as fatigue, engagement, confidence, and cognitive difficulty should be treated as **model estimates or behavioural indicators**, not absolute truths.

---

# 18. System Architecture

The conceptual architecture is:

```text
                    ┌──────────────────┐
                    │     Learner      │
                    └────────┬─────────┘
                             │
                             ▼
                 ┌─────────────────────┐
                 │ Learning Interface  │
                 └────────┬────────────┘
                          │
                          ▼
                 ┌─────────────────────┐
                 │ Interaction Event   │
                 │ Collection Layer    │
                 └────────┬────────────┘
                          │
                          ▼
                 ┌─────────────────────┐
                 │ Behavioural Feature │
                 │ Extraction          │
                 └────────┬────────────┘
                          │
                          ▼
                 ┌─────────────────────┐
                 │ Learner State       │
                 │ Estimation Engine   │
                 │                     │
                 │ • Knowledge         │
                 │ • Confidence        │
                 │ • Engagement        │
                 │ • Fatigue           │
                 │ • Difficulty        │
                 └────────┬────────────┘
                          │
                          ▼
                 ┌─────────────────────┐
                 │ Adaptive Decision   │
                 │ Engine              │
                 └────────┬────────────┘
                          │
                          ▼
                 ┌─────────────────────┐
                 │ Next Learning       │
                 │ Action              │
                 └─────────────────────┘
```

---

# 19. Project Success Criteria

The project should ultimately demonstrate whether:

1. Behavioural data provides useful information beyond correctness.
2. Multiple behavioural signals can contribute to learner modelling.
3. A behavioural learner model can improve adaptation decisions.
4. The proposed system performs better than appropriate baseline approaches.
5. The contribution can be experimentally validated.
6. The system can operate dynamically during a learning session.
7. The research claims are supported by evidence rather than assumptions.

---

# 20. Important Research Constraints

The project must avoid the following mistakes:

## Do Not Assume Behaviour Equals Psychological State

For example:

```text
Long response time ≠ automatically low knowledge.
```

It may indicate:

* Difficult question
* Careful thinking
* Distraction
* Uncertainty
* Cognitive load

The system should use multiple signals and contextual information where possible.

---

## Do Not Assume More Data Is Better

Every behavioural signal should justify its inclusion.

The research should investigate whether each signal contributes useful predictive or adaptive value.

---

## Do Not Force Deep Learning

Deep learning should only be used if:

* The dataset supports it
* Sequential or complex patterns justify it
* It performs better than simpler baselines

Simple and interpretable models should also be evaluated.

---

## Do Not Claim Novelty Without Evidence

The literature review must determine:

* What already exists
* Which behavioural signals have already been studied
* Whether multimodal learner modelling already exists
* Whether behavioural signals are already used for real-time adaptation
* What exact gap remains

The project should be willing to modify its proposed approach based on existing research.

---

# 21. Expected Final Research Contribution

The exact contribution is currently open and must be determined through the literature review.

Potential contributions may include:

* A multidimensional learner state model combining knowledge and behavioural indicators
* A behavioural feature fusion approach for adaptive learning
* A framework for real-time behaviour-aware adaptation
* An empirical comparison between performance-only and behaviour-aware adaptation
* An analysis of which behavioural signals provide the greatest value
* An explainable adaptive learning framework

The final research contribution must be selected based on the strongest defensible research gap.

---

# 22. Current Development Philosophy

The priority order for this project is:

```text
1. Research Validity
        ↓
2. Clear Research Gap
        ↓
3. Measurable Contribution
        ↓
4. Experimental Evaluation
        ↓
5. System Design
        ↓
6. Implementation Complexity
```

The goal is not simply to build a feature-rich learning platform.

The goal is to build a system that can support a meaningful research contribution and answer a clearly defined research question.

---

# 23. Open Questions to Resolve Through Literature Review

The following decisions are intentionally not fixed.

Research should help determine:

* Which behavioural signals are scientifically useful?
* Which signals can be reliably measured?
* Which learner states can realistically be inferred?
* How should learner states be represented?
* Should knowledge and behavioural states be modelled separately or jointly?
* What machine learning or deep learning approaches are appropriate?
* Is reinforcement learning necessary?
* How should adaptation decisions be evaluated?
* What datasets are available?
* Is a new dataset required?
* How can learner states such as fatigue or confidence be validated?
* Which baseline models should be used?
* What constitutes a meaningful improvement?
* What is the strongest defensible research contribution?

---

# 24. Final Project Objective

The final objective of this research is to investigate and develop an adaptive learning system that goes beyond correctness-based personalization.

The proposed system will explore whether combining:

```text
Academic Performance
+
Interaction Behaviour
+
Temporal Behaviour
+
Session Behaviour
```

can produce a richer learner representation.

This representation may then be used to generate more context-aware and personalized learning decisions.

The fundamental hypothesis to investigate is:

> Behavioural context may provide information that is not captured by correctness and performance alone, and this additional information may improve learner modelling and adaptive learning decisions.

This is a hypothesis to be tested through literature review and experimentation, not an assumption or established conclusion.
