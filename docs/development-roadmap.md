# Adaptive Learning Platform Development Roadmap

## Research question and hypothesis

**Question:** Do adaptive learning systems that use behavioural analysis improve learning efficiency compared with conventional adaptation?

**Primary hypothesis:** Under the same question bank and mastery criterion, behavioural-state adaptation reduces questions and elapsed time to concept mastery compared with static and correctness-only adaptive policies.

**Primary outcome:** time to mastery. Secondary outcomes are questions to mastery, final concept mastery, calibration, and per-step predictive performance. Synthetic results are not claims about human learners.

## Ordered stages

| Stage | Deliverable and acceptance criteria |
| --- | --- |
| 0. Foundation | This review, a stable API contract, documented state/action semantics, and automated unit/integration tests |
| 1. Explainable rule engine | Calibrated feature transforms, deterministic latent-state updates, action reasons, confidence/uncertainty, concept-aware selection, and test fixtures |
| 2. Student simulator | Seeded archetypes with independent learning, confidence, behaviour, fatigue, motivation, retention, and attention parameters; reproducible interaction trajectories |
| 3. Synthetic data | Versioned records containing pre/post learner state, aggregated features, item metadata, rule output, selected action, and outcome; schema and data card |
| 4. ML pipeline | Leakage-safe split by learner, baseline and advanced models, saved preprocessing/model artifacts, fixed seeds, and a model card |
| 5. Model comparison | Logistic regression, decision tree, random forest, XGBoost, LightGBM, CatBoost, MLP; accuracy, precision, recall, F1, ROC-AUC, training/inference time, and feature importance |
| 6. Dynamic roadmap | Versioned concept prerequisite graph, mastery beliefs per concept, recommendation actions, and explainable roadmap updates |
| 7. Experimental evaluation | Static, existing-rule, improved-rule, and ML policies assessed on matched seeded cohorts; bootstrap confidence intervals and paired tests where assumptions hold |
| 8. Paper | Methods, results, limitations, reproducibility appendix, and only evidence supported by executed experiments |

## Policy baselines

1. **Static curriculum:** fixed prerequisite order and fixed difficulty progression.
2. **Existing rule baseline:** freeze the original thresholds currently in `backend/src/utils/ruleEngine.js`; do not tune it after outcomes are inspected.
3. **Improved rule policy:** documented feature transformations and state/action logic designed before outcome evaluation.
4. **ML policy:** trained solely from the designated training learners, evaluated on held-out learner cohorts.

## Dataset and reproducibility protocol

- All generators accept an explicit seed; a manifest records generator version, seed, parameters, row count, and checksum.
- Split by learner before fit/transform operations. No events from the same simulated learner may span train and test.
- The target must be available at decision time; post-answer fields cannot predict a pre-answer action.
- Store only aggregate behavioural values, not raw mouse paths or key content.
- Fix dependency versions, write configuration files, and log runtime/hardware metadata for each run.

## Statistical protocol

- For each policy, run the same seeded learner cohorts and item graph.
- Report mean/median time to mastery, questions to mastery, failure rate, and 95% bootstrap confidence intervals.
- Use paired comparisons across identical learner seeds; test distributional assumptions before selecting paired t-test or Wilcoxon signed-rank test.
- Report effect sizes and multiple-comparison corrections for the four policy comparisons.
- Treat simulator outcomes as construct-validation results, not evidence of human learning efficacy.

## Key decisions and trade-offs

| Decision | Alternatives considered | Chosen rationale |
| --- | --- | --- |
| State model | opaque end-to-end predictor; five latent variables | Retain five interpretable variables to satisfy explainability and enable rule baseline analysis |
| Behaviour capture | raw interaction logs; aggregate features | Aggregate consented features to reduce privacy risk and simplify reproducibility |
| Storage | JSON-only state; normalized event model | Add immutable interaction events while retaining session snapshots for fast serving |
| ML objective | predict correctness; predict action/outcome | Train decision-time models and evaluate closed-loop learning efficiency, not classification alone |
| Evaluation data | only synthetic; immediate human trial | Begin synthetic with explicit limitations; human study requires separate ethics, consent, and power planning |

## Risks and mitigations

- **Sim-to-real gap:** publish parameter sensitivity and avoid human-effect claims.
- **Circular labels:** define the decision target independently of the rule engine where possible and keep rule outputs as features only in ablation experiments.
- **Model availability:** make gradient boosting optional adapters with recorded availability; required comparisons fail visibly rather than silently substituting another model.
- **Privacy:** consent gate, aggregation, retention limits, and no content-level keystroke capture.
- **Question quality:** validate generated schema and retain item provenance; use curated fixtures when generation is unavailable.
