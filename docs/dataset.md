# Synthetic Interaction Dataset

Generate a deterministic dataset from the repository root:

```bash
PYTHONPATH=ml-service ml-service/venv/bin/python -m app.research.generate_dataset --learners 200 --items 24 --seed 20260726
```

The output contains one aggregate per-question event per simulated learner, plus a JSON manifest with the generator schema version, seed, dimensions, and SHA-256 checksum. The generator has four learner archetypes (`steady`, `rapid`, `careful`, `fatigable`) with independently sampled learning rate, confidence, guessing, reading/typing speed, fatigue growth, motivation, persistence, retention, and attention span.

This is synthetic data designed for pipeline and benchmark validation. It must not be represented as an observation of human learners or as evidence that behavioural analytics improves real learning outcomes.
