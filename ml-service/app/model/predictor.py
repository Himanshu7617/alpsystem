from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


ARTIFACT_DIR = Path(__file__).resolve().parents[3] / "artifacts/models"
#: Pre-Phase-0 paths. The research pipeline stopped producing these before
#: Phase 0 and nothing noticed, because `load_model()` returning ``None`` is
#: silent and `/predict` falls back to a rule. That drift is exactly what
#: Phase 8.5's registry exists to prevent, so the registry is tried first and
#: these remain only so an old checkout still serves something.
ARTIFACT_CAL = ARTIFACT_DIR / "best-next-correct-calibrated.joblib"
ARTIFACT_RAW = ARTIFACT_DIR / "best-next-correct.joblib"


@lru_cache
def load_model():
    """The Study 1 artifact through the registry; legacy joblibs as a fallback.

    Returns the same ``{"model": ..., "model_name": ...}`` shape either way, plus
    ``features`` when the registry supplied it, so a caller can check whether it
    actually holds the columns the model was fitted on.
    """
    from app.model import registry

    for name in registry.available():
        if not name.startswith("study1-"):
            continue
        loaded = registry.load(name)
        return {"model": loaded.model, "model_name": name,
                "features": loaded.features, "source": "registry",
                "metrics": loaded.metrics}
    for path in (ARTIFACT_CAL, ARTIFACT_RAW):
        if path.exists():
            artifact = joblib.load(path)
            return {**artifact, "source": "legacy", "features": artifact.get("features")}
    return None


def predict_difficulty(data):
    artifact = load_model()
    base = data.model_dump() if artifact else None
    if artifact and artifact.get("features"):
        # A Study 1 model is fitted on the full offline feature row. This
        # endpoint receives a much smaller payload, so rather than feed the
        # estimator columns it never saw — and return a confident wrong number —
        # fall through to the rule and say which features were missing. Phase 10
        # replaces this endpoint with /v1/state and /v1/decide, which build the
        # full row before calling the model.
        missing = [name for name in artifact["features"] if name not in base]
        if missing:
            fallback = _rule_fallback(data)
            return {**fallback, "policy": "fallback_rule",
                    "model_unavailable_reason":
                        f"{artifact['model_name']} needs {len(artifact['features'])} "
                        f"features; the request supplies neither {len(missing)} of them "
                        f"(e.g. {missing[:3]}) — see BUILD.md Phase 10"}
    if artifact:
        candidates = {"easy": 2, "medium": 5, "hard": 8}
        probabilities = {}
        for difficulty, score in candidates.items():
            row = {**base, "difficulty_score": score}
            probabilities[difficulty] = float(
                artifact["model"].predict_proba(pd.DataFrame([row]))[0, 1]
            )

        # Check confidence: if the spread across candidates is too narrow,
        # the model can't meaningfully differentiate difficulty levels.
        spread = max(probabilities.values()) - min(probabilities.values())
        low_confidence = spread < 0.05

        # Choose a productive-success band instead of maximising easy-item accuracy.
        difficulty = min(probabilities, key=lambda name: abs(probabilities[name] - 0.72))

        return {
            "nextDifficulty": difficulty,
            "predictedSuccess": probabilities[difficulty],
            "allProbabilities": probabilities,
            "spread": round(spread, 4),
            "low_confidence": low_confidence,
            "policy": "ml",
            "model": artifact["model_name"],
        }

    return {**_rule_fallback(data), "policy": "fallback_rule"}


def _rule_fallback(data):
    """The rule used whenever no usable model prediction is available."""
    if data.isCorrect and data.timeTaken < 60:
        return {"nextDifficulty": "hard", "low_confidence": False}
    if data.isCorrect:
        return {"nextDifficulty": "medium", "low_confidence": False}
    return {"nextDifficulty": "easy", "low_confidence": False}
