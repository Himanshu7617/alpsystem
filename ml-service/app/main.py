from fastapi import FastAPI
import logging
import json

from app.features.extract import FEATURE_VERSION, extract
from app.model.predictor import ARTIFACT_DIR, load_model, predict_difficulty
from app.schemas.events import ExtractRequest
from app.schemas.input_schema import PredictionInput
from app.schemas.serving import DecideRequest, StateRequest


class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_obj = {
            "level": record.levelname,
            "message": record.getMessage(),
            "timestamp": self.formatTime(record),
        }
        if hasattr(record, "extra_data"):
            log_obj.update(record.extra_data)
        return json.dumps(log_obj)


handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logging.basicConfig(level=logging.INFO, handlers=[handler])
ml_logger = logging.getLogger("ml-service")

app = FastAPI()


@app.get("/")
def home():
    return {"message": "ML Service Running"}


@app.get("/health")
def health():
    """Liveness plus whether a trained artifact is currently loadable."""
    artifact = load_model()
    return {
        "status": "ok",
        "model_loaded": artifact is not None,
        # Which artifact, and from where. `model_loaded: false` alone hid the
        # fact that the service was pointing at a file the pipeline had stopped
        # producing (Phase 8.5 audit).
        "model_name": (artifact or {}).get("model_name"),
        "model_source": (artifact or {}).get("source"),
        "artifact_dir": str(ARTIFACT_DIR),
    }


@app.post("/v1/features/extract")
def extract_features(request: ExtractRequest):
    """Raw events -> the per-attempt feature row.

    The backend calls this per attempt; offline training imports
    `app.features.extract` directly. One definition, two callers.
    """
    result = extract(request)
    ml_logger.info(
        "extract",
        extra={
            "extra_data": {
                "item_id": request.item.item_id,
                "n_events": len(request.events),
                "available_classes": result["available_classes"],
                "feature_version": FEATURE_VERSION,
            }
        },
    )
    return result


@app.post("/predict")
def predict(data: PredictionInput):
    result = predict_difficulty(data)
    ml_logger.info(
        "predict",
        extra={
            "extra_data": {
                "predicted_difficulty": result.get("nextDifficulty"),
                "predicted_success": result.get("predictedSuccess"),
                "low_confidence": result.get("low_confidence", False),
            }
        },
    )
    return result


@app.get("/v1/decisions/{decision_id}/explanation")
def decision_explanation(decision_id: str):
    """Why one adaptive decision was made, decomposed over validated states.

    Phase 8 writes the sampled decision traces to
    ``artifacts/evaluation/decision-explanations.json``; this endpoint serves
    them. Two guarantees, both checked by ``app.research.test_policies``:

    * an explanation never cites a state that failed the validation gate, and
    * the rules that were *excluded* — including the two joint-state rules the
      gate disabled — are served alongside the ones that fired, because a
      system that silently drops a rule cannot be audited for having done so.
    """
    from fastapi import HTTPException

    from app.policy.explanations import find, render

    trace = find(decision_id)
    if trace is None:
        raise HTTPException(status_code=404, detail=f"no decision {decision_id!r}")
    return {**trace, "narrative": render(trace)}


# --------------------------------------------------------------- serving

@app.post("/v1/state")
def state(request: StateRequest):
    """Session history -> the gated multi-state estimate with its uncertainty.

    The caller sends every attempt of the session so far; the service replays
    the ones its slot has not seen. That is what makes the estimate survive a
    restart of this process without a session store.

    A state Phase 7's gate rejected is **absent** from the response, and the
    reason it was rejected is returned alongside so the platform can say so
    rather than showing a blank.
    """
    from app.policy import live

    engine = live.engine()
    index = engine.sync(request.session_id, [attempt.model_dump() for attempt in request.attempts])
    result = engine.states(index)
    ml_logger.info("state", extra={"extra_data": {
        "session_id": request.session_id, "attempts": len(request.attempts),
        "steps": result["steps"], "states": sorted(result["states"])}})
    return result


@app.post("/v1/decide")
def decide(request: DecideRequest):
    """The next action, its propensity and its explanation.

    The action is the shared Phase 8 action space — difficulty bin, concept
    move, intervention — not an item id: which item realises a difficulty in a
    concept is the platform's bank lookup, exactly as it is the environment's
    in the closed loop.
    """
    from app.policy import live

    engine = live.engine()
    if request.attempts is not None:
        engine.sync(request.session_id, [attempt.model_dump() for attempt in request.attempts])
    action = engine.decide(request.session_id, request.view.model_dump())
    ml_logger.info("decide", extra={"extra_data": {
        "session_id": request.session_id, "policy": action["policy"],
        "difficulty": action["difficulty"], "intervention": action["intervention"],
        "propensity": action["propensity"], "explored": action["explored"]}})
    return action


@app.get("/model-info")
def model_info():
    """Which artifacts this process is serving, and what the gate allows.

    `model_loaded: false` on its own hid a service pointing at a file the
    pipeline had stopped producing (Phase 8.5 audit), so every version the
    decision depends on is named here.
    """
    from app.policy import live

    artifact = load_model()
    payload = {
        "feature_version": FEATURE_VERSION,
        "predictor": {"model_loaded": artifact is not None,
                      "model_name": (artifact or {}).get("model_name"),
                      "source": (artifact or {}).get("source")},
        "artifact_dir": str(ARTIFACT_DIR),
    }
    try:
        payload["serving"] = live.engine().info()
    except (OSError, SystemExit, RuntimeError) as error:
        # A checkout without trained artifacts must still answer this endpoint:
        # "which policy is live" is exactly the question being asked when the
        # answer is "none of them".
        payload["serving"] = {"error": str(error), "policy": live.policy_key()}
    return payload
