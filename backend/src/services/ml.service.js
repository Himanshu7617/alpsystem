import logger from "../config/logger.js";

const ML_SERVICE_URL = process.env.ML_SERVICE_URL ?? "http://localhost:8000";
const TIMEOUT_MS = Number(process.env.ML_TIMEOUT_MS ?? 2000);

/**
 * The two calls the adaptive loop makes into the research pipeline.
 *
 * Both return `null` when the service is unreachable, slow or unhappy, and the
 * caller falls back to the rule policy. A learner is never blocked on a model:
 * that is a Phase 10 requirement, not a nicety, and the fallback is logged and
 * surfaced in the response rather than hidden.
 */
const post = async (route, payload) => {
    const started = Date.now();
    try {
        const response = await fetch(`${ML_SERVICE_URL}${route}`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload),
            signal: AbortSignal.timeout(TIMEOUT_MS)
        });
        if (!response.ok) {
            logger.warn({ route, status: response.status }, "ml-call-failed");
            return null;
        }
        const body = await response.json();
        return { ...body, latency_ms: Date.now() - started };
    } catch (error) {
        logger.warn({ route, error: error.message, ms: Date.now() - started }, "ml-unreachable");
        return null;
    }
};

/** Session history → the gated state estimate with its uncertainty. */
export const estimate_state = (session_id, attempts) =>
    post("/v1/state", { session_id, attempts });

/** State + context → the next action, its propensity and its explanation. */
export const decide = (session_id, view, attempts) =>
    post("/v1/decide", { session_id, view, attempts });

/** Which policy and which artifacts the service is currently running. */
export const model_info = async () => {
    try {
        const response = await fetch(`${ML_SERVICE_URL}/model-info`,
                                     { signal: AbortSignal.timeout(TIMEOUT_MS) });
        return response.ok ? await response.json() : null;
    } catch (error) {
        logger.warn({ error: error.message }, "ml-model-info-unreachable");
        return null;
    }
};

/**
 * Legacy difficulty prediction, still used by the pre-/v1 `/question` path.
 * The research loop does not go through it.
 */
export const predict_next_difficulty = async (features) => {
    const response = await fetch(`${ML_SERVICE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(features)
    });
    return response.json();
};
