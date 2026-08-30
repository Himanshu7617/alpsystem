import logger from "../config/logger.js";

const ML_SERVICE_URL = process.env.ML_SERVICE_URL ?? "http://localhost:8000";

/**
 * Per-attempt feature extraction.
 *
 * The backend never computes a feature itself: it posts the raw event stream to
 * the ML service, which owns the single definition of every feature
 * (`ml-service/app/features/extract.py`). Offline training imports that same
 * module directly. Two implementations of one feature is how train/serve skew
 * starts, so there is exactly one.
 *
 * Returns `null` when the service is unreachable — a learner is never blocked
 * on the feature pipeline; the attempt is stored with empty features and the
 * failure is logged.
 */
export const extract_features = async (payload) => {
    try {
        const response = await fetch(`${ML_SERVICE_URL}/v1/features/extract`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });
        if (!response.ok) {
            logger.warn({ status: response.status }, "feature-extract-failed");
            return null;
        }
        return await response.json();
    } catch (error) {
        logger.warn({ error: error.message }, "feature-extract-unreachable");
        return null;
    }
};
