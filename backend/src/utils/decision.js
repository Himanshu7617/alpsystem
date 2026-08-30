/**
 * Action selection with a logged propensity.
 *
 * The rule policy scores candidate items deterministically (questionPolicy.js).
 * A deterministic policy logs propensity 1.0 for every decision, which makes
 * inverse-propensity estimators degenerate: Study 3 needs the logging policy to
 * have put non-zero probability on the actions it did not take. So the served
 * policy is epsilon-greedy over the top-K of the same ranking, and both the
 * chosen action's probability and the whole action space are written to the
 * `Decision` row.
 *
 * ponytail: epsilon-greedy over top-K, not a softmax over all candidates. It is
 * the smallest thing that produces valid propensities. Upgrade to a softmax (or
 * a learned behaviour policy) in Phase 8, where the arms are defined.
 */

export const DEFAULT_EPSILON = Number(process.env.ALP_EPSILON ?? 0.1);
export const TOP_K = Number(process.env.ALP_TOP_K ?? 5);

/** Deterministic 32-bit PRNG so a session's exploration replays exactly. */
export const seededRandom = (seed) => {
    let state = (seed >>> 0) || 1;
    return () => {
        state = (state + 0x6d2b79f5) >>> 0;
        let t = state;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
};

/**
 * @param {Array<{question: object, score: number}>} scored - descending by score
 * @param {number} seed - session seed
 * @param {number} step - 1-based decision index within the session
 * @returns {{question: object, propensity: number, actionSpace: Array, epsilon: number, explored: boolean}}
 */
export const chooseWithPropensity = (scored, seed, step, epsilon = DEFAULT_EPSILON) => {
    const candidates = scored.slice(0, TOP_K);
    if (!candidates.length) return { question: undefined, propensity: 0, actionSpace: [], epsilon, explored: false };

    const k = candidates.length;
    // Greedy mass on the argmax, the exploration mass spread over all k
    // (including the argmax) so the probabilities sum to exactly 1.
    const probabilities = candidates.map((_, index) => (index === 0 ? 1 - epsilon : 0) + epsilon / k);

    const draw = seededRandom(seed + step * 7919)();
    let cumulative = 0;
    let picked = 0;
    for (let index = 0; index < k; index += 1) {
        cumulative += probabilities[index];
        if (draw <= cumulative) { picked = index; break; }
    }

    const actionSpace = candidates.map((candidate, index) => ({
        item_id: candidate.question.id,
        difficulty: candidate.question.difficulty,
        score: Number(candidate.score.toFixed(6)),
        probability: Number(probabilities[index].toFixed(6)),
        chosen: index === picked
    }));

    return {
        question: candidates[picked].question,
        propensity: probabilities[picked],
        actionSpace,
        epsilon,
        explored: picked !== 0
    };
};
