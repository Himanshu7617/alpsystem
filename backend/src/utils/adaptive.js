/**
 * The glue between the served policy and the item bank.
 *
 * The policy chooses an *action* — a difficulty bin, a concept move and an
 * intervention — exactly as it does in the closed loop. Which item realises
 * that action is the platform's job, and it is done the same way the
 * environment does it (`closed_loop.resolve`): move to the target concept,
 * skip concepts already mastered, then take the unasked item whose difficulty
 * bin is closest to the one asked for.
 *
 * Keeping the two resolutions the same is the point. If the live platform
 * resolved an action differently from the environment the policy was evaluated
 * in, the evaluation would not describe the deployment.
 */

/** Attempt records in the shape `/v1/state` and `/v1/decide` expect. */
export const attemptRecords = (session_id, attempts, bank) =>
    attempts.map((attempt, index) => {
        const item = bank.items.get(attempt.item_id) ?? {};
        return {
            session_id,
            item_id: attempt.item_id,
            concept_id: item.concept_id ?? "unknown",
            correct: attempt.correct,
            difficulty_score: item.difficultyScore ?? 1,
            response_time_ms: attempt.response_time_ms ?? 0,
            item_b: item.b ?? 0,
            timestamp: new Date(attempt.submitted_at).getTime(),
            position: index,
            question_number: attempt.seq,
            idle_time: attempt.features?.idle_time ?? null,
            features: attempt.features ?? {}
        };
    });

/**
 * The decision context. Everything in it is observable from the session log;
 * nothing in it is a state estimate — those reach the policy through the
 * service's own estimator, gated.
 */
export const decisionView = (records, curriculum, activeConcept) => {
    const outcomes = records.map(record => (record.correct ? 1 : 0));
    let incorrect_streak = 0;
    for (let index = outcomes.length - 1; index >= 0 && outcomes[index] === 0; index -= 1) {
        incorrect_streak += 1;
    }
    return {
        position: records.length,
        session_position: records.length,
        active_concept: activeConcept,
        concept_attempts: records.filter(record => record.concept_id === activeConcept).length,
        outcomes,
        incorrect_streak,
        curriculum
    };
};

/** Concepts whose roadmap status is `mastered`. */
export const masteredConcepts = (roadmap) =>
    new Set((roadmap?.concepts ?? []).filter(concept => concept.status === "mastered")
                                     .map(concept => concept.concept_id));

/**
 * Apply the concept move, then skip forward over anything already mastered —
 * the environment's `_advance_to_unmastered`, so a policy that says
 * "next_concept" lands where it would have landed in the experiment.
 */
export const targetConcept = (move, active, curriculum, mastered) => {
    const order = curriculum.length ? curriculum : [active];
    const current = Math.max(0, order.indexOf(active));
    let concept = order[current];
    if (move === "next_concept") {
        const rest = [...order.slice(current + 1), ...order.slice(0, current)];
        concept = rest.find(name => !mastered.has(name)) ?? concept;
    } else if (move === "prerequisite_concept") {
        concept = order.slice(0, current).reverse()[0] ?? concept;
    }
    if (!mastered.has(concept)) return concept;
    const from = order.indexOf(concept);
    const rest = [...order.slice(from + 1), ...order.slice(0, from + 1)];
    return rest.find(name => !mastered.has(name)) ?? concept;
};

/**
 * The item that realises an action.
 *
 * Returns the chosen question and the candidates it was chosen from, so the
 * `Decision` row records what the platform could have served as well as what
 * it did — the action space the policy scored is the 120-cell one, and this is
 * the part of it the bank could actually deliver.
 */
export const itemForAction = (action, questions, asked, concept) => {
    const unasked = questions.filter(question => !asked.includes(question.id));
    const pool = unasked.filter(question => (question.concepts ?? []).includes(concept));
    const candidates = pool.length ? pool : unasked;
    if (!candidates.length) return { question: undefined, candidates: [] };

    const wanted = action.difficulty;
    const distance = candidates.map(question => Math.abs((question.difficultyScore ?? 1) - wanted));
    const closest = Math.min(...distance);
    const tied = candidates.filter((_, index) => distance[index] === closest);
    return {
        question: tied[0],
        candidates: tied.map(question => ({
            item_id: question.id,
            difficulty: question.difficultyScore,
            concept_id: (question.concepts ?? [])[0] ?? null
        }))
    };
};

/**
 * Learner-facing text for the action.
 *
 * `ponytail:` the support content is the item's own explanation, revealed in
 * part for a hint and in full for a worked example. The bank has no authored
 * scaffolds and inventing them would put unmeasured pedagogy into the live
 * system; the pre-registered intervention effects (§9.3) are simulation
 * constants and are not claimed here. Author real scaffolds if the platform is
 * ever run with learners whose outcomes are analysed.
 */
export const interventionText = (intervention, question) => {
    const explanation = question?.explanation ?? "";
    if (intervention === "hint") {
        const first = explanation.split(/(?<=\.)\s/)[0] ?? "";
        return { kind: "hint", title: "A hint before you answer", body: first };
    }
    if (intervention === "worked_example") {
        return { kind: "worked_example", title: "Worked example", body: explanation };
    }
    if (intervention === "break_suggestion") {
        return {
            kind: "break_suggestion",
            title: "Take a short break",
            body: "Your recent pattern suggests a pause will help more than another item right now."
        };
    }
    return null;
};

/**
 * A plain-English rendering of why this item was served.
 *
 * It may name only states the validation gate admitted. The rules the gate
 * disabled are reported as excluded rather than dropped silently, because a
 * system that quietly stops applying a rule cannot be audited for having done
 * so.
 */
export const explain = (action, decision, concept) => {
    const parts = [];
    const rules = decision?.explanation?.rules_fired ?? [];
    for (const rule of rules) {
        parts.push(String(rule.rule ?? rule).replace(/_/g, " "));
    }
    if (!parts.length) parts.push("no adaptation rule applied to this attempt");
    const intervention = action.intervention !== "no_intervention"
        ? ` It also offers a ${action.intervention.replace(/_/g, " ")}.`
        : "";
    const because = parts.length ? ` because ${parts.join("; ")}` : "";
    return `Serving a difficulty-${action.difficulty} item on ${concept.replace(/_/g, " ")}`
        + `${because}.${intervention}`;
};
