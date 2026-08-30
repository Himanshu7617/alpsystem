/**
 * The /v1 research API: consent, session, item delivery, telemetry ingestion,
 * attempt submission.
 *
 * Three properties of this controller are research requirements rather than
 * engineering taste:
 *   1. Correctness is computed here from the stored answer key. The answer key
 *      is never sent to the client, so a client cannot report its own outcome.
 *   2. Every served item is a logged `Decision` carrying the propensity under
 *      which it was chosen. Study 3 cannot be run without it and it cannot be
 *      backfilled.
 *   3. Every signal class can be switched off by consent, and the loop must
 *      still complete. That switch is the mechanism behind the RQ4
 *      privacy-utility curve.
 */
import prisma from "../config/db.js";
import logger from "../config/logger.js";
import { create_session_with_bank } from "../services/session.service.js";
import { canonicalBank } from "../services/questionBank.service.js";
import { extract_features } from "../services/features.service.js";
import { decide as ml_decide, estimate_state } from "../services/ml.service.js";
import {
    attemptRecords, decisionView, explain, interventionText, itemForAction,
    masteredConcepts, targetConcept
} from "../utils/adaptive.js";
import { calculateStateAndNextQuestion } from "../utils/ruleEngine.js";
import { buildLearningRoadmap } from "../utils/conceptRoadmap.js";
import { chooseNextQuestion } from "../utils/questionPolicy.js";
import { chooseWithPropensity, seededRandom } from "../utils/decision.js";

// One name for one switch. `ADAPTIVE_POLICY` was the compose-file spelling and
// `ALP_POLICY` the ml-service one; two names for one switch is how a service
// ends up running a different policy from the one the paper reports, so the old
// name is now an error rather than a silent second opinion.
if (process.env.ADAPTIVE_POLICY && !process.env.ALP_POLICY) {
    throw new Error("ADAPTIVE_POLICY was renamed ALP_POLICY — set ALP_POLICY instead");
}
const POLICY_NAME = process.env.ALP_POLICY ?? "rule_improved";
const POLICY_VERSION = "phase2-rule-improved-eps";
//: The policy served when the ML service cannot be reached. Never block a
//: learner on a model: the loop degrades to the Phase 2 rule engine and says so.
const FALLBACK_POLICY = "rule_improved_fallback";
const DEFAULT_TOPIC = process.env.ALP_TOPIC ?? "item-bank-v1";
const PROBE_RATE = Number(process.env.ALP_PROBE_RATE ?? 0.12);
const EFFORT_PROBE_EVERY = Number(process.env.ALP_EFFORT_PROBE_EVERY ?? 15);

/** Signal class each event type belongs to. Events of a class the learner did
 *  not consent to are dropped at ingestion, not merely hidden downstream. */
const EVENT_SIGNAL_CLASS = {
    SESSION_STARTED: "core", SESSION_ENDED: "core", ITEM_PRESENTED: "core",
    ANSWER_SUBMITTED: "core", ITEM_SKIPPED: "core", DECISION_MADE: "core",
    FIRST_INTERACTION: "timing",
    OPTION_SELECTED: "interaction", OPTION_CHANGED: "interaction", HINT_REQUESTED: "interaction",
    IDLE_ENTERED: "interaction", IDLE_EXITED: "interaction", VISIBILITY_CHANGED: "interaction",
    CURSOR_SEGMENT: "motor",
    PROBE_SHOWN: "probes", PROBE_ANSWERED: "probes"
};

const DEFAULT_CONSENT = { correctness: true, timing: true, interaction: true, motor: true, probes: true };

/** A stable 32-bit seed per session so probe assignment and exploration replay. */
const seedFromId = (sessionId) => {
    let hash = 2166136261;
    for (const character of sessionId) {
        hash ^= character.charCodeAt(0);
        hash = Math.imul(hash, 16777619);
    }
    // 31 bits: the seed is stored in a Postgres integer column.
    return (hash >>> 0) & 0x7fffffff;
};

const consentFor = async (session) => {
    if (!session.consent_id) return DEFAULT_CONSENT;
    const consent = await prisma.consent.findUnique({ where: { consent_id: session.consent_id } });
    if (!consent) return DEFAULT_CONSENT;
    return {
        correctness: consent.correctness,
        timing: consent.timing,
        interaction: consent.interaction,
        motor: consent.motor,
        probes: consent.probes
    };
};

const publicItem = (question) => ({
    item_id: question.id,
    question_id: question.question_id,
    stem: question.question,
    options: question.options,
    difficulty: question.difficulty,
    difficulty_score: question.difficultyScore,
    concepts: question.concepts,
    format: question.questionType,
    estimated_time_seconds: question.estimatedTimeSeconds
});

/** Probe assignment: randomised, seeded, and logged so it stays analysable. */
const assignProbes = (seed, seq, consent) => {
    if (!consent.probes) return [];
    const draw = seededRandom(seed + seq * 104729)();
    const probes = [];
    if (draw < PROBE_RATE) {
        probes.push({ kind: "confidence", scale_points: 4, assign_p: PROBE_RATE, assign_draw: draw });
        probes.push({ kind: "perceived_difficulty", scale_points: 4, assign_p: PROBE_RATE, assign_draw: draw });
    }
    if (seq % EFFORT_PROBE_EVERY === 0) {
        probes.push({ kind: "effort", scale_points: 5, assign_p: 1, assign_draw: draw });
    }
    return probes;
};

// ---------------------------------------------------------------- consent

export const post_consent = async (req, res) => {
    const { learner_id, classes = {} } = req.body ?? {};
    if (!learner_id) return res.status(400).json({ error: "learner_id is required" });

    const consent = await prisma.consent.create({
        data: {
            learner_id,
            // correctness is not optional: without it there is no adaptive loop
            // to consent to. Every other class is the learner's choice.
            correctness: true,
            timing: classes.timing !== false,
            interaction: classes.interaction !== false,
            motor: classes.motor !== false,
            probes: classes.probes !== false,
            user_agent: req.headers["user-agent"] ?? null
        }
    });
    return res.status(201).json({ consent_id: consent.consent_id, classes: {
        correctness: consent.correctness, timing: consent.timing, interaction: consent.interaction,
        motor: consent.motor, probes: consent.probes
    } });
};

// ---------------------------------------------------------------- sessions

export const post_session = async (req, res) => {
    const { learner_id, topic = null, consent_id = null } = req.body ?? {};
    if (!learner_id) return res.status(400).json({ error: "learner_id is required" });
    // The bank is the server's choice, not the client's: the served policy acts
    // over the canonical bank's concept ids and difficulty bins, and a session
    // built from any other bank would log decisions the experiments cannot
    // interpret. A client-supplied topic is recorded and ignored.
    if (topic && topic !== DEFAULT_TOPIC) {
        logger.info({ requested: topic, using: DEFAULT_TOPIC }, "v1-topic-ignored");
    }

    const session = await create_session_with_bank(DEFAULT_TOPIC, { learner_id, consent_id, source: "pilot" });
    if (!session) return res.status(404).json({ error: `No question bank for topic: ${topic}` });

    const seed = seedFromId(session.session_id);
    await prisma.experimentAssignment.create({
        data: { learner_id, session_id: session.session_id, arm: POLICY_NAME, policy_name: POLICY_NAME, seed }
    });

    const consent = await consentFor(session);
    return res.status(201).json({
        session_id: session.session_id,
        seed,
        arm: POLICY_NAME,
        policy_name: POLICY_NAME,
        policy_version: POLICY_VERSION,
        consent,
        session_version: session.version
    });
};

// ---------------------------------------------------------------- next item

/**
 * The learned action, resolved to an item — or the rule fallback.
 *
 * Returns everything the caller needs to log a `Decision` and to tell the
 * learner what happened, including whether the ML service answered at all.
 */
const chooseItem = async ({ session, questions, asked, history, state, roadmap }) => {
    const bank = await canonicalBank();
    const attempts = await prisma.itemAttempt.findMany({
        where: { session_id: session.session_id }, orderBy: { seq: "asc" }
    });
    const records = attemptRecords(session.session_id, attempts, bank);
    const curriculum = bank.curriculum ?? [];
    const lastDecision = await prisma.decision.findFirst({
        where: { session_id: session.session_id }, orderBy: { createdAt: "desc" }
    });
    const active = lastDecision?.action?.concept_id ?? curriculum[0] ?? null;

    // The whole history goes with the request: the service keeps live sessions
    // in a fixed pool of slots, so one that was evicted — or lost to a restart
    // — rebuilds itself from this instead of answering from a stale estimate.
    const view = decisionView(records, curriculum, active);
    const action = curriculum.length
        ? await ml_decide(session.session_id, view, records)
        : null;

    if (action) {
        const concept = targetConcept(action.concept_move, active, curriculum,
                                      masteredConcepts(roadmap));
        const { question, candidates } = itemForAction(action, questions, asked, concept);
        if (question) {
            // The bank may have nothing left in the target concept, in which
            // case `itemForAction` serves the nearest item it does have. The
            // explanation and the logged action then name the concept actually
            // served, not the one the policy asked for.
            const served = (question.concepts ?? [])[0] ?? concept;
            return {
                question,
                concept: served,
                propensity: action.propensity,
                actionSpace: candidates,
                policy_name: action.policy,
                policy_version: `phase9-${action.policy}`,
                action: {
                    item_id: question.id, difficulty: action.difficulty,
                    concept_id: served, requested_concept: concept,
                    concept_move: action.concept_move,
                    intervention: action.intervention, action_index: action.action_index
                },
                explanation: {
                    ...action.explanation, explored: action.explored,
                    rung: action.rung, latency_ms: action.latency_ms,
                    narrative: explain(action, { explanation: action.explanation }, served)
                },
                intervention: interventionText(action.intervention, question),
                fallback: null
            };
        }
    }

    // Fail-safe: the Phase 2 rule engine over the same bank, epsilon-greedy so
    // the decision still carries a usable propensity.
    const reason = curriculum.length ? "ml-service-unavailable" : "no-canonical-curriculum";
    const { scored } = chooseNextQuestion(questions, session.current_difficulty, asked,
                                          history, state, roadmap);
    const seed = seedFromId(session.session_id);
    const chosen = chooseWithPropensity(scored, seed, asked.length + 1);
    if (!chosen.question) return { question: undefined, fallback: reason };
    const lastRecommendation = history.at(-1)?.recommendation ?? {};
    return {
        question: chosen.question,
        concept: (chosen.question.concepts ?? [])[0] ?? null,
        propensity: chosen.propensity,
        actionSpace: chosen.actionSpace,
        policy_name: FALLBACK_POLICY,
        policy_version: POLICY_VERSION,
        action: { item_id: chosen.question.id,
                  difficulty: chosen.question.difficultyScore ?? null,
                  difficulty_label: chosen.question.difficulty,
                  concept_id: (chosen.question.concepts ?? [])[0] ?? null },
        explanation: {
            epsilon: chosen.epsilon, explored: chosen.explored,
            reasons: lastRecommendation.reasons ?? [],
            target_concept: roadmap.target_concept ?? null,
            narrative: "The adaptive model was unavailable, so the rule engine chose this item."
        },
        intervention: null,
        fallback: reason
    };
};

export const get_next_item = async (req, res) => {
    const session_id = req.query?.session_id;
    if (!session_id) return res.status(400).json({ error: "session_id is required" });

    const session = await prisma.session.findUnique({ where: { session_id } });
    if (!session) return res.status(404).json({ error: "Session not found" });

    const questions = await prisma.question.findMany({ where: { session_id } });
    const asked = session.asked_questions ?? [];
    const history = session.history ?? [];
    const latest = await prisma.stateEstimate.findFirst({
        where: { session_id }, orderBy: { createdAt: "desc" }
    });
    const state = {
        knowledge: session.mastery, confidence: session.confidence, engagement: session.engagement,
        cognitive_load: session.cognitive_load, fatigue: session.fatigue,
        knowledge_validated: latest?.knowledge_validated ?? false
    };
    const roadmap = buildLearningRoadmap(questions, history, state);
    const chosen = await chooseItem({ session, questions, asked, history, state, roadmap });
    // Only the concepts this session can actually serve. The graph has 36; the
    // curriculum the policy acts over has six, and showing the learner a target
    // the policy will never choose is a second, contradictory roadmap.
    const bank = await canonicalBank();
    const sessionRoadmap = {
        ...roadmap,
        concepts: (roadmap.concepts ?? []).filter(concept =>
            (bank.curriculum ?? []).includes(concept.concept_id)),
        target_concept: chosen.concept ?? roadmap.target_concept
    };
    const { question } = chosen;
    if (!question) return res.status(404).json({ error: "No item available" });

    const seed = seedFromId(session_id);
    const step = asked.length + 1;
    const propensity = chosen.propensity;
    const decision = await prisma.decision.create({
        data: {
            session_id,
            policy_name: chosen.policy_name,
            policy_version: chosen.policy_version,
            action: chosen.action,
            action_space: chosen.actionSpace,
            propensity,
            state_snapshot: state,
            explanation: chosen.explanation
        }
    });

    const updated = await prisma.session.updateMany({
        where: { session_id, version: session.version },
        data: {
            asked_questions: [...asked, question.id],
            current_difficulty: question.difficulty,
            version: session.version + 1
        }
    });
    if (updated.count === 0) return res.status(409).json({ error: "session changed concurrently; re-read and retry" });
    await prisma.question.update({ where: { question_id: question.question_id }, data: { asked: true } });

    const consent = await consentFor(session);
    return res.json({
        item: publicItem(question),
        seq: step,
        decision_id: decision.decision_id,
        propensity,
        policy: chosen.policy_name,
        explanation: chosen.explanation.narrative,
        intervention: chosen.intervention,
        target_concept: chosen.concept,
        roadmap: sessionRoadmap,
        fallback: chosen.fallback,
        probes: assignProbes(seed, step, consent),
        consent,
        session_version: session.version + 1,
        total_items: questions.length,
        items_asked: step
    });
};

// ---------------------------------------------------------------- events

export const post_events = async (req, res) => {
    const { session_id, learner_id = null, events = [] } = req.body ?? {};
    if (!session_id) return res.status(400).json({ error: "session_id is required" });
    if (!Array.isArray(events)) return res.status(400).json({ error: "events must be an array" });

    const session = await prisma.session.findUnique({ where: { session_id } });
    if (!session) return res.status(404).json({ error: "Session not found" });
    const consent = await consentFor(session);

    const accepted = [];
    const rejected = [];
    for (const event of events) {
        const signalClass = EVENT_SIGNAL_CLASS[event.type];
        if (!signalClass) { rejected.push({ event_id: event.event_id, reason: "unknown type" }); continue; }
        if (signalClass !== "core" && consent[signalClass] === false) {
            rejected.push({ event_id: event.event_id, reason: `consent denied for ${signalClass}` });
            continue;
        }
        accepted.push(event);
    }

    // Idempotent on event_id: a client retry after a dropped response replays
    // the same batch and writes nothing twice — including the derived
    // CursorSegment and Probe rows, which is why the replay is filtered out
    // here rather than relying on createMany's skipDuplicates alone.
    const seen = await prisma.interactionEvent.findMany({
        where: { event_id: { in: accepted.map(event => event.event_id) } },
        select: { event_id: true }
    });
    const alreadyStored = new Set(seen.map(row => row.event_id));
    const fresh = accepted.filter(event => !alreadyStored.has(event.event_id));

    const written = await prisma.interactionEvent.createMany({
        data: fresh.map(event => ({
            event_id: event.event_id,
            session_id,
            learner_id: learner_id ?? session.learner_id,
            item_id: event.item_id ?? null,
            attempt_id: event.attempt_id ?? null,
            type: event.type,
            client_ts: new Date(event.client_ts),
            seq: event.seq ?? 0,
            payload: event.payload ?? {}
        })),
        skipDuplicates: true
    });

    await materialise(session_id, fresh);

    return res.status(202).json({
        received: events.length,
        stored: written.count,
        rejected,
        session_version: session.version
    });
};

/**
 * Derives the typed rows the analysis reads (`CursorSegment`, `Probe`) from the
 * event stream, so ingestion stays a single endpoint and the typed tables can
 * always be rebuilt from `InteractionEvent`.
 */
const materialise = async (session_id, events) => {
    for (const event of events) {
        if (event.type === "CURSOR_SEGMENT") {
            const trajectory = event.payload ?? {};
            await prisma.cursorSegment.create({
                data: {
                    session_id,
                    item_id: event.item_id ?? null,
                    auc_toward_nonchosen: trajectory.auc_toward_nonchosen ?? 0,
                    max_deviation: trajectory.max_deviation ?? 0,
                    x_flips: trajectory.x_flips ?? 0,
                    sample_entropy: trajectory.sample_entropy ?? 0,
                    velocity_peak: trajectory.velocity_peak ?? 0,
                    velocity_mean: trajectory.velocity_mean ?? 0,
                    pause_count: trajectory.pause_count ?? 0,
                    path_ratio: trajectory.path_ratio ?? 1,
                    time_to_first_movement_ms: Math.round(trajectory.time_to_first_movement_ms ?? 0),
                    time_to_first_selection_ms: Math.round(trajectory.time_to_first_selection_ms ?? 0),
                    hover_time_ms: trajectory.hover_time_ms ?? [],
                    n_samples: trajectory.n_samples ?? 0,
                    sample_interval_ms: trajectory.sample_interval_ms ?? 50
                }
            });
        }
        if (event.type === "PROBE_SHOWN") {
            const probe = event.payload ?? {};
            await prisma.probe.upsert({
                where: { probe_id: probe.probe_id },
                update: {},
                create: {
                    probe_id: probe.probe_id,
                    session_id,
                    item_id: event.item_id ?? null,
                    kind: probe.kind,
                    scale_points: probe.scale_points ?? 4,
                    shown_at: new Date(event.client_ts),
                    assign_p: probe.assign_p ?? 0,
                    assign_draw: probe.assign_draw ?? 0
                }
            });
        }
        if (event.type === "PROBE_ANSWERED") {
            const probe = event.payload ?? {};
            await prisma.probe.updateMany({
                where: { probe_id: probe.probe_id },
                data: {
                    answered_at: new Date(event.client_ts),
                    response: probe.response ?? null,
                    skipped: Boolean(probe.skipped)
                }
            });
        }
    }
};

// ---------------------------------------------------------------- attempts

export const post_attempt = async (req, res) => {
    const {
        session_id, item_id, selected_index = null, skipped = false,
        presented_at = null, response_time_ms = null, session_version = null
    } = req.body ?? {};

    if (!session_id || !item_id) return res.status(400).json({ error: "session_id and item_id are required" });

    const session = await prisma.session.findUnique({ where: { session_id } });
    if (!session) return res.status(404).json({ error: "Session not found" });
    if (session_version !== null && session_version !== session.version) {
        return res.status(409).json({ error: "stale session_version", session_version: session.version });
    }

    const questions = await prisma.question.findMany({ where: { session_id } });
    const question = questions.find(candidate => candidate.id === item_id);
    if (!question) return res.status(404).json({ error: "Item not in this session" });

    // Correctness is a server-side fact derived from the stored answer key.
    // The key never leaves the server before this point.
    const options = question.options ?? [];
    const correct_index = options.indexOf(question.correctAnswer);
    const correct = !skipped && selected_index !== null && Number(selected_index) === correct_index;

    const seq = await prisma.itemAttempt.count({ where: { session_id } }) + 1;
    const attempt = await prisma.itemAttempt.create({
        data: {
            session_id, item_id, question_id: question.question_id, seq,
            correct, selected_index: selected_index === null ? null : Number(selected_index),
            skipped: Boolean(skipped),
            presented_at: presented_at ? new Date(presented_at) : null,
            response_time_ms: response_time_ms === null ? null : Math.round(response_time_ms)
        }
    });

    // Everything logged for this item before the attempt existed now points at it.
    const link = { attempt_id: attempt.attempt_id };
    await prisma.interactionEvent.updateMany({ where: { session_id, item_id, attempt_id: null }, data: link });
    await prisma.cursorSegment.updateMany({ where: { session_id, item_id, attempt_id: null }, data: link });
    await prisma.probe.updateMany({ where: { session_id, item_id, attempt_id: null }, data: link });

    const consent = await consentFor(session);
    const events = await prisma.interactionEvent.findMany({
        where: { session_id, item_id }, orderBy: { seq: "asc" }
    });

    const extracted = await extract_features({
        consent,
        item: {
            item_id, difficulty: question.difficulty, difficulty_score: question.difficultyScore,
            estimated_time_seconds: question.estimatedTimeSeconds, n_options: options.length
        },
        attempt: {
            attempt_id: attempt.attempt_id, seq, correct, skipped: Boolean(skipped),
            selected_index, response_time_ms, presented_at,
            session_started_at: session.started_at.toISOString(),
            submitted_at: attempt.submitted_at.toISOString()
        },
        events: events.map(event => ({
            event_id: event.event_id,
            type: event.type,
            client_ts: event.client_ts.toISOString(),
            seq: event.seq,
            payload: event.payload
        }))
    });

    const features = extracted?.features ?? {};
    if (extracted) {
        await prisma.itemAttempt.update({
            where: { attempt_id: attempt.attempt_id },
            data: { features, feature_version: extracted.feature_version }
        });
    }

    // The rule engine is the Phase 2 state estimator. It is deliberately not a
    // model: Phase 7 replaces it and gates it. Nothing here is marked validated.
    const { studentState, recommendation, updatedHistory } = calculateStateAndNextQuestion(
        session, question,
        {
            ...features,
            isCorrect: correct,
            question_id: item_id,
            total_response_time: (features.total_response_time ?? (response_time_ms ?? 0) / 1000),
            skip: Boolean(skipped)
        }
    );
    updatedHistory.at(-1).recommendation = recommendation;

    // The validated estimate. `/v1/state` replays the session through Phase 7's
    // encoder and returns only what the gate admitted; the rule engine's other
    // three numbers are still stored, but stored as unvalidated, which is the
    // difference between a measurement and a heuristic.
    const bank = await canonicalBank();
    const priorAttempts = await prisma.itemAttempt.findMany({
        where: { session_id }, orderBy: { seq: "asc" }
    });
    const estimate = await estimate_state(session_id, attemptRecords(session_id, priorAttempts, bank));
    const gated = estimate?.states ?? {};
    const errors = estimate?.standard_error ?? {};
    const knowledge = gated.knowledge ?? studentState.knowledge;

    await prisma.stateEstimate.create({
        data: {
            session_id, attempt_id: attempt.attempt_id,
            knowledge, knowledge_se: errors.knowledge ?? null,
            knowledge_validated: gated.knowledge !== undefined,
            engagement: gated.engagement ?? studentState.engagement,
            engagement_se: errors.engagement ?? null,
            engagement_validated: gated.engagement !== undefined,
            confidence: gated.confidence ?? studentState.confidence,
            confidence_se: errors.confidence ?? null,
            confidence_validated: gated.confidence !== undefined,
            fatigue: gated.fatigue ?? studentState.fatigue,
            fatigue_se: errors.fatigue ?? null,
            fatigue_validated: gated.fatigue !== undefined,
            estimator: estimate ? `phase7-gru-${estimate.rung}` : "rule_improved",
            estimator_version: estimate ? `gate-${(estimate.gate?.admitted ?? []).join("+") || "none"}`
                                        : POLICY_VERSION
        }
    });

    const updated = await prisma.session.updateMany({
        where: { session_id, version: session.version },
        data: {
            // The validated estimate when there is one; the rule engine's own
            // number only while the model is unreachable.
            mastery: knowledge,
            confidence: studentState.confidence,
            engagement: studentState.engagement,
            cognitive_load: studentState.cognitive_load,
            fatigue: studentState.fatigue,
            history: updatedHistory,
            correct_answers: session.correct_answers + (correct ? 1 : 0),
            wrong_answers: session.wrong_answers + (correct ? 0 : 1),
            current_difficulty: recommendation.difficulty ?? session.current_difficulty,
            version: session.version + 1
        }
    });
    if (updated.count === 0) return res.status(409).json({ error: "session changed concurrently; re-read and retry" });

    logger.info({
        session_id, attempt_id: attempt.attempt_id, seq, correct,
        features_extracted: Boolean(extracted), policy: POLICY_NAME
    }, "v1-attempt");

    return res.status(201).json({
        attempt_id: attempt.attempt_id,
        seq,
        correct,
        correct_index,
        explanation: question.explanation,
        state: { ...studentState, knowledge },
        // Estimates, with their uncertainty and the gate's verdict. The UI
        // labels them estimates; a state the gate rejected is absent here and
        // the reason it is absent travels with the response.
        estimate: estimate
            ? { states: gated, standard_error: errors, rung: estimate.rung, gate: estimate.gate }
            : { states: {}, standard_error: {}, rung: null, gate: null, unavailable: true },
        recommendation,
        features_extracted: Boolean(extracted),
        session_version: session.version + 1
    });
};

export const post_session_end = async (req, res) => {
    const { session_id } = req.body ?? {};
    if (!session_id) return res.status(400).json({ error: "session_id is required" });
    const ended = await prisma.session.updateMany({ where: { session_id }, data: { ended_at: new Date() } });
    if (!ended.count) return res.status(404).json({ error: "Session not found" });
    return res.json({ session_id, ended: true });
};
