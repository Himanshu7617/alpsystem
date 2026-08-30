/**
 * Rule Engine for Inferred Variables and Next Question Selection
 */

/**
 * Preprocesses and normalizes the incoming features from request body.
 * Ensures the system does not fail by applying defaults for any missing features.
 * 
 * @param {Object} reqBody - Request body containing features
 * @param {Array} sessionHistory - Array of history objects from the session
 * @param {number} estimatedTime - Estimated time for the current question
 * @returns {Object} Preprocessed and normalized features
 */
export const preprocessFeatures = (reqBody, sessionHistory = [], estimatedTime = 30) => {
    const toFiniteNumber = (value, fallback = 0) => {
        const numeric = Number(value);
        return Number.isFinite(numeric) ? numeric : fallback;
    };

    const toBoolean = (value, fallback = false) => {
        if (typeof value === "boolean") return value;
        if (typeof value === "number") return value !== 0;
        if (typeof value === "string") {
            const normalized = value.trim().toLowerCase();
            if (["true", "1", "yes"].includes(normalized)) return true;
            if (["false", "0", "no", ""].includes(normalized)) return false;
        }
        return fallback;
    };
    const getVal = (keys, defaultVal = 0) => {
        for (const key of keys) {
            // Direct checks
            if (reqBody[key] !== undefined && reqBody[key] !== null) {
                return reqBody[key];
            }
            // Case-insensitive direct checks
            const lowerKey = key.toLowerCase();
            for (const bodyKey in reqBody) {
                if (bodyKey.toLowerCase() === lowerKey && reqBody[bodyKey] !== undefined && reqBody[bodyKey] !== null) {
                    return reqBody[bodyKey];
                }
            }
            // Nested checks under common object roots
            for (const root of ['time_spent', 'timeSpent', 'mouse_behavior', 'mouseBehavior', 'keyboard_behavior', 'keyboardBehavior', 'fatigue_features', 'fatigueFeatures']) {
                const rootObj = reqBody[root];
                if (rootObj && typeof rootObj === 'object') {
                    if (rootObj[key] !== undefined && rootObj[key] !== null) {
                        return rootObj[key];
                    }
                    const lowerKeySub = key.toLowerCase();
                    for (const subKey in rootObj) {
                        if (subKey.toLowerCase() === lowerKeySub && rootObj[subKey] !== undefined && rootObj[subKey] !== null) {
                            return rootObj[subKey];
                        }
                    }
                }
            }
        }
        return defaultVal;
    };

    // 1. Time spent features
    const total_response_time = Math.max(0.1, toFiniteNumber(getVal(['total_response_time', 'totalResponseTime', 'timeTaken', 'time_taken'], estimatedTime), estimatedTime));
    const reading_time = Math.max(0, toFiniteNumber(getVal(['reading_time', 'readingTime'], Math.min(5, total_response_time * 0.2))));
    const time_after_last_interaction = Math.max(0, toFiniteNumber(getVal(['time_after_last_interaction', 'timeAfterLastInteraction'], Math.min(2, total_response_time * 0.1))));

    // 2. Correctness & Basic properties
    let isCorrect = false;
    const getCorrectnessKey = () => {
        const checkKeys = ['correct', 'is_correct', 'iscorrect', 'isCorrect'];
        for (const key of checkKeys) {
            if (reqBody[key] !== undefined && reqBody[key] !== null) {
                return reqBody[key];
            }
            const lowerKey = key.toLowerCase();
            for (const bodyKey in reqBody) {
                if (bodyKey.toLowerCase() === lowerKey && reqBody[bodyKey] !== undefined && reqBody[bodyKey] !== null) {
                    return reqBody[bodyKey];
                }
            }
        }
        return null;
    };

    const correctnessVal = getCorrectnessKey();
    if (correctnessVal !== null) {
        isCorrect = toBoolean(correctnessVal);
    } else if (reqBody.selected_answer !== undefined && reqBody.correctAnswer !== undefined) {
        isCorrect = String(reqBody.selected_answer).trim() === String(reqBody.correctAnswer).trim();
    } else if (reqBody.answer !== undefined && reqBody.correctAnswer !== undefined) {
        isCorrect = String(reqBody.answer).trim() === String(reqBody.correctAnswer).trim();
    }

    const difficulty = String(getVal(['difficulty'], 'easy')).toLowerCase();
    const attempts = Math.max(1, Math.floor(toFiniteNumber(getVal(['attempts', 'numAttempts', 'attemptsCount'], 1), 1)));
    const skip = toBoolean(getVal(['skip', 'skipped', 'skipQuestion'], false));
    const option_changes = Math.max(0, toFiniteNumber(getVal(['option_changes', 'optionChanges'], 0)));

    // 3. Mouse behavior features
    const mouse_distance = Math.max(0, toFiniteNumber(getVal(['mouse_distance', 'mouseDistance'], 0)));
    const mouse_speed = Math.max(0, toFiniteNumber(getVal(['mouse_speed', 'mouseSpeed'], 0)));
    const hover_time = Math.max(0, toFiniteNumber(getVal(['hover_time', 'hoverTime'], 0)));

    // 4. Keyboard behavior features (for typing answers)
    const typing_speed = Math.max(0, toFiniteNumber(getVal(['typing_speed', 'typingSpeed'], 0)));
    const backspaces = Math.max(0, toFiniteNumber(getVal(['backspaces', 'backspacesCount'], 0)));
    const delete_frequency = Math.max(0, toFiniteNumber(getVal(['delete_frequency', 'deleteFrequency'], 0)));
    const pause_duration = Math.max(0, toFiniteNumber(getVal(['pause_duration', 'pauseDuration'], 0)));

    // 5. Fatigue features
    const question_number = Math.max(1, Math.floor(toFiniteNumber(getVal(['question_number', 'questionNumber'], sessionHistory.length + 1), sessionHistory.length + 1)));
    const session_duration = Math.max(0, toFiniteNumber(getVal(['session_duration', 'sessionDuration'], total_response_time), total_response_time));
    const accuracy_decay = Math.max(0, toFiniteNumber(getVal(['accuracy_decay', 'accuracyDecay'], 0)));

    // Extra web logs
    const tab_switches = Math.max(0, toFiniteNumber(getVal(['tab_switches', 'tabSwitches', 'window_blurs', 'windowBlurs'], 0)));

    // Normalize time ratio (actual response time vs estimated time)
    const timeRatio = total_response_time / (estimatedTime || 30);

    return {
        total_response_time,
        reading_time,
        time_after_last_interaction,
        isCorrect,
        difficulty,
        attempts,
        skip,
        option_changes,
        mouse_distance,
        mouse_speed,
        hover_time,
        typing_speed,
        backspaces,
        delete_frequency,
        pause_duration,
        question_number,
        session_duration,
        accuracy_decay,
        tab_switches,
        timeRatio
    };
};

/**
 * Knowledge (Mastery) Rule Engine
 */
export const updateKnowledge = (prevKnowledge, features, history = []) => {
    if (features.skip) {
        // Skipping indicates low confidence/knowledge
        return Math.max(0.0, prevKnowledge - 0.05);
    }

    let delta = 0;
    if (features.isCorrect) {
        if (features.difficulty === "hard") {
            delta = 0.18;
        } else if (features.difficulty === "medium") {
            delta = 0.12;
        } else {
            delta = 0.08; // easy
        }

        // Penalty for multiple attempts
        if (features.attempts > 1) {
            delta = delta / features.attempts;
        }
    } else {
        if (features.difficulty === "easy") {
            delta = -0.15;
        } else if (features.difficulty === "medium") {
            delta = -0.10;
        } else {
            delta = -0.05; // hard incorrect is penalized less
        }
    }

    let knowledge = prevKnowledge + delta;

    // Trend updates: check previous correctness
    if (history.length > 0) {
        const lastQuestion = history[history.length - 1];
        // Previous Correctness: Track learning trend
        if (!lastQuestion.isCorrect && features.isCorrect) {
            // Trend is improving
            knowledge += 0.03;
        } else if (lastQuestion.isCorrect && !features.isCorrect) {
            // Trend is declining
            knowledge -= 0.03;
        }

        // Rolling accuracy: calculate last 3 items
        const rollingSize = Math.min(3, history.length);
        const lastQuestions = history.slice(-rollingSize);
        const prevAcc = lastQuestions.filter(h => h.isCorrect).length / rollingSize;

        const currentAndLast = [...lastQuestions, features];
        const currentAcc = currentAndLast.filter(h => h.isCorrect).length / currentAndLast.length;

        if (currentAcc > prevAcc) {
            knowledge += 0.02; // Rolling accuracy went up
        } else if (currentAcc < prevAcc) {
            knowledge -= 0.02; // Rolling accuracy went down
        }
    }

    return Math.max(0.0, Math.min(1.0, knowledge));
};

/**
 * Confidence Rule Engine
 */
export const updateConfidence = (prevConfidence, features) => {
    if (features.skip) {
        return Math.max(0.0, prevConfidence - 0.15);
    }

    let delta = 0;

    // Correct & low time -> High confidence
    if (features.isCorrect && features.timeRatio < 1.0) {
        delta += 0.10;
    }
    // Wrong & high time -> Low confidence
    if (!features.isCorrect && features.timeRatio > 1.3) {
        delta -= 0.10;
    }
    // Many option changes -> Low confidence
    if (features.option_changes > 3) {
        delta -= 0.08;
    }
    // Few changes + quick submit -> High confidence
    if (features.option_changes <= 1 && features.timeRatio < 0.7) {
        delta += 0.05;
    }
    // Long time before first interaction (reading time) -> Low confidence
    if (features.reading_time > 15) {
        delta -= 0.05;
    }
    // Long time after selecting option -> Low confidence
    if (features.time_after_last_interaction > 10) {
        delta -= 0.05;
    }

    return Math.max(0.0, Math.min(1.0, prevConfidence + delta));
};

/**
 * Engagement Rule Engine
 */
export const updateEngagement = (prevEngagement, features) => {
    if (features.skip) {
        return Math.max(0.0, prevEngagement - 0.10);
    }

    let delta = 0;

    // Tab switch / window blur -> Low
    if (features.tab_switches > 0) {
        delta -= 0.20;
    }

    // Low mouse movement -> Low
    if (features.mouse_distance < 30 && features.total_response_time > 12) {
        delta -= 0.10;
    }

    // Very high random movement -> Low
    if (features.mouse_distance > 4000 && features.mouse_speed > 400) {
        delta -= 0.05;
    }

    // Consistent activity -> High
    if (features.mouse_distance >= 100 && features.mouse_distance <= 3000 && features.mouse_speed > 10 && features.mouse_speed < 300) {
        delta += 0.05;
    }

    // No skips + steady interaction -> High
    if (!features.skip && features.mouse_distance > 50 && features.total_response_time < 90) {
        delta += 0.03;
    }

    // Long idle time -> Low
    if (features.pause_duration > 10 || features.time_after_last_interaction > 8) {
        delta -= 0.05;
    }

    // Blend previous with delta
    return Math.max(0.0, Math.min(1.0, prevEngagement + delta));
};

/**
 * Cognitive Load Rule Engine
 */
export const updateCognitiveLoad = (prevCognitiveLoad, features) => {
    let clInstant = 0.2; // Baseline

    // High time + wrong -> High
    if (features.timeRatio > 1.3 && !features.isCorrect) {
        clInstant = 0.85;
    }
    // High time + correct -> Medium
    else if (features.timeRatio > 1.3 && features.isCorrect) {
        clInstant = 0.55;
    }
    // Normal/Low time + correct -> Low
    else if (features.timeRatio <= 1.0 && features.isCorrect) {
        clInstant = 0.20;
    }
    // Easy question but high time -> High
    if (features.difficulty === "easy" && features.timeRatio > 1.4) {
        clInstant = Math.max(clInstant, 0.75);
    }

    // Many option changes or attempts -> High cognitive load (scrolling & confusion)
    if (features.option_changes > 3 || features.attempts > 2) {
        clInstant = Math.min(1.0, clInstant + 0.10);
    }

    // Keyboard features (backspaces, deletes, pauses in typing indicate high cognitive load)
    if (features.backspaces > 6 || features.delete_frequency > 6) {
        clInstant = Math.min(1.0, clInstant + 0.08);
    }
    if (features.pause_duration > 8) {
        clInstant = Math.min(1.0, clInstant + 0.07);
    }

    // Smooth over time
    return Math.max(0.0, Math.min(1.0, prevCognitiveLoad * 0.4 + clInstant * 0.6));
};

/**
 * Fatigue Rule Engine
 */
export const updateFatigue = (prevFatigue, features, history = []) => {
    // 1. Session duration increases fatigue
    const durationFatigue = Math.min(0.6, features.session_duration / 1800); // 30 mins max baseline

    // 2. Question number increases fatigue
    const questionFatigue = Math.min(0.6, features.question_number / 30); // 30 questions max baseline

    // Base fatigue accumulates from time and questions
    let fatigue = (durationFatigue * 0.5) + (questionFatigue * 0.5);

    // 3. Accuracy Decay (declining accuracy over session)
    let accuracyDecayVal = 0;
    if (history.length >= 4) {
        const half = Math.floor(history.length / 2);
        const firstHalf = history.slice(0, half);
        const secondHalf = history.slice(half);
        
        const accFirst = firstHalf.filter(h => h.isCorrect).length / firstHalf.length;
        const accSecond = secondHalf.filter(h => h.isCorrect).length / secondHalf.length;
        
        if (accFirst > accSecond) {
            accuracyDecayVal = (accFirst - accSecond) * 0.25; // max 0.25
        }
    }

    // 4. Response Time Trend (getting slower on same tasks)
    let rtTrend = 0;
    if (history.length >= 4) {
        const half = Math.floor(history.length / 2);
        const firstHalf = history.slice(0, half);
        const secondHalf = history.slice(half);

        const avgRtFirst = firstHalf.reduce((acc, h) => acc + h.total_response_time, 0) / firstHalf.length;
        const avgRtSecond = secondHalf.reduce((acc, h) => acc + h.total_response_time, 0) / secondHalf.length;

        if (avgRtSecond > avgRtFirst * 1.25) {
            rtTrend = 0.15;
        }
    }

    // 5. Frequent long idle times indicate fatigue
    let longIdleBonus = 0;
    if (features.pause_duration > 12 || features.time_after_last_interaction > 8) {
        longIdleBonus = 0.10;
    }

    return Math.max(0.0, Math.min(1.0, fatigue + accuracyDecayVal + rtTrend + longIdleBonus));
};

/**
 * Next Question Selection Logic
 * 
 * Balances the 5 merits to select next difficulty
 */
const difficultyToLevel = (difficulty) => ({ easy: 1, medium: 2, hard: 3 }[difficulty] ?? 2);
const levelToDifficulty = (level) => ({ 1: "easy", 2: "medium", 3: "hard" }[level]);

export const summarizeRecentPerformance = (history = [], currentFeatures = null) => {
    const entries = [...history, ...(currentFeatures ? [currentFeatures] : [])].slice(-5);
    if (!entries.length) return { observations: 0, accuracy: null, timeRatio: null, accuracyTrend: 0 };
    const accuracy = entries.filter(entry => entry.isCorrect).length / entries.length;
    const timeRatio = entries.reduce((sum, entry) => sum + (entry.timeRatio ?? entry.features?.timeRatio ?? 1), 0) / entries.length;
    const midpoint = Math.ceil(entries.length / 2);
    const early = entries.slice(0, midpoint);
    const late = entries.slice(midpoint);
    const earlyAccuracy = early.filter(entry => entry.isCorrect).length / early.length;
    const lateAccuracy = late.filter(entry => entry.isCorrect).length / late.length;
    return { observations: entries.length, accuracy, timeRatio, accuracyTrend: lateAccuracy - earlyAccuracy };
};

/**
 * Builds an auditable difficulty decision from student state *and* a rolling
 * response history. It limits each transition to one level so a single noisy
 * interaction cannot cause an extreme curriculum change.
 */
export const buildDifficultyDecision = (state, history = [], currentFeatures = {}) => {
    const { knowledge, confidence, engagement, cognitive_load, fatigue } = state;
    const recent = summarizeRecentPerformance(history, currentFeatures);
    const reasons = [];

    let desiredLevel = 1;
    if (knowledge < 0.4) {
        desiredLevel = 1;
    } else if (knowledge >= 0.4 && knowledge < 0.75) {
        desiredLevel = 2;
    } else {
        desiredLevel = 3;
    }

    if (cognitive_load > 0.7) {
        desiredLevel -= 1;
        reasons.push("high_cognitive_load");
    }
    if (fatigue > 0.7) {
        desiredLevel -= 1;
        reasons.push("high_fatigue");
    }
    if (engagement < 0.4) {
        desiredLevel -= 1;
        reasons.push("low_engagement");
    }
    if (recent.observations >= 3 && recent.accuracy <= 0.4) {
        desiredLevel -= 1;
        reasons.push("recent_struggle");
    }
    if (recent.observations >= 3 && recent.timeRatio > 1.25) {
        desiredLevel -= 1;
        reasons.push("slowing_response_time");
    }
    if ((recent.observations >= 2 && recent.accuracy >= 0.66 && confidence > 0.5) || (knowledge >= 0.4 && recent.accuracy >= 0.5)) {
        desiredLevel += 1;
        reasons.push("sustained_mastery");
    }

    desiredLevel = Math.max(1, Math.min(3, desiredLevel));
    const currentLevel = difficultyToLevel(currentFeatures.difficulty);
    const targetLevel = Math.max(currentLevel - 1, Math.min(currentLevel + 1, desiredLevel));
    if (!reasons.length) reasons.push("knowledge_band");

    return {
        difficulty: levelToDifficulty(targetLevel),
        current_difficulty: levelToDifficulty(currentLevel),
        desired_level: desiredLevel,
        recent_performance: recent,
        reasons,
    };
};

export const selectNextDifficulty = (state, history = [], currentFeatures = {}) => {
    return buildDifficultyDecision(state, history, currentFeatures).difficulty;
};

/**
 * Produces an auditable action alongside the legacy difficulty value.
 * This is intentionally deterministic: the rule policy is a benchmark, not
 * an opaque model. Reasons expose only aggregate behavioural signals.
 */
export const recommendLearningAction = (state, history = [], currentFeatures = {}) => {
    const decision = buildDifficultyDecision(state, history, currentFeatures);
    const reasons = [...decision.reasons];
    const difficulty = decision.difficulty;
    let action = "advance";

    if (state.cognitive_load > 0.7) {
        action = "scaffold";
        reasons.push("high_cognitive_load");
    }
    if (state.fatigue > 0.7) {
        action = "offer_break";
        reasons.push("high_fatigue");
    }
    if (state.engagement < 0.4) {
        action = action === "advance" ? "reengage" : action;
        reasons.push("low_engagement");
    }
    if (state.knowledge < 0.4) {
        action = action === "advance" ? "practice_foundation" : action;
        reasons.push("low_knowledge");
    }
    if (action === "advance" && reasons.includes("recent_struggle")) action = "review_prerequisite";
    if (action === "advance" && reasons.includes("slowing_response_time")) action = "reduce_pacing";

    return {
        policy_version: "rule-v1.2",
        action,
        difficulty,
        reasons,
        decision_context: decision.recent_performance,
    };
};

/**
 * Orchestrator to calculate next student state and next difficulty
 */
export const calculateStateAndNextQuestion = (session, currentQuestion, rawFeatures) => {
    // 1. Preprocess and normalize features
    const estimatedTime = currentQuestion?.estimatedTimeSeconds ?? 30;
    const history = session.history ? (typeof session.history === 'string' ? JSON.parse(session.history) : session.history) : [];
    
    const preprocessed = preprocessFeatures(rawFeatures, history, estimatedTime);

    // Override difficulty/correctness details from the DB question for absolute safety if missing
    preprocessed.difficulty = currentQuestion?.difficulty ?? preprocessed.difficulty;
    
    // 2. Run rule engine calculations
    const nextKnowledge = updateKnowledge(session.mastery, preprocessed, history);
    const nextConfidence = updateConfidence(session.confidence, preprocessed);
    const nextEngagement = updateEngagement(session.engagement, preprocessed);
    const nextCognitiveLoad = updateCognitiveLoad(session.cognitive_load, preprocessed);
    const nextFatigue = updateFatigue(session.fatigue, preprocessed, history);

    const studentState = {
        knowledge: nextKnowledge,
        confidence: nextConfidence,
        engagement: nextEngagement,
        cognitive_load: nextCognitiveLoad,
        fatigue: nextFatigue
    };

    // 3. Create the event before making the next decision, so rolling history
    // includes the response that just occurred.
    const historyEntry = {
        question_id: currentQuestion?.id || rawFeatures.question_id,
        topics: currentQuestion?.concepts ?? [],
        difficulty: preprocessed.difficulty,
        isCorrect: preprocessed.isCorrect,
        total_response_time: preprocessed.total_response_time,
        features: preprocessed,
        state_after: studentState,
        timestamp: new Date().toISOString()
    };

    const updatedHistory = [...history, historyEntry];

    // 4. Determine a bounded, history-aware next action and difficulty.
    const recommendation = recommendLearningAction(studentState, history, preprocessed);
    const nextDifficulty = recommendation.difficulty;

    return {
        studentState,
        nextDifficulty,
        recommendation,
        updatedHistory,
        isCorrect: preprocessed.isCorrect,
        normalizedFeatures: preprocessed
    };
};
