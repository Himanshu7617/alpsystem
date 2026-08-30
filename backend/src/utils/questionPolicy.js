const TOPIC_ALIASES = {
    "Binary Tree": "binary_trees",
    "Binary Trees": "binary_trees",
    "Recursion": "recursion",
    "Algorithm": "algorithms",
    "Algorithms": "algorithms",
    "Data Structures": "data_structures",
    "Database": "dbms",
    "DBMS": "dbms",
    "Operating Systems": "operating_systems",
    "Computer Networks": "networks",
    "OOP": "object_oriented_programming",
};

const CO_TEACHING = {
    recursion: ["binary_trees", "algorithms"],
    binary_trees: ["recursion", "data_structures", "algorithms"],
    data_structures: ["algorithms", "binary_trees"],
    algorithms: ["data_structures", "recursion", "networks"],
    dbms: ["object_oriented_programming", "operating_systems"],
    operating_systems: ["networks", "dbms"],
    networks: ["operating_systems", "algorithms"],
    object_oriented_programming: ["dbms", "data_structures"],
};

export const canonicalTopic = (value) => TOPIC_ALIASES[value] ?? String(value ?? "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");

export const questionTopics = (question) => [...new Set((question.concepts ?? []).map(canonicalTopic).filter(Boolean))];

export const calculateTopicWeights = (questions, history = [], studentState = {}, roadmap = {}) => {
    const weights = new Map();
    const recentTopics = history.slice(-3).flatMap(entry => (entry.topics ?? []).map(canonicalTopic));
    for (const question of questions) {
        for (const topic of questionTopics(question)) weights.set(topic, weights.get(topic) ?? 1);
    }
    for (const [topic, base] of weights) {
        const topicAttempts = history.filter(entry => (entry.topics ?? []).map(canonicalTopic).includes(topic));
        const accuracy = topicAttempts.length ? topicAttempts.filter(entry => entry.isCorrect).length / topicAttempts.length : 0;
        let weight = base + (1 - accuracy) * 2;
        if (topic === canonicalTopic(roadmap.target_concept)) weight += 2;
        if (recentTopics.includes(topic)) weight -= 0.6; // avoid repetitive topic blocks
        if (studentState.cognitive_load > .7 || studentState.fatigue > .7) weight *= .8;
        weights.set(topic, Math.max(.1, weight));
    }
    return Object.fromEntries(weights);
};

export const chooseNextQuestion = (questions, desiredDifficulty, askedQuestions, history, studentState, roadmap) => {
    const unasked = questions.filter(question => !askedQuestions.includes(question.id));
    const candidates = unasked.length ? unasked : questions;
    const topicWeights = calculateTopicWeights(questions, history, studentState, roadmap);
    const difficultyOrder = ["easy", "medium", "hard"];
    const desiredIndex = difficultyOrder.indexOf(desiredDifficulty);
    const scored = candidates.map(question => {
        const topics = questionTopics(question);
        const topicScore = topics.reduce((sum, topic) => sum + (topicWeights[topic] ?? 1), 0) / Math.max(1, topics.length);
        const difficultyPenalty = Math.abs(difficultyOrder.indexOf(question.difficulty) - desiredIndex);
        const coTeachingBonus = topics.some(topic => CO_TEACHING[canonicalTopic(roadmap.target_concept)]?.includes(topic)) ? .5 : 0;
        const alreadyAskedPenalty = askedQuestions.includes(question.id) ? 10.0 : 0;
        return { question, score: topicScore + coTeachingBonus - difficultyPenalty * 1.5 - alreadyAskedPenalty };
    });
    scored.sort((left, right) => right.score - left.score || left.question.id.localeCompare(right.question.id));
    // `scored` is returned so a stochastic policy can compute a propensity over
    // the same ranking the deterministic caller uses. One scoring definition.
    return { question: scored[0]?.question, topicWeights, scored };
};
