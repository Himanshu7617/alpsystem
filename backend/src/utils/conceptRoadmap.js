import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Load concept graph
const graphData = JSON.parse(fs.readFileSync(path.resolve(__dirname, "../data/concept_graph.json"), "utf8"));
const { concepts: conceptGraph, question_to_concept } = graphData;

export const getConceptForQuestion = (questionId) => {
    return question_to_concept[questionId] || null;
};

export const buildLearningRoadmap = (questions, history = [], studentState = {}) => {
    const conceptAttempts = {};
    const conceptCorrect = {};

    // Compute stats from history based on question_to_concept or question concept tags
    const questionMap = new Map((questions || []).map(q => [q.id, q]));

    for (const entry of history) {
        // Find concept for question
        let conceptId = getConceptForQuestion(entry.question_id);

        if (!conceptId && questionMap.has(entry.question_id)) {
            const q = questionMap.get(entry.question_id);
            if (q.concepts && q.concepts.length > 0) {
                // Find matching concept in graph by ID or name, or match first concept
                for (const cTag of q.concepts) {
                    if (conceptGraph[cTag]) {
                        conceptId = cTag;
                        break;
                    }
                    const matchKey = Object.keys(conceptGraph).find(
                        k => conceptGraph[k].name.toLowerCase() === cTag.toLowerCase() ||
                             k.toLowerCase() === cTag.toLowerCase()
                    );
                    if (matchKey) {
                        conceptId = matchKey;
                        break;
                    }
                }
                // Fallback: if concept is a custom name not in graph (e.g. unit tests), use raw tag
                if (!conceptId) conceptId = q.concepts[0];
            }
        }

        if (!conceptId) continue;

        if (!conceptAttempts[conceptId]) {
            conceptAttempts[conceptId] = 0;
            conceptCorrect[conceptId] = 0;
        }

        conceptAttempts[conceptId]++;
        if (entry.isCorrect) {
            conceptCorrect[conceptId]++;
        }
    }

    const conceptMastery = studentState.concept_mastery || {};
    if (!studentState.concept_mastery) {
        for (const conceptId of Object.keys(conceptGraph)) {
            if (conceptAttempts[conceptId] > 0) {
                conceptMastery[conceptId] = conceptCorrect[conceptId] / conceptAttempts[conceptId];
            } else {
                conceptMastery[conceptId] = 0;
            }
        }
    }

    const conceptsList = [];
    const progress = { mastered: 0, in_progress: 0, eligible: 0, locked: 0, total: Object.keys(conceptGraph).length };
    const subjects = {};

    const conceptStatus = {};
    const sortedConceptIds = Object.keys(conceptGraph).sort((a, b) => conceptGraph[a].level - conceptGraph[b].level);

    for (const conceptId of sortedConceptIds) {
        const concept = conceptGraph[conceptId];
        const mastery = conceptMastery[conceptId] || 0;
        const attempts = studentState.concept_mastery ? (mastery > 0 || conceptMastery.hasOwnProperty(conceptId) ? 1 : 0) : (conceptAttempts[conceptId] || 0);
        
        let status = "locked";
        
        if (attempts > 0 || (studentState.concept_mastery && conceptMastery[conceptId] !== undefined)) {
            if (mastery >= 0.7) {
                status = "mastered";
                progress.mastered++;
            } else {
                status = "in_progress";
                progress.in_progress++;
            }
        } else {
            // Check prerequisites
            const prereqsMet = concept.prerequisites.every(prereqId => {
                return conceptStatus[prereqId] === "mastered";
            });
            
            if (prereqsMet) {
                status = "eligible";
                progress.eligible++;
            } else {
                status = "locked";
                progress.locked++;
            }
        }
        
        conceptStatus[conceptId] = status;
        
        if (!subjects[concept.subject]) {
            subjects[concept.subject] = { mastered: 0, total: 0, progress: 0 };
        }
        subjects[concept.subject].total++;
        if (status === "mastered") {
            subjects[concept.subject].mastered++;
        }
        
        conceptsList.push({
            concept_id: conceptId,
            name: concept.name,
            subject: concept.subject,
            level: concept.level,
            status: status,
            mastery: mastery,
            prerequisites: concept.prerequisites
        });
    }

    for (const subject in subjects) {
        subjects[subject].progress = subjects[subject].total > 0 ? subjects[subject].mastered / subjects[subject].total : 0;
    }

    let target = null;

    // Check if passed questions map to conceptGraph
    const anyMapped = (questions || []).some(q => getConceptForQuestion(q.id) || (q.concepts && q.concepts.some(c => conceptGraph[c])));

    if (anyMapped || !questions || questions.length === 0) {
        const candidates = conceptsList.filter(c => c.status === "in_progress" || c.status === "eligible");
        if (candidates.length > 0) {
            // Find the lowest-mastery eligible concept. 
            candidates.sort((a, b) => {
                if (a.mastery !== b.mastery) return a.mastery - b.mastery;
                return a.level - b.level;
            });
            target = candidates[0];
        }
    } else {
        // Fallback for custom question banks / unit tests without conceptGraph mapping
        const conceptStats = new Map();
        const attempted = new Map();
        for (const entry of history) {
            const existing = attempted.get(entry.question_id) ?? [];
            existing.push(entry.isCorrect ? 1 : 0);
            attempted.set(entry.question_id, existing);
        }
        for (const question of questions) {
            for (const concept of question.concepts ?? []) {
                const results = attempted.get(question.id) ?? [];
                const current = conceptStats.get(concept) ?? { concept_id: concept, name: concept, attempts: 0, correct: 0 };
                current.attempts += results.length;
                current.correct += results.reduce((total, result) => total + result, 0);
                conceptStats.set(concept, current);
            }
        }
        const fallbackConcepts = [...conceptStats.values()].map(stat => ({
            ...stat,
            mastery: stat.attempts ? stat.correct / stat.attempts : 0,
        }));
        fallbackConcepts.sort((a, b) => a.mastery - b.mastery);
        if (fallbackConcepts.length > 0) {
            target = fallbackConcepts[0];
        }
    }

    let action = "practice_target_concept";
    if (studentState && studentState.fatigue > 0.7) action = "offer_break";
    else if (studentState && studentState.cognitive_load > 0.7) action = "scaffold";

    return {
        roadmap_version: "concept-roadmap-v1",
        target_concept: target ? (target.concept_id || target.concept) : null,
        target_concept_name: target ? (target.name || target.concept) : null,
        target_estimated_mastery: target ? target.mastery : null,
        action: action,
        progress: progress,
        concepts: conceptsList,
        subjects: subjects
    };
};
