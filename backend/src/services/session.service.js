import { v4 as uuidv4 } from "uuid";
import prisma from "../config/db.js";
import { get_question_by_topic } from "./questionBank.service.js";

export const MASTERY_THRESHOLD = 0.9;

/**
 * Creates a session and materialises the topic's question bank into it.
 * Shared by the legacy `/question` path and the `/v1` research path so the two
 * cannot drift apart on how a session starts.
 */
export const create_session_with_bank = async (topic, extra = {}) => {
    const bank = await get_question_by_topic(topic);
    if (!bank) return null;

    const questions = bank.questions ?? [];
    return prisma.session.create({
        data: {
            session_id: uuidv4(),
            topic,
            asked_questions: [],
            mastery: 0.2,
            confidence: 0.5,
            engagement: 0.8,
            cognitive_load: 0.2,
            fatigue: 0.0,
            history: [],
            ...extra,
            questions: {
                create: questions.map(question => ({
                    id: question.id,
                    difficulty: question.difficulty,
                    questionType: question.questionType ?? "mcq",
                    question: question.question,
                    options: question.options,
                    correctAnswer: question.correctAnswer,
                    explanation: question.explanation ?? "",
                    estimatedTimeSeconds: question.estimatedTimeSeconds ?? 30,
                    concepts: question.concepts ?? [],
                    tags: question.tags ?? [],
                    learningObjective: question.learningObjective ?? "",
                    prerequisiteLevel: question.prerequisiteLevel ?? 1,
                    difficultyScore: question.difficultyScore ?? 1,
                    sourceType: question.sourceType ?? "generated",
                    asked: false
                }))
            }
        }
    });
};
