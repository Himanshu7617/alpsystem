/**
 * Seeds the static, version-controlled question bank into the database.
 *
 * The current schema attaches every Question to a Session, so the bank lives
 * under one canonical seed session rather than in a standalone table. The
 * schema gains a first-class item bank in Phase 2; until then this keeps the
 * question table populated and the seed idempotent.
 *
 * ponytail: one seed session as the bank holder. Replace with a real Item
 * table when Phase 2 extends the schema.
 */
import prisma from "../src/config/db.js";
import { get_question_by_topic } from "../src/services/questionBank.service.js";

const SEED_SESSION_ID = "seed-question-bank";
const SEED_TOPIC = "multi-topic cse";

const main = async () => {
    const bank = await get_question_by_topic(SEED_TOPIC);
    if (!bank) throw new Error(`No static bank for topic: ${SEED_TOPIC}`);

    await prisma.session.upsert({
        where: { session_id: SEED_SESSION_ID },
        update: {},
        create: {
            session_id: SEED_SESSION_ID,
            topic: SEED_TOPIC,
            asked_questions: [],
            history: [],
            source: "seed"
        }
    });

    await prisma.question.deleteMany({ where: { session_id: SEED_SESSION_ID } });

    const { count } = await prisma.question.createMany({
        data: bank.questions.map(q => ({
            session_id: SEED_SESSION_ID,
            id: q.id,
            difficulty: q.difficulty,
            questionType: q.questionType ?? "mcq",
            question: q.question,
            options: q.options,
            correctAnswer: q.correctAnswer,
            explanation: q.explanation ?? "",
            estimatedTimeSeconds: q.estimatedTimeSeconds ?? 30,
            concepts: q.concepts ?? [],
            tags: q.tags ?? [],
            learningObjective: q.learningObjective ?? "",
            prerequisiteLevel: q.prerequisiteLevel ?? 1,
            difficultyScore: q.difficultyScore ?? 1,
            sourceType: q.sourceType ?? "curated",
            asked: false
        }))
    });

    console.log(`seed: wrote ${count} questions to session ${SEED_SESSION_ID}`);
};

main()
    .catch((error) => {
        console.error("seed failed:", error);
        process.exit(1);
    })
    .finally(() => prisma.$disconnect());
