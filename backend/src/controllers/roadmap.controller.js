import prisma from "../config/db.js";
import { buildLearningRoadmap, getConceptForQuestion } from "../utils/conceptRoadmap.js";

export const getRoadmap = async (req, res) => {
    try {
        const { username } = req.query;
        if (!username) {
            return res.status(400).json({ error: "username is required" });
        }

        let learner = await prisma.learner.findUnique({ where: { username } });
        if (!learner) {
            learner = await prisma.learner.create({
                data: {
                    username,
                    concept_mastery: {},
                    roadmap_state: {}
                }
            });
        }

        // Generate roadmap using current mastery
        const roadmap = buildLearningRoadmap([], [], { concept_mastery: learner.concept_mastery });

        // Update the learner's roadmap_state if it's new
        await prisma.learner.update({
            where: { username },
            data: { roadmap_state: roadmap }
        });

        res.status(200).json(roadmap);
    } catch (error) {
        console.error("Error fetching roadmap:", error);
        res.status(500).json({ error: "Internal server error" });
    }
};

export const updateRoadmap = async (req, res) => {
    try {
        const { username, question_id, is_correct } = req.body;
        if (!username || !question_id || is_correct === undefined) {
            return res.status(400).json({ error: "username, question_id, and is_correct are required" });
        }

        let learner = await prisma.learner.findUnique({ where: { username } });
        if (!learner) {
            learner = await prisma.learner.create({
                data: {
                    username,
                    concept_mastery: {},
                    roadmap_state: {}
                }
            });
        }

        // Figure out which concept this question maps to
        const conceptId = getConceptForQuestion(question_id);
        
        let newMastery = { ...learner.concept_mastery };
        if (conceptId) {
            // Update mastery logic. We'll use a simple EMA (Exponential Moving Average)
            const currentMastery = newMastery[conceptId] || 0.0;
            const target = is_correct ? 1.0 : 0.0;
            const alpha = 0.3; // learning rate
            newMastery[conceptId] = currentMastery + alpha * (target - currentMastery);
        }

        const roadmap = buildLearningRoadmap([], [], { concept_mastery: newMastery });

        const updatedLearner = await prisma.learner.update({
            where: { username },
            data: {
                concept_mastery: newMastery,
                roadmap_state: roadmap,
                total_questions: learner.total_questions + 1,
                total_correct: learner.total_correct + (is_correct ? 1 : 0)
            }
        });

        res.status(200).json(roadmap);
    } catch (error) {
        console.error("Error updating roadmap:", error);
        res.status(500).json({ error: "Internal server error" });
    }
};
