import fs from "fs/promises";

/**
 * @deprecated Use calculateStateAndNextQuestion from ./ruleEngine.js instead.
 */
export const get_mastery = async (features, filePath) => {
    console.warn("get_mastery is deprecated. Use calculateStateAndNextQuestion instead.");
    try {
        const raw = await fs.readFile(filePath, "utf-8");
        const file_content = JSON.parse(raw);
        const curr_question = file_content.questions.find(
            question => question.id === features.question_id
        );
        if (!curr_question) return -0.08;
        const submittedAnswer = features.selected_answer ?? features.answer;
        if (submittedAnswer === curr_question.correctAnswer) {
            return 0.1;
        }
        return -0.08;
    } catch (e) {
        return -0.08;
    }
};

/**
 * @deprecated Use selectNextDifficulty from ./ruleEngine.js instead.
 */
export const predict_next_difficulty = (mastery) => {
    console.warn("predict_next_difficulty is deprecated. Use selectNextDifficulty instead.");
    if (mastery < 0.5) {
        return "easy";
    } else if (mastery >= 0.5 && mastery <= 1) {
        return "medium";
    }
    return "hard";
};
