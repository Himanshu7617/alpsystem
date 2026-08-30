/**
 * question_bank_loader.js
 *
 * Reads question_bank.csv and produces the JSON structure the backend expects.
 *
 * Usage (standalone):
 *   node backend/src/data/questions/question_bank_loader.js
 *
 * Or import { loadQuestionBank } from "./question_bank_loader.js"
 */

import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const CSV_PATH = path.resolve(__dirname, "question_bank.csv");

/**
 * Parse a single CSV line respecting quoted fields that may contain commas.
 */
function parseCSVLine(line) {
    const fields = [];
    let current = "";
    let inQuotes = false;

    for (let i = 0; i < line.length; i++) {
        const ch = line[i];
        if (ch === '"') {
            if (inQuotes && i + 1 < line.length && line[i + 1] === '"') {
                current += '"';
                i++; // skip escaped quote
            } else {
                inQuotes = !inQuotes;
            }
        } else if (ch === "," && !inQuotes) {
            fields.push(current.trim());
            current = "";
        } else {
            current += ch;
        }
    }
    fields.push(current.trim());
    return fields;
}

/**
 * Load question_bank.csv and return { questions: [...] } in the schema the
 * backend expects (matching multitopic_cse.json format).
 */
export function loadQuestionBank(csvPath = CSV_PATH) {
    const raw = fs.readFileSync(csvPath, "utf-8");
    const lines = raw.split(/\r?\n/).filter((l) => l.trim());

    // First line is the header
    const header = parseCSVLine(lines[0]);
    const colIndex = Object.fromEntries(header.map((h, i) => [h, i]));

    const questions = [];

    for (let i = 1; i < lines.length; i++) {
        const cols = parseCSVLine(lines[i]);
        if (cols.length < header.length) continue; // skip malformed rows

        const col = (name) => cols[colIndex[name]] ?? "";

        const id = col("id");
        const difficulty = col("difficulty");
        const difficultyScore = parseInt(col("difficulty_score"), 10) || 1;
        const estimatedTimeSeconds = parseInt(col("estimated_time_seconds"), 10) || 30;
        const topics = col("topics")
            .split(/[;|]/)
            .map((t) => t.trim())
            .filter(Boolean);

        const options = [
            col("option_a"),
            col("option_b"),
            col("option_c"),
            col("option_d"),
        ];

        questions.push({
            id,
            difficulty,
            questionType: "mcq",
            question: col("question"),
            options,
            correctAnswer: col("correct_answer"),
            explanation: col("explanation"),
            estimatedTimeSeconds,
            concepts: topics,
            tags: topics.map((t) => t.toLowerCase()),
            learningObjective: "",
            prerequisiteLevel: difficulty === "easy" ? 1 : difficulty === "medium" ? 2 : 3,
            difficultyScore,
            sourceType: "curated",
        });
    }

    return { questions };
}

// If run directly, write the JSON file
if (process.argv[1] && path.resolve(process.argv[1]) === path.resolve(fileURLToPath(import.meta.url))) {
    const bank = loadQuestionBank();
    const outPath = path.resolve(__dirname, "multitopic_cse.json");
    fs.writeFileSync(outPath, JSON.stringify(bank, null, 2));
    console.log(`✅ Wrote ${bank.questions.length} questions to ${outPath}`);
}
