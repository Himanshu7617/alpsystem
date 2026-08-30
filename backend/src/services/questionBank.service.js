import fs from "fs/promises";
import path from "path";
import { fileURLToPath } from "url";

const BANK_DIR = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "../data/questions");

// Topic strings the frontend sends, mapped onto the version-controlled bank
// files. Anything not listed here is an unknown topic, not a generation request.
const TOPIC_FILES = {
    // The canonical Phase 1 bank, generated from data/items/item-bank-v1.json by
    // `make bank`. It is the only bank whose items carry a concept id, a
    // difficulty bin and a calibrated b, which is what the served policy's
    // action space is defined over — so it is what a /v1 session gets.
    "item-bank-v1": "item-bank-v1.json",
    "multi-topic cse": "multitopic_cse.json",
    "multi-topic cse curriculum": "multitopic_cse.json",
    "cse curriculum": "multitopic_cse.json",
    "recursion": "recursion.json",
    "binary trees": "binary_trees.json",
    "binary_trees": "binary_trees.json",
    "transportability among animals": "transportability_among_animals.json",
};

export const listTopics = () => Object.keys(TOPIC_FILES);

/**
 * Load a static question bank by topic.
 * Returns { questions: [...] }, or null when the topic has no bank file.
 */
export const get_question_by_topic = async (topic) => {
    const file = TOPIC_FILES[String(topic ?? "").trim().toLowerCase()];
    if (!file) return null;
    return JSON.parse(await fs.readFile(path.join(BANK_DIR, file), "utf-8"));
};


/**
 * The canonical bank, kept in memory.
 *
 * The Question rows a session materialises cannot carry every field the policy
 * needs — `concept_id` and the calibrated `b` have no column — and adding two
 * columns to store what a versioned file already says would be a migration for
 * nothing. So the file is the lookup, and the rows stay as they are.
 */
let canonical = null;

export const canonicalBank = async () => {
    if (canonical === null) {
        const bank = JSON.parse(await fs.readFile(path.join(BANK_DIR, TOPIC_FILES["item-bank-v1"]), "utf-8"));
        canonical = {
            version: bank.bank_version,
            curriculum: bank.curriculum,
            items: new Map(bank.questions.map(question => [question.id, question]))
        };
    }
    return canonical;
};
