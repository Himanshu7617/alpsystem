import { mkdir, writeFile } from "node:fs/promises";
import path from "node:path";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const output = path.join(root, "backend/src/data/questions/multitopic_cse.json");
const templates = [
  { topics: ["Algorithms", "Data Structures"], concept: "Graph traversal with queue discipline", prompt: "When exploring the shortest number of edges from a source in an unweighted graph, which data structure should the algorithm use?", options: ["A stack", "A queue", "A hash function", "A database index"], answer: "A queue" },
  { topics: ["Recursion", "Binary Trees"], concept: "Recursive tree traversal", prompt: "Which recursive traversal visits a binary-search-tree's keys in ascending order?", options: ["Preorder", "Inorder", "Postorder", "Level order"], answer: "Inorder" },
  { topics: ["Algorithms", "Binary Trees"], concept: "Balanced tree search complexity", prompt: "Why does a balanced binary search tree support logarithmic search?", options: ["It stores all keys twice", "Its height remains logarithmic", "It never uses comparisons", "Every node has one child"], answer: "Its height remains logarithmic" },
  { topics: ["Operating Systems", "Computer Networks"], concept: "Blocking I/O and scheduling", prompt: "When a process waits for a network read that has not completed, which scheduler state is most appropriate?", options: ["Running", "Ready", "Blocked", "Terminated"], answer: "Blocked" },
  { topics: ["DBMS", "Object Oriented Programming"], concept: "Transactional persistence boundary", prompt: "Which database property should an ORM-backed money transfer preserve so it is all-or-nothing?", options: ["Atomicity", "Inheritance", "Overloading", "Caching"], answer: "Atomicity" },
  { topics: ["Operating Systems", "DBMS"], concept: "Concurrency control and isolation", prompt: "Which principle prevents one concurrent transaction from observing another transaction's partial updates?", options: ["Isolation", "Paging", "Encapsulation", "Recursion"], answer: "Isolation" },
  { topics: ["Computer Networks", "Algorithms"], concept: "Shortest-path routing", prompt: "Which algorithmic property is required by Dijkstra-style shortest-path routing?", options: ["Non-negative edge weights", "A binary tree only", "Recursive base cases", "SQL joins"], answer: "Non-negative edge weights" },
  { topics: ["Data Structures", "Object Oriented Programming"], concept: "Encapsulated stack abstraction", prompt: "Which design keeps a stack's internal array hidden while exposing push and pop operations?", options: ["Encapsulation", "Denormalization", "Packet routing", "Deadlock"], answer: "Encapsulation" },
  { topics: ["Recursion", "Algorithms"], concept: "Divide-and-conquer recurrence", prompt: "Which recurrence models a divide-and-conquer algorithm that solves two half-sized problems and combines in linear time?", options: ["T(n)=T(n-1)+O(1)", "T(n)=2T(n/2)+O(n)", "T(n)=O(1)", "T(n)=nT(n/2)"], answer: "T(n)=2T(n/2)+O(n)" },
  { topics: ["Binary Trees", "Data Structures"], concept: "Tree representation", prompt: "Which property distinguishes a binary tree node from a general rooted-tree node?", options: ["At most two children", "Always sorted values", "No references", "Exactly one parent pointer"], answer: "At most two children" }
];
const contexts = ["a compiler course", "a systems design exercise", "an online assessment", "a debugging session", "a distributed service", "a database-backed application", "a data-processing pipeline", "a university lab", "a performance review", "a reliability test", "a code review", "a technical interview", "a cloud deployment", "a mobile application", "a search service", "a classroom lab", "a graph analytics job", "a transaction-processing system", "a real-time dashboard", "a file-sharing service", "a multiplayer game", "a monitoring platform", "a build pipeline", "a robotics controller", "an embedded device", "a scheduling system", "a logistics application", "a payment platform", "a social-network service", "a version-control tool", "a machine-learning feature store", "an operating-systems practical", "a networking lab", "a database migration", "a security review", "a reliability exercise", "a distributed cache", "a university examination", "an API gateway", "a command-line tool"];
const levels = [
  { difficulty: "easy", score: 2, time: 30, suffix: "Identify the core principle." },
  { difficulty: "medium", score: 5, time: 50, suffix: "Apply the principle to the stated situation." },
  { difficulty: "hard", score: 8, time: 75, suffix: "Justify the choice while considering its cross-topic implication." }
];
const questions = [];
for (const [templateIndex, template] of templates.entries()) {
  for (const [contextIndex, context] of contexts.entries()) {
    for (const level of levels) {
      questions.push({
        id: `mt-${templateIndex + 1}-${contextIndex + 1}-${level.difficulty}`,
        difficulty: level.difficulty,
        questionType: "mcq",
        question: `In ${context}, ${template.prompt} ${level.suffix}`,
        options: template.options,
        correctAnswer: template.answer,
        explanation: `${template.answer} is correct because it applies ${template.concept.toLowerCase()} across ${template.topics.join(" and ")}.`,
        estimatedTimeSeconds: level.time,
        concepts: template.topics,
        topicWeights: Object.fromEntries(template.topics.map((topic, index) => [topic, index === 0 ? 0.6 : 0.4])),
        tags: ["multi-topic", "synthetic-template", ...template.topics.map(topic => topic.toLowerCase().replaceAll(" ", "-"))],
        learningObjective: `Apply ${template.concept.toLowerCase()} across ${template.topics.join(" and ")}.`,
        prerequisiteLevel: Math.ceil(level.score / 2), difficultyScore: level.score, sourceType: "synthetic-template"
      });
    }
  }
}
await mkdir(path.dirname(output), { recursive: true });
await writeFile(output, JSON.stringify({ topic: "Multi-topic CSE curriculum", datasetVersion: "multitopic-cse-v1", generatedBy: "scripts/generate-multitopic-questions.mjs", questions }, null, 2) + "\n");
console.log(`Generated ${questions.length} questions at ${output}`);
