import { 
    preprocessFeatures, 
    updateKnowledge, 
    updateConfidence, 
    updateEngagement, 
    updateCognitiveLoad, 
    updateFatigue, 
    selectNextDifficulty, 
    recommendLearningAction,
    buildDifficultyDecision,
    calculateStateAndNextQuestion 
} from "./ruleEngine.js";
import { buildLearningRoadmap } from "./conceptRoadmap.js";
import { chooseNextQuestion } from "./questionPolicy.js";

// Utility for assertions
function assert(condition, message) {
    if (!condition) {
        console.error(`❌ Assertion Failed: ${message}`);
        process.exit(1);
    }
    console.log(`✅ Passed: ${message}`);
}

function assertApprox(val1, val2, message) {
    if (Math.abs(val1 - val2) > 0.0001) {
        console.error(`❌ Assertion Failed: ${message} (Expected ${val2}, got ${val1})`);
        process.exit(1);
    }
    console.log(`✅ Passed: ${message}`);
}

function runTests() {
    console.log("==========================================");
    console.log("RUNNING RULE ENGINE DIAGNOSTIC TESTS");
    console.log("==========================================\n");

    // ----------------------------------------------------
    // Test 1: Preprocessor Robustness & Defaults
    // ----------------------------------------------------
    console.log("--- Test 1: Preprocessor Robustness ---");
    const emptyPayload = {};
    const preprocessed = preprocessFeatures(emptyPayload, [], 30);
    
    assert(preprocessed.total_response_time === 30, "Should fallback to estimated time for total response time");
    assert(preprocessed.isCorrect === false, "Should default to incorrect");
    assert(preprocessed.difficulty === "easy", "Should default to easy difficulty");
    assert(preprocessed.attempts === 1, "Should default to 1 attempt");
    assert(preprocessed.mouse_distance === 0, "Should default to 0 mouse distance");
    assert(preprocessed.tab_switches === 0, "Should default to 0 tab switches");
    
    // Case insensitivity and nesting support
    const mixedPayload = {
        timeSpent: {
            totalResponseTime: 45,
            readingTime: 12
        },
        ISCORRECT: true,
        mouse_behavior: {
            mouseDistance: 250
        }
    };
    const prepMixed = preprocessFeatures(mixedPayload, [], 30);
    assert(prepMixed.total_response_time === 45, "Should extract nested camelCase totalResponseTime");
    assert(prepMixed.reading_time === 12, "Should extract nested camelCase readingTime");
    assert(prepMixed.isCorrect === true, "Should extract case-insensitive ISCORRECT");
    assert(prepMixed.mouse_distance === 250, "Should extract nested mouseDistance");
    assert(preprocessFeatures({ isCorrect: "false" }).isCorrect === false, "Should parse string false safely");
    assert(preprocessFeatures({ isCorrect: "yes", attempts: "2" }).isCorrect === true, "Should parse supported boolean strings");
    console.log("");

    // ----------------------------------------------------
    // Test 2: Knowledge / Mastery Logic
    // ----------------------------------------------------
    console.log("--- Test 2: Knowledge Score Calculations ---");
    let knowledge = 0.1;
    
    // Correct on easy
    knowledge = updateKnowledge(knowledge, { isCorrect: true, difficulty: "easy", attempts: 1, skip: false });
    assert(knowledge > 0.1, `Knowledge should increase on easy correct: ${knowledge}`);
    
    // Correct on hard
    let prev = knowledge;
    knowledge = updateKnowledge(knowledge, { isCorrect: true, difficulty: "hard", attempts: 1, skip: false });
    assertApprox(knowledge - prev, 0.18, `Knowledge should increase by 0.18 on hard correct: ${knowledge}`);
    
    // Attempt penalty
    prev = knowledge;
    knowledge = updateKnowledge(knowledge, { isCorrect: true, difficulty: "hard", attempts: 3, skip: false });
    assertApprox(knowledge - prev, 0.06, `Knowledge increase should be penalized by attempts (0.18 / 3 = 0.06): ${knowledge}`);
    
    // Wrong on easy
    prev = knowledge;
    knowledge = updateKnowledge(knowledge, { isCorrect: false, difficulty: "easy", attempts: 1, skip: false });
    assertApprox(prev - knowledge, 0.15, `Knowledge should decrease by 0.15 on easy wrong: ${knowledge}`);
    
    // Skip penalty
    prev = knowledge;
    knowledge = updateKnowledge(knowledge, { skip: true });
    assertApprox(prev - knowledge, 0.05, `Knowledge should decrease by 0.05 on skip: ${knowledge}`);
    console.log("");

    // ----------------------------------------------------
    // Test 3: Confidence Logic
    // ----------------------------------------------------
    console.log("--- Test 3: Confidence Score Calculations ---");
    let confidence = 0.5;
    
    // Correct + low time
    confidence = updateConfidence(confidence, { isCorrect: true, timeRatio: 0.5, skip: false });
    assert(confidence > 0.5, `Confidence increases for correct + low time: ${confidence}`);
    
    // Many changes
    prev = confidence;
    confidence = updateConfidence(confidence, { option_changes: 4, skip: false });
    assert(confidence < prev, `Confidence decreases for many option changes: ${confidence}`);
    
    // Skip
    confidence = updateConfidence(confidence, { skip: true });
    assertApprox(confidence, 0.37, `Confidence should drop significantly on skip: ${confidence}`);
    console.log("");

    // ----------------------------------------------------
    // Test 4: Engagement Logic
    // ----------------------------------------------------
    console.log("--- Test 4: Engagement Score Calculations ---");
    let engagement = 0.8;
    
    // Tab switches drop engagement
    engagement = updateEngagement(engagement, { tab_switches: 1, skip: false });
    assertApprox(engagement, 0.6, `Engagement drops on tab switch: ${engagement}`);
    
    // Low mouse movement on long response
    engagement = updateEngagement(engagement, { mouse_distance: 10, total_response_time: 15, skip: false });
    assertApprox(engagement, 0.5, `Engagement drops on low movement + long response: ${engagement}`);
    
    // Steady interaction increases engagement
    engagement = updateEngagement(engagement, { mouse_distance: 500, mouse_speed: 50, skip: false });
    assertApprox(engagement, 0.55, `Engagement increases on steady interaction: ${engagement}`);
    console.log("");

    // ----------------------------------------------------
    // Test 5: Cognitive Load Logic
    // ----------------------------------------------------
    console.log("--- Test 5: Cognitive Load Calculations ---");
    let cogLoad = 0.2;
    
    // High time + wrong
    cogLoad = updateCognitiveLoad(cogLoad, { timeRatio: 1.5, isCorrect: false });
    assert(cogLoad > 0.5, `Cognitive load increases for high time + wrong: ${cogLoad}`);
    
    // Easy question + high time
    cogLoad = updateCognitiveLoad(0.2, { difficulty: "easy", timeRatio: 1.6, isCorrect: true });
    assert(cogLoad >= 0.5, `Cognitive load increases for easy question taken slowly: ${cogLoad}`);
    console.log("");

    // ----------------------------------------------------
    // Test 6: Fatigue Logic
    // ----------------------------------------------------
    console.log("--- Test 6: Fatigue Calculations ---");
    let fatigue = updateFatigue(0.0, { session_duration: 600, question_number: 10 });
    assert(fatigue > 0.0, `Fatigue accumulates over duration & question number: ${fatigue}`);
    
    // Accuracy decay fatigue boost
    const historyWithDecay = [
        { isCorrect: true, total_response_time: 15 },
        { isCorrect: true, total_response_time: 15 },
        { isCorrect: false, total_response_time: 15 },
        { isCorrect: false, total_response_time: 15 }
    ];
    let decayFatigue = updateFatigue(0.0, { session_duration: 600, question_number: 10 }, historyWithDecay);
    assert(decayFatigue > fatigue, `Fatigue increases on accuracy decay: ${decayFatigue}`);
    console.log("");

    // ----------------------------------------------------
    // Test 7: Next Difficulty Selection Logic
    // ----------------------------------------------------
    console.log("--- Test 7: Next Difficulty Logic ---");
    
    // Base knowledge low
    assert(selectNextDifficulty({ knowledge: 0.2, confidence: 0.5, engagement: 0.8, cognitive_load: 0.2, fatigue: 0.1 }) === "easy", "Should select easy for low knowledge");
    
    // Base knowledge medium
    assert(selectNextDifficulty({ knowledge: 0.5, confidence: 0.5, engagement: 0.8, cognitive_load: 0.2, fatigue: 0.1 }) === "medium", "Should select medium for medium knowledge");
    
    // Base knowledge hard
    assert(selectNextDifficulty({ knowledge: 0.8, confidence: 0.5, engagement: 0.8, cognitive_load: 0.2, fatigue: 0.1 }) === "hard", "Should select hard for high knowledge");
    
    // High Cognitive Load reduction
    assert(selectNextDifficulty({ knowledge: 0.8, confidence: 0.5, engagement: 0.8, cognitive_load: 0.8, fatigue: 0.1 }) === "medium", "Should reduce difficulty if cognitive load is high");
    
    // High Fatigue reduction
    assert(selectNextDifficulty({ knowledge: 0.8, confidence: 0.5, engagement: 0.8, cognitive_load: 0.2, fatigue: 0.8 }) === "medium", "Should reduce difficulty if fatigue is high");
    
    // A single high-confidence state does not cause an unsupported jump.
    assert(selectNextDifficulty({ knowledge: 0.72, confidence: 0.8, engagement: 0.8, cognitive_load: 0.2, fatigue: 0.1 }) === "medium", "Should require sustained evidence before increasing difficulty");
    const action = recommendLearningAction({ knowledge: 0.8, confidence: 0.8, engagement: 0.8, cognitive_load: 0.2, fatigue: 0.1 });
    assert(action.action === "advance" && action.difficulty === "hard", "Should return an explainable challenge recommendation");
    const contextualDecision = buildDifficultyDecision(
        { knowledge: 0.7, confidence: 0.8, engagement: 0.8, cognitive_load: 0.2, fatigue: 0.1 },
        [{ isCorrect: true, timeRatio: 0.8 }, { isCorrect: true, timeRatio: 0.9 }],
        { isCorrect: true, timeRatio: 0.8, difficulty: "medium" }
    );
    assert(contextualDecision.difficulty === "hard", "Should advance after sustained accurate, efficient performance");
    console.log("");

    // ----------------------------------------------------
    // Test 8: End-to-end Session Orchestration
    // ----------------------------------------------------
    console.log("--- Test 8: End-to-end Session Calculation ---");
    const mockSession = {
        mastery: 0.4,
        confidence: 0.5,
        engagement: 0.8,
        cognitive_load: 0.3,
        fatigue: 0.1,
        history: []
    };
    const mockQuestion = {
        id: "q_test_1",
        difficulty: "medium",
        estimatedTimeSeconds: 40
    };
    const mockFeatures = {
        question_id: "q_test_1",
        correct: true,
        time_taken: 20,
        mouse_distance: 120,
        mouse_speed: 15
    };

    const result = calculateStateAndNextQuestion(mockSession, mockQuestion, mockFeatures);
    assert(result.isCorrect === true, "Result isCorrect should be true");
    assert(result.studentState.knowledge > 0.4, "Knowledge should increase");
    assert(result.studentState.confidence > 0.5, "Confidence should increase");
    assert(result.updatedHistory.length === 1, "History should contain the new entry");
    assert(result.nextDifficulty === "medium" || result.nextDifficulty === "hard", `Should output valid difficulty: ${result.nextDifficulty}`);
    assert(result.recommendation.policy_version === "rule-v1.2", "Should include policy provenance");

    const roadmap = buildLearningRoadmap([
        { id: "q1", concepts: ["Foundations"] },
        { id: "q2", concepts: ["Applications"] }
    ], [{ question_id: "q1", isCorrect: true }], result.studentState);
    assert(roadmap.target_concept === "Applications", "Should recommend the weakest eligible concept");
    assert(roadmap.roadmap_version === "concept-roadmap-v1", "Should include roadmap provenance");

    const selection = chooseNextQuestion([
        { id: "tree-review", difficulty: "medium", concepts: ["Binary Trees"] },
        { id: "recursion-bridge", difficulty: "medium", concepts: ["Recursion", "Binary Trees"] }
    ], "medium", [], [{ topics: ["Binary Trees"], isCorrect: false }], result.studentState, { target_concept: "Recursion" });
    assert(selection.question.id === "recursion-bridge", "Should prioritize a weak target concept and its co-teaching topic");
    assert(selection.topicWeights.recursion > 0, "Should expose dynamic topic weights");

    console.log("==========================================");
    console.log("ALL TESTS COMPLETED SUCCESSFULLY!");
    console.log("==========================================");
}

runTests();
