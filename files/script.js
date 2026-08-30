// Configuration
const apiOverride = new URLSearchParams(window.location.search).get('api');
const API_URLS = [...new Set([
    apiOverride,
    window.location.origin,
    'http://localhost:4000'
].filter(Boolean).map(url => url.replace(/\/$/, '')))];

async function apiFetch(path, options) {
    let lastError;

    for (const apiUrl of API_URLS) {
        try {
            const response = await fetch(`${apiUrl}${path}`, options);

            if (response.ok) {
                return response;
            }

            lastError = new Error(`Request failed with status ${response.status}`);
        } catch (error) {
            lastError = error;
        }
    }

    throw lastError || new Error('Request failed');
}

// Subject metadata for the frontend subject selector and landing page.
// The actual question bank (126 curated questions) lives on the backend
// at backend/src/data/questions/question_bank.csv — edit that file to
// add, remove, or modify questions.
const cseSubjects = [
    { name: 'Data Structures', count: 21, description: 'Arrays, linked lists, trees, heaps, hash tables, graphs, and advanced structures.' },
    { name: 'Algorithms', count: 21, description: 'Sorting, searching, graph algorithms, dynamic programming, and complexity analysis.' },
    { name: 'DBMS', count: 21, description: 'SQL, normalization, transactions, indexing, concurrency, and distributed databases.' },
    { name: 'Operating Systems', count: 21, description: 'Processes, scheduling, memory management, deadlocks, and file systems.' },
    { name: 'Computer Networks', count: 21, description: 'OSI/TCP model, routing, DNS, TCP/UDP, security, and congestion control.' },
    { name: 'OOP', count: 21, description: 'Encapsulation, inheritance, polymorphism, SOLID principles, and design patterns.' }
];

// Legacy questionBank kept minimal — used only for the offline fallback
// subject list and getSubjects(). Not used in research mode.
const questionBank = cseSubjects.map(s => ({
    id: `${s.name.toLowerCase().replace(/\s+/g, '-')}-placeholder`,
    subject: s.name,
    difficulty: 'easy',
    text: `Placeholder for ${s.name}`,
    options: ['A', 'B', 'C', 'D'],
    correct: 0,
    explanation: ''
}));

const difficultyOrder = ['easy', 'medium', 'hard'];
const difficultyLabels = {
    easy: 'Easy',
    medium: 'Medium',
    hard: 'Hard'
};

let appState = {
    userName: '',
    stats: {
        totalQuestions: 0,
        correctAnswers: 0,
        currentDifficulty: 'medium',
        accuracy: 0,
        streak: 0,
        totalTime: 0,
        lastSuggestion: ''
    },
    selectedSubject: 'all',
    attempts: [],
    // Maps question IDs → concept IDs for the roadmap (loaded from backend
    // or hardcoded fallback from concept_graph.json question_to_concept).
    roadmapConceptMap: {
        "ds-e-1":"stacks_and_queues","ds-e-2":"arrays_and_lists","ds-e-3":"trees_and_bst",
        "ds-e-4":"stacks_and_queues","ds-e-5":"hash_tables","ds-e-6":"trees_and_bst",
        "ds-e-7":"arrays_and_lists","ds-m-1":"hash_tables","ds-m-2":"trees_and_bst",
        "ds-m-3":"heaps","ds-m-4":"arrays_and_lists","ds-m-5":"hash_tables",
        "ds-m-6":"graphs","ds-m-7":"heaps","ds-h-1":"advanced_ds","ds-h-2":"heaps",
        "ds-h-3":"advanced_ds","ds-h-4":"advanced_ds","ds-h-5":"advanced_ds",
        "ds-h-6":"advanced_ds","ds-h-7":"advanced_ds",
        "algo-e-1":"complexity_analysis","algo-e-2":"sorting_searching","algo-e-3":"sorting_searching",
        "algo-e-4":"sorting_searching","algo-e-5":"divide_and_conquer","algo-e-6":"sorting_searching",
        "algo-e-7":"sorting_searching","algo-m-1":"sorting_searching","algo-m-2":"dynamic_programming",
        "algo-m-3":"graph_algorithms","algo-m-4":"divide_and_conquer","algo-m-5":"graph_algorithms",
        "algo-m-6":"greedy_algorithms","algo-m-7":"sorting_searching","algo-h-1":"advanced_algorithms",
        "algo-h-2":"advanced_algorithms","algo-h-3":"complexity_analysis","algo-h-4":"complexity_analysis",
        "algo-h-5":"graph_algorithms","algo-h-6":"graph_algorithms","algo-h-7":"graph_algorithms",
        "db-e-1":"sql_basics","db-e-2":"sql_basics","db-e-3":"sql_basics","db-e-4":"sql_basics",
        "db-e-5":"sql_basics","db-e-6":"sql_basics","db-e-7":"sql_basics","db-m-1":"normalization",
        "db-m-2":"normalization","db-m-3":"transactions_acid","db-m-4":"sql_basics",
        "db-m-5":"indexing_optimization","db-m-6":"sql_basics","db-m-7":"concurrency_control",
        "db-h-1":"indexing_optimization","db-h-2":"concurrency_control","db-h-3":"concurrency_control",
        "db-h-4":"transactions_acid","db-h-5":"concurrency_control","db-h-6":"indexing_optimization",
        "db-h-7":"distributed_databases",
        "os-e-1":"processes_threads","os-e-2":"processes_threads","os-e-3":"scheduling",
        "os-e-4":"memory_management","os-e-5":"processes_threads","os-e-6":"processes_threads",
        "os-e-7":"processes_threads","os-m-1":"synchronization","os-m-2":"synchronization",
        "os-m-3":"memory_management","os-m-4":"memory_management","os-m-5":"deadlocks",
        "os-m-6":"scheduling","os-m-7":"memory_management","os-h-1":"deadlocks",
        "os-h-2":"synchronization","os-h-3":"memory_management","os-h-4":"memory_management",
        "os-h-5":"processes_threads","os-h-6":"memory_management","os-h-7":"scheduling",
        "cn-e-1":"network_fundamentals","cn-e-2":"network_fundamentals","cn-e-3":"application_protocols",
        "cn-e-4":"transport_protocols","cn-e-5":"network_fundamentals","cn-e-6":"network_fundamentals",
        "cn-e-7":"ip_addressing","cn-m-1":"transport_protocols","cn-m-2":"network_fundamentals",
        "cn-m-3":"ip_addressing","cn-m-4":"network_fundamentals","cn-m-5":"ip_addressing",
        "cn-m-6":"application_protocols","cn-m-7":"network_security","cn-h-1":"congestion_control_net",
        "cn-h-2":"routing","cn-h-3":"routing","cn-h-4":"network_security","cn-h-5":"routing",
        "cn-h-6":"transport_protocols","cn-h-7":"network_security",
        "oop-e-1":"classes_and_objects","oop-e-2":"inheritance_polymorphism","oop-e-3":"inheritance_polymorphism",
        "oop-e-4":"classes_and_objects","oop-e-5":"classes_and_objects","oop-e-6":"classes_and_objects",
        "oop-e-7":"classes_and_objects","oop-m-1":"inheritance_polymorphism","oop-m-2":"inheritance_polymorphism",
        "oop-m-3":"inheritance_polymorphism","oop-m-4":"solid_principles","oop-m-5":"solid_principles",
        "oop-m-6":"classes_and_objects","oop-m-7":"design_patterns","oop-h-1":"solid_principles",
        "oop-h-2":"solid_principles","oop-h-3":"design_patterns","oop-h-4":"inheritance_polymorphism",
        "oop-h-5":"classes_and_objects","oop-h-6":"design_patterns","oop-h-7":"inheritance_polymorphism"
    },
    quiz: {
        questions: [],
        currentQuestionIndex: 0,
        selectedAnswer: null,
        isAnswered: false,
        startTime: null,
        nextDifficulty: null,
        researchSessionId: null,
        pendingQuestion: null,
        researchMode: true,
        // Phase 10: what the served policy decided, and what it estimated.
        decision: null,
        estimate: null,
        roadmap: null
    }
};

function getSubjects() {
    return [...new Set(questionBank.map(question => question.subject))].sort();
}

function getSubjectLabel(value) {
    return value === 'all' ? 'All CSE topics' : value;
}

function normalizeDifficulty(value) {
    return difficultyOrder.includes(value) ? value : 'medium';
}

function getFilteredQuestions(difficulty = appState.stats.currentDifficulty) {
    return questionBank.filter(question => {
        const subjectMatches = appState.selectedSubject === 'all' || question.subject === appState.selectedSubject;
        return subjectMatches && question.difficulty === difficulty;
    });
}

function buildQuizQuestions() {
    const primary = getFilteredQuestions(appState.stats.currentDifficulty);
    const fallback = questionBank.filter(question => {
        return appState.selectedSubject === 'all' || question.subject === appState.selectedSubject;
    });

    return (primary.length ? primary : fallback).slice();
}

function showPage(pageName) {
    document.querySelectorAll('.page').forEach(page => page.classList.remove('active'));
    document.getElementById(pageName).classList.add('active');
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

function goHome() {
    showPage(appState.userName ? 'dashboard-page' : 'landing-page');
}

function requireLogin(pageName) {
    if (!appState.userName) {
        showPage('landing-page');
        focusLogin();
        return;
    }

    if (pageName === 'review-page') {
        renderAttempts('review-list', appState.attempts);
    }

    showPage(pageName);
}

function focusLogin() {
    document.getElementById('username').focus();
}

function showQuestionPreview() {
    document.getElementById('subject-preview').scrollIntoView({ behavior: 'smooth' });
}

async function handleLogin(event) {
    event.preventDefault();

    const usernameInput = document.getElementById('username');
    const loginStatus = document.getElementById('login-status');
    const loginButton = document.getElementById('login-btn');
    const username = usernameInput.value.trim();

    if (!username) {
        return false;
    }

    loginButton.disabled = true;
    loginStatus.textContent = 'Checking backend login...';

    try {
        const response = await apiFetch('/user/login', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ username })
        });

        const result = await response.json();
        const loggedInUser = result.user?.username || username;

        appState.userName = loggedInUser;
        usernameInput.value = '';
        loginStatus.textContent = '';
        updateSessionUI();
        updateDashboard();
        showPage('dashboard-page');
    } catch (error) {
        console.error('Error logging in:', error);
        loginStatus.textContent = 'Could not log in. Make sure the backend is running.';
    } finally {
        loginButton.disabled = false;
    }

    return false;
}

function logout() {
    appState.userName = '';
    appState.attempts = [];
    appState.stats = {
        totalQuestions: 0,
        correctAnswers: 0,
        currentDifficulty: 'medium',
        accuracy: 0,
        streak: 0,
        totalTime: 0,
        lastSuggestion: ''
    };
    appState.selectedSubject = 'all';
    document.getElementById('subject-select').value = 'all';
    document.getElementById('difficulty-select').value = 'medium';
    updateSessionUI();
    updateDashboard();
    showPage('landing-page');
}

function updateSessionUI() {
    document.getElementById('nav-user').textContent = appState.userName || 'Guest';
    document.getElementById('nav-logout').hidden = !appState.userName;
    document.getElementById('display-name').textContent = appState.userName;
}

function updateDashboard() {
    const stats = appState.stats;
    const averageTime = stats.totalQuestions ? Math.round(stats.totalTime / stats.totalQuestions) : 0;
    const currentQuestions = getFilteredQuestions(stats.currentDifficulty);

    document.getElementById('total-questions').textContent = stats.totalQuestions;
    document.getElementById('correct-answers').textContent = stats.correctAnswers;
    document.getElementById('accuracy').textContent = `${stats.accuracy}%`;
    document.getElementById('current-level').textContent = stats.currentDifficulty.toUpperCase();
    document.getElementById('selected-topic').textContent = getSubjectLabel(appState.selectedSubject);
    document.getElementById('average-time').textContent = `${averageTime}s`;
    document.getElementById('streak-label').textContent = `${stats.streak} streak`;
    document.getElementById('last-suggestion').textContent = stats.lastSuggestion || 'Not available yet';
    document.getElementById('question-bank-count').textContent = `${currentQuestions.length || buildQuizQuestions().length} questions ready`;
    document.getElementById('difficulty-select').value = stats.currentDifficulty;
    renderAttempts('recent-attempts', appState.attempts.slice(0, 5));
    renderAttempts('review-list', appState.attempts);
    renderRoadmap();
}

// ─── Learning Roadmap ───
// Embedded concept graph for client-side rendering. This mirrors
// backend/src/data/concept_graph.json so the roadmap works even offline.
const CONCEPT_GRAPH = {
    "arrays_and_lists":        { name: "Arrays & Lists",           subject: "Data Structures",   level: 1, prerequisites: [] },
    "stacks_and_queues":       { name: "Stacks & Queues",          subject: "Data Structures",   level: 1, prerequisites: ["arrays_and_lists"] },
    "hash_tables":             { name: "Hash Tables",              subject: "Data Structures",   level: 2, prerequisites: ["arrays_and_lists"] },
    "trees_and_bst":           { name: "Trees & BSTs",             subject: "Data Structures",   level: 2, prerequisites: ["stacks_and_queues"] },
    "heaps":                   { name: "Heaps",                    subject: "Data Structures",   level: 2, prerequisites: ["trees_and_bst"] },
    "graphs":                  { name: "Graphs",                   subject: "Data Structures",   level: 3, prerequisites: ["trees_and_bst", "stacks_and_queues"] },
    "advanced_ds":             { name: "Advanced DS",              subject: "Data Structures",   level: 3, prerequisites: ["hash_tables", "heaps", "graphs"] },
    "sorting_searching":       { name: "Sorting & Searching",      subject: "Algorithms",        level: 2, prerequisites: ["arrays_and_lists"] },
    "complexity_analysis":     { name: "Complexity Analysis",      subject: "Algorithms",        level: 2, prerequisites: ["sorting_searching"] },
    "divide_and_conquer":      { name: "Divide & Conquer",         subject: "Algorithms",        level: 2, prerequisites: ["sorting_searching"] },
    "greedy_algorithms":       { name: "Greedy Algorithms",        subject: "Algorithms",        level: 3, prerequisites: ["sorting_searching"] },
    "dynamic_programming":     { name: "Dynamic Programming",      subject: "Algorithms",        level: 3, prerequisites: ["divide_and_conquer"] },
    "graph_algorithms":        { name: "Graph Algorithms",         subject: "Algorithms",        level: 3, prerequisites: ["graphs", "sorting_searching"] },
    "advanced_algorithms":     { name: "Advanced Algorithms",      subject: "Algorithms",        level: 4, prerequisites: ["dynamic_programming", "graph_algorithms"] },
    "classes_and_objects":     { name: "Classes & Objects",         subject: "OOP",               level: 1, prerequisites: [] },
    "inheritance_polymorphism":{ name: "Inheritance",               subject: "OOP",               level: 2, prerequisites: ["classes_and_objects"] },
    "solid_principles":        { name: "SOLID Principles",         subject: "OOP",               level: 3, prerequisites: ["inheritance_polymorphism"] },
    "design_patterns":         { name: "Design Patterns",          subject: "OOP",               level: 3, prerequisites: ["solid_principles"] },
    "sql_basics":              { name: "SQL Fundamentals",         subject: "DBMS",              level: 1, prerequisites: [] },
    "normalization":           { name: "Normalization",            subject: "DBMS",              level: 2, prerequisites: ["sql_basics"] },
    "transactions_acid":       { name: "Transactions & ACID",      subject: "DBMS",              level: 2, prerequisites: ["sql_basics"] },
    "indexing_optimization":   { name: "Indexing",                 subject: "DBMS",              level: 3, prerequisites: ["normalization"] },
    "concurrency_control":     { name: "Concurrency Control",      subject: "DBMS",              level: 3, prerequisites: ["transactions_acid"] },
    "distributed_databases":   { name: "Distributed DBs",          subject: "DBMS",              level: 4, prerequisites: ["concurrency_control", "indexing_optimization"] },
    "processes_threads":       { name: "Processes & Threads",      subject: "Operating Systems", level: 1, prerequisites: [] },
    "scheduling":              { name: "CPU Scheduling",           subject: "Operating Systems", level: 2, prerequisites: ["processes_threads"] },
    "synchronization":         { name: "Synchronization",          subject: "Operating Systems", level: 2, prerequisites: ["processes_threads"] },
    "memory_management":       { name: "Memory Management",        subject: "Operating Systems", level: 2, prerequisites: ["processes_threads"] },
    "deadlocks":               { name: "Deadlocks",                subject: "Operating Systems", level: 3, prerequisites: ["synchronization", "scheduling"] },
    "network_fundamentals":    { name: "Network Basics",           subject: "Computer Networks", level: 1, prerequisites: [] },
    "ip_addressing":           { name: "IP & Subnetting",          subject: "Computer Networks", level: 2, prerequisites: ["network_fundamentals"] },
    "transport_protocols":     { name: "TCP & UDP",                subject: "Computer Networks", level: 2, prerequisites: ["network_fundamentals"] },
    "application_protocols":   { name: "App Protocols",            subject: "Computer Networks", level: 2, prerequisites: ["transport_protocols"] },
    "routing":                 { name: "Routing",                  subject: "Computer Networks", level: 3, prerequisites: ["ip_addressing"] },
    "congestion_control_net":  { name: "Congestion Control",       subject: "Computer Networks", level: 3, prerequisites: ["transport_protocols"] },
    "network_security":        { name: "Network Security",         subject: "Computer Networks", level: 3, prerequisites: ["transport_protocols", "routing"] }
};

const MASTERY_THRESHOLD = 0.7;

function computeRoadmapState() {
    // Build per-concept mastery from attempt history
    const conceptMastery = {}; // concept_id → { correct, total }
    for (const attempt of appState.attempts) {
        // Try to map question ID to concept via the backend roadmap data
        const conceptId = appState.roadmapConceptMap?.[attempt.questionId];
        if (conceptId) {
            if (!conceptMastery[conceptId]) conceptMastery[conceptId] = { correct: 0, total: 0 };
            conceptMastery[conceptId].total++;
            if (attempt.isCorrect) conceptMastery[conceptId].correct++;
        }
    }

    const concepts = [];
    const masteredSet = new Set();

    // First pass: compute mastery values
    for (const [id, info] of Object.entries(CONCEPT_GRAPH)) {
        const stats = conceptMastery[id];
        const mastery = stats ? stats.correct / stats.total : 0;
        if (mastery >= MASTERY_THRESHOLD) masteredSet.add(id);
    }

    // Second pass: compute statuses
    let targetConcept = null;
    let targetMastery = Infinity;

    for (const [id, info] of Object.entries(CONCEPT_GRAPH)) {
        const stats = conceptMastery[id];
        const mastery = stats ? stats.correct / stats.total : 0;
        const prereqsMet = info.prerequisites.every(p => masteredSet.has(p));

        let status;
        if (mastery >= MASTERY_THRESHOLD) {
            status = 'mastered';
        } else if (stats && stats.total > 0) {
            status = 'in_progress';
        } else if (prereqsMet) {
            status = 'eligible';
        } else {
            status = 'locked';
        }

        // Find target: lowest mastery among eligible/in_progress
        if ((status === 'eligible' || status === 'in_progress') && mastery < targetMastery) {
            targetMastery = mastery;
            targetConcept = id;
        }

        concepts.push({ id, ...info, mastery, status });
    }

    return { concepts, targetConcept };
}

function renderRoadmap() {
    const container = document.getElementById('roadmap-container');
    const progressLabel = document.getElementById('roadmap-progress');

    const { concepts, targetConcept } = computeRoadmapState();

    const mastered = concepts.filter(c => c.status === 'mastered').length;
    const total = concepts.length;
    progressLabel.textContent = `${mastered} / ${total} concepts mastered`;

    // Group by subject
    const subjectOrder = ['Data Structures', 'Algorithms', 'OOP', 'DBMS', 'Operating Systems', 'Computer Networks'];
    const bySubject = {};
    for (const c of concepts) {
        if (!bySubject[c.subject]) bySubject[c.subject] = [];
        bySubject[c.subject].push(c);
    }

    const statusIcons = { mastered: '✓', in_progress: '◔', eligible: '○', locked: '🔒' };

    // Overall progress bar
    const pct = total > 0 ? Math.round((mastered / total) * 100) : 0;
    let html = `
        <div class="roadmap-overall" style="grid-column: 1 / -1;">
            <div class="progress-bar-wrap">
                <div class="progress-bar-fill" style="width: ${pct}%"></div>
            </div>
            <span style="font-size: 13px; font-weight: 600; color: var(--primary);">${pct}%</span>
            ${targetConcept ? `<span class="roadmap-target">Next: <strong>${CONCEPT_GRAPH[targetConcept].name}</strong></span>` : ''}
        </div>
    `;

    for (const subject of subjectOrder) {
        const group = (bySubject[subject] || []).sort((a, b) => a.level - b.level);
        const subMastered = group.filter(c => c.status === 'mastered').length;
        const badgeClass = subMastered === group.length ? 'complete' : subMastered > 0 ? 'partial' : 'none';

        html += `
            <div class="roadmap-subject">
                <div class="roadmap-subject-header">
                    <h3>${subject}</h3>
                    <span class="progress-badge ${badgeClass}">${subMastered}/${group.length}</span>
                </div>
                <div class="roadmap-concepts">
                    ${group.map(c => `
                        <span class="concept-node ${c.status}${c.id === targetConcept ? ' target' : ''}"
                              title="${c.name} — ${c.status === 'mastered' ? 'Mastered' : c.status === 'in_progress' ? Math.round(c.mastery * 100) + '% mastery' : c.status === 'eligible' ? 'Ready to learn' : 'Prerequisites needed'}">
                            <span class="status-icon">${statusIcons[c.status]}</span>
                            ${c.name}
                        </span>
                    `).join('')}
                </div>
            </div>
        `;
    }

    container.innerHTML = html;
}

function setSubject() {
    appState.selectedSubject = document.getElementById('subject-select').value;
    updateDashboard();
}

function setStartingDifficulty() {
    appState.stats.currentDifficulty = normalizeDifficulty(document.getElementById('difficulty-select').value);
    updateDashboard();
}

function resetProgress() {
    appState.stats.totalQuestions = 0;
    appState.stats.correctAnswers = 0;
    appState.stats.accuracy = 0;
    appState.stats.streak = 0;
    appState.stats.totalTime = 0;
    appState.stats.lastSuggestion = '';
    appState.attempts = [];
    updateDashboard();
}

// ---------------------------------------------------------------------------
// Research session flow — the /v1 API.
//
// The client never learns the answer key: correctness comes back from
// POST /v1/attempts, which computes it from the stored item. Consent is
// collected before the first session and gates collection at the source.
// ---------------------------------------------------------------------------

const CONSENT_STORAGE_KEY = 'alp.consent.v1';
const SIGNAL_CLASSES = ['timing', 'interaction', 'motor', 'probes'];

function storedConsent() {
    try {
        return JSON.parse(localStorage.getItem(CONSENT_STORAGE_KEY) || 'null');
    } catch (error) {
        return null;
    }
}

function openConsent() {
    const consent = storedConsent();
    SIGNAL_CLASSES.forEach(signalClass => {
        const checkbox = document.getElementById(`consent-${signalClass}`);
        if (checkbox) checkbox.checked = consent ? consent.classes[signalClass] !== false : true;
    });
    showPage('consent-page');
}

async function saveConsent() {
    if (!appState.userName) {
        requireLogin('dashboard-page');
        return;
    }
    const classes = {};
    SIGNAL_CLASSES.forEach(signalClass => {
        classes[signalClass] = document.getElementById(`consent-${signalClass}`)?.checked !== false;
    });

    try {
        const response = await apiFetch('/v1/consent', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ learner_id: appState.userName, classes })
        });
        const result = await response.json();
        localStorage.setItem(CONSENT_STORAGE_KEY, JSON.stringify({
            consent_id: result.consent_id, classes: result.classes, learner_id: appState.userName
        }));
        document.getElementById('consent-status').textContent = 'Saved. You can change this at any time.';
        showPage('dashboard-page');
    } catch (error) {
        console.error('Unable to record consent:', error);
        document.getElementById('consent-status').textContent = 'Could not reach the server. Consent was not recorded.';
    }
}

function toUiItem(item) {
    return {
        id: item.item_id,
        subject: (item.concepts || []).join(' + ') || 'Multi-topic CSE',
        difficulty: normalizeDifficulty(item.difficulty),
        text: item.stem,
        options: item.options,
        explanation: '',
        correct: null
    };
}

async function startQuiz() {
    if (!appState.userName) {
        requireLogin('dashboard-page');
        return;
    }

    const consent = storedConsent();
    if (!consent) {
        openConsent();
        return;
    }

    try {
        const response = await apiFetch('/v1/sessions', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                learner_id: appState.userName,
                // The bank is the server's: a research session always runs the
                // canonical item bank the policies were evaluated over.
                consent_id: consent.consent_id
            })
        });
        const session = await response.json();

        appState.quiz = {
            questions: [], currentQuestionIndex: 0, selectedAnswer: null, isAnswered: false,
            sessionId: session.session_id, sessionVersion: session.session_version,
            seq: 0, probes: [], pendingAttempt: null, presentedAt: null,
            totalSessionQuestions: 0, sessionQuestionCount: 0
        };

        await ALPTelemetry.init({
            endpoint: '',
            sessionId: session.session_id,
            learnerId: appState.userName,
            consent: session.consent
        });

        showPage('quiz-page');
        await loadNextItem();
    } catch (error) {
        console.error('Unable to start research session:', error);
        alert('Unable to start the session. Is the backend running (make up)?');
    }
}

async function loadNextItem() {
    try {
        const response = await apiFetch(`/v1/next-item?session_id=${encodeURIComponent(appState.quiz.sessionId)}`);
        const payload = await response.json();

        appState.quiz.questions = [toUiItem(payload.item)];
        appState.quiz.currentQuestionIndex = 0;
        appState.quiz.sessionVersion = payload.session_version;
        appState.quiz.seq = payload.seq;
        appState.quiz.probes = payload.probes ?? [];
        appState.quiz.sessionQuestionCount = payload.items_asked;
        appState.quiz.totalSessionQuestions = payload.total_items;
        ALPTelemetry.decisionMade(payload.item.item_id, {
            decision_id: payload.decision_id,
            propensity: payload.propensity,
            difficulty: payload.item.difficulty
        });
        appState.quiz.decision = {
            policy: payload.policy,
            explanation: payload.explanation,
            fallback: payload.fallback,
            target_concept: payload.target_concept
        };
        appState.quiz.roadmap = payload.roadmap ?? null;
        loadQuestion();
        renderSupport(payload.intervention);
        renderDecision();
    } catch (error) {
        console.error('Unable to fetch the next item:', error);
        backToDashboard();
    }
}

function loadQuestion() {
    const currentQuestion = getCurrentQuestion();

    if (!currentQuestion) {
        backToDashboard();
        return;
    }

    const currentNum = appState.quiz.sessionQuestionCount || 1;
    const totalNum = appState.quiz.totalSessionQuestions || 1;
    const progress = Math.min(100, Math.round((currentNum / totalNum) * 100));

    document.getElementById('question-number').textContent = currentNum;
    document.getElementById('quiz-level').textContent = currentQuestion.difficulty.toUpperCase();
    document.getElementById('quiz-subject').textContent = currentQuestion.subject;
    document.getElementById('question-topic').textContent = `${currentQuestion.subject} · ${difficultyLabels[currentQuestion.difficulty]}`;
    document.getElementById('quiz-progress-label').textContent = `${currentNum} / ${totalNum}`;
    document.getElementById('quiz-progress').style.width = `${progress}%`;
    document.getElementById('question-text').textContent = currentQuestion.text;

    const optionsContainer = document.getElementById('options-container');
    optionsContainer.innerHTML = '';

    const optionElements = currentQuestion.options.map((option, index) => {
        const optionButton = document.createElement('button');
        optionButton.className = 'option';
        optionButton.type = 'button';
        optionButton.textContent = option;
        optionButton.onclick = () => selectOption(index);
        optionsContainer.appendChild(optionButton);
        return optionButton;
    });

    appState.quiz.selectedAnswer = null;
    appState.quiz.isAnswered = false;

    document.getElementById('result-box').hidden = true;
    document.getElementById('probe-box').hidden = true;
    document.getElementById('submit-btn').hidden = false;
    document.getElementById('next-btn').hidden = true;

    // Layout is settled by now, so the option rectangles the trajectory
    // features are measured against are the ones the learner actually sees.
    appState.quiz.presentedAt = ALPTelemetry.startItem(currentQuestion.id, optionElements);
}

function getCurrentQuestion() {
    return appState.quiz.questions[appState.quiz.currentQuestionIndex];
}

function selectOption(index) {
    if (appState.quiz.isAnswered) return;

    appState.quiz.selectedAnswer = index;
    ALPTelemetry.selectOption(index);

    document.querySelectorAll('.option').forEach((option, optionIndex) => {
        option.classList.toggle('selected', optionIndex === index);
    });
}

async function submitAnswer() {
    if (appState.quiz.selectedAnswer === null) {
        alert('Please select an answer');
        return;
    }
    if (appState.quiz.isAnswered) return;
    appState.quiz.isAnswered = true;

    const currentQuestion = getCurrentQuestion();
    const selectedAnswer = appState.quiz.selectedAnswer;

    // Flushes the event stream (including the aggregated trajectory) so the
    // server holds every event for this item before the attempt is posted.
    const timing = await ALPTelemetry.submitItem(selectedAnswer);

    let result;
    try {
        const response = await apiFetch('/v1/attempts', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                session_id: appState.quiz.sessionId,
                item_id: currentQuestion.id,
                selected_index: selectedAnswer,
                presented_at: timing.presented_at,
                response_time_ms: timing.response_time_ms,
                session_version: appState.quiz.sessionVersion
            })
        });
        result = await response.json();
    } catch (error) {
        console.error('Attempt submission failed:', error);
        alert('The attempt could not be recorded. Check the backend and try again.');
        appState.quiz.isAnswered = false;
        return;
    }

    appState.quiz.sessionVersion = result.session_version;
    appState.quiz.estimate = result.estimate ?? null;
    renderStatePanel();
    currentQuestion.correct = result.correct_index;
    currentQuestion.explanation = result.explanation ?? '';

    const timeTaken = Math.max(1, Math.round((timing.response_time_ms ?? 0) / 1000));
    appState.stats.totalQuestions += 1;
    appState.stats.totalTime += timeTaken;
    appState.stats.streak = result.correct ? appState.stats.streak + 1 : 0;
    if (result.correct) appState.stats.correctAnswers += 1;
    appState.stats.accuracy = Math.round((appState.stats.correctAnswers / appState.stats.totalQuestions) * 100);
    appState.stats.currentDifficulty = normalizeDifficulty(result.recommendation?.difficulty ?? appState.stats.currentDifficulty);
    appState.stats.lastSuggestion = result.recommendation
        ? `${difficultyLabels[appState.stats.currentDifficulty]} · ${result.recommendation.action} · ${(result.recommendation.reasons || []).join(', ')}`
        : `${difficultyLabels[appState.stats.currentDifficulty]} after this attempt`;

    appState.attempts.unshift({
        question: currentQuestion.text,
        subject: currentQuestion.subject,
        difficulty: currentQuestion.difficulty,
        selected: currentQuestion.options[selectedAnswer],
        correct: currentQuestion.options[result.correct_index],
        isCorrect: Boolean(result.correct),
        timeTaken,
        nextDifficulty: appState.stats.currentDifficulty,
        explanation: currentQuestion.explanation
    });

    // Probes are asked after submission and *before* feedback, so the answer
    // key cannot colour the self-report.
    showProbes(currentQuestion.id, appState.quiz.probes, () => {
        showResult(Boolean(result.correct), currentQuestion, timeTaken);
    });
    updateDashboard();
}

const PROBE_SCALES = {
    confidence: { question: 'How sure were you of that answer?', labels: ['Guessing', 'Not sure', 'Fairly sure', 'Certain'] },
    perceived_difficulty: { question: 'How difficult was that question?', labels: ['Very easy', 'Easy', 'Hard', 'Very hard'] },
    effort: { question: 'How much effort are you putting in right now?', labels: ['None', 'A little', 'Some', 'A lot', 'Everything'] }
};

/** Renders assigned probes one at a time; skipping is allowed and is logged. */
function showProbes(itemId, probes, done) {
    const box = document.getElementById('probe-box');
    const queue = probes.slice();

    const next = () => {
        if (!queue.length) {
            box.hidden = true;
            done();
            return;
        }
        const probe = queue.shift();
        const scale = PROBE_SCALES[probe.kind] ?? { question: probe.kind, labels: [] };
        const probeId = ALPTelemetry.probeShown(itemId, probe);

        box.hidden = false;
        box.innerHTML = '';
        const heading = document.createElement('h3');
        heading.textContent = scale.question;
        box.appendChild(heading);

        const row = document.createElement('div');
        row.className = 'probe-options';
        for (let point = 1; point <= probe.scale_points; point += 1) {
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'btn btn-secondary btn-small';
            button.textContent = scale.labels[point - 1] ?? String(point);
            button.onclick = () => {
                ALPTelemetry.probeAnswered(itemId, probeId, point, false);
                next();
            };
            row.appendChild(button);
        }
        box.appendChild(row);

        const skip = document.createElement('button');
        skip.type = 'button';
        skip.className = 'btn-link';
        skip.textContent = 'Skip this question';
        skip.onclick = () => {
            ALPTelemetry.probeAnswered(itemId, probeId, null, true);
            next();
        };
        box.appendChild(skip);
    };

    next();
}

function showResult(isCorrect, currentQuestion, timeTaken) {
    const resultBox = document.getElementById('result-box');
    const resultMessage = document.getElementById('result-message');
    const explanation = document.getElementById('answer-explanation');
    const nextDifficulty = document.getElementById('next-difficulty');

    document.querySelectorAll('.option').forEach((option, index) => {
        option.classList.remove('selected');
        option.disabled = true;

        if (index === currentQuestion.correct) {
            option.classList.add('correct');
        }

        if (index === appState.quiz.selectedAnswer && index !== currentQuestion.correct) {
            option.classList.add('incorrect');
        }
    });

    resultMessage.textContent = isCorrect ? 'Correct answer' : 'Needs revision';
    resultMessage.className = isCorrect ? 'correct-text' : 'incorrect-text';
    explanation.textContent = currentQuestion.explanation;
    nextDifficulty.textContent = `Time: ${timeTaken}s · Next level: ${difficultyLabels[appState.stats.currentDifficulty]}`;

    resultBox.hidden = false;
    document.getElementById('submit-btn').hidden = true;
    document.getElementById('next-btn').hidden = false;
}

function nextQuestion() {
    loadNextItem();
}

async function backToDashboard() {
    if (appState.quiz.sessionId) {
        await ALPTelemetry.endSession();
        try {
            await apiFetch('/v1/sessions/end', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: appState.quiz.sessionId })
            });
        } catch (error) {
            console.warn('Could not mark the session ended:', error);
        }
        appState.quiz.sessionId = null;
    }
    updateDashboard();
    showPage('dashboard-page');
}

function renderAttempts(containerId, attempts) {
    const container = document.getElementById(containerId);

    if (!container) return;

    if (!attempts.length) {
        container.className = 'attempt-list empty-state';
        container.textContent = containerId === 'recent-attempts' ? 'No attempts yet.' : 'No attempts to review yet.';
        return;
    }

    container.className = 'attempt-list';
    container.innerHTML = attempts.map((attempt, index) => `
        <article class="attempt-item">
            <div>
                <strong>${index + 1}. ${attempt.subject}</strong>
                <p>${attempt.question}</p>
                <small>${difficultyLabels[attempt.difficulty]} · ${attempt.timeTaken}s · next ${difficultyLabels[attempt.nextDifficulty]}</small>
            </div>
            <span class="${attempt.isCorrect ? 'badge badge-correct' : 'badge badge-wrong'}">
                ${attempt.isCorrect ? 'Correct' : 'Review'}
            </span>
        </article>
    `).join('');
}

function renderSubjects() {
    const subjectSelect = document.getElementById('subject-select');
    const subjectPreview = document.getElementById('subject-preview');

    cseSubjects.forEach(subject => {
        const option = document.createElement('option');
        option.value = subject.name;
        option.textContent = subject.name;
        subjectSelect.appendChild(option);
    });

    subjectPreview.innerHTML = cseSubjects.map(subject => `
        <article class="subject-card">
            <h3>${subject.name}</h3>
            <p>${subject.count} adaptive MCQs — ${subject.description}</p>
        </article>
    `).join('');

    const totalQuestions = cseSubjects.reduce((sum, s) => sum + s.count, 0);
    document.getElementById('landing-question-count').textContent = totalQuestions;
    document.getElementById('landing-subject-count').textContent = cseSubjects.length;
}

document.addEventListener('DOMContentLoaded', () => {
    renderSubjects();
    updateSessionUI();
    updateDashboard();
    showPage('landing-page');
});


// ---------------------------------------------------------------- explanations

const STATE_LABELS = {
    knowledge: 'Knowledge of the current concept',
    engagement: 'Engagement',
    confidence: 'Confidence',
    fatigue: 'Fatigue'
};

/** The support the policy chose to offer: a hint, a worked example, a break. */
function renderSupport(intervention) {
    const box = document.getElementById('support-box');
    if (!box) return;
    if (!intervention || !intervention.body) {
        box.hidden = true;
        return;
    }
    document.getElementById('support-title').textContent = intervention.title;
    document.getElementById('support-body').textContent = intervention.body;
    box.dataset.kind = intervention.kind;
    box.hidden = false;
}

/** Why this item, in plain English, plus a fallback notice when one applies. */
function renderDecision() {
    const line = document.getElementById('decision-explanation');
    const panel = document.getElementById('state-panel');
    if (!line || !panel) return;
    const decision = appState.quiz.decision ?? {};
    const fallback = decision.fallback
        ? ' The adaptive model was unavailable for this item, so the rule engine chose it.'
        : '';
    line.textContent = `${decision.explanation ?? ''}${fallback}`.trim();
    if (line.textContent) panel.hidden = false;
    renderRoadmapMini();
}

/**
 * The state panel.
 *
 * A state the validation gate rejected is listed as withheld, with the reason
 * the gate gave. Showing it as a number would present a construct the project
 * measured and refused as though it had passed.
 */
function renderStatePanel() {
    const panel = document.getElementById('state-panel');
    const list = document.getElementById('state-list');
    const source = document.getElementById('state-panel-source');
    if (!panel || !list) return;

    const estimate = appState.quiz.estimate;
    list.innerHTML = '';
    if (!estimate || estimate.unavailable) {
        source.textContent = 'estimator unavailable';
        panel.hidden = false;
        return;
    }

    source.textContent = `estimates from the ${estimate.rung} model — not diagnoses`;
    for (const [name, value] of Object.entries(estimate.states ?? {})) {
        const error = estimate.standard_error?.[name];
        const term = document.createElement('dt');
        term.textContent = STATE_LABELS[name] ?? name;
        const detail = document.createElement('dd');
        detail.textContent = error === undefined
            ? `${Math.round(value * 100)}%`
            : `${Math.round(value * 100)}% ± ${Math.round(error * 100)}`;
        list.append(term, detail);
    }
    for (const [name, reason] of Object.entries(estimate.gate?.rejected ?? {})) {
        const term = document.createElement('dt');
        term.textContent = STATE_LABELS[name] ?? name;
        const detail = document.createElement('dd');
        detail.className = 'state-withheld';
        detail.textContent = `withheld — ${reason}`;
        list.append(term, detail);
    }
    panel.hidden = false;
}

/** The session's concepts and how far along each one is. */
function renderRoadmapMini() {
    const container = document.getElementById('roadmap-mini-list');
    if (!container) return;
    const roadmap = appState.quiz.roadmap;
    const concepts = (roadmap?.concepts ?? []).filter(concept => concept.status !== 'locked');
    container.innerHTML = '';
    if (!concepts.length) {
        container.textContent = 'No concept progress yet.';
        return;
    }
    for (const concept of concepts) {
        const row = document.createElement('div');
        row.className = 'roadmap-mini-row';
        const active = concept.concept_id === appState.quiz.decision?.target_concept;
        row.innerHTML = `<span>${active ? '\u25b6 ' : ''}${concept.name}</span>`
            + `<span>${Math.round((concept.mastery ?? 0) * 100)}%</span>`;
        container.appendChild(row);
    }
}
