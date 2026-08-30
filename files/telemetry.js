/**
 * Client telemetry for the adaptive learning platform.
 *
 * Two design constraints come from the research plan, not from engineering
 * taste, and neither can be retrofitted later:
 *
 *   1. **Raw pointer samples never leave the browser.** They live in a
 *      session-scoped buffer, are reduced on submit to the pre-registered
 *      trajectory features in `docs/preregistration.md`, and are then dropped.
 *      Only the aggregate is sent.
 *   2. **Consent gates collection at the source.** A signal class the learner
 *      switched off is never recorded, not merely hidden downstream. The whole
 *      loop must still work with every optional class off — that is the
 *      mechanism behind the RQ4 privacy-utility curve.
 */
const ALPTelemetry = (() => {
    const SAMPLE_INTERVAL_MS = 50;      // pointermove sampling rate
    const IDLE_THRESHOLD_MS = 3000;     // no interaction for this long -> IDLE_ENTERED
    const PAUSE_THRESHOLD_MS = 300;     // velocity below PAUSE_VELOCITY for this long -> a pause
    const PAUSE_VELOCITY = 0.05;        // px/ms
    const FLUSH_INTERVAL_MS = 5000;

    const uuid = () => (crypto.randomUUID
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}-${Math.random().toString(16).slice(2)}`);

    const state = {
        endpoint: '',
        sessionId: null,
        learnerId: null,
        consent: { correctness: true, timing: true, interaction: true, motor: true, probes: true },
        seq: 0,
        buffer: [],
        flushTimer: null,
        item: null
    };

    // ---------------------------------------------------------------- events

    const signalClass = {
        SESSION_STARTED: 'core', SESSION_ENDED: 'core', ITEM_PRESENTED: 'core',
        ANSWER_SUBMITTED: 'core', ITEM_SKIPPED: 'core', DECISION_MADE: 'core',
        FIRST_INTERACTION: 'timing',
        OPTION_SELECTED: 'interaction', OPTION_CHANGED: 'interaction', HINT_REQUESTED: 'interaction',
        IDLE_ENTERED: 'interaction', IDLE_EXITED: 'interaction', VISIBILITY_CHANGED: 'interaction',
        CURSOR_SEGMENT: 'motor',
        PROBE_SHOWN: 'probes', PROBE_ANSWERED: 'probes'
    };

    function record(type, payload = {}, itemId = state.item?.itemId ?? null) {
        const consentClass = signalClass[type];
        if (!consentClass) return;
        if (consentClass !== 'core' && state.consent[consentClass] === false) return;
        if (!state.sessionId) return;

        state.seq += 1;
        state.buffer.push({
            event_id: uuid(),
            session_id: state.sessionId,
            learner_id: state.learnerId,
            item_id: itemId,
            type,
            client_ts: new Date().toISOString(),
            seq: state.seq,
            payload
        });
    }

    async function flush() {
        if (!state.buffer.length || !state.sessionId) return { stored: 0 };
        const batch = state.buffer;
        state.buffer = [];
        try {
            const response = await fetch(`${state.endpoint}/v1/events`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: state.sessionId, learner_id: state.learnerId, events: batch })
            });
            if (!response.ok) throw new Error(`events ${response.status}`);
            return await response.json();
        } catch (error) {
            // Re-queue: /v1/events is idempotent on event_id, so a replay is safe.
            state.buffer = batch.concat(state.buffer);
            console.warn('telemetry flush failed, will retry', error);
            return { stored: 0, error: String(error) };
        }
    }

    // ------------------------------------------------------------ trajectory

    /** Perpendicular signed distance of (x, y) from the line a->b. */
    function signedDistance(point, a, b) {
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        const length = Math.hypot(dx, dy);
        if (length === 0) return 0;
        return ((point.x - a.x) * dy - (point.y - a.y) * dx) / length;
    }

    /**
     * Sample entropy (SampEn, m = 2, r = 0.2 sigma) of the speed series.
     * ponytail: the textbook O(n^2) definition over a few hundred samples per
     * item — roughly 10^5 comparisons, far below anything worth optimising.
     * Use a sliding-window/kd-tree implementation only if items ever run long
     * enough to produce thousands of samples.
     */
    function sampleEntropy(series, m = 2, rFactor = 0.2) {
        const n = series.length;
        if (n < m + 2) return 0;
        const mean = series.reduce((sum, value) => sum + value, 0) / n;
        const sd = Math.sqrt(series.reduce((sum, value) => sum + (value - mean) ** 2, 0) / n);
        if (sd === 0) return 0;
        const r = rFactor * sd;

        const count = (length) => {
            let matches = 0;
            for (let i = 0; i + length <= n; i += 1) {
                for (let j = i + 1; j + length <= n; j += 1) {
                    let within = true;
                    for (let k = 0; k < length; k += 1) {
                        if (Math.abs(series[i + k] - series[j + k]) > r) { within = false; break; }
                    }
                    if (within) matches += 1;
                }
            }
            return matches;
        };

        const a = count(m + 1);
        const b = count(m);
        if (a === 0 || b === 0) return 0;
        return Number((-Math.log(a / b)).toFixed(4));
    }

    /** Reduce the raw sample buffer to the pre-registered aggregate. */
    function trajectoryFeatures(item, selectedIndex) {
        const samples = item.samples;
        const hover = item.hoverMs.slice();
        if (samples.length < 2) {
            return {
                auc_toward_nonchosen: 0, max_deviation: 0, x_flips: 0, sample_entropy: 0,
                velocity_peak: 0, velocity_mean: 0, pause_count: 0, path_ratio: 1,
                time_to_first_movement_ms: item.firstMovementMs ?? 0,
                time_to_first_selection_ms: item.firstSelectionMs ?? 0,
                hover_time_ms: hover, n_samples: samples.length, sample_interval_ms: SAMPLE_INTERVAL_MS
            };
        }

        const start = samples[0];
        const target = item.optionCentres[selectedIndex] ?? samples[samples.length - 1];
        // The competing option: the non-chosen option the cursor lingered on
        // longest. The AUC is signed positive toward it.
        let competitor = null;
        let bestHover = -1;
        item.optionCentres.forEach((centre, index) => {
            if (index === selectedIndex || !centre) return;
            if ((hover[index] ?? 0) > bestHover) { bestHover = hover[index] ?? 0; competitor = centre; }
        });
        const competitorSign = competitor ? Math.sign(signedDistance(competitor, start, target)) || 1 : 1;

        let pathLength = 0;
        let maxDeviation = 0;
        let auc = 0;
        let flips = 0;
        let lastDirection = 0;
        const speeds = [];
        let pauseCount = 0;
        let pauseRun = 0;

        for (let index = 1; index < samples.length; index += 1) {
            const previous = samples[index - 1];
            const current = samples[index];
            const dt = Math.max(1, current.t - previous.t);
            const step = Math.hypot(current.x - previous.x, current.y - previous.y);
            pathLength += step;

            const speed = step / dt;                       // px/ms
            speeds.push(speed);
            if (speed < PAUSE_VELOCITY) {
                pauseRun += dt;
            } else {
                if (pauseRun > PAUSE_THRESHOLD_MS) pauseCount += 1;
                pauseRun = 0;
            }

            const direction = Math.sign(current.x - previous.x);
            if (direction !== 0 && lastDirection !== 0 && direction !== lastDirection) flips += 1;
            if (direction !== 0) lastDirection = direction;

            const deviation = signedDistance(current, start, target);
            maxDeviation = Math.max(maxDeviation, Math.abs(deviation));
            auc += competitorSign * deviation * dt / 1000;   // px-seconds, signed toward the competitor
        }
        if (pauseRun > PAUSE_THRESHOLD_MS) pauseCount += 1;

        const straight = Math.hypot(target.x - start.x, target.y - start.y);
        const peak = speeds.length ? Math.max(...speeds) : 0;
        const mean = speeds.length ? speeds.reduce((sum, value) => sum + value, 0) / speeds.length : 0;

        return {
            auc_toward_nonchosen: Number(auc.toFixed(4)),
            max_deviation: Number(maxDeviation.toFixed(4)),
            x_flips: flips,
            sample_entropy: sampleEntropy(speeds),
            velocity_peak: Number((peak * 1000).toFixed(4)),   // px/s
            velocity_mean: Number((mean * 1000).toFixed(4)),   // px/s
            pause_count: pauseCount,
            path_ratio: straight > 0 ? Number((pathLength / straight).toFixed(4)) : 1,
            time_to_first_movement_ms: item.firstMovementMs ?? 0,
            time_to_first_selection_ms: item.firstSelectionMs ?? 0,
            hover_time_ms: hover.map(value => Math.round(value)),
            n_samples: samples.length,
            sample_interval_ms: SAMPLE_INTERVAL_MS
        };
    }

    // -------------------------------------------------------------- listeners

    function onPointerMove(event) {
        const item = state.item;
        if (!item) return;
        const now = performance.now();
        markInteraction(now, 'pointermove');
        if (!state.consent.motor) return;
        if (now - item.lastSampleAt < SAMPLE_INTERVAL_MS) return;

        item.lastSampleAt = now;
        const point = { t: Math.round(now - item.startedAt), x: event.clientX, y: event.clientY };
        item.samples.push(point);
        if (item.firstMovementMs === null) item.firstMovementMs = point.t;

        item.optionRects.forEach((rect, index) => {
            if (!rect) return;
            const inside = event.clientX >= rect.left && event.clientX <= rect.right
                && event.clientY >= rect.top && event.clientY <= rect.bottom;
            if (inside) item.hoverMs[index] += SAMPLE_INTERVAL_MS;
        });
    }

    function markInteraction(now, kind) {
        const item = state.item;
        if (!item) return;
        if (!item.firstInteractionAt) {
            item.firstInteractionAt = now;
            record('FIRST_INTERACTION', { kind, ms_since_presented: Math.round(now - item.startedAt) });
        }
        if (item.idle) {
            item.idle = false;
            record('IDLE_EXITED', { ms_since_presented: Math.round(now - item.startedAt) });
        }
        item.lastInteractionAt = now;
    }

    function idleTick() {
        const item = state.item;
        if (!item || item.idle || !item.lastInteractionAt) return;
        if (performance.now() - item.lastInteractionAt > IDLE_THRESHOLD_MS) {
            item.idle = true;
            record('IDLE_ENTERED', { threshold_ms: IDLE_THRESHOLD_MS });
        }
    }

    function onVisibilityChange() {
        // Focus state is recorded alongside visibility so idle-with-focus and
        // idle-without-focus stay distinguishable.
        record('VISIBILITY_CHANGED', {
            visible: document.visibilityState === 'visible',
            focused: document.hasFocus(),
            ms_since_presented: state.item ? Math.round(performance.now() - state.item.startedAt) : null
        });
    }

    // --------------------------------------------------------------- public

    return {
        SAMPLE_INTERVAL_MS,

        init({ endpoint = '', sessionId, learnerId, consent }) {
            state.endpoint = endpoint;
            state.sessionId = sessionId;
            state.learnerId = learnerId;
            if (consent) state.consent = { ...state.consent, ...consent };
            state.seq = 0;
            state.buffer = [];

            window.addEventListener('pointermove', onPointerMove, { passive: true });
            document.addEventListener('visibilitychange', onVisibilityChange);
            window.addEventListener('blur', onVisibilityChange);
            window.addEventListener('focus', onVisibilityChange);
            window.addEventListener('pagehide', () => { flush(); });
            clearInterval(state.flushTimer);
            state.flushTimer = setInterval(() => { idleTick(); flush(); }, FLUSH_INTERVAL_MS);

            record('SESSION_STARTED', { consent: state.consent, user_agent: navigator.userAgent });
            return flush();
        },

        get consent() { return { ...state.consent }; },

        /** Called when an item is rendered. `optionElements` index-aligns with the item's options. */
        startItem(itemId, optionElements) {
            const now = performance.now();
            const rects = optionElements.map(element => element.getBoundingClientRect());
            state.item = {
                itemId,
                startedAt: now,
                presentedAt: new Date().toISOString(),
                samples: [],
                optionRects: rects,
                optionCentres: rects.map(rect => ({ x: rect.left + rect.width / 2, y: rect.top + rect.height / 2 })),
                hoverMs: rects.map(() => 0),
                lastSampleAt: 0,
                firstMovementMs: null,
                firstSelectionMs: null,
                firstInteractionAt: null,
                lastInteractionAt: now,
                idle: false,
                selections: [],
                selectedIndex: null
            };
            record('ITEM_PRESENTED', { presented_at: state.item.presentedAt });
            return state.item.presentedAt;
        },

        /** Every selection, in order, with its timing — not just a change count. */
        selectOption(index) {
            const item = state.item;
            if (!item) return;
            const now = performance.now();
            markInteraction(now, 'select');
            const msSincePresented = Math.round(now - item.startedAt);
            const previous = item.selectedIndex;
            item.selections.push({ index, ms: msSincePresented });
            item.selectedIndex = index;

            if (previous === null) {
                item.firstSelectionMs = msSincePresented;
                record('OPTION_SELECTED', { to_index: index, ms_since_presented: msSincePresented });
            } else if (previous !== index) {
                record('OPTION_CHANGED', {
                    from_index: previous,
                    to_index: index,
                    change_number: item.selections.filter((_, position) => position > 0).length,
                    ms_since_presented: msSincePresented
                });
            }
        },

        /**
         * Ends the item: emits the aggregated trajectory, drops the raw samples,
         * and flushes so the server holds the full stream before the attempt is
         * posted. Returns the timing facts the attempt endpoint needs.
         */
        async submitItem(selectedIndex, { skipped = false } = {}) {
            const item = state.item;
            if (!item) return { presented_at: null, response_time_ms: null };
            const responseTimeMs = Math.round(performance.now() - item.startedAt);

            if (state.consent.motor && item.samples.length) {
                record('CURSOR_SEGMENT', trajectoryFeatures(item, selectedIndex ?? item.selectedIndex ?? 0));
            }
            record(skipped ? 'ITEM_SKIPPED' : 'ANSWER_SUBMITTED', {
                selected_index: selectedIndex,
                response_time_ms: responseTimeMs,
                option_change_sequence: item.selections
            });

            // Raw samples are discarded here, before anything is sent. This is
            // the privacy requirement in docs/data-privacy-policy.md.
            item.samples = [];
            const presentedAt = item.presentedAt;
            const itemId = item.itemId;
            state.item = null;
            await flush();
            return { presented_at: presentedAt, response_time_ms: responseTimeMs, item_id: itemId };
        },

        probeShown(itemId, probe) {
            const probeId = uuid();
            record('PROBE_SHOWN', { probe_id: probeId, ...probe }, itemId);
            return probeId;
        },

        probeAnswered(itemId, probeId, response, skipped = false) {
            record('PROBE_ANSWERED', { probe_id: probeId, response, skipped }, itemId);
        },

        decisionMade(itemId, decision) {
            record('DECISION_MADE', decision, itemId);
        },

        async endSession() {
            record('SESSION_ENDED', {});
            const result = await flush();
            clearInterval(state.flushTimer);
            state.sessionId = null;
            return result;
        },

        flush
    };
})();
