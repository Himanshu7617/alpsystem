/**
 * Phase 10's end-to-end check: consent → session → twenty items → session end,
 * driven through the real HTTP surface against a real database.
 *
 * The three assertions the phase names, and one it implies:
 *
 *  1. every `Decision` row carries a propensity in (0, 1] — Study 3 cannot be
 *     re-run on a log that lacks one, and it cannot be backfilled;
 *  2. no explanation shown to a learner names a state the validation gate
 *     rejected;
 *  3. the roadmap advances — concepts move off zero as the session runs; and
 *  4. the loop completes whether or not the ML service is reachable, because a
 *     learner is never blocked on a model. When it is unreachable the decision
 *     is logged as the fallback policy, which is asserted rather than assumed.
 */
import { jest } from "@jest/globals";
import request from "supertest";
import app from "../app.js";
import prisma from "../config/db.js";
import { model_info } from "../services/ml.service.js";

const ITEMS = 20;
const REJECTED_STATES = ["engagement", "confidence", "fatigue"];

jest.setTimeout(180000);

const learner_id = `e2e_${Date.now()}`;
let session_id;
let sessionVersion;
let servedPolicy = null;   // which arm the ml-service says it is running
const served = [];
const decisionLatency = [];

describe("live adaptive loop", () => {
    test("the ml-service reports which policy it serves", async () => {
        const info = await model_info();
        servedPolicy = info?.serving?.policy ?? null;
        // Not an assertion: the loop must run without the service, and the
        // fallback assertions below are what covers that case.
        console.log(servedPolicy ? `ml-service serving ${servedPolicy}`
                                 : "ml-service unreachable — exercising the fallback path");
    });

    test("consent is recorded with its signal classes", async () => {
        const response = await request(app)
            .post("/v1/consent")
            .send({ learner_id, classes: { timing: true, interaction: true, motor: true, probes: true } })
            .expect(201);
        expect(response.body.consent_id).toBeTruthy();
        expect(response.body.classes.correctness).toBe(true);
    });

    test("a session materialises the canonical bank", async () => {
        const response = await request(app)
            .post("/v1/sessions")
            .send({ learner_id })
            .expect(201);
        session_id = response.body.session_id;
        sessionVersion = response.body.session_version;
        const questions = await prisma.question.count({ where: { session_id } });
        expect(questions).toBeGreaterThan(ITEMS);
    });

    test("twenty items are served, answered and logged", async () => {
        for (let index = 0; index < ITEMS; index += 1) {
            const started = Date.now();
            const next = await request(app)
                .get(`/v1/next-item?session_id=${session_id}`)
                .expect(200);
            decisionLatency.push(Date.now() - started);
            expect(next.body.item.item_id).toBeTruthy();
            expect(next.body.propensity).toBeGreaterThan(0);
            served.push(next.body);
            sessionVersion = next.body.session_version;

            // Synthetic telemetry: enough of the event stream that the
            // extractor produces a real feature row rather than an empty one.
            const presented_at = new Date().toISOString();
            const events = [
                { event_id: `${session_id}-${index}-p`, type: "ITEM_PRESENTED",
                  item_id: next.body.item.item_id, client_ts: presented_at, seq: index * 3 },
                { event_id: `${session_id}-${index}-f`, type: "FIRST_INTERACTION",
                  item_id: next.body.item.item_id, client_ts: new Date(Date.now() + 1200).toISOString(),
                  seq: index * 3 + 1 },
                { event_id: `${session_id}-${index}-s`, type: "OPTION_SELECTED",
                  item_id: next.body.item.item_id, client_ts: new Date(Date.now() + 2600).toISOString(),
                  seq: index * 3 + 2, payload: { option_index: index % 4 } }
            ];
            await request(app).post("/v1/events").send({ session_id, learner_id, events }).expect(202);

            const attempt = await request(app)
                .post("/v1/attempts")
                .send({
                    session_id,
                    item_id: next.body.item.item_id,
                    selected_index: index % 4,
                    presented_at,
                    response_time_ms: 8000 + index * 250,
                    session_version: sessionVersion
                })
                .expect(201);
            sessionVersion = attempt.body.session_version;
            expect(attempt.body).toHaveProperty("estimate");
        }

        await request(app).post("/v1/sessions/end").send({ session_id }).expect(200);
    });

    test("every logged decision carries a usable propensity", async () => {
        const decisions = await prisma.decision.findMany({ where: { session_id } });
        expect(decisions).toHaveLength(ITEMS);
        for (const decision of decisions) {
            expect(decision.propensity).toBeGreaterThan(0);
            expect(decision.propensity).toBeLessThanOrEqual(1);
            expect(decision.action).toHaveProperty("item_id");
        }
    });

    test("the served policy is the one the ml-service reports", async () => {
        const decisions = await prisma.decision.findMany({ where: { session_id } });
        const policies = new Set(decisions.map(decision => decision.policy_name));
        if (servedPolicy) {
            // Every decision came from the arm the service says it runs. A
            // fallback here would mean the loop silently degraded.
            expect([...policies]).toEqual([servedPolicy]);
        } else {
            expect([...policies]).toEqual(["rule_improved_fallback"]);
        }
    });

    test("the decision path stays inside its latency budget", () => {
        const sorted = [...decisionLatency].sort((a, b) => a - b);
        const p95 = sorted[Math.floor(sorted.length * 0.95)] ?? sorted.at(-1);
        console.log(`next-item latency: p50 ${sorted[Math.floor(sorted.length / 2)]} ms, p95 ${p95} ms`);
        // BUILD.md Phase 10: p95 under 300 ms. The budget is per item because
        // a session-level detector resolves too slowly to act on.
        expect(p95).toBeLessThan(300);
    });

    test("no learner-facing explanation names a gate-rejected state", async () => {
        const decisions = await prisma.decision.findMany({ where: { session_id } });
        const narratives = [
            ...served.map(item => item.explanation ?? ""),
            ...decisions.map(decision => decision.explanation?.narrative ?? "")
        ];
        for (const narrative of narratives) {
            for (const state of REJECTED_STATES) {
                expect(narrative.toLowerCase()).not.toContain(state);
            }
        }
    });

    test("state estimates record whether the gate validated them", async () => {
        const estimates = await prisma.stateEstimate.findMany({ where: { session_id } });
        expect(estimates).toHaveLength(ITEMS);
        for (const estimate of estimates) {
            // Whatever the estimator was, a rejected construct is never stored
            // as validated.
            expect(estimate.engagement_validated).toBe(false);
            expect(estimate.confidence_validated).toBe(false);
            expect(estimate.fatigue_validated).toBe(false);
        }
    });

    test("the roadmap advanced", async () => {
        const last = served.at(-1);
        const touched = (last.roadmap?.concepts ?? []).filter(concept => concept.status !== "locked");
        expect(touched.length).toBeGreaterThan(0);
        const attempts = await prisma.itemAttempt.count({ where: { session_id } });
        expect(attempts).toBe(ITEMS);
    });

    afterAll(async () => {
        await prisma.$disconnect();
    });
});
