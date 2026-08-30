/**
 * The researcher dashboard: one server-rendered page over the tables the
 * research questions are answered from.
 *
 * No framework and no client-side charting library. The page reads Postgres,
 * renders HTML and draws its trajectories as inline SVG, which is the whole
 * requirement — a second frontend stack to display four queries would be more
 * to maintain than the queries.
 *
 * It is read-only and it never shows an answer key.
 */
import prisma from "../config/db.js";
import { model_info } from "../services/ml.service.js";

const escape = (value) => String(value ?? "")
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");

const percent = (value) => `${Math.round((value ?? 0) * 100)}%`;

/** A knowledge trajectory as an inline sparkline. */
const sparkline = (values, width = 220, height = 34) => {
    if (values.length < 2) return '<span class="muted">not enough points</span>';
    const step = width / (values.length - 1);
    const points = values
        .map((value, index) => `${(index * step).toFixed(1)},${(height - value * height).toFixed(1)}`)
        .join(" ");
    return `<svg width="${width}" height="${height}" viewBox="0 0 ${width} ${height}" role="img"
                 aria-label="knowledge estimate over the session">
              <polyline points="${points}" fill="none" stroke="#4c8bf5" stroke-width="2" />
            </svg>`;
};

export const get_dashboard = async (req, res) => {
    const limit = Math.min(50, Number(req.query?.limit ?? 20));
    const sessions = await prisma.session.findMany({
        orderBy: { started_at: "desc" },
        take: limit,
        include: {
            _count: { select: { attempts: true, events: true, decisions: true } },
            assignment: true
        }
    });
    const ids = sessions.map(session => session.session_id);

    const estimates = await prisma.stateEstimate.findMany({
        where: { session_id: { in: ids } },
        orderBy: { createdAt: "asc" },
        select: { session_id: true, knowledge: true, knowledge_se: true, knowledge_validated: true, estimator: true }
    });
    // One session's whole log when asked for it (the Phase 10 trace figure);
    // the latest hundred across sessions otherwise (the page).
    const focus = req.query?.session_id ? String(req.query.session_id) : null;
    const decisions = await prisma.decision.findMany({
        where: { session_id: focus ? focus : { in: ids } },
        orderBy: { createdAt: focus ? "asc" : "desc" },
        take: focus ? 500 : 100
    });
    const probes = await prisma.probe.findMany({
        where: { session_id: { in: ids } },
        select: { kind: true, answered_at: true, skipped: true }
    });
    const serving = await model_info();

    if (req.query?.format === "json") {
        // The Phase 10 figure script reads the live session trace from here
        // rather than opening a second database connection from Python.
        return res.json({
            serving: serving?.serving ?? null,
            sessions: sessions.map(session => ({
                session_id: session.session_id,
                policy: session.assignment?.policy_name ?? session.source,
                started_at: session.started_at,
                attempts: session._count.attempts,
                decisions: session._count.decisions,
                estimates: estimates.filter(row => row.session_id === session.session_id)
            })),
            decisions: decisions.map(decision => ({
                session_id: decision.session_id, policy: decision.policy_name,
                action: decision.action, propensity: decision.propensity,
                created_at: decision.createdAt, explanation: decision.explanation
            })),
            probes
        });
    }

    const trajectory = new Map();
    for (const estimate of estimates) {
        const points = trajectory.get(estimate.session_id) ?? [];
        points.push(estimate);
        trajectory.set(estimate.session_id, points);
    }

    const probeRates = {};
    for (const probe of probes) {
        const row = probeRates[probe.kind] ?? { shown: 0, answered: 0, skipped: 0 };
        row.shown += 1;
        if (probe.answered_at && !probe.skipped) row.answered += 1;
        if (probe.skipped) row.skipped += 1;
        probeRates[probe.kind] = row;
    }

    const sessionRows = sessions.map(session => {
        const points = trajectory.get(session.session_id) ?? [];
        const last = points.at(-1);
        return `<tr>
            <td><code>${escape(session.session_id.slice(0, 8))}</code></td>
            <td>${escape(session.assignment?.policy_name ?? session.source)}</td>
            <td>${session._count.attempts}</td>
            <td>${session._count.decisions}</td>
            <td>${session._count.events}</td>
            <td>${last ? percent(last.knowledge) : "—"}
                ${last?.knowledge_se != null ? `<span class="muted">± ${Math.round(last.knowledge_se * 100)}</span>` : ""}
                ${last && !last.knowledge_validated ? '<span class="warn">unvalidated</span>' : ""}</td>
            <td>${sparkline(points.map(point => point.knowledge))}</td>
            <td>${escape(last?.estimator ?? "—")}</td>
        </tr>`;
    }).join("\n");

    const decisionRows = decisions.map(decision => `<tr>
            <td>${escape(new Date(decision.createdAt).toISOString().slice(11, 19))}</td>
            <td><code>${escape(decision.session_id.slice(0, 8))}</code></td>
            <td>${escape(decision.policy_name)}</td>
            <td>${escape(decision.action?.item_id)}</td>
            <td>${escape(decision.action?.difficulty)}</td>
            <td>${escape(decision.action?.intervention ?? "—")}</td>
            <td>${decision.propensity.toFixed(4)}</td>
            <td class="muted">${escape(decision.explanation?.narrative ?? "")}</td>
        </tr>`).join("\n");

    const probeRows = Object.entries(probeRates).map(([kind, row]) => `<tr>
            <td>${escape(kind)}</td><td>${row.shown}</td><td>${row.answered}</td>
            <td>${row.skipped}</td><td>${percent(row.shown ? row.answered / row.shown : 0)}</td>
        </tr>`).join("\n");

    res.type("html").send(`<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<title>ALP — researcher dashboard</title>
<style>
 body { font: 14px/1.5 system-ui, sans-serif; margin: 2rem; color: #16202c; }
 h1 { font-size: 1.4rem; } h2 { font-size: 1.05rem; margin-top: 2rem; }
 table { border-collapse: collapse; width: 100%; margin-top: 0.5rem; }
 th, td { text-align: left; padding: 0.35rem 0.6rem; border-bottom: 1px solid #e4e8ef; vertical-align: middle; }
 th { font-weight: 600; background: #f6f8fb; }
 code { font-size: 0.85em; } .muted { opacity: 0.6; } .warn { color: #b45309; font-size: 0.8em; }
 .serving { background: #f6f8fb; border: 1px solid #e4e8ef; border-radius: 8px; padding: 0.75rem 1rem; }
</style></head><body>
<h1>Adaptive Learning Platform — researcher dashboard</h1>
<p class="muted">Read-only. Every number comes from the tables the studies are run on.</p>

<div class="serving">
  <strong>Serving:</strong> ${escape(serving?.serving?.policy ?? "ml-service unreachable")}
  · estimator rung ${escape(serving?.serving?.estimator_rung ?? "—")}
  · admitted states: ${escape((serving?.serving?.gate?.admitted ?? []).join(", ") || "none")}
  · live slots in use ${escape(serving?.serving?.slots_in_use ?? "—")}/${escape(serving?.serving?.live_slots ?? "—")}
</div>

<h2>Sessions</h2>
<table><thead><tr><th>session</th><th>policy</th><th>attempts</th><th>decisions</th><th>events</th>
<th>knowledge</th><th>trajectory</th><th>estimator</th></tr></thead>
<tbody>${sessionRows || '<tr><td colspan="8" class="muted">no sessions yet</td></tr>'}</tbody></table>

<h2>Decision log (latest 100)</h2>
<table><thead><tr><th>time</th><th>session</th><th>policy</th><th>item</th><th>difficulty</th>
<th>intervention</th><th>propensity</th><th>why</th></tr></thead>
<tbody>${decisionRows || '<tr><td colspan="8" class="muted">no decisions yet</td></tr>'}</tbody></table>

<h2>Probe response rates</h2>
<table><thead><tr><th>kind</th><th>shown</th><th>answered</th><th>skipped</th><th>response rate</th></tr></thead>
<tbody>${probeRows || '<tr><td colspan="5" class="muted">no probes yet</td></tr>'}</tbody></table>
</body></html>`);
};
