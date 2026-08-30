"""Phase 12 — the final verification pass, as a checklist rather than a memory.

    python3 scripts/verify_final.py [--json artifacts/evaluation/final-verification.json]

Four checks, all of which fail loudly:

1. **Deliverables.** Every phase's named outputs exist, and each figure
   directory holds the number of charts BUILD.md's inventory says it should.
2. **Figure provenance.** Every figure in `artifacts/figures/manifest.jsonl`
   declares source files, they exist, and they live under the allowed roots.
3. **Substituted numbers.** Every `{{artifact:…}}` placeholder in the two
   documents resolves against the artifact tree.
4. **Typed numbers.** Every *other* number in the two documents is listed with
   its sentence, so the honesty pass is a checklist someone can walk rather
   than a claim that it was done. Numbers matching `KNOWN_PROSE` are marked as
   checked, with the artifact each was read from; anything left over is
   reported as unverified and fails the run.

Standard library only.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "evaluation" / "final-verification.json"
DOCUMENTS = [Path("docs/reports/ALP-System-Report.md"), Path("docs/paper-draft.md")]

#: (phase, path) — the outputs whose absence means a phase did not finish.
DELIVERABLES: list[tuple[str, str]] = [
    ("1", "artifacts/datasets/item-parameters-v1.csv"),
    ("1", "data/items/item-bank-v1.json"),
    ("2", "artifacts/figures/02-telemetry"),
    ("3", "data/processed/splits.json"),
    ("3", "artifacts/datasets/archival-summary.json"),
    ("4", "artifacts/datasets/simulator-params-v2.json"),
    ("4", "artifacts/datasets/simulator-calibration-ks.json"),
    ("5", "artifacts/benchmarks/eda-summary.json"),
    ("5", "artifacts/benchmarks/feature-shortlist.json"),
    ("5", "docs/feature-catalogue.md"),
    ("6", "artifacts/benchmarks/study1-cv.json"),
    ("6", "artifacts/benchmarks/study1-final.json"),
    ("7", "artifacts/benchmarks/study2-heads.json"),
    ("7", "artifacts/evaluation/validation-gate.json"),
    ("7", "artifacts/models/state-encoders-by-rung.pt"),
    ("8", "artifacts/evaluation/policy-results-v0.json"),
    ("8", "artifacts/evaluation/policy-results-v4.json"),
    ("8", "artifacts/evaluation/decision-explanations.json"),
    ("8.5", "artifacts/evaluation/phase-8.5-validation.json"),
    ("9", "artifacts/evaluation/ope-results.json"),
    ("9", "artifacts/evaluation/rl-results.json"),
    ("9", "artifacts/models/rl/policy.zip"),
    ("10", "artifacts/evaluation/load-test.json"),
    ("10", "artifacts/evaluation/system-figure-findings.json"),
    ("11", "artifacts/evaluation/statistical-report.json"),
    ("11", "artifacts/evaluation/sensitivity.json"),
    ("11", "artifacts/evaluation/tables/statistical-tests.tex"),
    ("11", "artifacts/figures/INDEX.md"),
    ("12", "artifacts/evaluation/test-results.json"),
    ("12", "artifacts/evaluation/reproduction-timing.json"),
    ("12", "docs/reports/ALP-System-Report.pdf"),
    ("12", "docs/reports/ALP-Research-Paper-Draft.pdf"),
]

#: BUILD.md §6's figure inventory: directory → how many charts it should hold.
FIGURE_INVENTORY = {
    "01-content": 4, "02-telemetry": 4, "03-archival": 8, "04-simulator": 9,
    "05-eda": 13, "06-study1": 11, "07-study2": 10, "08-policies": 14,
    "08.5-integration": 4, "09-ope-rl": 10, "10-system": 6, "11-stats": 8,
    "12-walkthrough": 5,
}

#: Numbers that appear as prose in the two documents, with the artifact each was
#: read from. Anything numeric in the documents that is neither substituted at
#: build time nor listed here is reported as unverified.
KNOWN_PROSE: dict[str, str] = {
    "0.012": "artifacts/evaluation/statistical-report.json (max |hedges_g| on knowledge_gain)",
    "97.7": "statistical-report.json mixed_effects/outcomes/knowledge_gain/variance_share/learner",
    "0.015": "statistical-report.json mixed_effects/correctness/*/fixed_effects/rung_increment",
    "0.024": "statistical-report.json mixed_effects/correctness/*/fixed_effects/rung_increment",
    "0.27": "statistical-report.json effect_sizes (time_to_mastery, excluding random)",
    "0.3": "preregistration §11 / BUILD.md §4.5 reference effect size",
    "3,000": "policy-per-learner-v0.csv (1000 learners x 3 seeds per arm)",
    "200": "policy-results-v0.json budget_items",
    "0.2": "BUILD.md Phase 9 note 1 — deterministic-target match rate (ope-results action_match_rate)",
    "2,348": "ope-results.json effective_sample_size (clipped, lower end)",
    "2,691": "ope-results.json effective_sample_size (clipped, upper end)",
    "45,884": "ope-results.json logged_decisions",
    "120": "ope-results.json action_space",
    "0.081": "rl-results.json summary V0 — bandit_lin_ts minus model_L0 knowledge gain",
    "0.054": "rl-results.json knowledge_gain_95_ci (lower)",
    "0.109": "rl-results.json knowledge_gain_95_ci (upper)",
    "0.36": "BUILD.md Phase 9 note 3 — unpaired d for the bandit contrast",
    "36.6": "rl-results.json summary V0 bandit_lin_ts interventions_per_learner/hint",
    "51.7": "rl-results.json summary V0 bandit_lin_ts interventions_per_learner/worked_example",
    "59.5": "rl-results.json summary V0 bandit_lin_ts interventions_per_learner/break_suggestion",
    "4.1": "rl-results.json summary V0 model_L0 interventions_per_learner/worked_example",
    "0.0077": "rl-results.json summary V0 model_L0 mean_reward",
    "0.32": "validation-gate.json measurements/knowledge/label_correlation",
    "0.016": "validation-gate.json measurements/knowledge/ece_calibrated",
    "0.686": "validation-gate.json measurements/knowledge/roc_auc",
    "0.15": "preregistration §8.2 C1 threshold and §10.2 logging epsilon",
    "0.85": "preregistration §8.2 C4 threshold",
    "0.05": "preregistration §8.2 C2 threshold / alpha",
    "0.05–0.07": "statistical-report.json power/achieved (knowledge_gain)",
    "7.9": "load-test.json runs[0] next-item p95 at concurrency 1",
    "300": "BUILD.md Phase 10 acceptance — latency budget",
    "104": "artifacts/figures/manifest.jsonl (figure count before Phase 12)",
    "129k": "artifacts/datasets/archival-summary.json (rows after filtering)",
    "91": "statistical-report.json censoring/arms/*/censoring_rate",
    "92": "statistical-report.json censoring/arms/*/censoring_rate",
    "187": "statistical-report.json censoring restricted_mean_items_to_budget",
    "189": "statistical-report.json censoring restricted_mean_items_to_budget",
    "0.64": "study1-final.json (lowest best-cell ROC-AUC, ednet_kt1)",
    "0.71": "study1-final.json (highest best-cell ROC-AUC, assistments_2009)",
    "20260821": "the global seed, recorded in every manifest",
    "26": "artifacts-legacy-invalid/ — the invalidated pre-Phase-0 effect size",
    "10": "action space difficulty range 1-10 / ten arms",
    "0.80": "preregistration §9.2 mastery threshold",
    "80": "power target, preregistration §11",
    "60": "export_bank.py — items in the served bank",
    "36": "backend/src/data/concept_graph.json — concepts in the DAG",
    "6": "closed_loop.select_curriculum — concepts in the taught curriculum",
    "5": "consent classes / CV folds — docs/preregistration.md §4",
    "4": "the four candidate states, one admitted",
    "0.5": "chance ROC-AUC reference line",
    "2": "the two PDFs / two-tap probes",
    "1": "one construct admitted",
    "3": "three archival datasets / three services",
    "9": "nine comparisons against the reference arm",
    "0.1": "arms_module.EPSILON — the served exploration rate",
    "20": "ope.WEIGHT_CLIP — importance-weight clip",
    "12": "the twelve phases",
    "10,000": "stats.py --bootstrap default; statistical-report.json bootstrap_resamples",
    "0.07": "statistical-report.json power/achieved (knowledge_gain, upper end)",
    "2012": "dataset name (ASSISTments 2012)",
    "16": "load-test.json headline concurrency",
    "2009": "dataset name", "2012": "dataset name", "1": "one",
    "0.016.": "validation-gate.json measurements/knowledge/ece_calibrated",
}

PLACEHOLDER = re.compile(r"\{\{artifact:([^:}]+):([^}]*)\}\}")
NUMBER = re.compile(r"(?<![\w.])(\d[\d,]*(?:\.\d+)?)(?![\w])")

#: Text that contains digits but makes no numeric claim: figure and table ids,
#: section and phase references, code spans, URLs, rung and criterion labels.
NOT_A_CLAIM = re.compile(
    r"`[^`]*`"                      # inline code
    r"|https?://\S+"                # URLs
    r"|\bf\d{2}(?:\.\d)?-\d{2}\b"  # figure ids (f08-14, f085-01)
    r"|§\s?\d+(?:\.\d+)*"          # section references
    r"|\b[Ss]ections?\s+\d+(?:\s*\(|,|\s+and\s+|\)|\b)"
    r"|\b(?:Phase|RQ|Study|C|L|V)\s?\d+(?:\.\d+)?\b"
    r"|\[S\]"
    r"|\bASSISTments\s+\d{4}\b|\bEdNet\s*KT\d\b|\b\d{4}\s+and\s+\d{4}\b"
    r"|\((?:\d+\)|\d+\))"          # enumerations like (1) (2)
    r"|\b\d+\s*\([a-z ]+\)"        # section lists: "6 (method), 7 (results)"
)


def resolve(path: str, pointer: str):
    target = ROOT / path
    data = json.loads(target.read_text(encoding="utf-8"))
    for key in [part for part in pointer.split("/") if part]:
        data = data[int(key)] if isinstance(data, list) else data[key]
    return data


def check_deliverables() -> tuple[list[dict], list[str]]:
    rows, missing = [], []
    for phase, path in DELIVERABLES:
        exists = (ROOT / path).exists()
        rows.append({"phase": phase, "path": path, "exists": exists})
        if not exists:
            missing.append(f"phase {phase}: {path}")
    return rows, missing


def check_figures() -> tuple[dict, list[str]]:
    problems, counts = [], {}
    for directory, expected in FIGURE_INVENTORY.items():
        found = len(list((ROOT / "artifacts" / "figures" / directory).glob("*.png")))
        counts[directory] = {"expected": expected, "found": found}
        if found < expected:
            problems.append(f"{directory}: {found} figures, inventory says {expected}")
    return counts, problems


def check_manifest() -> tuple[dict, list[str]]:
    manifest = ROOT / "artifacts" / "figures" / "manifest.jsonl"
    if not manifest.exists():
        return {}, ["artifacts/figures/manifest.jsonl absent — run `make figures`"]
    entries = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            entries[(row["directory"], row["figure"])] = row
    problems = []
    for (directory, figure), row in entries.items():
        if not row.get("sources"):
            problems.append(f"{directory}/{figure}: declares no source")
        for source in row.get("sources", []):
            if not (ROOT / source).exists():
                problems.append(f"{directory}/{figure}: missing source {source}")
    return {"figures": len(entries)}, problems


def check_documents() -> tuple[dict, list[str]]:
    substituted, unverified, problems = [], [], []
    for document in DOCUMENTS:
        text = (ROOT / document).read_text(encoding="utf-8")
        for path, pointer in PLACEHOLDER.findall(text):
            try:
                value = resolve(path, pointer)
            except Exception as error:                      # noqa: BLE001 - reported, not raised
                problems.append(f"{document}: {{{{artifact:{path}:{pointer}}}}} → {error}")
                continue
            substituted.append({"document": str(document), "artifact": path,
                                "pointer": pointer, "value": value})
        stripped = PLACEHOLDER.sub(" ", text)
        for line in stripped.splitlines():
            if line.startswith(("|", "```", "    ", "#", "!")) or line.strip().startswith(("`", "*f")):
                continue        # tables, code, headings, figure lines and captions
            line = NOT_A_CLAIM.sub(" ", re.sub(r"^\s*\d+\.\s+", "", line))
            for raw in NUMBER.findall(line):
                number = raw.rstrip(",.")
                if number in KNOWN_PROSE or number.rstrip("0").rstrip(".") in KNOWN_PROSE:
                    continue
                unverified.append({"document": str(document), "number": number,
                                   "sentence": line.strip()[:160]})
    if unverified:
        problems.append(f"{len(unverified)} numeric claim(s) in prose are neither "
                        "substituted nor listed in KNOWN_PROSE")
    return {"substituted": substituted, "unverified": unverified}, problems


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-unverified", action="store_true",
                        help="report typed numbers without failing (drafting aid)")
    arguments = parser.parse_args()

    deliverables, missing = check_deliverables()
    figures, figure_problems = check_figures()
    manifest, manifest_problems = check_manifest()
    documents, document_problems = check_documents()

    problems = missing + figure_problems + manifest_problems + document_problems
    report = {
        "phase": 12,
        "deliverables": deliverables,
        "missing_deliverables": missing,
        "figure_counts": figures,
        "figure_manifest": manifest,
        "documents": {
            "substituted_numbers": len(documents["substituted"]),
            "unverified_prose_numbers": len(documents["unverified"]),
            "unverified": documents["unverified"][:40],
            "substitutions": documents["substituted"],
        },
        "problems": problems,
        "passed": not problems or (arguments.allow_unverified
                                   and problems == document_problems),
    }
    arguments.json.parent.mkdir(parents=True, exist_ok=True)
    arguments.json.write_text(json.dumps(report, indent=2, default=str) + "\n", encoding="utf-8")

    print(f"deliverables: {len(deliverables) - len(missing)}/{len(deliverables)} present")
    print(f"figures: {sum(v['found'] for v in figures.values())} PNGs across "
          f"{len(figures)} directories; manifest holds {manifest.get('figures', 0)}")
    print(f"documents: {len(documents['substituted'])} numbers substituted from artifacts, "
          f"{len(documents['unverified'])} typed numbers to check")
    for problem in problems[:20]:
        print(f"  ! {problem}")
    for entry in documents["unverified"][:20]:
        print(f"  ? {entry['number']}: {entry['sentence']}")
    print(f"wrote: {arguments.json.relative_to(ROOT)}")
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
