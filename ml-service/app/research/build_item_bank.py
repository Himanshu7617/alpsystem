"""Consolidate every question source into one canonical, versioned item bank.

Sources are merged in a fixed order; the first item to claim a normalised stem
hash wins, so the output is deterministic and re-running the script is a no-op.

    python -m app.research.build_item_bank

Writes:
    data/items/item-bank-v1.json
    data/items/item-bank-v1.manifest.json
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
QUESTION_DIR = ROOT / "backend" / "src" / "data" / "questions"
GRAPH_PATH = ROOT / "backend" / "src" / "data" / "concept_graph.json"
OUT_DIR = ROOT / "data" / "items"
BANK_PATH = OUT_DIR / "item-bank-v1.json"
MANIFEST_PATH = OUT_DIR / "item-bank-v1.manifest.json"

BANK_VERSION = "item-bank-v1"

# Merge order. Earlier sources win a stem-hash collision, so the richest and
# most authoritative source is listed first: multitopic_cse.json carries the
# same 126 items as question_bank.csv but with concepts, tags and objectives.
SOURCES = [
    ("multitopic_cse.json", "json", None),
    ("question_bank.csv", "csv", None),
    ("binary_trees.json", "json", "trees_and_bst"),
    ("recursion.json", "json", "divide_and_conquer"),
    ("transportability_among_animals.json", "json", None),
]

# The topical banks label items with free text rather than graph concept ids.
# First matching keyword wins; anything unmatched falls back to the source's
# topic default above. A source with no topic default and no keyword match is
# out of scope for the CSE graph and is excluded, not silently mislabelled.
# ponytail: keyword table, not a classifier. 30 items do not justify one.
KEYWORD_CONCEPTS = [
    ("memoization", "dynamic_programming"),
    ("dynamic programming", "dynamic_programming"),
    ("backtracking", "advanced_algorithms"),
    ("quick sort", "sorting_searching"),
    ("dfs", "graph_algorithms"),
    ("graph algorithms", "graph_algorithms"),
    ("time complexity", "complexity_analysis"),
    ("space complexity", "complexity_analysis"),
]

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[^a-z0-9 ]+")

# Namespace prefixes: the topical banks all number their items q1..q30.
ID_PREFIX = {
    "binary_trees.json": "bt",
    "recursion.json": "rec",
    "transportability_among_animals.json": "tra",
}


def normalise_stem(stem: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace. Used only for dedup."""
    return _WS.sub(" ", _PUNCT.sub(" ", stem.lower())).strip()


def stem_hash(stem: str) -> str:
    return hashlib.sha256(normalise_stem(stem).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def concept_for(raw: dict, source: str, topic_default: str | None, q2c: dict) -> str | None:
    """Resolve an item to a graph concept id: explicit map, keyword, then default."""
    mapped = q2c.get(raw["id"])
    if mapped:
        return mapped
    haystack = " ".join(raw.get("concepts", []) + raw.get("tags", [])).lower()
    for keyword, concept_id in KEYWORD_CONCEPTS:
        if keyword in haystack:
            return concept_id
    return topic_default


def read_json_source(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["questions"]


def read_csv_source(path: Path) -> list[dict]:
    """question_bank.csv, reshaped into the same dict shape as the JSON banks."""
    with path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return [
        {
            "id": row["id"],
            "question": row["question"],
            "options": [row["option_a"], row["option_b"], row["option_c"], row["option_d"]],
            "correctAnswer": row["correct_answer"],
            "explanation": row["explanation"],
            "difficultyScore": int(row["difficulty_score"]),
            "questionType": "mcq",
            "concepts": [row["topics"]] if row.get("topics") else [],
            "tags": [],
        }
        for row in rows
    ]


def build(seed: int) -> dict:
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    concepts = graph["concepts"]
    q2c = graph.get("question_to_concept", {})

    items: list[dict] = []
    seen: dict[str, str] = {}
    duplicates: list[dict] = []
    excluded: list[dict] = []
    per_source: dict[str, int] = {}

    for filename, kind, topic_default in SOURCES:
        path = QUESTION_DIR / filename
        raws = read_json_source(path) if kind == "json" else read_csv_source(path)
        kept = 0
        for raw in raws:
            digest = stem_hash(raw["question"])
            item_id = f"{ID_PREFIX[filename]}-{raw['id']}" if filename in ID_PREFIX else raw["id"]

            if digest in seen:
                duplicates.append({"item_id": item_id, "source": filename, "duplicate_of": seen[digest]})
                continue

            concept_id = concept_for(raw, filename, topic_default, q2c)
            if concept_id is None or concept_id not in concepts:
                excluded.append(
                    {
                        "item_id": item_id,
                        "source": filename,
                        "reason": "no concept in " + graph["version"],
                    }
                )
                continue

            options = list(raw["options"])
            if raw["correctAnswer"] not in options:
                excluded.append(
                    {"item_id": item_id, "source": filename, "reason": "correct answer not among options"}
                )
                continue

            seen[digest] = item_id
            kept += 1
            items.append(
                {
                    "item_id": item_id,
                    "concept_id": concept_id,
                    "subject": concepts[concept_id]["subject"],
                    "stem": raw["question"],
                    "options": options,
                    "correct_index": options.index(raw["correctAnswer"]),
                    "explanation": raw.get("explanation", ""),
                    "author_difficulty": int(raw["difficultyScore"]),
                    "prerequisite_concepts": list(concepts[concept_id]["prerequisites"]),
                    "format": raw.get("questionType", "mcq"),
                }
            )
        per_source[filename] = kept

    bank = {
        "version": BANK_VERSION,
        "graph_version": graph["version"],
        "seed": seed,
        "items": items,
    }
    coverage: dict[str, int] = {cid: 0 for cid in concepts}
    for item in items:
        coverage[item["concept_id"]] += 1

    manifest = {
        "version": BANK_VERSION,
        "graph_version": graph["version"],
        "seed": seed,
        "bank_file": str(BANK_PATH.relative_to(ROOT)),
        "item_count": len(items),
        "sources": per_source,
        "duplicates_dropped": duplicates,
        "excluded": excluded,
        "concept_coverage": coverage,
        "keyword_concept_rules": [list(rule) for rule in KEYWORD_CONCEPTS],
        "topic_defaults": {name: default for name, _, default in SOURCES if default},
    }
    return {"bank": bank, "manifest": manifest}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    result = build(args.seed)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    BANK_PATH.write_text(json.dumps(result["bank"], indent=2) + "\n", encoding="utf-8")

    manifest = result["manifest"]
    manifest["bank_sha256"] = sha256_file(BANK_PATH)
    # Preserve fields written by later phases (calibrate_items adds the bins).
    if MANIFEST_PATH.exists():
        existing = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        for key in ("difficulty_bins", "calibration"):
            if key in existing:
                manifest[key] = existing[key]
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote: {BANK_PATH.relative_to(ROOT)} ({manifest['item_count']} items)")
    print(f"wrote: {MANIFEST_PATH.relative_to(ROOT)} (sha256 {manifest['bank_sha256'][:12]}…)")
    print(f"dropped: {len(manifest['duplicates_dropped'])} duplicate stems")
    print(f"excluded: {len(manifest['excluded'])} items with no concept in the graph")


if __name__ == "__main__":
    main()
