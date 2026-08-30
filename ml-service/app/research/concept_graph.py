"""Small, versioned prerequisite graph used by synthetic experiments."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Concept:
    concept_id: str
    prerequisites: tuple[str, ...]
    level: int
    subject: str
    name: str

_DATA_PATH = Path(__file__).resolve().parent.parent.parent.parent / "backend" / "src" / "data" / "concept_graph.json"

with open(_DATA_PATH, "r", encoding="utf-8") as f:
    _data = json.load(f)

CONCEPT_GRAPH_VERSION = "cse-prerequisite-graph-v1"

_concepts_list = []
for cid, info in _data["concepts"].items():
    _concepts_list.append(
        Concept(
            concept_id=cid,
            prerequisites=tuple(info["prerequisites"]),
            level=info["level"],
            subject=info["subject"],
            name=info["name"]
        )
    )

CONCEPTS = tuple(_concepts_list)

def eligible_concepts(mastery: dict[str, float], threshold: float = .70) -> list[Concept]:
    return [concept for concept in CONCEPTS if all(mastery.get(item, 0) >= threshold for item in concept.prerequisites)]


def next_roadmap_item(mastery: dict[str, float]) -> dict:
    eligible = eligible_concepts(mastery)
    if not eligible:
        # fallback if all are > threshold somehow or no eligible
        eligible = CONCEPTS
    target = min(eligible, key=lambda concept: mastery.get(concept.concept_id, 0))
    return {
        "graph_version": CONCEPT_GRAPH_VERSION,
        "concept_id": target.concept_id,
        "prerequisites": list(target.prerequisites),
        "mastery": round(mastery.get(target.concept_id, 0), 6),
        "reason": "lowest_mastery_eligible_concept",
    }
