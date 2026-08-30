"""Validate the item bank against the prerequisite graph.

    python -m app.research.validate_bank

Hard failures (exit 1): a cyclic graph, an item whose concept is not in the
graph, a malformed item. Concept coverage below the 5-item floor is reported,
not fatal — the bank is what it is and hiding the shortfall is worse than
printing it, because mastery estimation on a 1-item concept is meaningless.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
BANK_PATH = ROOT / "data" / "items" / "item-bank-v1.json"
GRAPH_PATH = ROOT / "backend" / "src" / "data" / "concept_graph.json"

MIN_ITEMS_PER_CONCEPT = 5


def topological_order(concepts: dict) -> list[str]:
    """Kahn's algorithm. Raises ValueError naming the concepts left in a cycle."""
    indegree = {cid: len(info["prerequisites"]) for cid, info in concepts.items()}
    dependents: dict[str, list[str]] = {cid: [] for cid in concepts}
    for cid, info in concepts.items():
        for prereq in info["prerequisites"]:
            dependents[prereq].append(cid)

    queue = [cid for cid, deg in indegree.items() if deg == 0]
    order = []
    while queue:
        cid = queue.pop()
        order.append(cid)
        for dependent in dependents[cid]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                queue.append(dependent)
    if len(order) != len(concepts):
        raise ValueError("prerequisite graph is cyclic: " + ", ".join(sorted(set(concepts) - set(order))))
    return order


def validate() -> int:
    graph = json.loads(GRAPH_PATH.read_text(encoding="utf-8"))
    concepts = graph["concepts"]
    bank = json.loads(BANK_PATH.read_text(encoding="utf-8"))
    items = bank["items"]
    errors: list[str] = []

    for cid, info in concepts.items():
        for prereq in info["prerequisites"]:
            if prereq not in concepts:
                errors.append(f"concept {cid} names unknown prerequisite {prereq}")
    if errors:
        for message in errors:
            print("FAIL:", message)
        return 1

    topological_order(concepts)  # raises on a cycle
    print(f"ok: {graph['version']} is acyclic ({len(concepts)} concepts)")

    seen_ids: set[str] = set()
    coverage = {cid: 0 for cid in concepts}
    for item in items:
        if item["item_id"] in seen_ids:
            errors.append(f"duplicate item_id {item['item_id']}")
        seen_ids.add(item["item_id"])
        if item["concept_id"] not in concepts:
            errors.append(f"item {item['item_id']} has concept {item['concept_id']} not in the graph")
            continue
        coverage[item["concept_id"]] += 1
        if len(item["options"]) != 4:
            errors.append(f"item {item['item_id']} has {len(item['options'])} options, expected 4")
        if not 0 <= item["correct_index"] < len(item["options"]):
            errors.append(f"item {item['item_id']} has correct_index out of range")
        if not 1 <= item["author_difficulty"] <= 10:
            errors.append(f"item {item['item_id']} has author_difficulty out of 1..10")

    if not errors:
        print(f"ok: {len(items)} items, every concept_id resolves")

    width = max(len(cid) for cid in concepts)
    print()
    print(f"{'concept':{width}}  {'subject':20}  {'lvl':>3}  {'items':>5}  status")
    short = 0
    for cid in sorted(concepts, key=lambda c: (concepts[c]["subject"], concepts[c]["level"], c)):
        count = coverage[cid]
        status = "ok" if count >= MIN_ITEMS_PER_CONCEPT else f"SHORT (< {MIN_ITEMS_PER_CONCEPT})"
        short += count < MIN_ITEMS_PER_CONCEPT
        print(f"{cid:{width}}  {concepts[cid]['subject']:20}  {concepts[cid]['level']:>3}  {count:>5}  {status}")
    print()
    print(f"total items {len(items)}; concepts below the {MIN_ITEMS_PER_CONCEPT}-item floor: {short}/{len(concepts)}")
    if short:
        print("WARNING: mastery estimates on the short concepts are not trustworthy. Authoring more items is the fix.")

    if errors:
        print()
        for message in errors:
            print("FAIL:", message)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(validate())
