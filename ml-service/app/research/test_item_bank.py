"""Phase 1 checks: bank invariants, graph acyclicity, Rasch recovery, binning."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from app.research import build_item_bank, validate_bank
from app.research.calibrate_items import assign_bins, fit_rasch

SEED = 20260821


@pytest.fixture(scope="module")
def built():
    return build_item_bank.build(SEED)


def test_bank_items_are_well_formed(built):
    graph = json.loads(build_item_bank.GRAPH_PATH.read_text(encoding="utf-8"))["concepts"]
    seen_ids, seen_stems = set(), set()
    for item in built["bank"]["items"]:
        assert item["item_id"] not in seen_ids
        seen_ids.add(item["item_id"])
        digest = build_item_bank.stem_hash(item["stem"])
        assert digest not in seen_stems, f"duplicate stem survived dedup: {item['item_id']}"
        seen_stems.add(digest)
        assert item["concept_id"] in graph
        assert item["prerequisite_concepts"] == list(graph[item["concept_id"]]["prerequisites"])
        assert len(item["options"]) == 4
        assert 0 <= item["correct_index"] < 4
        assert 1 <= item["author_difficulty"] <= 10


def test_build_is_deterministic(built):
    assert build_item_bank.build(SEED)["bank"] == built["bank"]


def test_out_of_scope_items_are_excluded_not_mislabelled(built):
    excluded = built["manifest"]["excluded"]
    assert excluded, "the biology bank has no CSE concept and must be excluded"
    assert all(entry["reason"].startswith("no concept") for entry in excluded)


def test_graph_is_acyclic_and_cycles_are_caught():
    graph = json.loads(build_item_bank.GRAPH_PATH.read_text(encoding="utf-8"))["concepts"]
    validate_bank.topological_order(graph)
    cyclic = {"a": {"prerequisites": ["b"]}, "b": {"prerequisites": ["a"]}}
    with pytest.raises(ValueError):
        validate_bank.topological_order(cyclic)


def test_rasch_recovers_known_difficulty():
    rng = np.random.default_rng(SEED)
    true_b = np.linspace(-2, 2, 20)
    theta = rng.normal(0, 1, 400)
    rows = []
    for learner, ability in enumerate(theta):
        probability = 1 / (1 + np.exp(-(ability - true_b)))
        correct = rng.random(len(true_b)) < probability
        for index, is_correct in enumerate(correct):
            rows.append((f"L{learner}", f"I{index:02d}", int(is_correct)))
    responses = pd.DataFrame(rows, columns=["learner_id", "item_id", "correct"])

    fitted = fit_rasch(responses, SEED).sort_values("item_id")
    assert spearmanr(fitted["b"], true_b).statistic > 0.9
    assert (fitted["n_responses"] == len(theta)).all()
    assert fitted["se_b"].notna().all()


def test_bins_span_the_full_scale_and_are_monotone():
    b = pd.Series(np.linspace(-3, 3, 50))
    bins, edges = assign_bins(b)
    assert len(edges) == 9
    assert bins.min() == 1 and bins.max() == 10
    assert (bins.diff().dropna() >= 0).all()
