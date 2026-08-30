"""Checks on the Phase 3 transforms that a wrong result would slip past silently.

The loaders need the multi-gigabyte archival downloads; these run on tiny frames
built here, so the logic is covered without the data being present.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from app.research.prep_archival import (
    N_FOLDS,
    drop_short_learners,
    make_splits,
    response_time_effort,
    winsorise_response_time,
)


def frame(learners: list[str], items: list[str], response_times: list[float]) -> pd.DataFrame:
    return pd.DataFrame({
        "learner_id": learners,
        "item_id": items,
        "response_time_ms_raw": response_times,
        "correct": [1] * len(learners),
    })


def test_short_learners_are_dropped_whole():
    data = frame(["a"] * 12 + ["b"] * 3, ["i"] * 15, [1000.0] * 15)
    kept, notes = drop_short_learners(data, minimum=10)
    assert set(kept["learner_id"]) == {"a"}
    assert notes["rows_dropped_short_learners"] == 3
    assert notes["learners_dropped_short"] == 1


def test_winsorisation_clips_only_the_tails_and_counts_them():
    times = [1.0] + [1000.0] * 98 + [10_000_000.0]
    data = frame(["a"] * 100, ["i"] * 100, times)
    clipped, notes = winsorise_response_time(data)
    assert clipped.max() < 10_000_000.0
    assert clipped.min() > 1.0
    assert notes["rows_winsorised"] == 2
    assert clipped.iloc[1:99].tolist() == [1000.0] * 98


def test_thin_items_fall_back_to_source_level_bounds():
    # One item with 100 responses sets the source bounds; a 3-response item
    # cannot estimate its own 1st percentile and must borrow them.
    times = [1000.0] * 100 + [5.0, 1000.0, 1000.0]
    data = frame(["a"] * 103, ["big"] * 100 + ["thin"] * 3, times)
    _, notes = winsorise_response_time(data)
    assert notes["items_using_source_level_bounds"] == 1


def test_rapid_guesses_are_flagged_and_rte_is_per_learner():
    # Item mean is ~5 s, so the normative threshold is the 1 s floor.
    data = frame(["a"] * 8 + ["b"] * 2, ["i"] * 10, [5000.0] * 8 + [100.0, 5000.0])
    solution, learner_rte, notes = response_time_effort(data)
    assert solution.tolist() == [1] * 8 + [0, 1]
    assert learner_rte[data["learner_id"] == "a"].eq(1.0).all()
    assert learner_rte[data["learner_id"] == "b"].eq(0.5).all()
    assert notes["rapid_guess_rate"] == 0.1


def test_missing_response_time_is_not_a_rapid_guess():
    data = frame(["a"] * 3, ["i"] * 3, [np.nan, 5000.0, 5000.0])
    solution, _, _ = response_time_effort(data)
    assert pd.isna(solution.iloc[0])


def test_splits_are_disjoint_cover_everyone_and_are_seed_stable():
    learners = np.array([f"L{i}" for i in range(100)])
    splits = make_splits(learners, seed=20260821)
    groups = [set(splits["test"]), *(set(fold) for fold in splits["cv_folds"])]
    assert len(groups) == N_FOLDS + 1
    assert set().union(*groups) == set(learners)
    assert sum(len(group) for group in groups) == len(learners)  # no learner twice
    assert make_splits(learners[::-1], seed=20260821) == splits  # input order cannot matter
    assert make_splits(learners, seed=1) != splits
