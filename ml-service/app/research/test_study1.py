"""Phase 6 checks. Run by `pytest -q`, which is what `make test` runs.

These are checks on the *protocol*, not on the results. A result that moves
when the data is rebuilt is fine; a protocol that lets a test learner into
training, or reports a ΔAUC without a paired interval, is not.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.research import baselines, build_features, feature_catalogue as catalogue, study1

BENCH = study1.BENCH_DIR
SEED = 20260821


# ------------------------------------------------------------------ protocol

def test_rungs_skip_what_adds_nothing():
    """EdNet has no attempt or hint counters, so its L2 is its L1."""
    live = [level for level, _ in study1.rungs_for("ednet_kt1")]
    assert "L2" not in live, "a rung that adds no column must not be fitted twice"
    assert "L4" not in live, "no archival source carries motor telemetry"
    assert study1.skipped_rungs("ednet_kt1")["L2"].startswith("adds no feature")
    # The simulator carries every class, so nothing is skipped there.
    assert [level for level, _ in study1.rungs_for("sim-V0")] == catalogue.LEVELS


def test_rungs_are_cumulative():
    for source in ("assistments_2012", "sim-V0"):
        previous: list[str] = []
        for _, columns in study1.rungs_for(source):
            assert set(previous) < set(columns)
            previous = columns


def test_no_rung_contains_latent_ground_truth():
    for source in study1.DEFAULT_SOURCES:
        for _, columns in study1.rungs_for(source):
            for name in columns:
                assert not any(name.startswith(prefix)
                               for prefix in catalogue.FORBIDDEN_PREFIXES)


def test_next_item_id_is_not_a_feature():
    """The base-rate floor may use it; the ladder may not."""
    assert "next_item_id" in catalogue.PASSTHROUGH
    assert "next_item_id" not in catalogue.names()
    for source in study1.DEFAULT_SOURCES:
        for _, columns in study1.rungs_for(source):
            assert "next_item_id" not in columns
            assert "next_skill_id" not in columns


# ---------------------------------------------------------------- statistics

def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(SEED)
    learners = np.repeat(np.arange(200), 25)
    labels = rng.integers(0, 2, len(learners))
    scores = labels * 0.3 + rng.random(len(learners)) * 0.7
    result = study1.bootstrap_auc(labels, scores, learners, SEED, draws=80)
    assert result["ci_low"] <= result["auc"] <= result["ci_high"]
    assert result["learners"] == 200


def test_paired_delta_is_zero_and_insignificant_for_identical_models():
    rng = np.random.default_rng(SEED)
    learners = np.repeat(np.arange(150), 20)
    labels = rng.integers(0, 2, len(learners))
    scores = labels * 0.3 + rng.random(len(learners)) * 0.7
    result = study1.paired_delta_auc(labels, scores, scores, learners, SEED, draws=80)
    assert result["delta_auc"] == 0
    assert not result["excludes_zero"]
    assert not result["clears_protocol_band"]


def test_protocol_band_rejects_a_small_but_significant_increment():
    """The rule that makes a null a null.

    A tiny increment can be statistically significant on a million rows and
    still be inside pyKT's 1–2 % protocol noise. `clears_protocol_band` is what
    the paper reports, and it must not be satisfied by significance alone.
    """
    rng = np.random.default_rng(SEED)
    learners = np.repeat(np.arange(400), 40)
    labels = rng.integers(0, 2, len(learners))
    weak = labels * 0.30 + rng.random(len(learners)) * 0.70
    strong = labels * 0.31 + rng.random(len(learners)) * 0.70
    result = study1.paired_delta_auc(labels, weak, strong, learners, SEED, draws=120)
    assert result["delta_auc"] < study1.PROTOCOL_NOISE_BAND
    assert not result["clears_protocol_band"]


def test_holm_is_monotone_and_never_below_the_raw_p():
    raw = {"a": 0.001, "b": 0.02, "c": 0.4}
    adjusted = study1.holm(raw)
    assert all(adjusted[name] >= raw[name] for name in raw)
    assert adjusted["a"] <= adjusted["b"] <= adjusted["c"]


def test_ece_is_zero_for_a_perfectly_calibrated_model():
    rng = np.random.default_rng(SEED)
    probabilities = rng.random(200_000)
    labels = (rng.random(200_000) < probabilities).astype(int)
    assert study1.expected_calibration_error(labels, probabilities) < 0.01
    # ... and large for a confidently wrong one.
    assert study1.expected_calibration_error(labels, np.full(200_000, 0.99)) > 0.4


# ----------------------------------------------------------------- baselines

def test_upstream_shims_are_still_needed():
    """The shim works around an upstream bug. If it is fixed, delete the shim.

    A workaround nobody revisits is how a codebase accumulates cargo, so this
    asserts the bug is still there: a dependency bump that fixes it fails here
    instead of leaving dead code behind.

    The bug is pykt's own source line, **not** whether the runtime happens to
    ship a working ``turtle``. An earlier version of this test asserted that
    ``import pykt.models`` raises, which made it a test of the image rather
    than of pykt: `python:3.11-slim` now resolves ``turtle``, so the import
    succeeded and the test failed while the upstream defect was untouched.
    """
    import inspect

    baselines._stub_turtle()
    from pykt.models import qdkt
    from pykt.models.dkt import DKT  # noqa: F401

    source = inspect.getsource(qdkt).splitlines()
    assert any(line.strip() == "from turtle import forward" for line in source[:10]), (
        "pykt no longer imports `turtle` — delete baselines._stub_turtle() and this test")


def test_pybkt_imports_through_the_shim():
    assert baselines._import_pybkt() is not None


def test_trivial_floors_behave_like_floors():
    frame = build_features.read_features("assistments_2009").head(5_000).copy()
    frame[study1.LABEL] = frame[study1.LABEL].astype(int)
    train, validate = frame.head(4_000), frame.tail(1_000)

    majority = baselines.majority_class(train, validate)
    assert len(set(majority.probabilities)) == 1, "a constant prediction, by definition"
    assert baselines.auc_or_none(validate[study1.LABEL].to_numpy(), majority.probabilities) is None \
        or abs(baselines.auc_or_none(validate[study1.LABEL].to_numpy(),
                                     majority.probabilities) - 0.5) < 1e-9

    rate = baselines.item_base_rate(train, validate)
    assert not np.isnan(rate.probabilities).any(), "unseen items must fall back, not go null"


def test_sequences_align_the_label_one_step_ahead():
    frame = build_features.read_features("sim-V0").head(2_000).copy()
    frame[study1.LABEL] = frame[study1.LABEL].astype(int)
    skills = {value: index + 1 for index, value in enumerate(sorted(set(frame["skill_id"].astype(str))))}
    items = {value: index + 1 for index, value in enumerate(sorted(set(frame["item_id"].astype(str))))}
    built = baselines._sequences(frame, skills, items, length=50)
    assert built["concepts"].shape == built["targets"].shape == built["masks"].shape
    # Every masked position's target is the label of the row it came from.
    rows, targets, mask = built["rows"], built["targets"], built["masks"]
    for row, target in zip(rows[mask], targets[mask]):
        assert frame.loc[row, study1.LABEL] == int(target)


# --------------------------------------------------------------- the outputs

@pytest.fixture(scope="module")
def report() -> dict:
    path = BENCH / "study1-cv.json"
    if not path.exists():
        pytest.skip("study1 has not been run — `make train-baselines`")
    return json.loads(path.read_text(encoding="utf-8"))


def test_every_reported_delta_carries_an_interval(report):
    for source, result in report["sources"].items():
        for model, steps in result["ladder"].items():
            for name, step in steps.items():
                assert step["ci_low"] <= step["delta_auc"] <= step["ci_high"], (
                    f"{source}/{model}/{name}: point estimate outside its own CI")
                assert "clears_protocol_band" in step, "a delta without the band verdict"


def test_the_ladder_verdict_matches_its_own_numbers(report):
    """`clears_protocol_band` must mean exactly what it says."""
    for result in report["sources"].values():
        for steps in result["ladder"].values():
            for step in steps.values():
                assert step["clears_protocol_band"] == (step["ci_low"] > study1.PROTOCOL_NOISE_BAND)


def test_cross_validation_never_scored_a_test_learner(report):
    """The out-of-fold file must contain no test rows at all."""
    path = BENCH / "study1-predictions.parquet"
    if not path.exists():
        pytest.skip("no predictions file")
    predictions = pd.read_parquet(path)
    assert "test" not in set(predictions["split"]), (
        "a test learner was scored during cross-validation")
    splits = json.loads(build_features.SPLITS_PATH.read_text(encoding="utf-8"))["sources"]
    for source, frame in predictions.groupby("source"):
        held_out = set(splits[source]["test"])
        assert not set(frame["learner_id"]) & held_out


def test_final_file_reports_the_test_set_only():
    path = BENCH / "study1-final.json"
    if not path.exists():
        pytest.skip("final pass has not been run")
    final = json.loads(path.read_text(encoding="utf-8"))
    splits = json.loads(build_features.SPLITS_PATH.read_text(encoding="utf-8"))["sources"]
    for source, result in final["sources"].items():
        if "error" in result:
            continue
        assert result["learners"] <= len(splits[source]["test"])
        assert result["cells"], "no test-set metrics recorded"
