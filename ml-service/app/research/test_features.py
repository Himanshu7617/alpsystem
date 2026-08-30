"""Phase 5 checks. Run by `pytest -q`, which is what `make test` runs.

The leakage audit is invoked from here on purpose: BUILD.md Phase 5 step 3
requires it to fail the build, and a script nobody runs is not a guard. The
last test plants a lookahead and asserts the audit catches it — a guard that
has never been shown to fire is a guard nobody should trust.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from app.research import audit_leakage, build_features, feature_catalogue as catalogue

SPLITS = json.loads((build_features.SPLITS_PATH).read_text(encoding="utf-8"))["sources"]
SEED = 20260821
#: The two sources the audit runs against by default: the smallest real table
#: and the calibrated simulator. Between them they exercise every feature —
#: the real one has no motor or interaction telemetry, the simulated one has
#: no wall-clock gaps between sessions.
AUDITED = ["assistments_2009", "sim-V0"]


# --------------------------------------------------------------- the catalogue

def test_catalogue_validates():
    catalogue.validate()


def test_every_feature_has_a_rationale():
    """BUILD.md Phase 5 acceptance: zero rows with an empty rationale field."""
    empty = [f.name for f in catalogue.CATALOGUE if not f.rationale.strip()]
    assert not empty, f"features with no rationale must not be built: {empty}"


def test_ladder_is_cumulative():
    ladder = catalogue.ladder()
    for lower, higher in zip(catalogue.LEVELS, catalogue.LEVELS[1:]):
        assert set(ladder[lower]) < set(ladder[higher])
    assert set(ladder["L4"]) == set(catalogue.names())


def test_generated_doc_matches_the_catalogue():
    """`docs/feature-catalogue.md` is generated — a stale copy is a lie."""
    if not catalogue.DOC_PATH.exists():
        pytest.skip("docs/ is not mounted in this container")
    assert catalogue.DOC_PATH.read_text(encoding="utf-8").strip() == catalogue.to_markdown().strip(), (
        "docs/feature-catalogue.md is out of date — "
        "run `python -m app.research.feature_catalogue --write-doc`"
    )


# ------------------------------------------------------------- built matrices

@pytest.fixture(scope="module")
def matrices() -> dict[str, pd.DataFrame]:
    frames = {}
    for source in build_features.ALL_SOURCES:
        path = build_features.PROCESSED_DIR / f"features-{source}.parquet"
        if path.exists():
            frames[source] = pd.read_parquet(path)
    if not frames:
        pytest.skip("no feature matrices built — run `make features`")
    return frames


def test_every_source_was_built(matrices):
    assert set(matrices) == set(build_features.ALL_SOURCES)


def test_no_latent_ground_truth_reaches_a_matrix(matrices):
    for source, frame in matrices.items():
        audit_leakage.check_no_latent_features(frame, where=source)


def test_absent_features_are_null_not_zero(matrices):
    """A source that cannot measure something must not report it as a zero."""
    for source, frame in matrices.items():
        for name, feature in catalogue.BY_NAME.items():
            if source in feature.unavailable:
                assert name not in frame.columns or frame[name].isna().all(), (
                    f"{source}: {name} is documented as unavailable but carries values"
                )


def test_label_is_the_next_attempt(matrices):
    """`next_correct` must be the following row's `correct`, within a learner."""
    frame = matrices["sim-V0"]
    one = frame[frame["learner_id"] == frame["learner_id"].iloc[0]].sort_values("order_index")
    assert (one["next_correct"].to_numpy()[:-1] == one["correct"].to_numpy()[1:]).all()


def test_history_features_exclude_their_own_row(matrices):
    """`prior_accuracy` at row t is the mean over rows before t, not including t."""
    frame = matrices["sim-V0"]
    one = frame[frame["learner_id"] == frame["learner_id"].iloc[0]].sort_values("order_index")
    expected = one["correct"].expanding().mean().shift(1)
    assert np.allclose(one["prior_accuracy"].astype(float), expected.astype(float),
                       rtol=1e-5, atol=1e-5, equal_nan=True)
    assert pd.isna(one["prior_accuracy"].iloc[0]), "the first attempt has no history"


def test_test_learners_are_held_out(matrices):
    for source, frame in matrices.items():
        held_out = set(SPLITS[source]["test"])
        marked = set(frame.loc[frame["split"] == "test", "learner_id"])
        assert marked <= held_out
        assert not marked & {learner for fold in SPLITS[source]["cv_folds"] for learner in fold}


def test_read_features_restores_the_full_schema():
    """A source's parquet may omit columns; every consumer still sees them."""
    frame = build_features.read_features("assistments_2009")
    assert list(frame.columns) == catalogue.PASSTHROUGH + catalogue.names()
    assert frame["auc_toward_nonchosen"].isna().all()


# ---------------------------------------------------------------- the audit

@pytest.mark.parametrize("source", AUDITED)
def test_no_feature_reads_the_future(source):
    audit_leakage.check_no_future_information(source, SPLITS, SEED, rows=1_500)


@pytest.mark.parametrize("source", AUDITED)
def test_item_statistics_never_see_a_test_learner(source):
    audit_leakage.check_item_statistics_exclude_test(source, SPLITS, SEED)


def test_audit_catches_a_planted_lookahead(monkeypatch):
    """The guard must be shown to fire, not merely to pass.

    A leading (rather than lagging) prior accuracy is the smallest honest
    mistake that reproduces Defect 2's shape: a feature that knows the answer
    it is about to be scored against. If check B cannot see this, it cannot see
    anything.
    """
    real = build_features._prior_mean

    def leaking(frame, keys, column):
        mean, count = real(frame, keys, column)
        if column == "correct":
            # The next row's outcome, handed to this row.
            group = [frame[key] for key in keys]
            mean = frame[column].astype(float).groupby(group, sort=False, observed=True).shift(-1)
        return mean, count

    monkeypatch.setattr(build_features, "_prior_mean", leaking)
    with pytest.raises(audit_leakage.LeakageError, match="read the future"):
        audit_leakage.check_no_future_information("sim-V0", SPLITS, SEED, rows=1_500)


def test_audit_main_exits_zero():
    """The exact invocation `make test` depends on."""
    audit_leakage.catalogue.validate()
    audit_leakage.check_legacy_features_are_not_used()
    assert audit_leakage.check_built_matrices()
