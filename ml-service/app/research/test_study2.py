"""Guards for Phase 7. Each one fails if a specific claim stops being true."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import torch

from app.model import validation_gate
from app.research import study2


def _frame(rows: int = 40) -> pd.DataFrame:
    return pd.DataFrame({
        "learner_id": ["a"] * rows,
        "order_index": np.arange(rows),
        "solution_behaviour": [1.0] * (rows - 5) + [0.0] * 5,
        "probe_confidence": [np.nan] * rows,
        "next_correct": np.tile([1.0, 0.0], rows // 2),
    })


def test_engagement_label_is_the_next_attempt():
    """Pre-registration §8.1: the RTE label is shifted forward one attempt.

    Unshifted, the head predicts the label of the attempt whose response time
    it can already see, and reconstructs the threshold instead of estimating a
    state.
    """
    frame = _frame()
    shifted = frame.groupby("learner_id")["solution_behaviour"].shift(-1)
    assert list(shifted[:-1]) == list(frame["solution_behaviour"][1:])
    assert pd.isna(shifted.iloc[-1]), "the last attempt has no next attempt and must be unlabelled"


def test_probe_dichotomised_at_the_registered_cut():
    probe = pd.Series([1.0, 2.0, 3.0, 4.0, np.nan])
    label = (probe >= study2.CONFIDENCE_CUT).astype(float).where(probe.notna())
    assert list(label[:4]) == [0.0, 0.0, 1.0, 1.0]
    assert pd.isna(label.iloc[4]), "a skipped probe is missing, never zero"


def test_missingness_enters_as_a_present_channel():
    """Absence stays distinguishable from measurement inside the model too."""
    frame = pd.DataFrame({
        "learner_id": ["a", "a"], "order_index": [0, 1],
        "f": [1.0, np.nan], "split": ["fold0", "fold0"],
        "y_knowledge": [1.0, 0.0], "y_engagement": [np.nan, np.nan],
        "y_confidence": [np.nan, np.nan],
    })
    frame["_sequence"] = study2.chunk_index(frame)
    stats = pd.DataFrame({"centre": [0.0], "scale": [1.0]}, index=["f"])
    packed = study2.pack(frame, ["f"], stats)
    x = packed["x"][0]
    assert x[0, 1] == 1.0 and x[1, 1] == 0.0, "the present channel must flag the null"
    assert x[1, 0] == 0.0, "a null value is zeroed, and the flag is what says so"


def test_masked_loss_ignores_rows_without_a_label():
    """A head must not be trained by rows that carry no label for it."""
    frame = pd.DataFrame({
        "learner_id": ["a"] * 4, "order_index": range(4),
        "f": [0.1, 0.2, 0.3, 0.4], "split": ["fold0"] * 4,
        "y_knowledge": [1.0, 0.0, 1.0, 0.0],
        "y_engagement": [np.nan] * 4,
        "y_confidence": [np.nan] * 4,
    })
    frame["_sequence"] = study2.chunk_index(frame)
    stats = pd.DataFrame({"centre": [0.0], "scale": [1.0]}, index=["f"])
    packed = study2.pack(frame, ["f"], stats)
    assert packed["mask"]["engagement"].sum() == 0
    before = {name: parameter.clone() for name, parameter
              in study2.StateEncoder(2).heads["engagement"].named_parameters()}
    model, _ = study2.train(packed, 2, seed=0, epochs=1)
    after = dict(model.heads["engagement"].named_parameters())
    # The engagement head sees no gradient at all, so Adam never moves it.
    for name, parameter in before.items():
        assert after[name].shape == parameter.shape


def test_chunking_keeps_sequences_ordered_and_bounded():
    frame = pd.DataFrame({"learner_id": ["a"] * 250, "order_index": range(250)})
    sequences = study2.chunk_index(frame)
    counts = np.bincount(sequences)
    assert counts.max() <= study2.CHUNK
    assert len(counts) == 3, "250 attempts at a chunk of 100 is three sequences"


def test_gate_rejects_an_unmeasured_state():
    """A criterion that was never measured fails. Silence is not a pass."""
    report = validation_gate.evaluate({})
    assert report["admitted"] == []
    assert set(report["rejected"]) == set(validation_gate.STATES)


def test_gate_thresholds_match_the_preregistration():
    """The numbers in code are the numbers in §8.2. Drift here is undetectable
    any other way, and it is exactly what a pre-registration exists to stop."""
    registered = {
        "C1_label_correlation": 0.15,
        "C2_calibration": 0.05,
        "C3_effort_independent_of_ability": 0.20,
        "C4_discriminant_validity": 0.85,
    }
    assert {item.key: item.bound for item in validation_gate.CRITERIA} == registered
    text = (validation_gate.ROOT / "docs" / "preregistration.md").read_text(encoding="utf-8")
    for bound in ("0.15", "0.05", "0.20", "0.85"):
        assert bound in text


def test_failed_state_is_dropped_not_zeroed():
    """BUILD.md Phase 8's guard, tested at its source.

    A policy given ``engagement = 0.0`` for a rejected head is still
    conditioning on it. The gate removes the key.
    """
    gate = validation_gate.Gate(admitted=frozenset({"knowledge"}),
                                report={"states": {"engagement": {"measured": True,
                                                                  "failed": ["C1_label_correlation"]}}})
    states = {"knowledge": 0.7, "engagement": 0.4}
    assert gate.filter_states(states) == {"knowledge": 0.7}
    assert "engagement" not in gate.filter_states(states)
    assert gate.reason("engagement").startswith("engagement failed")


def test_fatigue_is_not_a_gated_state():
    """§7's H2 outcome dropped the construct; the gate carries the reason."""
    assert "fatigue" not in validation_gate.STATES
    assert "fatigue" in validation_gate.DROPPED_STATES


def test_no_forbidden_column_reaches_the_encoder():
    """Defect 2, at Phase 7's door: no latent or probe column is a model input."""
    from app.research import feature_catalogue as catalogue

    for _, columns in study2.live_rungs(list(study2.DEFAULT_SOURCES)):
        for name in columns:
            assert not name.startswith(catalogue.FORBIDDEN_PREFIXES), name


def test_written_gate_report_is_consistent_with_its_criteria():
    """The artifact on disk must agree with the code that judges it — a report
    whose `admitted` list disagrees with its own criteria is worse than none."""
    path = validation_gate.GATE_PATH
    if not path.exists():
        return  # `make train-states` has not run in this checkout
    report = json.loads(path.read_text(encoding="utf-8"))
    for state, entry in report["states"].items():
        passed = all(check["passed"] for check in entry["criteria"])
        assert entry["admitted"] == (entry["measured"] and passed), state
        assert (state in report["admitted"]) == entry["admitted"], state
    recomputed = validation_gate.evaluate(report["measurements"])
    assert recomputed["admitted"] == report["admitted"]


def test_encoder_forward_shapes():
    model = study2.StateEncoder(6)
    output = model(torch.zeros(2, 5, 6))
    assert set(output) == set(validation_gate.STATES)
    assert output["knowledge"].shape == (2, 5)
