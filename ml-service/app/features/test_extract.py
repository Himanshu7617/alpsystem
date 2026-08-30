"""Synthetic telemetry conformance test.

Replays a recorded event stream through the extractor and asserts the feature
row still matches the golden fixture. This is the runnable check for Phase 2: if
a feature definition drifts, the diff shows up here before it reaches a model.

Regenerate the golden file **deliberately** (never to make a red test green):

    PYTHONPATH=. python -c "import json;from app.features.extract import extract;\
    json.dump(extract(json.load(open('app/features/testdata/attempt-events.json'))),\
    open('app/features/testdata/golden-features.json','w'),indent=2,sort_keys=True)"
"""
import copy
import json
from pathlib import Path

from app.features.extract import FEATURE_CLASSES, extract

TESTDATA = Path(__file__).parent / "testdata"
REQUEST = json.loads((TESTDATA / "attempt-events.json").read_text())
GOLDEN = json.loads((TESTDATA / "golden-features.json").read_text())


def test_recorded_stream_matches_golden_features():
    assert extract(copy.deepcopy(REQUEST)) == GOLDEN


def test_extraction_is_order_independent():
    shuffled = copy.deepcopy(REQUEST)
    shuffled["events"] = list(reversed(shuffled["events"]))
    assert extract(shuffled) == GOLDEN


def test_no_latent_ground_truth_is_ever_produced():
    """Standing guardrail 1: latent state is a target, never a feature."""
    banned = ("knowledge", "fatigue", "confidence", "engagement", "archetype", "guessed")
    produced = extract(copy.deepcopy(REQUEST))["features"]
    assert not [name for name in produced if any(word in name for word in banned)]


def test_withdrawn_consent_yields_null_not_zero():
    """A signal class the learner switched off must be absent, not zero: the
    RQ4 privacy-utility curve depends on telling absence from measurement."""
    for signal_class in ("timing", "interaction", "motor"):
        request = copy.deepcopy(REQUEST)
        request["consent"][signal_class] = False
        result = extract(request)
        assert signal_class not in result["available_classes"]
        for name, owner in FEATURE_CLASSES.items():
            if owner == signal_class:
                assert result["features"][name] is None, f"{name} should be null without {signal_class} consent"
            elif owner in ("correctness", "session"):
                assert result["features"][name] is not None


def test_loop_still_completes_with_only_correctness():
    request = copy.deepcopy(REQUEST)
    request["consent"] = {"correctness": True, "timing": False, "interaction": False, "motor": False, "probes": False}
    result = extract(request)
    assert result["features"]["correct"] == 1
    assert result["available_classes"] == ["correctness", "session"]


def test_missing_cursor_segment_is_null_motor_not_absent_keys():
    request = copy.deepcopy(REQUEST)
    request["events"] = [event for event in request["events"] if event["type"] != "CURSOR_SEGMENT"]
    result = extract(request)
    assert set(result["features"]) == set(FEATURE_CLASSES)
    assert result["features"]["max_deviation"] is None
