"""Phase 8.5 — the standing guards.

Each of these is a property the whole pipeline has to keep, not a unit test of
one function. They are the checks BUILD.md Phase 8.5 step 5 asks for: no hidden
latent state anywhere near a model or a policy, no test data during tuning,
every training script recording its seed, a propensity on every logged decision,
and one action space that the policies and the report agree on.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
ARTIFACTS = ROOT / "artifacts"
EVAL = ARTIFACTS / "evaluation"
BENCH = ARTIFACTS / "benchmarks"
POLICY_DIR = ROOT / "ml-service" / "app" / "policy"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


# ------------------------------------------------------------- the registry

def test_registry_round_trips_and_refuses_a_half_documented_artifact():
    """save → load → same metric, and an incomplete manifest is an error."""
    from app.model import registry

    registry.demo()


def test_every_registry_artifact_records_its_seed_and_commit():
    from app.model import registry

    names = registry.available()
    if not names:
        pytest.skip("no model artifact yet — run `make train-baselines`")
    for name in names:
        manifest = registry.load(name).manifest
        for key in ("seed", "dataset", "split_strategy", "split_sizes", "git_sha"):
            assert manifest.get(key) is not None, f"{name} manifest is missing {key}"


# --------------------------------------------------------------- no leakage

def test_no_latent_variable_can_reach_a_model_feature():
    """`latent_*` is the simulator's ground truth. It is never a model input.

    Checked against the catalogue itself rather than a manifest field name, so
    a new latent column added to the simulator fails here even if nobody
    remembers to update a list.
    """
    from app.research import feature_catalogue as catalogue

    for rung in ("L0", "L1", "L2", "L3", "L4"):
        for source in ("sim-V0", "assistments_2012"):
            offenders = [name for name in catalogue.available(source, upto=rung)
                         if str(name).startswith("latent_")]
            assert not offenders, f"{source} {rung} exposes latent ground truth: {offenders}"


def test_the_policy_package_cannot_read_the_simulator():
    """The guard Phase 8 established, re-asserted here as a standing property."""
    for path in POLICY_DIR.glob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert "app.research.simulator" not in source, f"{path.name} imports the simulator"
        assert "from app.research import simulator" not in source, \
            f"{path.name} imports the simulator"


# ----------------------------------------------------------- split hygiene

def test_cross_validation_report_never_scored_the_test_split():
    report = BENCH / "study1-cv.json"
    if not report.exists():
        pytest.skip("no study1-cv.json — run `make train-baselines`")
    protocol = read(report).get("protocol", {})
    assert protocol, "the cross-validation report must record its protocol"
    assert "learner-grouped" in protocol.get("cross_validation", ""), \
        "tuning must be grouped by learner, or the same learner tunes and scores"
    assert "never recomputed" in protocol.get("splits", ""), \
        "the split assignment must be read from splits.json, not re-derived per run"


def test_the_final_pass_reports_only_held_out_rows():
    final = BENCH / "study1-final.json"
    if not final.exists():
        pytest.fail("artifacts/benchmarks/study1-final.json is missing — Phase 8.5 step 1 "
                    "is `make train-baselines`; no Study 1 number is quotable without it")
    report = read(final)
    assert report.get("seed"), "the final pass must record its seed"
    for source, entry in report.get("sources", {}).items():
        if "error" in entry:
            continue
        assert entry["rows"] > 0 and entry["learners"] > 0, f"{source} scored nothing"


# ------------------------------------------------------------- propensities

def test_every_logged_decision_carries_a_usable_propensity():
    import pandas as pd

    path = EVAL / "decision-log-v0.parquet"
    if not path.exists():
        pytest.skip("no decision log — run `make eval-policies`")
    frame = pd.read_parquet(path, columns=["propensity", "policy"])
    assert len(frame) > 0
    assert frame["propensity"].notna().all(), "a decision without a propensity is unusable"
    assert (frame["propensity"] > 0).all(), "a zero propensity divides by zero in IPS"
    assert (frame["propensity"] <= 1).all(), "a propensity above 1 is not a probability"


# -------------------------------------------------------- one action space

def test_the_reported_action_space_is_the_one_the_policies_emit():
    from app.policy import rules as rules_module

    results = sorted(EVAL.glob("policy-results-v*.json"))
    if not results:
        pytest.skip("no policy results — run `make eval-policies`")
    reported = read(results[0])["action_space"]
    assert reported["difficulty"] == list(rules_module.DIFFICULTIES)
    assert reported["concept_move"] == list(rules_module.CONCEPT_MOVES)
    assert reported["intervention"] == list(rules_module.INTERVENTIONS)
    assert reported["size"] == (len(rules_module.DIFFICULTIES)
                                * len(rules_module.CONCEPT_MOVES)
                                * len(rules_module.INTERVENTIONS))


def test_a_published_report_says_which_profile_produced_it():
    """A dev run writes to the same paths a full run does — say which it was."""
    for path in sorted(EVAL.glob("policy-results-v*.json")):
        report = read(path)
        assert "run_profile" in report, f"{path.name} does not record its run profile"
        assert report["learners_per_seed"] > 0
