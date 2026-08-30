"""Phase 11 guards: the statistics that would be wrong quietly.

Every check here is one the prior version of this repository failed, or one
where a wrong answer looks plausible in a figure: the paired/unpaired effect
size mix-up that produced *d* = 26, a Holm correction that does not correct, a
power calculation off by a factor, and a figure that reads a file nobody can
reproduce.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.research import figure_index, stats, study1

ROOT = Path(__file__).resolve().parents[3]


def test_hedges_g_recovers_a_known_effect():
    rng = np.random.default_rng(0)
    control = rng.normal(0.0, 1.0, 20_000)
    treatment = rng.normal(0.8, 1.0, 20_000)
    d, g = stats.hedges_g(treatment, control)
    assert 0.75 < g < 0.85, g
    assert g < d, "Hedges' correction shrinks Cohen's d"


def test_unpaired_effect_size_is_smaller_than_the_paired_one():
    """The Defect-1 lesson, as a test.

    With a strong learner effect and a small treatment effect, the paired d
    divides by the sd of the *difference* and explodes. The unpaired g is the
    one this project reports; if the two ever agree here, the fixture is wrong.
    """
    rng = np.random.default_rng(7)
    learner = rng.normal(0.0, 3.0, 2_000)       # between-learner variance
    control = learner + rng.normal(0.0, 0.1, 2_000)
    treatment = learner + 0.2 + rng.normal(0.0, 0.1, 2_000)

    difference = treatment - control
    paired_d = float(np.mean(difference) / np.std(difference, ddof=1))
    _, unpaired_g = stats.hedges_g(treatment, control)

    assert paired_d > 1.0, paired_d
    assert abs(unpaired_g) < 0.2, unpaired_g
    assert abs(unpaired_g) < paired_d / 5


def test_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(3)
    treatment, control = rng.normal(0.5, 1.0, 800), rng.normal(0.0, 1.0, 800)
    _, g = stats.hedges_g(treatment, control)
    low, high = stats.bootstrap_g(treatment, control, seed=1, draws=500)
    assert low < g < high
    assert low > 0, "a half-sd effect at n = 800 per arm excludes zero"


def test_holm_never_lowers_a_p_value_and_is_monotone():
    raw = {"a": 0.001, "b": 0.02, "c": 0.04, "d": 0.9}
    adjusted = study1.holm(raw)
    assert all(adjusted[name] >= raw[name] for name in raw)
    ordered = sorted(raw, key=lambda name: raw[name])
    values = [adjusted[name] for name in ordered]
    assert values == sorted(values), "Holm output must be monotone in the raw order"
    assert adjusted["a"] == pytest.approx(0.004)      # 4 x the smallest


def test_required_sample_size_matches_the_textbook_value():
    """g = 0.3 at 80 % power, α = 0.05 two-sided: 176 per arm."""
    report = stats.power_analysis([{
        "comparison": "fixture", "outcome": "knowledge_gain",
        "hedges_g": 0.3, "n_treatment": 176, "n_control": 176}])
    assert report["rct_requirement"]["n_per_arm"] == pytest.approx(176, abs=1)
    assert report["achieved"][0]["power"] == pytest.approx(0.8, abs=0.02)


def test_reward_weights_in_the_preregistration_match_the_code():
    """§10.1's constants are the ones the sensitivity sweep varies."""
    from app.rl import reward

    text = (ROOT / "docs" / "preregistration.md").read_text(encoding="utf-8")
    assert f"MASTERY_SCALE = {reward.MASTERY_SCALE}" in text
    assert f"LAMBDA_TIME = {reward.LAMBDA_TIME}" in text
    assert f"MU_FRUSTRATION = {reward.MU_FRUSTRATION}" in text


def test_every_figure_declares_reproducible_sources():
    """No figure reads a file outside the repository's data and artifacts."""
    if not figure_index.MANIFEST.exists():
        pytest.skip("no figure manifest — run `make figures`")
    entries = figure_index.read_manifest()
    assert entries, "manifest exists but is empty"
    assert figure_index.check_sources(entries) == []


@pytest.mark.skipif(not (ROOT / "artifacts" / "evaluation" / "statistical-report.json").exists(),
                    reason="run `make stats` first")
def test_statistical_report_is_internally_consistent():
    report = json.loads((ROOT / "artifacts" / "evaluation" / "statistical-report.json")
                        .read_text(encoding="utf-8"))
    for entry in report["effect_sizes"]:
        assert entry["g_ci_low"] <= entry["hedges_g"] <= entry["g_ci_high"], entry["comparison"]
        # study1.holm rounds to five places; the guard is "never smaller", not "exact".
        assert entry["p_holm"] >= entry["p_welch"] - 1e-5, entry["comparison"]
        # BUILD.md §4.5: an effect this large is a bug until proven otherwise.
        assert abs(entry["cohens_d_unpaired"]) <= 3, entry["comparison"]
    for outcome, block in report["multiple_comparisons"].items():
        assert set(block["significant_holm"]) <= set(block["significant_raw"]), outcome
    for name, model in report["mixed_effects"]["outcomes"].items():
        shares = model["variance_share"]
        assert sum(shares.values()) == pytest.approx(1.0, abs=1e-6), name
