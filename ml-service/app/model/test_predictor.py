"""Tests for the predictor module."""
from __future__ import annotations

import pytest
from unittest.mock import patch, MagicMock
import numpy as np

from app.schemas.input_schema import PredictionInput


@pytest.fixture
def sample_input():
    return PredictionInput(
        isCorrect=True,
        timeTaken=25,
        difficulty_score=5,
        knowledge_before=0.6,
        fatigue_before=0.1,
        total_response_time=25,
        reading_time=5,
        time_after_last_interaction=1,
        question_number=5,
        session_duration=120,
    )


class FakeModel:
    """Mock model that returns controlled probabilities."""

    def __init__(self, prob_map):
        """prob_map: {difficulty_score: P(correct)}"""
        self.prob_map = prob_map

    def predict_proba(self, df):
        score = df["difficulty_score"].iloc[0]
        p = self.prob_map.get(score, 0.5)
        return np.array([[1 - p, p]])


def _make_artifact(prob_map, name="test_model"):
    return {
        "model": FakeModel(prob_map),
        "model_name": name,
        "features": [],
        "target": "next_correct",
        "seed": 42,
    }


class TestPredictorNormalCase:
    """When the model returns well-separated probabilities, low_confidence should be False."""

    def test_returns_low_confidence_false(self, sample_input):
        # Spread: 0.9 - 0.3 = 0.6 >> 0.05
        artifact = _make_artifact({2: 0.9, 5: 0.7, 8: 0.3})

        with patch("app.model.predictor.load_model", return_value=artifact):
            # Need to reimport after patching
            from app.model.predictor import predict_difficulty

            result = predict_difficulty(sample_input)

        assert result["low_confidence"] is False
        assert result["policy"] == "ml"
        assert "nextDifficulty" in result
        assert "spread" in result
        assert result["spread"] >= 0.05

    def test_selects_difficulty_closest_to_072(self, sample_input):
        # easy=0.9, medium=0.72, hard=0.4 → medium is closest to 0.72
        artifact = _make_artifact({2: 0.9, 5: 0.72, 8: 0.4})

        with patch("app.model.predictor.load_model", return_value=artifact):
            from app.model.predictor import predict_difficulty

            result = predict_difficulty(sample_input)

        assert result["nextDifficulty"] == "medium"
        assert abs(result["predictedSuccess"] - 0.72) < 0.01


class TestPredictorLowConfidence:
    """When all candidates return nearly the same probability, low_confidence should be True."""

    def test_returns_low_confidence_true_when_spread_below_threshold(self, sample_input):
        # All ~0.65, spread = 0.02 < 0.05
        artifact = _make_artifact({2: 0.66, 5: 0.65, 8: 0.64})

        with patch("app.model.predictor.load_model", return_value=artifact):
            from app.model.predictor import predict_difficulty

            result = predict_difficulty(sample_input)

        assert result["low_confidence"] is True
        assert result["spread"] < 0.05


class TestPredictorFallback:
    """When no model is loaded, the fallback rule should apply."""

    def test_correct_fast_returns_hard(self, sample_input):
        with patch("app.model.predictor.load_model", return_value=None):
            from app.model.predictor import predict_difficulty

            result = predict_difficulty(sample_input)

        assert result["nextDifficulty"] == "hard"
        assert result["policy"] == "fallback_rule"
        assert result["low_confidence"] is False

    def test_incorrect_returns_easy(self):
        inp = PredictionInput(isCorrect=False, timeTaken=30)
        with patch("app.model.predictor.load_model", return_value=None):
            from app.model.predictor import predict_difficulty

            result = predict_difficulty(inp)

        assert result["nextDifficulty"] == "easy"
        assert result["policy"] == "fallback_rule"
