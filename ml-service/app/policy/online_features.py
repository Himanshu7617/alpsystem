"""The feature catalogue, computed one attempt at a time.

``build_features.py`` computes history features with vectorised pandas over a
finished log. A closed-loop policy has no finished log: at step *t* it has the
rows up to *t* and must produce the same numbers a batch build would have
produced for row *t*. This module is that, and it is deliberately a
reimplementation of only the *history* half — every per-attempt behavioural
column still comes from ``app.features.extract`` via the environment, so there
is still one definition of ``max_deviation``.

The columns computed here are exactly the ones ``build_features.build`` adds on
top of the extractor's output:

``prior_accuracy``, ``prior_attempts``, ``skill_prior_accuracy``,
``skill_prior_attempts``, ``log_rt_z_item``, ``rt_drift``, ``log_lag_time``,
``idle_fraction``, ``matched_difficulty_accuracy_slope`` and
``matched_difficulty_speed_slope``.

``test_policies.py`` checks the reimplementation against the batch build on the
same rows rather than trusting that two similar-looking pieces of arithmetic
agree.

Per-item response-time location/scale and per-item accuracy come from the
offline corpus, fitted on training learners only — the same statistics a
deployed system would hold before a session starts. They are cached in
``artifacts/models/item-response-stats.json`` so the closed loop does not
re-read a 20 MB parquet once per cell.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from app.research import build_features
from app.research import feature_catalogue as catalogue

ROOT = Path(__file__).resolve().parents[3]
STATS_PATH = ROOT / "artifacts" / "models" / "item-response-stats.json"

#: Matches ``build_features.MIN_SLOPE_POINTS``: a slope through three noisy
#: points is not a measurement, and the batch build reports null there.
MIN_SLOPE_POINTS = build_features.MIN_SLOPE_POINTS


def item_response_stats(source: str = "sim-V0", refresh: bool = False) -> dict:
    """Per-item accuracy and log-RT location/scale from the offline corpus.

    Fitted by ``build_features.item_statistics`` on the training learners of
    ``source`` — the identical function the feature matrices were built with,
    so the closed loop standardises response time the same way Study 2's
    training rows were standardised.
    """
    if STATS_PATH.exists() and not refresh:
        cached = json.loads(STATS_PATH.read_text(encoding="utf-8"))
        if cached.get("source") == source:
            return cached

    import pandas as pd

    frame = pd.read_parquet(
        build_features.PROCESSED_DIR / f"features-{source}.parquet",
        columns=["item_id", "correct", "log_response_time", "split"])
    frame["item_id"] = frame["item_id"].astype("category")
    stats = build_features.item_statistics(frame, frame["split"] != "test")
    payload = {
        "source": source,
        "fallback": stats.attrs["fallback"],
        "items": {str(item): {"accuracy": float(row.accuracy), "rt_mean": float(row.rt_mean),
                              "rt_sd": float(row.rt_sd)}
                  for item, row in stats.iterrows()},
    }
    STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


class _Slope:
    """OLS slope of y on x over the points seen so far, from running sums.

    The same closed form ``build_features._running_slope`` uses, minus the
    groupby: one instance per learner-session.
    """

    __slots__ = ("n", "sx", "sy", "sxy", "sxx")

    def __init__(self) -> None:
        self.n = self.sx = self.sy = self.sxy = self.sxx = 0.0

    def value(self) -> float:
        if self.n < MIN_SLOPE_POINTS:
            return math.nan
        denominator = self.n * self.sxx - self.sx * self.sx
        if denominator == 0:
            return math.nan
        return (self.n * self.sxy - self.sx * self.sy) / denominator

    def add(self, x: float, y: float) -> None:
        if not (math.isfinite(x) and math.isfinite(y)):
            return
        self.n += 1.0
        self.sx += x
        self.sy += y
        self.sxy += x * y
        self.sxx += x * x


class LearnerFeatures:
    """Running feature state for one learner.

    :meth:`row` returns the modelling row for the attempt just made; every
    history value in it describes attempts **strictly before** that attempt,
    because :meth:`row` reads the accumulators before updating them. That
    ordering is the online form of the batch build's ``shift(1)`` guard.
    """

    def __init__(self, stats: dict):
        self.items = stats["items"]
        self.fallback = stats["fallback"]
        self.correct_total = 0.0
        self.count = 0.0
        self.log_rt_total = 0.0
        self.skill_correct: dict[str, float] = {}
        self.skill_count: dict[str, float] = {}
        self.accuracy_slope = _Slope()
        self.speed_slope = _Slope()
        self.session_id: str | None = None
        self.last_timestamp_ms: float | None = None

    def _item(self, item_id: str) -> dict:
        return self.items.get(item_id, {"accuracy": None, **self.fallback})

    def row(self, attempt: dict) -> dict:
        """Catalogue columns for ``attempt`` (an environment attempt record)."""
        if attempt["session_id"] != self.session_id:
            self.session_id = attempt["session_id"]
            self.accuracy_slope = _Slope()
            self.speed_slope = _Slope()

        stats = self._item(attempt["item_id"])
        skill = attempt["concept_id"]
        response_time_s = attempt["response_time_ms"] / 1000.0
        log_rt = math.log1p(max(0.0, response_time_s))
        rt_sd = stats["rt_sd"] or self.fallback["rt_sd"]
        log_rt_z_item = (log_rt - stats["rt_mean"]) / (rt_sd or 1.0)
        lag_s = (math.nan if self.last_timestamp_ms is None
                 else max(0.0, (attempt["timestamp"] - self.last_timestamp_ms) / 1000.0))

        row = dict(attempt)
        row["item_difficulty"] = attempt["item_b"]
        row["attempt_count"] = 1.0
        row["log_response_time"] = log_rt
        row["log_rt_z_item"] = log_rt_z_item
        row["log_lag_time"] = math.log1p(lag_s) if math.isfinite(lag_s) else math.nan
        row["prior_attempts"] = self.count
        row["prior_accuracy"] = self.correct_total / self.count if self.count else math.nan
        row["skill_prior_attempts"] = self.skill_count.get(skill, 0.0)
        row["skill_prior_accuracy"] = (self.skill_correct[skill] / self.skill_count[skill]
                                       if self.skill_count.get(skill) else math.nan)
        row["rt_drift"] = (log_rt - self.log_rt_total / self.count) if self.count else math.nan
        idle = attempt.get("idle_time")
        row["idle_fraction"] = (min(1.0, max(0.0, idle / response_time_s))
                                if idle is not None and response_time_s > 0 else math.nan)
        row["matched_difficulty_accuracy_slope"] = self.accuracy_slope.value()
        row["matched_difficulty_speed_slope"] = self.speed_slope.value()

        # ---- accumulate, after the row has been read
        self.count += 1.0
        self.correct_total += float(attempt["correct"])
        self.log_rt_total += log_rt
        self.skill_count[skill] = self.skill_count.get(skill, 0.0) + 1.0
        self.skill_correct[skill] = self.skill_correct.get(skill, 0.0) + float(attempt["correct"])
        self.last_timestamp_ms = attempt["timestamp"]
        # "Matched difficulty" is the item's own base rate removed, so a policy
        # handing out harder items does not read as a decline.
        base_rate = stats["accuracy"]
        if base_rate is None:
            base_rate = 1 / (1 + math.exp(self.fallback["item_difficulty"]))
        position = float(attempt.get("question_number") or 0.0)
        self.accuracy_slope.add(position, float(attempt["correct"]) - base_rate)
        self.speed_slope.add(position, log_rt_z_item)
        return row


def matrix(rows: list[dict | None], columns: list[str]) -> np.ndarray:
    """``columns`` of every row as a float array, missing rows as all-null."""
    out = np.full((len(rows), len(columns)), np.nan, dtype="float32")
    for index, row in enumerate(rows):
        if row is None:
            continue
        for position, name in enumerate(columns):
            value = row.get(name)
            if value is not None:
                out[index, position] = value
    return out


def observable_view(row: dict | None, rung: str) -> dict:
    """The part of an attempt record an arm at ``rung`` is allowed to read.

    Signal classes above the arm's rung are removed, not zeroed: an arm that
    "sees correctness only" must not be able to reach response time through a
    side door. Latent columns are never in this dict at any rung.

    It lives here rather than in ``closed_loop`` because the live service needs
    it too, and the serving path must not import the simulator.
    """
    if row is None:
        return {}
    allowed = set(catalogue.available("sim-V0", upto=rung))
    return {name: value for name, value in row.items()
            if name in allowed and value is not None and not (
                isinstance(value, float) and math.isnan(value))}
