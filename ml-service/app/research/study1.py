"""Study 1 — predictive modelling, baselines and the ablation ladder.

    python -m app.research.study1 [--sources ...] [--seed 20260821]

Answers one question: **on real data, under a leakage-safe protocol, how much
does each signal class actually add?** The known result is that behavioural
features help a little (SAINT+ reports +1.25 % AUC); this reproduces it
honestly rather than claiming it as new, and reports a null as a null.

Writes:

* ``artifacts/benchmarks/study1-cv.json`` — every cross-validated cell. This is
  the file that gets regenerated during development.
* ``artifacts/benchmarks/study1-final.json`` — the test set, touched **once**,
  at the end, and never rewritten during tuning.
* ``artifacts/benchmarks/study1-predictions.parquet`` — out-of-fold predictions,
  so the figures and the paired tests read one file rather than refitting.
* MLflow runs under ``./mlruns``, one per (source, rung, model).

**The protocol, and why each part of it is there.**

*Splits are read, never made.* `data/processed/splits.json` fixes a 20 % test
set of learners and five grouped CV folds over the rest. Nothing here re-splits;
a row-level split would put the same person on both sides and inflate every
number (`docs/evaluation.md` §1).

*Hyperparameters are searched once per source, at the top rung, and then held
fixed across every rung.* Searching per rung would confound the ladder: the
increment from L3 to L4 has to be the increment from adding motor features, not
from a luckier random search. It is the same discipline RQ3 applies to the
policy architecture — vary one thing.

*A rung that adds no feature for a source is not re-run.* EdNet has no attempt
or hint counters, so its L2 is identical to its L1, and fitting it twice would
put a fake "no improvement" data point in the ladder. It is recorded as
``adds_nothing`` instead, which is a different statement from "we tried and it
did not help".

*ΔAUC is bootstrapped by learner, not by row.* Attempts within a learner are
correlated; a row bootstrap reports a confidence interval several times
narrower than the data supports, which is exactly how a null gets published as
a finding.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import warnings
from pathlib import Path

# mlflow imports GitPython, which prints a ten-line complaint on any machine
# without a git binary. The slim image has none; the git SHA comes from the
# feature manifest instead.
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, brier_score_loss, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from app.model import registry
from app.research import baselines, build_features, feature_catalogue as catalogue

ROOT = Path(__file__).resolve().parents[3]
BENCH_DIR = ROOT / "artifacts" / "benchmarks"
MODEL_DIR = ROOT / "artifacts" / "models"
CV_PATH = BENCH_DIR / "study1-cv.json"
FINAL_PATH = BENCH_DIR / "study1-final.json"
PREDICTIONS_PATH = BENCH_DIR / "study1-predictions.parquet"

LABEL = "next_correct"
DEFAULT_SOURCES = ["assistments_2009", "assistments_2012", "ednet_kt1", "sim-V0"]

#: Rows sampled per source before anything is fitted. Six models at five rungs
#: over EdNet's 4.9 M rows is hours of CPU to move an AUC in the third decimal;
#: `f06-10`'s learning curve is what actually establishes whether more data
#: would change the answer. Sampling is by learner so sequences stay whole, it
#: is seeded, and the cap is recorded in every output file.
#: 250 k rows is not a guess: `f06-10`'s learning curve on
#: `assistments_2009` is flat from roughly a thousand learners (AUC 0.7112 at
#: 949 learners, 0.7132 at 1,899), so the cap costs nothing measurable and the
#: figure is the evidence for that claim rather than an assertion about it.
MAX_TRAIN_LEARNERS = 5_000
MAX_ROWS = 250_000

#: RandomizedSearchCV budget. Fixed, small, and spent once per source at the
#: top rung — see the module docstring on why it is not spent per rung.
SEARCH_ITERATIONS = 12

BOOTSTRAP_DRAWS = 400

#: The band inside which an AUC increment is indistinguishable from protocol
#: variation. pyKT (K12) attributes 1–2 % to protocol choices alone, and
#: BUILD.md Phase 6 step 3 makes the consequence explicit: an increment whose
#: CI does not exclude this is not a result.
PROTOCOL_NOISE_BAND = 0.02

#: Reference point from the literature the ladder is read against: SAINT+ (K8)
#: reports +1.25 % AUC from adding timing features to a deep KT model.
SAINT_PLUS_REFERENCE = 0.0125


# --------------------------------------------------------------------- models

def tabular_models(seed: int) -> dict:
    """The six required tabular models, plus whatever native runtime is present.

    A gradient-boosting backend that cannot load is reported by name in the
    output rather than quietly skipped — "we ran six models" and "six models
    were installed" are different claims.
    """
    available = {
        "logistic_elasticnet": LogisticRegression(
            penalty="elasticnet", solver="saga", l1_ratio=0.5, C=1.0,
            max_iter=200, random_state=seed),
        "random_forest": RandomForestClassifier(
            n_estimators=150, max_depth=12, min_samples_leaf=20,
            n_jobs=-1, random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True,
                             n_iter_no_change=5, max_iter=80, random_state=seed),
    }
    unavailable = {}
    optional = {
        "xgboost": ("xgboost", "XGBClassifier", dict(
            n_estimators=300, max_depth=6, learning_rate=0.06, subsample=0.8,
            colsample_bytree=0.8, n_jobs=4, random_state=seed, eval_metric="logloss")),
        "lightgbm": ("lightgbm", "LGBMClassifier", dict(
            n_estimators=300, learning_rate=0.06, num_leaves=63, subsample=0.8,
            colsample_bytree=0.8, verbosity=-1, n_jobs=4, random_state=seed)),
        "catboost": ("catboost", "CatBoostClassifier", dict(
            iterations=300, depth=6, learning_rate=0.06, verbose=False,
            allow_writing_files=False, random_seed=seed)),
    }
    for name, (module_name, class_name, kwargs) in optional.items():
        try:
            module = __import__(module_name, fromlist=[class_name])
            available[name] = getattr(module, class_name)(**kwargs)
        except Exception as error:  # noqa: BLE001
            unavailable[name] = f"{type(error).__name__}: {error}".split("\n", 1)[0]
    return available, unavailable


#: Search spaces for the one model per source that gets tuned. Deliberately
#: narrow: a wide search on a null result is how a null becomes a finding.
SEARCH_SPACES = {
    "lightgbm": {
        "model__n_estimators": [200, 300, 500],
        "model__learning_rate": [0.03, 0.06, 0.1],
        "model__num_leaves": [31, 63, 127],
        "model__min_child_samples": [20, 50, 100],
    },
    "logistic_elasticnet": {
        "model__C": [0.03, 0.1, 0.3, 1.0, 3.0],
        "model__l1_ratio": [0.0, 0.25, 0.5, 0.75, 1.0],
    },
}


def pipeline_for(estimator, columns: list[str]) -> Pipeline:
    """Median-impute, standardise, fit. One preprocessing story for every model.

    Imputation happens **inside** the pipeline so the median is refitted on
    each training fold rather than leaking the validation fold's distribution
    into it. The median is never written back to a matrix (`docs/modeling.md`).
    """
    return Pipeline([
        ("prepare", ColumnTransformer([("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), columns)])),
        ("model", estimator),
    ])


# ---------------------------------------------------------------- statistics

def bootstrap_auc(labels: np.ndarray, probabilities: np.ndarray, learners: np.ndarray,
                  seed: int, draws: int = BOOTSTRAP_DRAWS) -> dict:
    """AUC with a learner-clustered bootstrap CI."""
    point = roc_auc_score(labels, probabilities)
    unique, inverse = np.unique(learners, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(unique)))
    ends = np.append(starts[1:], len(order))

    rng = np.random.default_rng(seed)
    drawn = []
    for _ in range(draws):
        picked = rng.integers(0, len(unique), len(unique))
        index = np.concatenate([order[starts[k]:ends[k]] for k in picked])
        if len(np.unique(labels[index])) < 2:
            continue
        drawn.append(roc_auc_score(labels[index], probabilities[index]))
    low, high = np.percentile(drawn, [2.5, 97.5]) if drawn else (np.nan, np.nan)
    return {"auc": round(float(point), 5), "ci_low": round(float(low), 5),
            "ci_high": round(float(high), 5), "learners": int(len(unique))}


def paired_delta_auc(labels: np.ndarray, lower: np.ndarray, upper: np.ndarray,
                     learners: np.ndarray, seed: int, draws: int = BOOTSTRAP_DRAWS) -> dict:
    """ΔAUC between two models on the same rows, bootstrapped by learner.

    Paired: the same resampled learners score both models, so the CI reflects
    the *difference*, not the sum of two independent uncertainties. Unpaired
    CIs on a small increment are wide enough to hide anything.
    """
    delta = roc_auc_score(labels, upper) - roc_auc_score(labels, lower)
    unique, inverse = np.unique(learners, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(unique)))
    ends = np.append(starts[1:], len(order))

    rng = np.random.default_rng(seed)
    drawn = []
    for _ in range(draws):
        picked = rng.integers(0, len(unique), len(unique))
        index = np.concatenate([order[starts[k]:ends[k]] for k in picked])
        if len(np.unique(labels[index])) < 2:
            continue
        drawn.append(roc_auc_score(labels[index], upper[index])
                     - roc_auc_score(labels[index], lower[index]))
    if not drawn:
        return {"delta_auc": None}
    drawn = np.asarray(drawn)
    low, high = np.percentile(drawn, [2.5, 97.5])
    # Two-sided bootstrap p against delta = 0.
    p_value = 2 * min((drawn <= 0).mean(), (drawn >= 0).mean())
    return {
        "delta_auc": round(float(delta), 5),
        "ci_low": round(float(low), 5),
        "ci_high": round(float(high), 5),
        "p_value": round(float(min(1.0, p_value)), 5),
        "clears_protocol_band": bool(low > PROTOCOL_NOISE_BAND),
        "excludes_zero": bool(low > 0 or high < 0),
    }


def holm(p_values: dict[str, float]) -> dict[str, float]:
    """Holm-Bonferroni adjusted p-values. Family = the rungs of one ladder."""
    ordered = sorted(p_values.items(), key=lambda item: item[1])
    total, adjusted, running = len(ordered), {}, 0.0
    for index, (name, value) in enumerate(ordered):
        running = max(running, min(1.0, (total - index) * value))
        adjusted[name] = round(running, 5)
    return adjusted


def expected_calibration_error(labels: np.ndarray, probabilities: np.ndarray,
                               bins: int = 15) -> float:
    """ECE with equal-width bins.

    Reported ahead of AUC because the model output drives a *decision* — an
    adaptive policy targeting a 0.7 success probability needs the 0.7 to mean
    0.7, and AUC is invariant to any monotone distortion of exactly that.
    """
    edges = np.linspace(0, 1, bins + 1)
    index = np.clip(np.digitize(probabilities, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for bucket in range(bins):
        mask = index == bucket
        if not mask.any():
            continue
        error += mask.mean() * abs(labels[mask].mean() - probabilities[mask].mean())
    return float(error)


def metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = (probabilities >= 0.5).astype(int)
    return {
        "roc_auc": round(float(roc_auc_score(labels, probabilities)), 5),
        "accuracy": round(float(accuracy_score(labels, predicted)), 5),
        "precision": round(float(precision_score(labels, predicted, zero_division=0)), 5),
        "recall": round(float(recall_score(labels, predicted, zero_division=0)), 5),
        "f1": round(float(f1_score(labels, predicted, zero_division=0)), 5),
        "log_loss": round(float(log_loss(labels, probabilities)), 5),
        "brier": round(float(brier_score_loss(labels, probabilities)), 5),
        "ece": round(expected_calibration_error(labels, probabilities), 5),
    }


# ------------------------------------------------------------------- loading

def load(source: str, seed: int) -> pd.DataFrame:
    """One feature matrix, capped by learner so sequences stay whole."""
    frame = build_features.read_features(source)
    frame = frame[frame[LABEL].notna()].copy()

    learners = frame["learner_id"].drop_duplicates()
    if len(learners) > MAX_TRAIN_LEARNERS:
        keep = set(learners.sample(MAX_TRAIN_LEARNERS, random_state=seed))
        frame = frame[frame["learner_id"].isin(keep)]
    if len(frame) > MAX_ROWS:
        # Drop whole learners rather than rows: half a sequence is a different
        # object from a short sequence, and every history feature knows it.
        sizes = frame.groupby("learner_id", observed=True).size().sample(frac=1, random_state=seed)
        keep = set(sizes[sizes.cumsum() <= MAX_ROWS].index)
        frame = frame[frame["learner_id"].isin(keep)]

    frame = frame.sort_values(["learner_id", "order_index"], kind="stable").reset_index(drop=True)
    frame[LABEL] = frame[LABEL].astype(int)
    # PFA's two counts, derived rather than stored: the catalogue holds the
    # mean and the denominator, and successes = mean x denominator.
    attempts = frame["skill_prior_attempts"].fillna(0)
    successes = (frame["skill_prior_accuracy"].fillna(0) * attempts).round()
    frame["skill_prior_successes"] = successes
    frame["skill_prior_failures"] = attempts - successes
    return frame


def rungs_for(source: str) -> list[tuple[str, list[str]]]:
    """(rung, columns) for every rung that adds something for this source.

    A rung that adds no column is skipped rather than fitted to an identical
    feature set. Reporting "L2 gave no improvement" when L2 was empty would be
    a claim about behaviour made from an absence of data.
    """
    found, previous = [], []
    for level in catalogue.LEVELS:
        columns = catalogue.available(source, upto=level)
        if columns != previous:
            found.append((level, columns))
        previous = columns
    return found


def skipped_rungs(source: str) -> dict[str, str]:
    live = {level for level, _ in rungs_for(source)}
    return {
        level: (f"adds no feature for {source} — identical to the rung below "
                f"({len(catalogue.available(source, upto=level))} columns)")
        for level in catalogue.LEVELS if level not in live
    }


# -------------------------------------------------------------------- search

def search_hyperparameters(frame: pd.DataFrame, columns: list[str], name: str,
                           estimator, seed: int) -> dict:
    """One RandomizedSearchCV, on training folds only, at the top rung.

    `optuna` is not in the pinned dependency set and BUILD.md Phase 6 step 5
    says to use `RandomizedSearchCV` with a fixed budget in that case. The
    budget is fixed at `SEARCH_ITERATIONS` and spent once per source.
    """
    space = SEARCH_SPACES.get(name)
    if not space:
        return {}
    validation = frame[frame["split"] == "fold0"]
    training = frame[~frame["split"].isin({"test", "fold0"})]
    if len(training) < 1_000 or len(validation) < 500:
        return {}

    from sklearn.model_selection import PredefinedSplit

    combined = pd.concat([training, validation])
    fold = np.concatenate([np.full(len(training), -1), np.zeros(len(validation), int)])
    search = RandomizedSearchCV(
        pipeline_for(estimator, columns), space, n_iter=SEARCH_ITERATIONS,
        scoring="roc_auc", cv=PredefinedSplit(fold), random_state=seed,
        n_jobs=1, refit=False,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        search.fit(combined[columns], combined[LABEL])
    return {key.removeprefix("model__"): value for key, value in search.best_params_.items()}


# ------------------------------------------------------------ the ladder runs

def cross_validate(frame: pd.DataFrame, columns: list[str], estimator, seed: int,
                   calibrate: bool = False) -> tuple[np.ndarray, dict]:
    """Out-of-fold predictions over the five grouped CV folds.

    Every training row gets a prediction from a model that never saw its
    learner. That is the array every AUC, ΔAUC and calibration number in this
    phase is computed from, and it is written to disk so the figures do not
    refit anything.
    """
    from sklearn.base import clone

    folds = sorted({value for value in frame["split"].unique() if str(value).startswith("fold")})
    predictions = np.full(len(frame), np.nan)
    seconds = 0.0
    for held_out in folds:
        train = frame["split"].isin(set(folds) - {held_out}).to_numpy()
        validate = (frame["split"] == held_out).to_numpy()
        if train.sum() < 500 or validate.sum() < 100:
            continue
        model = pipeline_for(clone(estimator), columns)
        if calibrate:
            # Isotonic on an inner split of the training folds — never on the
            # fold being predicted, which would be the calibration equivalent
            # of fitting on the test set.
            model = CalibratedClassifierCV(model, method="isotonic", cv=3)
        start = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(frame.loc[train, columns], frame.loc[train, LABEL])
        seconds += time.perf_counter() - start
        predictions[validate] = model.predict_proba(frame.loc[validate, columns])[:, 1]
    return predictions, {"folds": len(folds), "train_seconds": round(seconds, 2)}


def fit_final(frame: pd.DataFrame, columns: list[str], estimator, seed: int,
              calibrate: bool = False):
    """Fit on every training fold; the caller scores it on the test set once."""
    from sklearn.base import clone

    train = frame[frame["split"] != "test"]
    model = pipeline_for(clone(estimator), columns)
    if calibrate:
        model = CalibratedClassifierCV(model, method="isotonic", cv=3)
    start = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(train[columns], train[LABEL])
    return model, round(time.perf_counter() - start, 2)


def learning_curve(frame: pd.DataFrame, columns: list[str], estimator, seed: int) -> list[dict]:
    """AUC against training-set size, by learner. Feeds `f06-10`.

    The question it answers is whether a deep sequence model is even justified
    here: if the curve is flat by a few thousand learners, more data is not
    what is missing, and K13's "when is deep learning the best approach" answer
    is no.
    """
    from sklearn.base import clone

    folds = sorted({value for value in frame["split"].unique() if str(value).startswith("fold")})
    if len(folds) < 2:
        return []
    validate = frame[frame["split"] == folds[0]]
    pool = frame[frame["split"].isin(folds[1:])]
    learners = pool["learner_id"].drop_duplicates().sample(frac=1, random_state=seed).tolist()

    points = []
    for fraction in (0.1, 0.25, 0.5, 0.75, 1.0):
        subset = set(learners[: max(2, int(len(learners) * fraction))])
        train = pool[pool["learner_id"].isin(subset)]
        if len(train) < 500 or train[LABEL].nunique() < 2:
            continue
        model = pipeline_for(clone(estimator), columns)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(train[columns], train[LABEL])
        probabilities = model.predict_proba(validate[columns])[:, 1]
        points.append({
            "fraction": fraction,
            "train_learners": int(train["learner_id"].nunique()),
            "train_rows": int(len(train)),
            "auc": round(float(roc_auc_score(validate[LABEL], probabilities)), 5),
        })
    return points


# ---------------------------------------------------------------------- main

def run_source(source: str, seed: int, tracker, models: dict, quick: bool) -> tuple[dict, pd.DataFrame]:
    frame = load(source, seed)
    ladder = rungs_for(source)
    top_rung, top_columns = ladder[-1]
    print(f"\n=== {source}: {len(frame):,} rows, {frame['learner_id'].nunique():,} learners, "
          f"{len(ladder)} live rungs ({', '.join(level for level, _ in ladder)})")

    tuned = {}
    for name in SEARCH_SPACES:
        if name in models:
            found = search_hyperparameters(frame, top_columns, name, models[name], seed)
            if found:
                tuned[name] = found
                print(f"  tuned {name}: {found}")

    predictions = pd.DataFrame({
        "learner_id": frame["learner_id"], "split": frame["split"],
        LABEL: frame[LABEL], "prior_accuracy": frame["prior_accuracy"],
        "log_response_time": frame["log_response_time"], "source": source,
    })
    cells, curves = {}, {}
    for level, columns in ladder:
        for name, estimator in models.items():
            if quick and name not in {"logistic_elasticnet", "lightgbm"}:
                continue
            configured = _configure(estimator, tuned.get(name))
            probabilities, notes = cross_validate(frame, columns, configured, seed)
            scored = ~np.isnan(probabilities)
            if scored.sum() < 100:
                continue
            key = f"{level}|{name}"
            predictions[key] = probabilities
            cell = metrics(frame.loc[scored, LABEL].to_numpy(), probabilities[scored])
            cell |= bootstrap_auc(frame.loc[scored, LABEL].to_numpy(), probabilities[scored],
                                  frame.loc[scored, "learner_id"].to_numpy(), seed)
            cell |= {"features": len(columns), "rung": level, "model": name,
                     "tuned": tuned.get(name), **notes}
            cells[key] = cell
            tracker(source, level, name, cell, tuned.get(name))
            print(f"  {level:3s} {name:20s} AUC {cell['auc']:.4f} "
                  f"[{cell['ci_low']:.4f}, {cell['ci_high']:.4f}]  ECE {cell['ece']:.4f}")

    best_model = _best(cells)
    calibration = {}
    if best_model in models:
        configured = _configure(models[best_model], tuned.get(best_model))
        curves = learning_curve(frame, top_columns, configured, seed)
        pass_result = calibrated_pass(frame, top_columns, configured, seed)
        scored = pass_result.pop("predictions")
        predictions[f"{top_rung}|{best_model}|calibrated"] = scored["calibrated"]
        calibration = pass_result | {"model": best_model, "rung": top_rung}

    return {
        "rows": int(len(frame)), "learners": int(frame["learner_id"].nunique()),
        "test_learners": int((frame["split"] == "test").groupby(frame["learner_id"]).any().sum()),
        "live_rungs": [level for level, _ in ladder],
        "skipped_rungs": skipped_rungs(source),
        "rung_features": {level: len(columns) for level, columns in ladder},
        "tuned_hyperparameters": tuned,
        "cells": cells,
        "ladder": build_ladder(frame, predictions, cells, seed),
        "learning_curve": curves,
        "calibration": calibration,
        "selected_model": best_model,
    }, predictions


def _configure(estimator, parameters: dict | None):
    from sklearn.base import clone

    model = clone(estimator)
    if parameters:
        model.set_params(**parameters)
    return model


def _best(cells: dict) -> str | None:
    """The model with the highest out-of-fold AUC at any rung."""
    scored = [(cell["auc"], cell["model"]) for cell in cells.values() if cell.get("auc")]
    return max(scored)[1] if scored else None


def build_ladder(frame: pd.DataFrame, predictions: pd.DataFrame, cells: dict, seed: int) -> dict:
    """ΔAUC per rung, paired by learner, Holm-corrected across the ladder."""
    ladder = {}
    for model in sorted({cell["model"] for cell in cells.values()}):
        steps, p_values = {}, {}
        keys = [key for key in cells if key.endswith(f"|{model}")]
        levels = [cells[key]["rung"] for key in keys]
        for lower, upper in zip(levels, levels[1:]):
            low_key, high_key = f"{lower}|{model}", f"{upper}|{model}"
            scored = predictions[low_key].notna() & predictions[high_key].notna()
            if scored.sum() < 100:
                continue
            step = paired_delta_auc(
                frame.loc[scored, LABEL].to_numpy(),
                predictions.loc[scored, low_key].to_numpy(),
                predictions.loc[scored, high_key].to_numpy(),
                frame.loc[scored, "learner_id"].to_numpy(), seed)
            step["from"], step["to"] = lower, upper
            steps[f"{lower}->{upper}"] = step
            if step.get("p_value") is not None:
                p_values[f"{lower}->{upper}"] = step["p_value"]
        for name, adjusted in holm(p_values).items():
            steps[name]["p_holm"] = adjusted
            steps[name]["significant_holm"] = bool(adjusted < 0.05)
        ladder[model] = steps
    return ladder


def run_baselines(source: str, seed: int, quick: bool) -> dict:
    """The floors and the competitors, scored on the same out-of-fold protocol."""
    frame = load(source, seed)
    folds = sorted({value for value in frame["split"].unique() if str(value).startswith("fold")})
    names = list(baselines.BASELINES)
    if quick:
        names = [name for name in names if name not in baselines.DEEP_MODELS]

    results = {}
    for name in names:
        probabilities = np.full(len(frame), np.nan)
        seconds, notes = 0.0, {}
        for held_out in folds:
            train = frame[frame["split"].isin(set(folds) - {held_out})]
            validate = frame[frame["split"] == held_out]
            if len(train) < 500 or len(validate) < 100:
                continue
            if name in baselines.DEEP_MODELS and len(train["learner_id"].unique()) > baselines.DEEP_TRAIN_LEARNERS:
                keep = set(pd.Series(train["learner_id"].unique())
                           .sample(baselines.DEEP_TRAIN_LEARNERS, random_state=seed))
                train = train[train["learner_id"].isin(keep)]
            result = baselines.fit_baseline(name, train, validate)
            probabilities[(frame["split"] == held_out).to_numpy()] = result.probabilities
            seconds += result.train_seconds
            notes = result.notes
            if name in baselines.DEEP_MODELS:
                break  # one fold is enough for a CPU transformer; recorded below

        scored = ~np.isnan(probabilities)
        auc = baselines.auc_or_none(frame.loc[scored, LABEL].to_numpy(), probabilities[scored]) \
            if scored.sum() > 100 else None
        if name == "majority_class":
            # Each fold predicts its own training base rate, so pooling the
            # out-of-fold predictions makes a *constant* model look like a weak
            # fold-identity detector. Its AUC is 0.5 by definition and that is
            # what gets reported; the pooled value is kept beside it as the
            # artefact it is.
            notes = dict(notes) | {"pooled_auc_artefact": round(auc, 5) if auc else None,
                                   "why": "constant within a fold; between-fold base rates differ"}
            auc = 0.5
        entry = {"auc": round(auc, 5) if auc is not None else None,
                 "train_seconds": round(seconds, 2), "rows_scored": int(scored.sum()),
                 "notes": notes}
        if auc is not None:
            entry |= metrics(frame.loc[scored, LABEL].to_numpy(), probabilities[scored])
            entry |= bootstrap_auc(frame.loc[scored, LABEL].to_numpy(), probabilities[scored],
                                   frame.loc[scored, "learner_id"].to_numpy(), seed)
        if name in baselines.DEEP_MODELS:
            entry["protocol_note"] = (
                f"one CV fold, at most {baselines.DEEP_TRAIN_LEARNERS} training learners, "
                f"{baselines.DEEP_EPOCHS} epochs, CPU — a bounded comparison, not a tuned one")
        results[name] = entry
        print(f"  baseline {name:20s} AUC {entry['auc']} ({entry['train_seconds']}s)"
              + (f"  FAILED: {notes['failed']}" if notes.get("failed") else ""))
    return results


def calibrated_pass(frame: pd.DataFrame, columns: list[str], estimator, seed: int) -> dict:
    """The same model with and without isotonic calibration, out of fold.

    BUILD.md Phase 6 step 4 puts calibration ahead of AUC and it is right to:
    the output drives a decision. An adaptive policy that targets a 0.7 success
    probability needs 0.7 to mean 0.7, and AUC is invariant to any monotone
    distortion of precisely that.
    """
    raw, _ = cross_validate(frame, columns, estimator, seed)
    calibrated, _ = cross_validate(frame, columns, estimator, seed, calibrate=True)
    scored = ~np.isnan(raw) & ~np.isnan(calibrated)
    labels = frame.loc[scored, LABEL].to_numpy()
    return {
        "raw": metrics(labels, raw[scored]),
        "calibrated": metrics(labels, calibrated[scored]),
        "predictions": {"raw": raw, "calibrated": calibrated, "scored": scored},
    }


def final_test_pass(source: str, seed: int, models: dict, tuned: dict,
                    selected: str) -> dict:
    """The test set, touched once.

    BUILD.md Phase 6 step 9: final numbers only, at the end, and the file is
    never regenerated during tuning. Every model is scored at every live rung so
    the ladder exists on the test set too, and the selected model is persisted
    for Phase 8 — which needs a real trained artifact, because reading the
    simulator's own response function instead of one is Defect 1.
    """
    frame = load(source, seed)
    test = frame[frame["split"] == "test"]
    if len(test) < 100:
        return {"error": "no test rows"}

    results, saved = {}, None
    for level, columns in rungs_for(source):
        for name, estimator in models.items():
            configured = _configure(estimator, tuned.get(name))
            calibrate = name == selected and level == rungs_for(source)[-1][0]
            model, seconds = fit_final(frame, columns, configured, seed, calibrate=False)
            probabilities = model.predict_proba(test[columns])[:, 1]
            entry = metrics(test[LABEL].to_numpy(), probabilities)
            entry |= bootstrap_auc(test[LABEL].to_numpy(), probabilities,
                                   test["learner_id"].to_numpy(), seed)
            entry |= {"train_seconds": seconds, "features": len(columns)}
            if calibrate:
                isotonic, _ = fit_final(frame, columns, configured, seed, calibrate=True)
                entry["calibrated"] = metrics(
                    test[LABEL].to_numpy(), isotonic.predict_proba(test[columns])[:, 1])
                saved = (isotonic, columns, level)
            results[f"{level}|{name}"] = entry

    artifact = None
    if saved is not None:
        model, columns, level = saved
        # Phase 8.5's artifact contract, not a loose joblib: a consumer must be
        # able to read the feature list, the held-out metrics and how the model
        # was trained from the artifact itself, without re-reading this script.
        cell = results[f"{level}|{selected}"]
        directory = registry.save(
            f"study1-{source}-{selected}", model,
            features=list(columns), target=LABEL,
            metrics={"split": "held-out test, touched once", "rung": level,
                     "rows": int(len(test)),
                     "learners": int(test["learner_id"].nunique()),
                     **{key: value for key, value in cell.items()
                        if key != "calibrated"},
                     "calibrated": cell.get("calibrated")},
            calibration={"method": "isotonic",
                         "fitted_on": "training folds only, never the test set",
                         "uncalibrated_ece": cell.get("ece"),
                         "calibrated_ece": (cell.get("calibrated") or {}).get("ece")},
            manifest={"seed": seed, "dataset": source, "dataset_rows": int(len(frame)),
                      "split_strategy": "learner-grouped; 5 CV folds for tuning, "
                                        "a disjoint learner set held out for test",
                      "rung": level, "model_name": selected,
                      "hyperparameters": tuned.get(selected) or {},
                      "split_sizes": {split: int(count) for split, count
                                      in frame["split"].value_counts().items()},
                      "git_sha": registry.git_sha(), "phase": 6})
        artifact = str(directory.relative_to(ROOT))
    return {"rows": int(len(test)), "learners": int(test["learner_id"].nunique()),
            "cells": results, "artifact": artifact}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--quick", action="store_true",
                        help="two tabular models and no deep baselines — for a smoke run")
    parser.add_argument("--skip-baselines", action="store_true")
    parser.add_argument("--no-test", action="store_true",
                        help="cross-validation only — leaves study1-final.json untouched")
    args = parser.parse_args()

    BENCH_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    sources = [name.strip() for name in args.sources.split(",") if name.strip()]
    models, unavailable = tabular_models(args.seed)
    if unavailable:
        print(f"unavailable model backends: {unavailable}")

    import mlflow

    # BUILD.md §3 chose "MLflow file backend ./mlruns, no tracking server".
    # MLflow 3.15 refuses to open a file store at all — it is in maintenance
    # mode and raises unless MLFLOW_ALLOW_FILE_STORE is set. SQLite keeps the
    # actual decision (one file under ./mlruns, no server, `mlflow ui` works)
    # while using a backend that still receives fixes. Recorded as a deviation
    # in docs/evaluation.md rather than opted out of with an env var.
    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}")
    mlflow.set_experiment("study1-ablation-ladder")
    manifest = json.loads(build_features.MANIFEST_PATH.read_text(encoding="utf-8"))

    def tracker(source, level, name, cell, tuned):
        with mlflow.start_run(run_name=f"{source}|{level}|{name}"):
            mlflow.log_params({
                "source": source, "rung": level, "model": name,
                "features": cell["features"], "seed": args.seed,
                "git_sha": manifest["git_sha"],
                "dataset_rows": manifest["sources"][source]["rows"],
                **{f"hp_{key}": value for key, value in (tuned or {}).items()},
            })
            mlflow.log_metrics({key: value for key, value in cell.items()
                                if isinstance(value, (int, float)) and not isinstance(value, bool)})

    report = {
        "seed": args.seed, "git_sha": manifest["git_sha"],
        "label": LABEL,
        "protocol": {
            "splits": "data/processed/splits.json — read, never recomputed",
            "cross_validation": "5 learner-grouped folds; out-of-fold predictions",
            "max_train_learners": MAX_TRAIN_LEARNERS, "max_rows": MAX_ROWS,
            "search_iterations": SEARCH_ITERATIONS,
            "search_scope": "once per source at the top rung, then held fixed across rungs",
            "bootstrap": "learner-clustered, paired for every delta",
            "protocol_noise_band": PROTOCOL_NOISE_BAND,
            "saint_plus_reference": SAINT_PLUS_REFERENCE,
        },
        "unavailable_model_backends": unavailable,
        "sources": {},
    }
    predictions = []
    for source in sources:
        result, frame_predictions = run_source(source, args.seed, tracker, models, args.quick)
        if not args.skip_baselines:
            result["baselines"] = run_baselines(source, args.seed, args.quick)
        report["sources"][source] = result
        predictions.append(frame_predictions)

    CV_PATH.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"\nwrote: {CV_PATH.relative_to(ROOT)}")
    combined = pd.concat(predictions, ignore_index=True)
    assert not (combined["split"] == "unassigned").any(), (
        "a learner in a feature matrix is in neither the test set nor a CV fold")
    # Test rows carry no prediction — they are in no fold — but a file named
    # "out-of-fold predictions" must not contain them at all. Carrying null
    # rows for held-out learners is an invitation for a later phase to compute
    # a metric over the test set by accident.
    combined = combined[combined["split"] != "test"].copy()
    combined["split"] = combined["split"].cat.remove_unused_categories()
    combined.to_parquet(PREDICTIONS_PATH, index=False)
    print(f"wrote: {PREDICTIONS_PATH.relative_to(ROOT)} ({len(combined):,} out-of-fold rows, "
          f"test learners excluded)")

    if args.no_test:
        print("test set NOT touched (--no-test). study1-final.json left as it was.")
        return

    # ---- the test set, once. Everything above this line is cross-validated on
    # training folds only; nothing below it feeds back into a choice.
    print("\n=== final pass: the test set, touched once")
    final = {"seed": args.seed, "git_sha": manifest["git_sha"], "label": LABEL,
             "protocol": report["protocol"] | {
                 "warning": "these numbers are the held-out test set. BUILD.md Phase 6 step 9: "
                            "this file is written once and never regenerated during tuning."},
             "sources": {}}
    for source in sources:
        result = report["sources"][source]
        final["sources"][source] = final_test_pass(
            source, args.seed, models, result["tuned_hyperparameters"], result["selected_model"])
        best = max((cell["auc"], key) for key, cell in final["sources"][source]["cells"].items())
        print(f"  {source}: best {best[1]} AUC {best[0]:.4f}  "
              f"artifact {final['sources'][source]['artifact']}")
    FINAL_PATH.write_text(json.dumps(final, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote: {FINAL_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
