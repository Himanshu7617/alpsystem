"""Study 2 — multi-state estimation, calibration and the validation gate.

One shared GRU encoder over a learner's interaction sequence, three heads on
its hidden state, **each supervised by its own label**:

===========  =========================================================  ==============
head         label                                                      source
===========  =========================================================  ==============
knowledge    ``next_correct`` — the next attempt's outcome              real + simulated
engagement   ``solution_behaviour`` of the *next* attempt (RTE, §6)     real
confidence   the post-submission probe, dichotomised at "fairly sure"   simulated
===========  =========================================================  ==============

The point of the phase is not the architecture — a shared encoder with several
supervised heads is standard composition. It is that a state is only allowed to
drive a policy once it has passed pre-registered criteria: correlation with its
own label, calibration, Wise & Kong's negative criterion for effort, and
discriminant validity against the knowledge head. Those criteria live in
``app.model.validation_gate``; this module measures the inputs they need and
writes the verdict.

**Why the engagement label is shifted forward one attempt.** Rapid-guessing is
defined by a response-time cut and response time is an L1 model input, so a head
predicting the label of the attempt whose response time it can see would score
near-perfectly by re-deriving the threshold. That is arithmetic, not a state.
Pre-registration §8.1 fixes the shift; ``test_study2.py`` asserts it is applied.

**Fatigue is absent by pre-registration**, not by omission: §7's H2 outcome
dropped the construct when no archival source showed a within-session accuracy
decline. ``validation_gate.DROPPED_STATES`` carries the reason into the report.

Outputs
-------
* ``artifacts/models/state-encoder.pt`` — the top-rung encoder and its config.
* ``artifacts/models/state-calibrators.joblib`` — one isotonic map per head.
* ``artifacts/models/state-encoders-by-rung.pt`` and
  ``state-calibrators-by-rung.joblib`` — the same, one entry per rung, which is
  what Phase 8's ``model_L0``/``model_L4`` arms condition on.
* ``artifacts/evaluation/validation-gate.json`` — the verdict, with the measured
  value of every criterion, passes and failures alike.
* ``artifacts/benchmarks/study2-heads.json`` — per-head metrics at every rung.
* ``artifacts/benchmarks/study2-predictions.parquet`` — scoring-half estimates
  with their MC-dropout standard errors, so the figures refit nothing.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score
from torch import nn

from app.model import validation_gate
from app.research import build_features, feature_catalogue as catalogue
from app.research.study1 import expected_calibration_error, load as load_source

ROOT = Path(__file__).resolve().parents[3]
BENCH_DIR = ROOT / "artifacts" / "benchmarks"
MODEL_DIR = ROOT / "artifacts" / "models"
EVAL_DIR = ROOT / "artifacts" / "evaluation"

ENCODER_PATH = MODEL_DIR / "state-encoder.pt"
CALIBRATOR_PATH = MODEL_DIR / "state-calibrators.joblib"
#: Phase 8's ladder arms need the estimator *of their own rung*, not the top
#: one: `model_L0` conditioning on a knowledge estimate that read motor
#: features would not be a correctness-only arm. Every rung is trained here
#: anyway, so the checkpoints are kept rather than discarded.
RUNG_ENCODER_PATH = MODEL_DIR / "state-encoders-by-rung.pt"
RUNG_CALIBRATOR_PATH = MODEL_DIR / "state-calibrators-by-rung.joblib"
HEADS_PATH = BENCH_DIR / "study2-heads.json"
PREDICTIONS_PATH = BENCH_DIR / "study2-predictions.parquet"

#: Which source supplies which head's label. A head is trained on the rows that
#: carry its label and on no others (masked loss), and it is *scored* only on
#: the source named here, which is what every gate number is measured over.
HEAD_SOURCES = {
    "knowledge": ("assistments_2012", "sim-V0"),
    "engagement": ("assistments_2012",),
    "confidence": ("sim-V0",),
}
#: The single source each gate criterion is measured on. Pooling a real and a
#: simulated validity number into one figure would make it impossible to say
#: which one the paper is claiming, so the gate names its source explicitly and
#: the per-source breakdown stays in ``study2-heads.json``. Knowledge is judged
#: on real data because a real-data verdict is the stronger claim.
GATE_SOURCE = {
    "knowledge": "assistments_2012",
    "engagement": "assistments_2012",
    "confidence": "sim-V0",
}

#: Confidence probe is 4-point (Guessing / Not sure / Fairly sure / Certain);
#: the policy consumes a probability, so it is dichotomised at "fairly sure".
CONFIDENCE_CUT = 3

HEAD_LABEL = {
    "knowledge": "next_correct — the next attempt's outcome",
    "engagement": "solution_behaviour of the next attempt (response-time effort, "
                  "preregistration §6)",
    "confidence": f"probe_confidence >= {CONFIDENCE_CUT} (post-submission self-report, "
                  f"preregistration §3)",
}

DEFAULT_SOURCES = ["assistments_2012", "sim-V0"]
HEADS = tuple(validation_gate.STATES)

#: Sequence chunk length. A learner with 400 attempts becomes four independent
#: chunks rather than one 400-step backprop.
#: `ponytail:` chunking resets the hidden state at every boundary, so nothing
#: earlier than 100 attempts back can inform an estimate. Upgrade path is
#: carrying the final hidden state across a learner's chunks; it costs an
#: ordered pass and it is only worth it if f07-07 shows state still moving at
#: the chunk boundary.
CHUNK = 100

EPOCHS = 6
BATCH = 64
HIDDEN = 64
DROPOUT = 0.2
LEARNING_RATE = 3e-3
MC_PASSES = 20
BOOTSTRAP_DRAWS = 400


# ------------------------------------------------------------------ sequences

def label_frame(source: str, seed: int) -> pd.DataFrame:
    """One source's feature matrix plus the three head labels.

    ``load_source`` is Phase 6's loader — same learner cap, same seeded sample,
    same sort — so Study 1 and Study 2 are fitted on the same rows and a
    difference between them is a difference in method, not in data.
    """
    frame = load_source(source, seed)

    frame["y_knowledge"] = frame["next_correct"].astype(float)

    # Pre-registration §8.1: the engagement label is the *next* attempt's
    # effort classification. `load_source` has already sorted by
    # (learner_id, order_index), so a group-wise shift is the next attempt.
    if frame["solution_behaviour"].notna().any():
        shifted = frame.groupby("learner_id", observed=True)["solution_behaviour"].shift(-1)
        frame["y_engagement"] = pd.to_numeric(shifted, errors="coerce")
    else:
        frame["y_engagement"] = np.nan

    if frame["probe_confidence"].notna().any():
        probe = pd.to_numeric(frame["probe_confidence"], errors="coerce")
        frame["y_confidence"] = (probe >= CONFIDENCE_CUT).astype(float).where(probe.notna())
    else:
        frame["y_confidence"] = np.nan

    frame["source"] = source
    return frame


def chunk_index(frame: pd.DataFrame) -> np.ndarray:
    """A sequence id per row: learner, cut into ``CHUNK``-long pieces."""
    position = frame.groupby("learner_id", observed=True).cumcount()
    keys = frame["learner_id"].astype(str) + "#" + (position // CHUNK).astype(str)
    return pd.factorize(keys)[0]


def pack(frame: pd.DataFrame, columns: list[str], stats: pd.DataFrame) -> dict:
    """Rows → padded (sequence, step, feature) tensors.

    Every feature enters twice: a robust-standardised value with its NaNs
    replaced by zero, and a 0/1 *present* channel. Absence and measurement stay
    distinguishable inside the model, the way they already are in the parquet
    (pre-registration §4). Nothing imputed here is written to disk.
    """
    values = frame[columns].astype("float32").to_numpy()
    present = np.isfinite(values).astype("float32")
    centred = (values - stats["centre"].to_numpy()) / stats["scale"].to_numpy()
    centred = np.nan_to_num(np.clip(centred, -5, 5), nan=0.0, posinf=0.0, neginf=0.0)
    packed = np.concatenate([centred, present], axis=1).astype("float32")

    sequence = frame["_sequence"].to_numpy()
    order = np.argsort(sequence, kind="stable")
    counts = np.bincount(sequence)
    width = int(counts.max())

    step = np.concatenate([np.arange(count) for count in counts])
    x = np.zeros((len(counts), width, packed.shape[1]), dtype="float32")
    x[sequence[order], step] = packed[order]

    targets, masks = {}, {}
    for head in HEADS:
        y = np.zeros((len(counts), width), dtype="float32")
        m = np.zeros((len(counts), width), dtype="float32")
        column = frame[f"y_{head}"].to_numpy(dtype="float32")
        y[sequence[order], step] = np.nan_to_num(column[order])
        m[sequence[order], step] = np.isfinite(column[order]).astype("float32")
        targets[head], masks[head] = y, m

    row = np.full((len(counts), width), -1, dtype="int64")
    row[sequence[order], step] = np.asarray(frame.index)[order]
    return {"x": torch.from_numpy(x), "y": {k: torch.from_numpy(v) for k, v in targets.items()},
            "mask": {k: torch.from_numpy(v) for k, v in masks.items()},
            "row": row}


def feature_stats(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Median and IQR of the *training* rows only. Fitted once, reused everywhere."""
    train = frame[frame["split"].isin([f"fold{i}" for i in range(4)])]
    centre = train[columns].median(numeric_only=True)
    scale = (train[columns].quantile(0.75) - train[columns].quantile(0.25)).replace(0, np.nan)
    return pd.DataFrame({"centre": centre.fillna(0.0), "scale": scale.fillna(1.0)}).loc[columns]


# --------------------------------------------------------------------- model

class StateEncoder(nn.Module):
    """A GRU trunk and one linear head per state. Nothing novel, deliberately.

    Dropout is kept active at inference for the MC-dropout standard error
    (pre-registration §8.3), which is why it sits on the trunk output rather
    than inside the recurrence.
    """

    def __init__(self, n_features: int, hidden: int = HIDDEN, dropout: float = DROPOUT):
        super().__init__()
        self.gru = nn.GRU(n_features, hidden, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.heads = nn.ModuleDict({head: nn.Linear(hidden, 1) for head in HEADS})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        trunk, _ = self.gru(x)
        trunk = self.dropout(trunk)
        return {head: layer(trunk).squeeze(-1) for head, layer in self.heads.items()}


def train(packed: dict, n_features: int, seed: int, epochs: int = EPOCHS) -> tuple[StateEncoder, dict]:
    """Masked multi-task training. A row never trains a head it has no label for."""
    torch.manual_seed(seed)
    model = StateEncoder(n_features)
    optimiser = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss(reduction="none")

    total = len(packed["x"])
    generator = torch.Generator().manual_seed(seed)
    history, start = [], time.perf_counter()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(total, generator=generator)
        running = {head: 0.0 for head in HEADS}
        for begin in range(0, total, BATCH):
            index = order[begin:begin + BATCH]
            logits = model(packed["x"][index])
            loss = 0.0
            for head in HEADS:
                mask = packed["mask"][head][index]
                if mask.sum() == 0:
                    continue
                # Masked mean: the loss of a head is the mean over the rows that
                # carry its label, so a head with few labels is not drowned by
                # one with many.
                head_loss = (loss_fn(logits[head], packed["y"][head][index]) * mask).sum() / mask.sum()
                running[head] += float(head_loss.detach()) * float(mask.sum())
                loss = loss + head_loss
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        history.append({"epoch": epoch, **{f"loss_{head}": round(running[head] /
                        max(1.0, float(packed["mask"][head].sum())), 5) for head in HEADS}})
    return model, {"epochs": epochs, "train_seconds": round(time.perf_counter() - start, 2),
                   "history": history}


@torch.no_grad()
def predict(model: StateEncoder, packed: dict, passes: int = MC_PASSES,
            seed: int = 0) -> dict[str, np.ndarray]:
    """Per-row probability and MC-dropout standard error, flattened back to rows.

    The mean is taken over ``passes`` stochastic forward passes; the standard
    deviation is what Phase 8 reads when it needs to know a state is unreliable.
    """
    model.train()  # dropout stays on: that is the point of MC dropout
    torch.manual_seed(seed)
    stacked = {head: [] for head in HEADS}
    for _ in range(passes):
        logits = model(packed["x"])
        for head in HEADS:
            stacked[head].append(torch.sigmoid(logits[head]).numpy())
    model.eval()

    valid = packed["row"] >= 0
    rows = packed["row"][valid]
    output = {"row": rows}
    for head in HEADS:
        draws = np.stack(stacked[head])[:, valid]
        output[f"{head}_p"] = draws.mean(axis=0)
        output[f"{head}_se"] = draws.std(axis=0)
    return output


# ------------------------------------------------------------------ measuring

def clustered_correlation(x: np.ndarray, y: np.ndarray, learners: np.ndarray,
                          seed: int, draws: int = BOOTSTRAP_DRAWS) -> dict:
    """Pearson r with a learner-clustered bootstrap CI.

    Clustered because rows within a learner are not independent; a row
    bootstrap would report an interval several times narrower than the data
    supports, which is how a null gets published as a finding.
    """
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return {"r": None, "ci_low": None, "ci_high": None, "n": int(len(x))}
    point = float(np.corrcoef(x, y)[0, 1])

    unique, inverse = np.unique(learners, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    starts = np.searchsorted(inverse[order], np.arange(len(unique)))
    ends = np.append(starts[1:], len(order))
    rng = np.random.default_rng(seed)
    drawn = []
    for _ in range(draws):
        picked = rng.integers(0, len(unique), len(unique))
        index = np.concatenate([order[starts[k]:ends[k]] for k in picked])
        if np.std(x[index]) == 0 or np.std(y[index]) == 0:
            continue
        drawn.append(np.corrcoef(x[index], y[index])[0, 1])
    low, high = np.percentile(drawn, [2.5, 97.5]) if drawn else (None, None)
    return {"r": round(point, 5), "ci_low": round(float(low), 5) if low is not None else None,
            "ci_high": round(float(high), 5) if high is not None else None,
            "n": int(len(x)), "learners": int(len(unique))}


def head_metrics(labels: np.ndarray, raw: np.ndarray, calibrated: np.ndarray,
                 learners: np.ndarray, seed: int) -> dict:
    correlation = clustered_correlation(calibrated, labels, learners, seed)
    return {
        "n": int(len(labels)),
        "learners": int(len(np.unique(learners))),
        "base_rate": round(float(labels.mean()), 5),
        "roc_auc": round(float(roc_auc_score(labels, calibrated)), 5)
        if len(np.unique(labels)) > 1 else None,
        "ece_raw": round(expected_calibration_error(labels, raw), 5),
        "ece_calibrated": round(expected_calibration_error(labels, calibrated), 5),
        "label_correlation": correlation["r"],
        "label_correlation_ci_low": correlation["ci_low"],
        "label_correlation_ci_high": correlation["ci_high"],
    }


def ability_by_learner(rows: pd.DataFrame) -> pd.Series:
    """Wise & Kong's ability term: a learner's mean correctness over their
    *solution-behaviour* responses.

    Restricted to solution behaviour because correctness on a rapid guess is
    not evidence about ability — that is the whole premise of the RTE label.

    Measured on the same held-out rows the engagement estimate is scored on,
    because the splits are **by learner**: a learner in the scoring half has no
    rows in the training folds, so there is no earlier data to compute their
    ability from. The two quantities are not independent by construction —
    one is an observed label average, the other a model output — and the
    criterion is a *negative* one, so the shared rows work against passing it,
    which is the conservative direction. Recorded as a deviation in
    ``docs/preregistration.md`` §8.4.
    """
    engaged = rows[rows["solution_behaviour"].fillna(0) > 0]
    return engaged.groupby("learner_id", observed=True)["correct"].mean()


# ---------------------------------------------------------------------- rungs

def live_rungs(sources: list[str]) -> list[tuple[str, list[str]]]:
    """(rung, columns) for every rung that adds a column for *some* source.

    Pooled: the union over sources, because a rung that only the simulator can
    supply still changes the model — the archival rows simply carry it as
    absent, which the present-channel makes visible rather than silent.
    """
    found, previous = [], []
    for level in catalogue.LEVELS:
        columns = sorted({name for source in sources
                          for name in catalogue.available(source, upto=level)})
        if columns != previous:
            found.append((level, columns))
        previous = columns
    return found


def halve(frame: pd.DataFrame, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Split `fold4` by learner into a calibration half and a scoring half."""
    validation = frame["split"] == "fold4"
    learners = frame.loc[validation, "learner_id"].drop_duplicates()
    rng = np.random.default_rng(seed)
    calibration_learners = set(learners.sample(frac=0.5, random_state=int(rng.integers(1 << 30))))
    in_calibration = frame["learner_id"].isin(calibration_learners)
    return (validation & in_calibration).to_numpy(), (validation & ~in_calibration).to_numpy()


def run_rung(frame: pd.DataFrame, columns: list[str], seed: int,
             epochs: int) -> dict:
    """Train, calibrate and score one rung.

    Returns the trained model, its isotonic maps, the scoring-half predictions
    and the per-head metrics. Nothing here reads ``split == "test"``.
    """
    stats = feature_stats(frame, columns)
    frame = frame.copy()
    frame["_sequence"] = chunk_index(frame)

    train_rows = frame[frame["_half"] == "train"].reset_index(drop=True)
    packed_train = pack(train_rows, columns, stats)
    model, notes = train(packed_train, 2 * len(columns), seed, epochs)

    validation = frame[frame["_half"].isin(["calibration", "scoring"])].reset_index(drop=True)
    predicted = predict(model, pack(validation, columns, stats), seed=seed)
    scored = validation.iloc[predicted["row"]].copy()
    for head in HEADS:
        scored[f"{head}_p_raw"] = predicted[f"{head}_p"]
        scored[f"{head}_se"] = predicted[f"{head}_se"]

    calibrators, metrics = {}, {}
    for head in HEADS:
        labelled = scored[f"y_{head}"].notna() & scored["source"].isin(HEAD_SOURCES[head])
        calibration = labelled & scored["_half"].eq("calibration")
        scoring = labelled & scored["_half"].eq("scoring")
        scored[f"{head}_p"] = scored[f"{head}_p_raw"]
        if calibration.sum() < 50 or scoring.sum() < 50:
            metrics[head] = {"skipped": f"too few labelled rows "
                                        f"({int(calibration.sum())} calibration, "
                                        f"{int(scoring.sum())} scoring)"}
            continue
        # Isotonic on the calibration half only. Fitting it on the rows the ECE
        # is then read from would report the calibrator's own residual, not the
        # model's calibration.
        isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        isotonic.fit(scored.loc[calibration, f"{head}_p_raw"], scored.loc[calibration, f"y_{head}"])
        calibrators[head] = isotonic
        scored.loc[labelled, f"{head}_p"] = isotonic.predict(scored.loc[labelled, f"{head}_p_raw"])

        metrics[head] = {"overall": head_metrics(
            scored.loc[scoring, f"y_{head}"].to_numpy(dtype=float),
            scored.loc[scoring, f"{head}_p_raw"].to_numpy(dtype=float),
            scored.loc[scoring, f"{head}_p"].to_numpy(dtype=float),
            scored.loc[scoring, "learner_id"].to_numpy(), seed)}
        metrics[head]["mean_se"] = round(float(scored.loc[scoring, f"{head}_se"].mean()), 5)
        for source in HEAD_SOURCES[head]:
            rows = scoring & scored["source"].eq(source)
            if rows.sum() >= 50 and scored.loc[rows, f"y_{head}"].nunique() > 1:
                metrics[head][source] = head_metrics(
                    scored.loc[rows, f"y_{head}"].to_numpy(dtype=float),
                    scored.loc[rows, f"{head}_p_raw"].to_numpy(dtype=float),
                    scored.loc[rows, f"{head}_p"].to_numpy(dtype=float),
                    scored.loc[rows, "learner_id"].to_numpy(), seed)

    return {"model": model, "calibrators": calibrators, "scored": scored,
            "metrics": metrics, "notes": notes, "stats": stats, "columns": columns}


# ----------------------------------------------------------------- gate input

def gate_measurements(scored: pd.DataFrame, metrics: dict, seed: int) -> dict:
    """The four pre-registered criteria, measured on the scoring half.

    Every value here is a *measurement*; whether it passes is
    ``validation_gate``'s decision, kept separate on purpose so a threshold
    cannot be quietly adjusted next to the number it judges.
    """
    measurements = {}
    for head in HEADS:
        source = GATE_SOURCE[head]
        entry = metrics.get(head, {}).get(source)
        if entry is None:
            continue
        measurements[head] = {
            "source": source,
            "label": HEAD_LABEL[head],
            "n": entry["n"],
            "learners": entry["learners"],
            "base_rate": entry["base_rate"],
            "roc_auc": entry["roc_auc"],
            "ece_raw": entry["ece_raw"],
            "ece_calibrated": entry["ece_calibrated"],
            "label_correlation": entry["label_correlation"],
            "label_correlation_ci_low": entry["label_correlation_ci_low"],
            "label_correlation_ci_high": entry["label_correlation_ci_high"],
        }

    scoring = scored["_half"].eq("scoring")

    # C3 — Wise & Kong's negative criterion, at learner level: an effort index
    # must not track ability.
    if "engagement" in measurements:
        rows = scoring & scored["source"].eq(GATE_SOURCE["engagement"])
        ability = ability_by_learner(scored.loc[rows])
        per_learner = scored.loc[rows].groupby("learner_id", observed=True)["engagement_p"].mean()
        paired = pd.concat([per_learner.rename("estimate"), ability.rename("ability")],
                           axis=1, join="inner").dropna()
        correlation = clustered_correlation(paired["estimate"].to_numpy(),
                                            paired["ability"].to_numpy(),
                                            paired.index.to_numpy(), seed)
        measurements["engagement"]["ability_correlation"] = correlation["r"]
        measurements["engagement"]["ability_correlation_ci"] = [correlation["ci_low"],
                                                                correlation["ci_high"]]
        measurements["engagement"]["ability_learners"] = correlation["n"]

    # C4 — discriminant validity: is the confidence head a relabelled knowledge head?
    if "confidence" in measurements:
        rows = scoring & scored["source"].eq(GATE_SOURCE["confidence"])
        correlation = clustered_correlation(scored.loc[rows, "confidence_p"].to_numpy(dtype=float),
                                            scored.loc[rows, "knowledge_p"].to_numpy(dtype=float),
                                            scored.loc[rows, "learner_id"].to_numpy(), seed)
        measurements["confidence"]["knowledge_correlation"] = correlation["r"]
        measurements["confidence"]["knowledge_correlation_ci"] = [correlation["ci_low"],
                                                                  correlation["ci_high"]]
    return measurements


# ----------------------------------------------------------------------- main

def assemble(sources: list[str], seed: int, epochs: int, tracker=None,
             rungs: list[str] | None = None) -> dict:
    frames = [label_frame(source, seed) for source in sources]
    frame = pd.concat(frames, ignore_index=True)
    frame["_half"] = np.where(frame["split"].isin([f"fold{i}" for i in range(4)]), "train", "")
    calibration_mask, scoring_mask = halve(frame, seed)
    frame.loc[calibration_mask, "_half"] = "calibration"
    frame.loc[scoring_mask, "_half"] = "scoring"
    frame.loc[frame["split"].eq("test"), "_half"] = "test"
    # The test set is not read in this phase. Dropping it here rather than
    # remembering not to touch it is the version that cannot go wrong.
    frame = frame[frame["_half"] != "test"].reset_index(drop=True)

    ladder = [rung for rung in live_rungs(sources) if rungs is None or rung[0] in rungs]
    results, top, by_rung = {}, None, {}
    for level, columns in ladder:
        print(f"  rung {level}: {len(columns)} features", flush=True)
        rung = run_rung(frame, columns, seed, epochs)
        results[level] = {"features": len(columns), "columns": columns,
                          "heads": rung["metrics"], **rung["notes"]}
        by_rung[level] = rung
        if tracker is not None:
            tracker(level, rung)
        top = rung
    return {"frame": frame, "ladder": results, "top": top, "by_rung": by_rung,
            "top_rung": ladder[-1][0] if ladder else None}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=",".join(DEFAULT_SOURCES))
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--rungs", default="", help="comma-separated subset, for a smoke run")
    parser.add_argument("--quick", action="store_true",
                        help="top rung only, one epoch — smoke run, writes nothing")
    args = parser.parse_args()

    for directory in (BENCH_DIR, MODEL_DIR, EVAL_DIR):
        directory.mkdir(parents=True, exist_ok=True)
    sources = [name.strip() for name in args.sources.split(",") if name.strip()]
    rungs = [name.strip() for name in args.rungs.split(",") if name.strip()] or None
    manifest = json.loads(build_features.MANIFEST_PATH.read_text(encoding="utf-8"))

    import mlflow

    # Same backend decision as Phase 6: one sqlite file under ./mlruns, no
    # server. MLflow 3.15 refuses to open a file store at all.
    (ROOT / "mlruns").mkdir(exist_ok=True)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / 'mlruns' / 'mlflow.db'}")
    mlflow.set_experiment("study2-state-heads")

    def tracker(level, rung):
        for head in HEADS:
            entry = rung["metrics"].get(head, {})
            overall = entry.get("overall")
            if overall is None:
                continue
            with mlflow.start_run(run_name=f"{level}|{head}"):
                mlflow.log_params({"rung": level, "head": head, "seed": args.seed,
                                   "features": len(rung["columns"]),
                                   "git_sha": manifest["git_sha"],
                                   "label": HEAD_LABEL[head],
                                   "sources": ",".join(sources)})
                mlflow.log_metrics({key: value for key, value in overall.items()
                                    if isinstance(value, (int, float)) and value is not None})
                mlflow.log_metric("mean_se", entry.get("mean_se", float("nan")))

    if args.quick:
        rungs, args.epochs = [live_rungs(sources)[-1][0]], 1

    print(f"sources: {', '.join(sources)}")
    run = assemble(sources, args.seed, args.epochs, tracker, rungs)
    top = run["top"]
    if top is None:
        raise SystemExit("no rung produced a model")

    measurements = gate_measurements(top["scored"], top["metrics"], args.seed)
    report = validation_gate.evaluate(measurements)
    report = {"seed": args.seed, "git_sha": manifest["git_sha"],
              "rung": run["top_rung"],
              "protocol": {
                  "splits": "data/processed/splits.json — read, never recomputed",
                  "train": "fold0-fold3",
                  "calibration": "half of fold4, by learner",
                  "scoring": "the other half of fold4",
                  "test": "not touched in this phase",
                  "uncertainty": f"MC dropout, {MC_PASSES} passes",
                  "thresholds": "docs/preregistration.md §8.2",
              },
              "measurements": measurements, **report}

    if args.quick:
        print(json.dumps(report["admitted"]), json.dumps(report["rejected"]))
        raise SystemExit(0)

    def checkpoint(rung: dict, level: str) -> dict:
        return {"state_dict": rung["model"].state_dict(),
                "columns": rung["columns"],
                "centre": rung["stats"]["centre"].to_dict(),
                "scale": rung["stats"]["scale"].to_dict(),
                "hidden": HIDDEN, "dropout": DROPOUT, "chunk": CHUNK,
                "heads": list(HEADS), "rung": level, "seed": args.seed,
                "git_sha": manifest["git_sha"]}

    torch.save(checkpoint(top, run["top_rung"]), ENCODER_PATH)
    print(f"wrote: {ENCODER_PATH.relative_to(ROOT)}")
    joblib.dump(top["calibrators"], CALIBRATOR_PATH)
    print(f"wrote: {CALIBRATOR_PATH.relative_to(ROOT)} ({len(top['calibrators'])} heads)")

    torch.save({level: checkpoint(rung, level) for level, rung in run["by_rung"].items()},
               RUNG_ENCODER_PATH)
    print(f"wrote: {RUNG_ENCODER_PATH.relative_to(ROOT)} ({len(run['by_rung'])} rungs)")
    joblib.dump({level: rung["calibrators"] for level, rung in run["by_rung"].items()},
                RUNG_CALIBRATOR_PATH)
    print(f"wrote: {RUNG_CALIBRATOR_PATH.relative_to(ROOT)} ({len(run['by_rung'])} rungs)")

    validation_gate.GATE_PATH.write_text(json.dumps(report, indent=2, default=float) + "\n",
                                         encoding="utf-8")
    print(f"wrote: {validation_gate.GATE_PATH.relative_to(ROOT)} — "
          f"admitted {report['admitted'] or 'none'}, rejected {report['rejected'] or 'none'}")

    heads = {"seed": args.seed, "git_sha": manifest["git_sha"], "sources": sources,
             "labels": HEAD_LABEL, "gate_source": GATE_SOURCE,
             "confidence_cut": CONFIDENCE_CUT, "chunk": CHUNK,
             "ladder": run["ladder"]}
    HEADS_PATH.write_text(json.dumps(heads, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote: {HEADS_PATH.relative_to(ROOT)} ({len(run['ladder'])} rungs)")

    keep = ["learner_id", "source", "session_id", "order_index", "split", "_half",
            "correct", "item_difficulty", "solution_behaviour", "learner_rte",
            "log_response_time"] + [f"y_{head}" for head in HEADS] + \
           [f"{head}_{suffix}" for head in HEADS for suffix in ("p_raw", "p", "se")]
    columns = [name for name in keep if name in top["scored"].columns]
    top["scored"][columns].to_parquet(PREDICTIONS_PATH, index=False)
    print(f"wrote: {PREDICTIONS_PATH.relative_to(ROOT)} ({len(top['scored']):,} validation rows)")


if __name__ == "__main__":
    main()
