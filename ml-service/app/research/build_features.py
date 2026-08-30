"""Build the modelling matrices — `make features`.

    python -m app.research.build_features [--sources ...] [--seed 20260821]

Writes ``data/processed/features-<source>.parquet`` for the three archival
tables and the five simulator variants, plus one manifest recording the seed,
the git SHA, row counts and the per-feature null rate of every source.

Three properties this file exists to hold, all checked by
``app.research.audit_leakage`` and ``app.research.test_features``:

1. **The catalogue is the schema.** Columns are exactly
   :data:`~app.research.feature_catalogue.CATALOGUE` plus the declared
   passthroughs. Nothing is invented here and nothing latent crosses over.
2. **A row never sees its own future.** Every history feature is computed from
   attempts *strictly before* its own row, so truncating the data at row *k*
   leaves row *k* bit-identical. That is a runnable test, not a convention.
3. **Item statistics come from training learners only.** Item difficulty and
   the per-item response-time standardisation are fitted on the learners
   `splits.json` puts in the CV folds; the held-out test learners contribute
   nothing to them.

Per-attempt behavioural columns are **not** recomputed here. They come from
``app.features.extract`` via the simulator (Phase 4), which is the same
function the live backend calls. This file only adds what needs a *sequence*:
history aggregates, session structure and the label.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from app.research import feature_catalogue as catalogue
from app.research.prep_archival import make_splits

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"
DATASET_DIR = ROOT / "artifacts" / "datasets"
SPLITS_PATH = PROCESSED_DIR / "splits.json"
MANIFEST_PATH = DATASET_DIR / "features-manifest.json"

ARCHIVAL_SOURCES = ["assistments_2009", "assistments_2012", "ednet_kt1"]
SIM_SOURCES = ["sim-V0", "sim-V1", "sim-V2", "sim-V3", "sim-V4"]
ALL_SOURCES = ARCHIVAL_SOURCES + SIM_SOURCES

#: A gap this long starts a new session. 30 minutes is the convention the
#: MOOC/ITS log literature uses (B13 works on session logs cut the same way).
#: ponytail: a fixed threshold, not a per-learner fitted one. Upgrade to a
#: mixture over inter-attempt gaps only if session boundaries ever carry a
#: result rather than merely bounding one.
SESSION_GAP_MS = 30 * 60 * 1000

#: Shrinkage strength for the per-item difficulty estimate, and the minimum
#: number of training responses an item needs before its own response-time
#: mean and sd are trusted over the source's.
ITEM_PRIOR_RESPONSES = 20

#: A running slope needs enough points to mean anything. Below this the
#: feature is null, which is the honest value — not zero.
MIN_SLOPE_POINTS = 4


def git_sha() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        head = (ROOT / ".git" / "HEAD").read_text(encoding="utf-8").strip()
        if head.startswith("ref: "):
            head = (ROOT / ".git" / head[5:]).read_text(encoding="utf-8").strip()
        return head


# --------------------------------------------------------------------- loading

#: Read from the archival tables and nothing else. EdNet is 4.9 M rows; the
#: columns this file never touches are 900 MB it does not need to hold.
ARCHIVAL_COLUMNS = [
    "learner_id", "item_id", "skill_id", "order_index", "timestamp", "correct",
    "response_time_ms", "lag_time_ms", "attempt_count", "hint_count",
    "learner_rte", "solution_behaviour",
]


def load_archival(name: str) -> pd.DataFrame:
    """One archival table in the canonical shape the feature builder expects."""
    frame = pd.read_parquet(PROCESSED_DIR / f"{name}.parquet", columns=ARCHIVAL_COLUMNS)
    frame = frame.sort_values(["learner_id", "order_index"], kind="stable").reset_index(drop=True)

    def number(column: str) -> pd.Series:
        return frame.pop(column).astype("Float64").astype("float32")

    out = pd.DataFrame({
        "learner_id": frame.pop("learner_id").astype("category"),
        "item_id": frame.pop("item_id").astype("category"),
        "skill_id": frame.pop("skill_id").astype("category"),
        "order_index": frame.pop("order_index").astype("int32"),
        # Milliseconds since the epoch overflow float32's 24-bit mantissa, so
        # the clock stays float64 while everything derived from it does not.
        "timestamp_ms": frame.pop("timestamp").astype("Float64").astype(float),
        "correct": frame.pop("correct").astype("int8"),
        "response_time_s": number("response_time_ms") / 1000.0,
        "lag_time_s": number("lag_time_ms") / 1000.0,
        "attempt_count": number("attempt_count"),
        "hint_count": number("hint_count"),
        "learner_rte": number("learner_rte"),
        "solution_behaviour": number("solution_behaviour"),
        "source": name,
    })
    return out


def load_simulated(name: str) -> pd.DataFrame:
    """One simulator variant. Its behavioural columns are already extractor output."""
    variant = name.removeprefix("sim-")
    frame = pd.read_parquet(DATASET_DIR / f"sim-v2-{variant}.parquet")
    frame = frame.sort_values(["learner_id", "position"], kind="stable").reset_index(drop=True)

    out = frame.rename(columns={"concept_id": "skill_id", "position": "order_index",
                                "timestamp": "timestamp_ms", "item_b": "item_difficulty"}).copy()
    for column in ("learner_id", "item_id", "skill_id", "session_id"):
        out[column] = out[column].astype("category")
    out["response_time_s"] = out["response_time_ms"] / 1000.0
    # One attempt per item in the simulated interface — a measurement, not an absence.
    out["attempt_count"] = 1.0
    out["source"] = name
    out["learner_rte"] = np.nan
    out["solution_behaviour"] = np.nan
    out["lag_time_s"] = np.nan  # recomputed from timestamps below, like the archival path
    return out


# ---------------------------------------------------------------- session shape

def sessionise(frame: pd.DataFrame) -> pd.DataFrame:
    """Give every attempt a session, a position in it and an elapsed time.

    Simulated data already carries `session_id`. Archival data is cut on a
    30-minute gap where a wall clock exists; where it does not, the whole L3
    rung is null for that source rather than silently invented.

    Mutates ``frame`` in place: it is the loader's own fresh table and copying
    five million rows to be polite costs more than it buys.
    """
    has_clock = frame["timestamp_ms"].notna().any()

    if "session_id" not in frame.columns:
        if not has_clock:
            frame["session_id"] = pd.NA
        else:
            grouped = frame.groupby("learner_id", sort=False, observed=True)["timestamp_ms"]
            gap = frame["timestamp_ms"] - grouped.shift(1)
            starts = gap.isna() | (gap > SESSION_GAP_MS)
            index = starts.groupby(frame["learner_id"], sort=False, observed=True).cumsum()
            frame["session_id"] = (
                frame["learner_id"].cat.codes.astype(str) + "-s" + index.astype(int).astype(str)
            ).astype("category")

    if frame["session_id"].isna().all():
        frame["question_number"] = np.nan
        frame["session_duration"] = np.nan
    else:
        frame["session_id"] = frame["session_id"].astype("category")
        key = frame["session_id"]
        frame["question_number"] = frame.groupby(key, sort=False, observed=True).cumcount() + 1
        # `session_duration` may already be extractor output on simulated rows.
        if "session_duration" not in frame.columns:
            start = frame.groupby(key, sort=False, observed=True)["timestamp_ms"].transform("min")
            frame["session_duration"] = (frame["timestamp_ms"] - start) / 1000.0

    # The simulator's own `question_number` is the position inside its session,
    # which is what the archival path just computed. Keep one definition.
    if "lag_time_s" in frame and frame["lag_time_s"].isna().all() and has_clock:
        frame["lag_time_s"] = (
            frame["timestamp_ms"] - frame.groupby("learner_id", sort=False, observed=True)["timestamp_ms"].shift(1)
        ) / 1000.0
    return frame


# ------------------------------------------------------------ category lookups
# Both helpers below exist because the tables are up to 4.9 M rows and the
# obvious `Series.map` materialises that many Python strings. Working through
# the categorical's integer codes keeps every join at one small array plus an
# int lookup.

def _by_category(series: pd.Series, lookup: pd.Series, fallback: float) -> pd.Series:
    """Map a categorical column through a per-category table, filling gaps."""
    values = lookup.reindex(series.cat.categories).to_numpy(dtype="float32")
    values = np.where(np.isnan(values), np.float32(fallback), values)
    codes = series.cat.codes.to_numpy()
    return pd.Series(np.where(codes < 0, np.float32(np.nan), values[codes]), index=series.index)


def _split_labels(learners: pd.Series, split: dict) -> pd.Categorical:
    """`test` or `fold<k>` per row, assigned through the learner categories."""
    test = set(split["test"])
    fold_of = {learner: index for index, fold in enumerate(split["cv_folds"]) for learner in fold}
    labels = ["test"] + [f"fold{index}" for index in range(len(split["cv_folds"]))] + ["unassigned"]
    index_of = {label: position for position, label in enumerate(labels)}
    per_category = np.array([
        index_of["test"] if learner in test else index_of.get(
            f"fold{fold_of[learner]}" if learner in fold_of else "unassigned")
        for learner in learners.cat.categories
    ], dtype="int8")
    return pd.Categorical.from_codes(per_category[learners.cat.codes.to_numpy()], categories=labels)


# ------------------------------------------------------------- item statistics

def item_statistics(frame: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """Per-item difficulty and log-RT location/scale, fitted on training learners.

    Difficulty is `-logit(p)` with the item's accuracy shrunk toward the source
    mean by a 20-response prior: an item with four responses should not be
    declared the hardest in the bank. Items unseen in training fall back to the
    source mean, which is difficulty 0 on the logit scale by construction.
    """
    train = frame.loc[train_mask, ["item_id", "correct", "log_response_time"]]
    grouped = train.groupby("item_id", sort=False, observed=True)
    stats = grouped.agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        rt_mean=("log_response_time", "mean"),
        rt_sd=("log_response_time", "std"),
    )
    source_accuracy = float(train["correct"].mean())
    source_rt_mean = float(train["log_response_time"].mean())
    source_rt_sd = float(train["log_response_time"].std())

    shrunk = (
        (stats["accuracy"] * stats["n"] + source_accuracy * ITEM_PRIOR_RESPONSES)
        / (stats["n"] + ITEM_PRIOR_RESPONSES)
    ).clip(1e-3, 1 - 1e-3)
    stats["item_difficulty"] = -np.log(shrunk / (1 - shrunk))

    thin = stats["n"] < ITEM_PRIOR_RESPONSES
    stats["rt_mean"] = stats["rt_mean"].where(~thin, source_rt_mean).fillna(source_rt_mean)
    stats["rt_sd"] = stats["rt_sd"].where(~thin, source_rt_sd).fillna(source_rt_sd)
    stats["rt_sd"] = stats["rt_sd"].replace(0.0, source_rt_sd or 1.0)
    stats.attrs["fallback"] = {
        "item_difficulty": float(-np.log(source_accuracy / (1 - source_accuracy))),
        "rt_mean": source_rt_mean,
        "rt_sd": source_rt_sd or 1.0,
    }
    return stats


# ------------------------------------------------------------ history features
# Every function below answers the same question — "what did this learner do
# *before* this row?" — and each does it by subtracting the current row from a
# cumulative sum. That is the shift(1) guard, expressed in a form that survives
# nulls and costs one pass instead of a Python-level rolling window.

def _prior_sum(series: pd.Series, group: list[pd.Series]) -> pd.Series:
    """Cumulative sum of ``series`` over strictly earlier rows in each group."""
    return series.groupby(group, sort=False, observed=True).cumsum() - series


def _prior_mean(frame: pd.DataFrame, keys: list[str], column: str) -> tuple[pd.Series, pd.Series]:
    """Mean and count of ``column`` over strictly earlier rows in each group."""
    group = [frame[key] for key in keys]
    values = frame[column].astype(float)
    valid = values.notna().astype(float)
    prior_total = _prior_sum(values.fillna(0.0), group)
    prior_count = _prior_sum(valid, group)
    return prior_total / prior_count.replace(0.0, np.nan), prior_count


def _running_slope(key: pd.Series, x: pd.Series, y: pd.Series) -> pd.Series:
    """OLS slope of ``y`` on ``x`` over strictly earlier rows within each ``key``.

    Closed form from four cumulative sums, so it is one vectorised pass rather
    than a regression per row. Null until `MIN_SLOPE_POINTS` earlier points
    exist — a slope through three noisy binary outcomes is not a measurement.
    """
    valid = x.notna() & y.notna()
    x0, y0 = x.where(valid, 0.0).astype(float), y.where(valid, 0.0).astype(float)
    group = [key]

    n = _prior_sum(valid.astype(float), group)
    sx, sy = _prior_sum(x0, group), _prior_sum(y0, group)
    sxy, sxx = _prior_sum(x0 * y0, group), _prior_sum(x0 * x0, group)

    denominator = n * sxx - sx * sx
    slope = (n * sxy - sx * sy) / denominator.replace(0.0, np.nan)
    return slope.where(n >= MIN_SLOPE_POINTS)


# ------------------------------------------------------------------- the build

def build(name: str, splits: dict, seed: int) -> tuple[pd.DataFrame, dict]:
    frame = load_archival(name) if name in ARCHIVAL_SOURCES else load_simulated(name)
    frame = sessionise(frame)

    frame["log_response_time"] = np.log1p(frame["response_time_s"].clip(lower=0)).astype("float32")

    if name not in splits:
        splits[name] = make_splits(frame["learner_id"].cat.categories.to_numpy(), seed)
    frame["split"] = _split_labels(frame["learner_id"], splits[name])
    train_mask = frame["split"] != "test"

    stats = item_statistics(frame, train_mask)
    fallback = stats.attrs["fallback"]
    location = _by_category(frame["item_id"], stats["rt_mean"], fallback["rt_mean"])
    scale = _by_category(frame["item_id"], stats["rt_sd"], fallback["rt_sd"])
    frame["log_rt_z_item"] = ((frame["log_response_time"] - location) / scale).astype("float32")
    if "item_difficulty" not in frame.columns:  # simulated data brings its own `item_b`
        frame["item_difficulty"] = _by_category(
            frame["item_id"], stats["item_difficulty"], fallback["item_difficulty"])

    # ---- L0 history
    frame["prior_accuracy"], frame["prior_attempts"] = _prior_mean(frame, ["learner_id"], "correct")
    frame["skill_prior_accuracy"], frame["skill_prior_attempts"] = _prior_mean(
        frame, ["learner_id", "skill_id"], "correct")

    # ---- L1
    frame["log_lag_time"] = np.log1p(frame["lag_time_s"].clip(lower=0))
    prior_log_rt, _ = _prior_mean(frame, ["learner_id"], "log_response_time")
    frame["rt_drift"] = frame["log_response_time"] - prior_log_rt

    # ---- L2 / L3 that need a ratio rather than a raw duration
    if "idle_time" in frame.columns:
        frame["idle_fraction"] = (frame["idle_time"] / frame["response_time_s"].replace(0, np.nan)).clip(0, 1)

    # ---- L3 matched-difficulty decay. "Matched difficulty" means the item's
    # own base rate is removed first: raw accuracy falls when a policy hands out
    # harder items, which is a policy effect, not fatigue.
    if frame["session_id"].isna().all():
        frame["matched_difficulty_accuracy_slope"] = np.nan
        frame["matched_difficulty_speed_slope"] = np.nan
    else:
        item_accuracy = _by_category(frame["item_id"], stats["accuracy"],
                                     1 / (1 + np.exp(fallback["item_difficulty"])))
        accuracy_residual = frame["correct"] - item_accuracy
        speed_residual = frame["log_rt_z_item"]
        session = frame["session_id"]
        position = frame["question_number"].astype("float32")
        frame["matched_difficulty_accuracy_slope"] = _running_slope(
            session, position, accuracy_residual)
        frame["matched_difficulty_speed_slope"] = _running_slope(
            session, position, speed_residual)

    # ---- the label: the next attempt's outcome. Features describe attempt t
    # and the decision they inform is the selection of attempt t+1, so a
    # present-tense behavioural trace is available when the decision is made.
    by_learner = frame.groupby("learner_id", sort=False, observed=True)
    frame["next_correct"] = by_learner["correct"].shift(-1)
    # The item the label belongs to. Not a feature — see PASSTHROUGH's note —
    # but an adaptive system knows which item it is about to serve, so the
    # per-item base-rate floor is entitled to it.
    frame["next_item_id"] = by_learner["item_id"].shift(-1)
    frame["next_skill_id"] = by_learner["skill_id"].shift(-1)
    labelled = frame["next_correct"].notna()

    # A source that cannot supply a feature does not get an all-null column
    # written for it — the catalogue already says which are absent and why, and
    # 33 empty float columns over EdNet's 4.9 M rows is 900 MB of nothing.
    # `read_features()` re-adds them on the way in, so every consumer still
    # sees one schema.
    columns = [name for name in catalogue.names() if name in frame.columns]
    passthrough = [column for column in catalogue.PASSTHROUGH if column in frame.columns]
    output = frame.loc[labelled, passthrough + columns].copy()
    del frame
    output["learner_id"] = output["learner_id"].astype(str)
    output["next_correct"] = output["next_correct"].astype("int8")
    for column in columns:
        if output[column].dtype == np.float64:
            output[column] = output[column].astype("float32")

    notes = {
        "rows": int(len(output)),
        "rows_dropped_last_attempt": int((~labelled).sum()),
        "learners": int(output["learner_id"].nunique()),
        "items": int(output["item_id"].nunique()),
        "sessions": int(output["session_id"].nunique()) if output["session_id"].notna().any() else 0,
        "accuracy": round(float(output["correct"].mean()), 4),
        "test_learners": len(splits[name]["test"]),
        "features_available": len(catalogue.available(name)),
        "features_written": len(columns),
        "null_rate": {
            column: round(float(output[column].isna().mean()), 4)
            for column in columns
        },
    }
    return output, notes


def read_features(source: str, columns: list[str] | None = None) -> pd.DataFrame:
    """One feature matrix, reindexed to the full catalogue schema.

    Features a source cannot supply are not written to its parquet; they come
    back here as all-null columns, so every consumer sees the same schema
    whatever it loaded. Absence stays visible as null, never as zero.
    """
    path = PROCESSED_DIR / f"features-{source}.parquet"
    wanted = columns or (catalogue.PASSTHROUGH + catalogue.names())
    stored = pd.read_parquet(path, columns=None)
    missing = [name for name in wanted if name not in stored.columns]
    for name in missing:
        stored[name] = np.float32(np.nan) if name in catalogue.BY_NAME else pd.NA
    return stored[wanted]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=",".join(ALL_SOURCES))
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    catalogue.validate()
    sources = [name.strip() for name in args.sources.split(",") if name.strip()]
    unknown = [name for name in sources if name not in ALL_SOURCES]
    if unknown:
        raise SystemExit(f"unknown source(s): {', '.join(unknown)}. Known: {', '.join(ALL_SOURCES)}")

    splits_file = json.loads(SPLITS_PATH.read_text(encoding="utf-8"))
    splits = splits_file["sources"]
    known_before = set(splits)

    manifest = {
        "seed": args.seed,
        "git_sha": git_sha(),
        "features": len(catalogue.CATALOGUE),
        "ladder": {level: len(names) for level, names in catalogue.ladder().items()},
        "session_gap_ms": SESSION_GAP_MS,
        "item_prior_responses": ITEM_PRIOR_RESPONSES,
        "min_slope_points": MIN_SLOPE_POINTS,
        "label": "next_correct — the outcome of the learner's next attempt",
        "sources": {},
    }
    for name in sources:
        output, notes = build(name, splits, args.seed)
        path = PROCESSED_DIR / f"features-{name}.parquet"
        output.to_parquet(path, index=False)
        manifest["sources"][name] = notes
        print(f"wrote: {path.relative_to(ROOT)} ({notes['rows']:,} rows, "
              f"{notes['learners']:,} learners, {notes['features_available']}/{len(catalogue.CATALOGUE)} features)")

    if set(splits) != known_before:
        # New sources only. The archival splits were fixed in Phase 3 and are
        # never rewritten — "never re-split" (BUILD.md Phase 3).
        splits_file["sources"] = splits
        SPLITS_PATH.write_text(json.dumps(splits_file, indent=2) + "\n", encoding="utf-8")
        print(f"wrote: {SPLITS_PATH.relative_to(ROOT)} "
              f"(added {', '.join(sorted(set(splits) - known_before))})")

    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"wrote: {MANIFEST_PATH.relative_to(ROOT)} ({len(manifest['sources'])} sources)")


if __name__ == "__main__":
    main()
