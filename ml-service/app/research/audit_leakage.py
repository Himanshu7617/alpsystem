"""The leakage audit — the standing guard against Defect 2, run by `make test`.

    python -m app.research.audit_leakage [--sources ...]

Exits 0 when every check passes and non-zero, loudly, when one does not. It is
imported by ``test_features.py`` so it also fails ``pytest -q``, which is what
``make test`` runs. BUILD.md Phase 5 step 3 requires exactly that: a leakage
check that fails the build rather than one a reviewer is trusted to run.

BUILD.md Defect 2 is that the pre-Phase-0 pipeline listed ``knowledge_before``
and ``fatigue_before`` — the simulator's own latent variables — in its feature
list, and they were the largest importances in the reported model. Every
ROC-AUC in the quarantined `artifacts-legacy-invalid/` is inflated by that leak.
The three checks below are the three ways it could come back:

**A — no latent ground truth is a feature.** Checked against the catalogue, the
legacy ``BASE_FEATURES`` list, and the actual columns of every built matrix.

**B — no row sees its own future.** Checked by rebuilding a source truncated at
row *k* and asserting row *k* comes out bit-identical. This is stronger than
reading the code for ``shift(1)``: it catches a centred rolling window, a
whole-column ``mean()``, a sort that reorders, or an expanding statistic that
forgot to exclude its own row. The per-item statistics are held fixed while it
runs, so the comparison isolates the question it is asking — does a feature
look *forward in time* — from the separate question check C asks.

**C — no test learner informs a training-time statistic.** Item difficulty and
the per-item response-time standardisation are refitted with the test learners
removed; the surviving rows must be unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

from app.features.extract import FEATURE_CLASSES
from app.research import build_features, feature_catalogue as catalogue

ROOT = Path(__file__).resolve().parents[3]
PROCESSED_DIR = ROOT / "data" / "processed"

#: Rows the truncation check rebuilds. Small on purpose — the check is about
#: whether the *definition* looks forward, and one learner's worth of rows
#: exercises every history feature there is.
TRUNCATION_ROWS = 4_000

#: Features fitted across the **training population** rather than along a
#: learner's timeline: an item's difficulty and its response-time scale are
#: properties of the item, estimated once from every training learner who saw
#: it, and the two matched-difficulty slopes are residualised against them.
#: They move whenever the population moves, which is not a lookahead — so
#: check B freezes them rather than excluding them, and check C asserts the
#: thing that actually matters about them: no **test** learner contributes.
#: That is the condition under which a target-style encoding is legitimate.
POPULATION_STATISTICS = {"item_difficulty", "log_rt_z_item"}


class LeakageError(AssertionError):
    """A feature matrix that cannot be trusted. Never caught, always fatal."""


def _fail(message: str) -> None:
    raise LeakageError(message)


# ----------------------------------------------------------------- check A

def check_no_latent_features(frame: pd.DataFrame | None = None, where: str = "catalogue") -> list[str]:
    """No latent ground truth, probe or label may be a model input."""
    names = catalogue.names() if frame is None else [
        column for column in frame.columns if column in set(catalogue.names())
    ]
    for name in names:
        for prefix in catalogue.FORBIDDEN_PREFIXES:
            if name.startswith(prefix):
                _fail(f"{where}: {name!r} is latent ground truth and is listed as a feature")

    if frame is not None:
        # The passthrough columns are allowed to exist beside the features, but
        # nothing outside the two declared lists may ride along unnoticed.
        allowed = set(catalogue.names()) | set(catalogue.PASSTHROUGH)
        stowaways = sorted(set(frame.columns) - allowed)
        if stowaways:
            _fail(f"{where}: undeclared columns in the matrix: {stowaways}")

    # The extractor is the other place a feature can be born.
    for name in FEATURE_CLASSES:
        for prefix in catalogue.FORBIDDEN_PREFIXES:
            if name.startswith(prefix):
                _fail(f"app.features.extract produces {name!r}, which is latent ground truth")
    return names


def check_legacy_features_are_not_used() -> None:
    """The exact columns of Defect 2 must not have come back through the back door."""
    from app.model import features as legacy

    leaked = [name for name in legacy.ENGINEERED_FEATURES
              if any(name.startswith(prefix) for prefix in catalogue.FORBIDDEN_PREFIXES)]
    if not leaked:
        _fail("app.model.features no longer contains the Defect 2 columns — delete this check "
              "and the module together rather than leaving a guard over nothing")
    shared = sorted(set(legacy.ENGINEERED_FEATURES) & set(catalogue.names()))
    for name in shared:
        if any(name.startswith(prefix) for prefix in catalogue.FORBIDDEN_PREFIXES):
            _fail(f"catalogue re-adopted the leaking legacy feature {name!r}")


# ----------------------------------------------------------------- check B

def check_no_future_information(source: str, splits: dict, seed: int, rows: int = TRUNCATION_ROWS) -> dict:
    """Rebuild the source truncated at row *k*; row *k* must be unchanged.

    A feature that looked forward — a centred window, an un-shifted expanding
    mean, a statistic fitted over the whole column — changes when the rows
    after it are removed. One that does not, cannot have seen them.
    """
    with _frozen_item_statistics():
        full, _ = build_features.build(source, splits, seed)
        if len(full) < 10:
            _fail(f"{source}: too few rows to audit ({len(full)})")
        head = full.head(rows)
        truncated = _build_truncated(source, splits, seed, head)

    features = [name for name in catalogue.names()
                if name in head.columns and head[name].notna().any()]
    compared = truncated.head(len(head))
    offenders = []
    for name in features:
        left = head[name].to_numpy(dtype=float, na_value=np.nan)
        right = compared[name].to_numpy(dtype=float, na_value=np.nan)
        if not np.allclose(left, right, rtol=1e-9, atol=1e-9, equal_nan=True):
            differing = int(np.sum(~np.isclose(left, right, rtol=1e-9, atol=1e-9, equal_nan=True)))
            offenders.append(f"{name} ({differing} rows differ)")
    if offenders:
        _fail(f"{source}: features change when later rows are removed — they read the future: "
              + ", ".join(offenders))

    return {"rows_compared": int(len(head)), "features_compared": len(features)}


@contextmanager
def _frozen_item_statistics():
    """Fit the per-item table once, then hand the same one to every rebuild.

    Without this, truncating the data changes which learners answered each item
    and every population statistic shifts a little — a real effect, but not the
    one check B is looking for. Frozen, any difference left is a feature
    reading its own future.
    """
    real = build_features.item_statistics
    memo: dict[str, pd.DataFrame] = {}

    def frozen(frame: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
        if "stats" not in memo:
            memo["stats"] = real(frame, train_mask)
        return memo["stats"]

    build_features.item_statistics = frozen
    try:
        yield
    finally:
        build_features.item_statistics = real


def _build_truncated(source: str, splits: dict, seed: int, head: pd.DataFrame) -> pd.DataFrame:
    """The same build, over a prefix of the raw table.

    The prefix is taken by learner so a truncated sequence is still a whole
    learner's history up to a point, which is the situation a live system is
    actually in.
    """
    keep = set(head["learner_id"].astype(str))
    original_loader = build_features.load_archival if source in build_features.ARCHIVAL_SOURCES \
        else build_features.load_simulated

    def loader(name: str) -> pd.DataFrame:
        frame = original_loader(name)
        subset = frame[frame["learner_id"].astype(str).isin(keep)].copy()
        # Truncate the last learner's sequence mid-way: the row before the cut
        # must not know that the rows after it were removed.
        last = subset["learner_id"].astype(str).iloc[-1]
        tail = subset[subset["learner_id"].astype(str) == last]
        return pd.concat([subset[subset["learner_id"].astype(str) != last],
                          tail.head(max(2, len(tail) // 2))]).reset_index(drop=True)

    return _with_loaders(source, loader, lambda name: build_features.build(name, splits, seed)[0])


def _with_loaders(source: str, loader, run):
    """Run ``run`` with the source loaders swapped for ``loader``."""
    archival, simulated = build_features.load_archival, build_features.load_simulated
    build_features.load_archival = loader
    build_features.load_simulated = loader
    try:
        return run(source)
    finally:
        build_features.load_archival, build_features.load_simulated = archival, simulated


# ----------------------------------------------------------------- check C

def check_item_statistics_exclude_test(source: str, splits: dict, seed: int) -> dict:
    """Item difficulty and the RT standardisation must not see test learners."""
    full, _ = build_features.build(source, splits, seed)
    test_learners = set(splits[source]["test"])
    if not test_learners:
        _fail(f"{source}: no held-out test learners in splits.json")

    original = build_features.load_archival if source in build_features.ARCHIVAL_SOURCES \
        else build_features.load_simulated

    def loader(name: str) -> pd.DataFrame:
        frame = original(name)
        return frame[~frame["learner_id"].astype(str).isin(test_learners)].reset_index(drop=True)

    without = _with_loaders(source, loader, lambda name: build_features.build(name, splits, seed)[0])

    # Dropping learners preserves row order, so the two frames line up
    # positionally. A merge would need a key, and `order_index` restarts per
    # session in the simulated data — the cartesian product that produces is a
    # comparison of unrelated rows dressed up as a passing check.
    train_rows = full[full["split"] != "test"].reset_index(drop=True)
    if train_rows.empty:
        _fail(f"{source}: dropping the test learners left no comparable training rows")
    if len(train_rows) != len(without):
        _fail(f"{source}: removing test learners changed the training row count "
              f"({len(train_rows):,} -> {len(without):,})")
    if not train_rows["learner_id"].equals(without["learner_id"].reset_index(drop=True)):
        _fail(f"{source}: removing test learners reordered the training rows")

    for name in sorted(POPULATION_STATISTICS):
        left = train_rows[name].to_numpy(dtype=float, na_value=np.nan)
        right = without[name].to_numpy(dtype=float, na_value=np.nan)
        if not np.allclose(left, right, rtol=1e-9, atol=1e-9, equal_nan=True):
            _fail(f"{source}: {name} changes when test learners are removed — it was fitted on them")
    return {"train_rows_compared": int(len(train_rows))}


# ----------------------------------------------------------------- check D

def check_built_matrices() -> dict:
    """Whatever is on disk right now must obey check A and carry a label."""
    found = {}
    for source in build_features.ALL_SOURCES:
        path = PROCESSED_DIR / f"features-{source}.parquet"
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        check_no_latent_features(frame, where=str(path.relative_to(ROOT)))
        if "next_correct" not in frame.columns:
            _fail(f"{path.name}: no label column")
        if frame["next_correct"].isna().any():
            _fail(f"{path.name}: null labels — the last attempt of a learner must be dropped, not kept")
        if not set(frame["split"].unique()) >= {"test"}:
            _fail(f"{path.name}: no held-out test rows")
        found[source] = {"rows": int(len(frame)), "columns": int(frame.shape[1])}
    if not found:
        _fail("no feature matrices on disk — run `make features` first")
    return found


# --------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default="assistments_2009,sim-V0",
                        help="sources to rebuild for the truncation and split checks")
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--skip-rebuild", action="store_true",
                        help="run only the checks that read what is already on disk")
    args = parser.parse_args()

    catalogue.validate()
    print(f"ok  A  catalogue: {len(check_no_latent_features())} features, no latent ground truth")
    check_legacy_features_are_not_used()
    print("ok  A  legacy app.model.features stays quarantined and unadopted")

    on_disk = check_built_matrices()
    print(f"ok  A  built matrices: {len(on_disk)} sources, no undeclared or latent columns")

    if not args.skip_rebuild:
        splits = json.loads((PROCESSED_DIR / "splits.json").read_text(encoding="utf-8"))["sources"]
        for source in (name.strip() for name in args.sources.split(",") if name.strip()):
            future = check_no_future_information(source, splits, args.seed)
            print(f"ok  B  {source}: {future['features_compared']} features unchanged by truncation "
                  f"({future['rows_compared']:,} rows)")
            fitted = check_item_statistics_exclude_test(source, splits, args.seed)
            print(f"ok  C  {source}: item statistics unchanged by removing test learners "
                  f"({fitted['train_rows_compared']:,} rows)")

    print("leakage audit passed")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except LeakageError as error:
        print(f"LEAKAGE: {error}", file=sys.stderr)
        sys.exit(1)
