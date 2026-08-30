"""Rasch (1PL) item calibration for the canonical item bank.

    python -m app.research.calibrate_items [--responses data/processed/bank-responses.csv]

Fits  logit P(correct) = theta_learner - b_item  by L2-regularised logistic
regression over one-hot learner and item dummies. If no response file exists
yet (Phase 3 has not run), items are marked `author_prior` and b is the
z-scored author difficulty — a placeholder, clearly flagged, to be replaced by
re-running this script once archival responses are mapped onto bank items.

Writes:
    artifacts/datasets/item-parameters-v1.csv
    data/items/item-bank-v1.manifest.json  (difficulty_bins, calibration blocks)
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parents[3]
BANK_PATH = ROOT / "data" / "items" / "item-bank-v1.json"
MANIFEST_PATH = ROOT / "data" / "items" / "item-bank-v1.manifest.json"
OUT_PATH = ROOT / "artifacts" / "datasets" / "item-parameters-v1.csv"
DEFAULT_RESPONSES = ROOT / "data" / "processed" / "bank-responses.csv"

N_BINS = 10


def fit_rasch(responses: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Return item_id, b, se_b, n_responses for every item with responses.

    Design matrix: learner dummies (+1) and item dummies (-1), no intercept, so
    the fitted item coefficient is b directly. The L2 penalty is what makes the
    otherwise rank-deficient parameterisation identifiable.
    """
    learners = pd.Index(sorted(responses["learner_id"].unique()))
    items = pd.Index(sorted(responses["item_id"].unique()))

    learner_dummies = pd.get_dummies(responses["learner_id"]).reindex(columns=learners, fill_value=0)
    item_dummies = pd.get_dummies(responses["item_id"]).reindex(columns=items, fill_value=0)
    design = np.hstack([learner_dummies.to_numpy(float), -item_dummies.to_numpy(float)])
    target = responses["correct"].to_numpy(int)

    model = LogisticRegression(
        C=10.0, fit_intercept=False, max_iter=2000, solver="lbfgs", random_state=seed
    )
    model.fit(design, target)
    coefficients = model.coef_.ravel()
    b = coefficients[len(learners):]

    # ponytail: item-information standard error, se = 1/sqrt(sum p(1-p)), not the
    # full inverse Fisher information of the joint fit. It ignores the
    # uncertainty in theta, so it is mildly optimistic. Upgrade to the marginal
    # (MML) standard errors if a reviewer asks for interval estimates on b.
    probability = model.predict_proba(design)[:, 1]
    information = pd.Series(probability * (1 - probability)).groupby(
        responses["item_id"].to_numpy()
    ).sum()
    counts = responses.groupby("item_id").size()

    return pd.DataFrame(
        {
            "item_id": items,
            "b": b,
            "se_b": [1.0 / np.sqrt(information.get(i, np.nan)) for i in items],
            "n_responses": [int(counts.get(i, 0)) for i in items],
        }
    )


def author_prior(bank_items: list[dict]) -> pd.DataFrame:
    """Placeholder b: z-scored author difficulty. No responses, so no se_b."""
    difficulty = np.array([item["author_difficulty"] for item in bank_items], dtype=float)
    return pd.DataFrame(
        {
            "item_id": [item["item_id"] for item in bank_items],
            "b": (difficulty - difficulty.mean()) / difficulty.std(ddof=0),
            "se_b": np.nan,
            "n_responses": 0,
        }
    )


def assign_bins(b: pd.Series) -> tuple[pd.Series, list[float]]:
    """Equal-width edges over the bank's b range -> difficulty_score 1..10.

    Equal width, not deciles: the policies read difficulty_score as a scale, so
    the mapping has to preserve spacing and span the full 1..10 range. Deciles
    collapse under the ties that a discrete author prior produces, which would
    silently hand the policies a 7-point scale.
    """
    values = b.to_numpy(float)
    edges = np.linspace(values.min(), values.max(), N_BINS + 1)[1:-1]
    bins = np.clip(np.searchsorted(edges, values, side="right") + 1, 1, N_BINS)
    return pd.Series(bins, index=b.index), [float(edge) for edge in edges]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--responses", type=Path, default=DEFAULT_RESPONSES)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    bank = json.loads(BANK_PATH.read_text(encoding="utf-8"))
    items = bank["items"]

    parameters = author_prior(items)
    parameters["source"] = "author_prior"
    fitted = 0

    if args.responses.exists():
        responses = pd.read_csv(args.responses)
        responses = responses[responses["item_id"].isin(parameters["item_id"])]
        if len(responses):
            calibrated = fit_rasch(responses, args.seed).set_index("item_id")
            index = parameters.set_index("item_id").index
            covered = index.intersection(calibrated.index)
            parameters = parameters.set_index("item_id")
            parameters.loc[covered, ["b", "se_b", "n_responses"]] = calibrated.loc[
                covered, ["b", "se_b", "n_responses"]
            ]
            parameters.loc[covered, "source"] = "rasch_1pl"
            parameters = parameters.reset_index()
            fitted = len(covered)
        else:
            print(f"note: {args.responses} has no responses for bank items; keeping author priors")
    else:
        print(f"note: {args.responses.relative_to(ROOT)} does not exist.")
        print("      Every item is marked author_prior. Phase 3 writes that file only when an")
        print("      archival skill maps onto a bank concept; see")
        print("      artifacts/datasets/archival-bank-concept-map.json for whether any does.")

    parameters["difficulty_bin"], edges = assign_bins(parameters["b"])
    parameters = parameters[["item_id", "b", "se_b", "n_responses", "source", "difficulty_bin"]]
    parameters["b"] = parameters["b"].round(6)
    parameters["se_b"] = parameters["se_b"].round(6)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    parameters.to_csv(OUT_PATH, index=False)

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    manifest["difficulty_bins"] = {
        "n_bins": N_BINS,
        "method": "equal-width over the b range",
        "edges": edges,
    }
    manifest["calibration"] = {
        "seed": args.seed,
        "responses_file": str(args.responses.relative_to(ROOT)) if args.responses.exists() else None,
        "items_fitted": fitted,
        "items_author_prior": int(len(parameters) - fitted),
        "parameters_file": str(OUT_PATH.relative_to(ROOT)),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(f"wrote: {OUT_PATH.relative_to(ROOT)} ({len(parameters)} items, {fitted} Rasch-calibrated)")
    print(f"wrote: {MANIFEST_PATH.relative_to(ROOT)} (difficulty_bins, calibration)")


if __name__ == "__main__":
    main()
