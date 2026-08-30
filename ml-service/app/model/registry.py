"""The one way a trained model is saved and the one way it is loaded.

Phase 8.5 exists because "it trained once" and "another process can load it and
get the number the paper reports" had drifted apart. The concrete symptom was
`predictor.py`, which the FastAPI service still calls: it loads
``artifacts/models/best-next-correct.joblib``, a file the research pipeline
stopped producing before Phase 0. `load_model()` therefore returns ``None`` in
production and nobody notices, because the fallback is silent.

So every trained model is a **directory with a fixed contract**, and every
consumer — the policy arms, the ml-service, the figure scripts — goes through
:func:`load`. A model that cannot state its features, its metrics and how it
was trained is not usable as evidence, so saving one without them is an error
rather than a warning.

    artifacts/models/<name>/
        model.joblib            the fitted estimator (or pipeline)
        feature_schema.json     exact column list and dtypes it expects
        metrics.json            held-out numbers, and which split they came from
        calibration.json        the calibrator and its diagnostics, or why none
        training_manifest.json  seed, dataset, split strategy, features, target,
                                hyperparameters, split sizes, git sha, profile

Nothing here writes a wall-clock timestamp: two runs of the same seed on the
same commit must produce the same manifest, so the run identifier is derived
from the inputs instead.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "artifacts" / "models"

FILES = ("model.joblib", "feature_schema.json", "metrics.json",
         "calibration.json", "training_manifest.json")

#: A manifest missing any of these cannot support a claim about the model, so
#: :func:`save` refuses rather than writing a half-documented artifact.
#: `features` and `target` are separate arguments to :func:`save` and are
#: merged into the written manifest, so they are not listed here.
REQUIRED_MANIFEST = ("seed", "dataset", "split_strategy",
                     "hyperparameters", "split_sizes", "git_sha")


def git_sha() -> str:
    """The commit the artifact was produced from, or ``unknown`` outside git."""
    try:
        output = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=10)
        return output.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def run_id(name: str, manifest: dict, features: list[str]) -> str:
    """A reproducible identifier: same inputs, same commit, same id."""
    payload = json.dumps({"name": name, "features": features,
                          "seed": manifest.get("seed"),
                          "dataset": manifest.get("dataset"),
                          "git_sha": manifest.get("git_sha")}, sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class LoadedModel:
    """A trained artifact plus everything needed to use it honestly."""

    name: str
    path: Path
    model: object
    features: list[str]
    target: str
    schema: dict
    metrics: dict
    calibration: dict
    manifest: dict

    def frame_for(self, frame):
        """The estimator's own columns, in its own order.

        A loader that silently accepts differently-shaped input is how a
        service ends up serving predictions from columns the model never saw.
        """
        missing = [name for name in self.features if name not in frame.columns]
        if missing:
            raise KeyError(
                f"{self.name} needs {len(self.features)} features; "
                f"{len(missing)} missing, first few: {missing[:5]}")
        return frame[self.features]

    def predict_proba(self, frame):
        return self.model.predict_proba(self.frame_for(frame))[:, 1]


def save(name: str, model, *, features: list[str], target: str, metrics: dict,
         calibration: dict, manifest: dict, dtypes: dict | None = None) -> Path:
    """Write one artifact directory. Every field is required for a reason."""
    import joblib

    missing = [key for key in REQUIRED_MANIFEST if key not in manifest]
    if missing:
        raise ValueError(f"training_manifest for {name!r} is missing {missing} — "
                         "a model that cannot say how it was trained is not evidence")
    directory = MODEL_DIR / name
    directory.mkdir(parents=True, exist_ok=True)
    complete = {**manifest, "name": name, "features": list(features), "target": target,
                "n_features": len(features),
                "run_id": run_id(name, manifest, list(features))}

    joblib.dump(model, directory / "model.joblib")
    _write(directory / "feature_schema.json",
           {"target": target, "n_features": len(features),
            "features": [{"name": column, "dtype": (dtypes or {}).get(column, "float64")}
                         for column in features]})
    _write(directory / "metrics.json", metrics)
    _write(directory / "calibration.json", calibration)
    _write(directory / "training_manifest.json", complete)
    load.cache_clear()
    return directory


def _write(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")


@lru_cache(maxsize=None)
def load(name: str) -> LoadedModel:
    """Load one artifact. A directory missing part of the contract is an error."""
    import joblib

    directory = MODEL_DIR / name
    absent = [file for file in FILES if not (directory / file).exists()]
    if absent:
        raise FileNotFoundError(
            f"artifacts/models/{name}/ is not a complete model artifact — missing "
            f"{absent}. Train it with the `make` target that owns it; do not "
            f"hand-assemble the directory.")
    read = lambda file: json.loads((directory / file).read_text(encoding="utf-8"))
    schema, manifest = read("feature_schema.json"), read("training_manifest.json")
    return LoadedModel(
        name=name, path=directory, model=joblib.load(directory / "model.joblib"),
        features=[entry["name"] for entry in schema["features"]],
        target=schema["target"], schema=schema, metrics=read("metrics.json"),
        calibration=read("calibration.json"), manifest=manifest)


def available() -> list[str]:
    """Every complete artifact directory under `artifacts/models/`."""
    if not MODEL_DIR.exists():
        return []
    return sorted(entry.name for entry in MODEL_DIR.iterdir()
                  if entry.is_dir() and all((entry / file).exists() for file in FILES))


def demo() -> None:
    """Self-check: save, reload, reproduce the metrics, and reject bad input."""
    import numpy as np
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    rng = np.random.default_rng(20260821)
    frame = pd.DataFrame({"a": rng.normal(size=400), "b": rng.normal(size=400)})
    frame["y"] = (frame["a"] + 0.5 * rng.normal(size=400) > 0).astype(int)
    fitted = LogisticRegression().fit(frame[["a", "b"]], frame["y"])
    auc = float(roc_auc_score(frame["y"], fitted.predict_proba(frame[["a", "b"]])[:, 1]))

    saved = save("_registry_selfcheck", fitted, features=["a", "b"], target="y",
                 metrics={"roc_auc": auc, "split": "in-sample (self-check only)"},
                 calibration={"method": None, "reason": "self-check"},
                 manifest={"seed": 20260821, "dataset": "synthetic",
                           "split_strategy": "none", "hyperparameters": {},
                           "split_sizes": {"train": 400}, "git_sha": git_sha()})
    try:
        loaded = load("_registry_selfcheck")
        assert loaded.features == ["a", "b"] and loaded.target == "y"
        assert abs(float(roc_auc_score(frame["y"], loaded.predict_proba(frame))) - auc) < 1e-12, \
            "a reloaded artifact must reproduce its recorded metric"
        # Column order must not matter; a missing column must be an error.
        shuffled = frame[["b", "a", "y"]]
        assert np.allclose(loaded.predict_proba(shuffled), loaded.predict_proba(frame))
        try:
            loaded.predict_proba(frame[["a"]])
        except KeyError:
            pass
        else:
            raise AssertionError("a missing feature must raise, not be guessed")
        assert run_id("x", {"seed": 1}, ["a"]) == run_id("x", {"seed": 1}, ["a"]), \
            "the run id must be reproducible"
        try:
            save("_registry_selfcheck", fitted, features=["a"], target="y", metrics={},
                 calibration={}, manifest={"seed": 1})
        except ValueError:
            pass
        else:
            raise AssertionError("an incomplete manifest must be refused")
        print(f"registry: ok ({', '.join(available()) or 'no artifacts yet'})")
    finally:
        for file in FILES:
            (saved / file).unlink(missing_ok=True)
        saved.rmdir()
        load.cache_clear()


if __name__ == "__main__":
    demo()
