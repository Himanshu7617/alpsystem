"""Leakage-safe benchmark suite for synthetic ALP decision-time prediction."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


FEATURES = [
    "difficulty_score", "knowledge_before", "fatigue_before", "total_response_time",
    "reading_time", "time_after_last_interaction", "attempts", "skip", "option_changes",
    "mouse_distance", "mouse_speed", "hover_time", "typing_speed", "backspaces",
    "delete_frequency", "pause_duration", "question_number", "session_duration", "tab_switches",
]


def models(seed: int):
    available = {
        "logistic_regression": LogisticRegression(max_iter=1000, solver="liblinear", random_state=seed),
        "decision_tree": DecisionTreeClassifier(max_depth=7, random_state=seed),
        "random_forest": RandomForestClassifier(n_estimators=160, max_depth=10, n_jobs=-1, random_state=seed),
        "mlp": MLPClassifier(hidden_layer_sizes=(32, 16), early_stopping=True, max_iter=400, random_state=seed),
    }
    unavailable = {}
    optional = {
        "xgboost": ("xgboost", "XGBClassifier", dict(n_estimators=120, max_depth=5, learning_rate=.06, n_jobs=1, random_state=seed, eval_metric="logloss")),
        "lightgbm": ("lightgbm", "LGBMClassifier", dict(n_estimators=120, learning_rate=.06, max_depth=5, verbosity=-1, random_state=seed)),
        "catboost": ("catboost", "CatBoostClassifier", dict(iterations=120, depth=5, learning_rate=.06, verbose=False, random_seed=seed)),
    }
    for name, (module_name, class_name, kwargs) in optional.items():
        try:
            module = __import__(module_name, fromlist=[class_name])
            available[name] = getattr(module, class_name)(**kwargs)
        except Exception as exc:  # Native runtimes may be absent on a researcher machine.
            unavailable[name] = f"{type(exc).__name__}: {exc}".split("\n", 1)[0]
    return available, unavailable


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmarks/metrics.json"))
    parser.add_argument("--seed", type=int, default=20260726)
    parser.add_argument("--model-output", type=Path, default=Path("artifacts/models/best-next-correct.joblib"))
    args = parser.parse_args()
    data = pd.read_csv(args.dataset)
    missing = set(FEATURES + ["correct", "learner_id"]) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    # Predict the *next* response from information available immediately after
    # this interaction. Shifting within learner avoids using a response's own
    # behavioural trace to predict that same response.
    data = data.sort_values(["learner_id", "event_index"]).copy()
    data["next_correct"] = data.groupby("learner_id")["correct"].shift(-1)
    data = data.dropna(subset=["next_correct"])
    X, y, groups = data[FEATURES], data["next_correct"].astype(int), data["learner_id"]
    splitter = GroupShuffleSplit(n_splits=1, test_size=.25, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    numeric = Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    results = {"dataset": str(args.dataset), "seed": args.seed, "target": "next_correct", "split": {"train_rows": len(train_idx), "test_rows": len(test_idx), "train_learners": int(groups.iloc[train_idx].nunique()), "test_learners": int(groups.iloc[test_idx].nunique())}, "models": {}}
    benchmark_models, unavailable = models(args.seed)
    results["unavailable_models"] = unavailable
    fitted = {}
    for name, classifier in benchmark_models.items():
        pipeline = Pipeline([("features", ColumnTransformer([("numeric", numeric, FEATURES)])), ("classifier", classifier)])
        start = time.perf_counter()
        pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
        training_seconds = time.perf_counter() - start
        start = time.perf_counter()
        predictions = pipeline.predict(X.iloc[test_idx])
        probabilities = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
        fitted[name] = (pipeline, roc_auc_score(y.iloc[test_idx], probabilities))
        inference_ms_per_row = (time.perf_counter() - start) * 1000 / len(test_idx)
        # Feature importance extraction
        classifier_model = pipeline.named_steps["classifier"]
        importances = {}
        if hasattr(classifier_model, "feature_importances_"):
            raw_imp = classifier_model.feature_importances_
            total = sum(raw_imp) or 1.0
            importances = {feat: round(float(val / total), 6) for feat, val in zip(FEATURES, raw_imp)}
        elif hasattr(classifier_model, "coef_"):
            raw_imp = [abs(c) for c in classifier_model.coef_[0]]
            total = sum(raw_imp) or 1.0
            importances = {feat: round(float(val / total), 6) for feat, val in zip(FEATURES, raw_imp)}
        else:
            # Fallback for MLP or generic models via permutation importance
            from sklearn.inspection import permutation_importance
            perm = permutation_importance(pipeline, X.iloc[test_idx], y.iloc[test_idx], n_repeats=5, random_state=args.seed)
            raw_imp = [max(0, val) for val in perm.importances_mean]
            total = sum(raw_imp) or 1.0
            importances = {feat: round(float(val / total), 6) for feat, val in zip(FEATURES, raw_imp)}

        results["models"][name] = {
            "accuracy": round(accuracy_score(y.iloc[test_idx], predictions), 6),
            "precision": round(precision_score(y.iloc[test_idx], predictions, zero_division=0), 6),
            "recall": round(recall_score(y.iloc[test_idx], predictions, zero_division=0), 6),
            "f1": round(f1_score(y.iloc[test_idx], predictions, zero_division=0), 6),
            "roc_auc": round(roc_auc_score(y.iloc[test_idx], probabilities), 6),
            "training_seconds": round(training_seconds, 6),
            "inference_ms_per_row": round(inference_ms_per_row, 6),
            "feature_importance": importances,
        }
    best_name, (best_pipeline, best_auc) = max(fitted.items(), key=lambda item: item[1][1])
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": best_pipeline, "model_name": best_name, "roc_auc": best_auc, "features": FEATURES, "target": "next_correct", "seed": args.seed}, args.model_output)
    results["selected_model"] = {"name": best_name, "roc_auc": round(best_auc, 6), "artifact": str(args.model_output)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")

    # Export feature importance CSV
    fi_rows = []
    for model_name, model_meta in results["models"].items():
        for feat, imp_val in model_meta.get("feature_importance", {}).items():
            fi_rows.append({"model": model_name, "feature": feat, "importance_score": imp_val})
    fi_df = pd.DataFrame(fi_rows)
    fi_csv_path = args.output.parent / "feature-importance.csv"
    fi_df.to_csv(fi_csv_path, index=False)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
