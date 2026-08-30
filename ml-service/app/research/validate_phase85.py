"""Phase 8.5 — one end-to-end run, written down. `make integrate`.

    python -m app.research.validate_phase85 --seed 20260821

Proves the chain the phase exists to establish, at the `smoke` size:

    processed data ──► trained artifact ──► registry.load() ──► policy ──► closed loop

and writes the result to ``artifacts/evaluation/phase-8.5-validation.json``,
pass or fail. A check that cannot run because its input is missing is recorded
as ``blocked`` with the `make` target that produces the input — never as a pass.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "artifacts" / "evaluation"
OUT = EVAL_DIR / "phase-8.5-validation.json"


def check(name: str, detail: str = "", status: str = "pass", **extra) -> dict:
    return {"check": name, "status": status, "detail": detail, **extra}


def artifact_contract() -> list[dict]:
    """Every model artifact is complete, documented, and reloadable."""
    from app.model import registry

    names = registry.available()
    if not names:
        return [check("model_artifact_contract", status="blocked",
                      detail="no complete artifact under artifacts/models/ — "
                             "run `make train-baselines` (Phase 8.5 step 1)")]
    results = []
    for name in names:
        loaded = registry.load(name)
        missing = [key for key in registry.REQUIRED_MANIFEST if key not in loaded.manifest]
        results.append(check(
            f"artifact:{name}",
            status="pass" if not missing else "fail",
            detail=f"{len(loaded.features)} features, target {loaded.target!r}, "
                   f"seed {loaded.manifest.get('seed')}, "
                   f"git {str(loaded.manifest.get('git_sha'))[:8]}"
                   + (f"; manifest missing {missing}" if missing else ""),
            metrics={key: value for key, value in loaded.metrics.items()
                     if isinstance(value, (int, float))}))
    return results


def artifact_reproduces_its_metric(seed: int) -> dict:
    """Reload the saved model and re-score it on the rows metrics.json names.

    A metric a reloaded artifact cannot reproduce is a metric nobody can check.
    """
    from sklearn.metrics import roc_auc_score

    from app.model import registry
    from app.research import study1

    names = [name for name in registry.available() if name.startswith("study1-")]
    if not names:
        return check("artifact_reproduces_metric", status="blocked",
                     detail="no study1 artifact — run `make train-baselines`")
    name = names[0]
    loaded = registry.load(name)
    source = loaded.manifest["dataset"]
    frame = study1.load(source, seed)
    test = frame[frame["split"] == "test"]
    scored = float(roc_auc_score(test[loaded.target].to_numpy(),
                                 loaded.predict_proba(test)))
    recorded = loaded.metrics.get("auc") or loaded.metrics.get("roc_auc")
    if recorded is None:
        return check("artifact_reproduces_metric", status="fail",
                     detail=f"{name} records no AUC to compare against")
    gap = abs(scored - float(recorded))
    return check("artifact_reproduces_metric",
                 status="pass" if gap < 1e-6 else "fail",
                 detail=f"{name}: recorded {float(recorded):.6f}, "
                        f"reloaded {scored:.6f}, gap {gap:.2e}",
                 recorded=float(recorded), reloaded=scored)


def closed_loop_smoke(seed: int) -> dict:
    """data → gate → policy → environment, at the smallest useful size."""
    from app.model import validation_gate
    from app.policy import policies as arms
    from app.research.closed_loop import ClosedLoop
    from app.research.evaluate_policies import observable_thresholds

    started = time.perf_counter()
    gate = validation_gate.load()
    environment = ClosedLoop("V0", seed, 40, budget=40)
    policy = arms.build("model_L4", 40, seed, budget=40,
                        thresholds=observable_thresholds(), gate=gate)
    policy.epsilon = 0.1
    rows = [None] * 40
    decisions = 0
    for _ in range(40):
        if environment.done:
            break
        actions = policy.act(environment.views(policy.rung, rows))
        outcomes = environment.step(actions)
        policy.observe(outcomes)
        rows = getattr(policy, "last_rows", outcomes)
        decisions += sum(1 for action in actions if action is not None)
    results = environment.results()
    with_propensity = sum(1 for action in actions if action and "propensity" in action)
    return check(
        "closed_loop_smoke",
        status="pass" if decisions and results and with_propensity else "fail",
        detail=f"{len(results)} learners, {decisions} decisions, "
               f"gate admitted {sorted(gate.admitted) or 'nothing'}, "
               f"{round(time.perf_counter() - started, 1)}s",
        learners=len(results), decisions=decisions,
        admitted=sorted(gate.admitted), rejected=sorted(gate.report.get("rejected", [])))


def service_loads_the_same_artifact() -> dict:
    """The ml-service must serve the artifact the pipeline produced."""
    from app.model.predictor import load_model

    artifact = load_model()
    if artifact is None:
        return check("service_loads_registry_artifact", status="blocked",
                     detail="predictor.load_model() found nothing — expected until "
                            "`make train-baselines` writes a registry artifact")
    return check("service_loads_registry_artifact",
                 status="pass" if artifact.get("source") == "registry" else "fail",
                 detail=f"{artifact.get('model_name')} via {artifact.get('source')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260821)
    args = parser.parse_args()

    from app.research import runprofile

    started = time.perf_counter()
    checks = [*artifact_contract(), artifact_reproduces_its_metric(args.seed),
              closed_loop_smoke(args.seed), service_loads_the_same_artifact()]
    tally = {status: sum(1 for entry in checks if entry["status"] == status)
             for status in ("pass", "fail", "blocked")}
    report = {
        "phase": 8.5, "seed": args.seed, "profile": runprofile.name(),
        "chain": "data/processed → trained artifact → registry.load() → policy → closed loop",
        "seconds": round(time.perf_counter() - started, 1),
        "summary": tally, "checks": checks,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote: {OUT.relative_to(ROOT)} — "
          f"{tally['pass']} pass, {tally['fail']} fail, {tally['blocked']} blocked")
    for entry in checks:
        print(f"  [{entry['status']:7s}] {entry['check']}: {entry['detail']}")
    if tally["fail"]:
        raise SystemExit("Phase 8.5 validation failed — see the report above.")


if __name__ == "__main__":
    main()
