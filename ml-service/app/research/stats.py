"""Phase 11 — the statistical treatment a reviewer will ask for.

    python -m app.research.stats [--seed 20260821] [--bootstrap 10000]

Reads what the earlier phases wrote and writes two things:

* ``artifacts/evaluation/statistical-report.json`` — every number this phase
  computes, with the test that produced it, its assumption checks and its
  sample size.
* ``artifacts/evaluation/tables/*.csv`` and the matching ``*.tex`` — the result
  tables the paper needs, generated rather than transcribed.

Four decisions this module makes deliberately, because each one is a way the
prior version of this repository overstated its results:

1. **Effect sizes are unpaired.** The same-seed cohort makes a paired *d*
   available and it is the wrong statistic to headline: dividing by the
   standard deviation of the *difference* between two arms on the same learner
   removes the between-learner variance that a reader assumes is in the
   denominator, which is how the prior work reached *d* = 26. The paired test
   is still the correct **significance** test and Phase 8 already reports it;
   what is reported here as an effect size is Hedges' *g* on independent
   samples, with a 10,000-resample bootstrap CI.
2. **Interactions are nested.** Every per-item model has crossed random
   intercepts for learner and item. Treating 745,192 attempts by 3,000 learners
   as 745,192 independent observations is the standard error inflation this
   literature is full of.
3. **Items-to-mastery is censored,** heavily: the budget is 200 items and most
   learners do not master inside it. Its mean is a lower bound on a difference,
   not a difference, so it gets a Kaplan-Meier treatment and a log-rank test
   alongside the uncensored co-primary, knowledge gain.
4. **Corrected and raw p-values are both reported.** Holm across each family;
   the family is named in the output rather than left to the reader.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from app.research import build_features, study1

ROOT = Path(__file__).resolve().parents[3]
EVAL_DIR = ROOT / "artifacts" / "evaluation"
BENCH_DIR = ROOT / "artifacts" / "benchmarks"
TABLE_DIR = EVAL_DIR / "tables"
REPORT_PATH = EVAL_DIR / "statistical-report.json"

#: The arm every other arm is compared against on the forest plot. It is the
#: strongest hand-written rule, so "the model arm beats the reference" is the
#: claim a reviewer cares about — not "the model arm beats random".
REFERENCE_ARM = "rule_improved"

#: Pre-registration §9.2. `knowledge_gain` first: it is the co-primary that the
#: 200-item budget does not censor.
OUTCOMES = ["knowledge_gain", "items_to_mastery", "time_to_mastery_seconds"]

#: The effect size a real RCT in this literature would be powered for
#: (BUILD.md §4 guardrail 5: the reference points are +1.25 % AUC and g ≈ 0.3).
RCT_TARGET_G = 0.3
ALPHA = 0.05

#: Crossed random effects are fitted on a subsample. statsmodels builds the
#: variance-component design densely, so a fit over every learner and every
#: item of a 250,000-row matrix does not fit in memory on this machine.
# ponytail: subsampled crossed fit. Move to a sparse/Bayesian fitter (lme4,
# bambi) if a reviewer asks for the full-data variance decomposition.
MIXED_MAX_ROWS = 8_000
MIXED_MAX_LEARNERS = 200
MIXED_MAX_ITEMS = 200


# ------------------------------------------------------------- effect sizes

def hedges_g(treatment: np.ndarray, control: np.ndarray) -> tuple[float, float]:
    """(Cohen's d, Hedges' g) on **independent** samples."""
    n1, n2 = len(treatment), len(control)
    if n1 < 2 or n2 < 2:
        return float("nan"), float("nan")
    pooled = np.sqrt((((n1 - 1) * np.var(treatment, ddof=1))
                      + ((n2 - 1) * np.var(control, ddof=1))) / (n1 + n2 - 2))
    if pooled == 0:
        return 0.0, 0.0
    d = float((np.mean(treatment) - np.mean(control)) / pooled)
    correction = 1 - (3 / (4 * (n1 + n2) - 9))     # Hedges' small-sample factor
    return d, float(d * correction)


def bootstrap_g(treatment: np.ndarray, control: np.ndarray, seed: int,
                draws: int) -> list[float]:
    """Percentile CI for g, resampling each arm independently."""
    if len(treatment) < 2 or len(control) < 2:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    left = rng.choice(treatment, size=(draws, len(treatment)), replace=True)
    right = rng.choice(control, size=(draws, len(control)), replace=True)
    n1, n2 = len(treatment), len(control)
    pooled = np.sqrt((((n1 - 1) * left.var(axis=1, ddof=1))
                      + ((n2 - 1) * right.var(axis=1, ddof=1))) / (n1 + n2 - 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        values = (left.mean(axis=1) - right.mean(axis=1)) / pooled
    values = values[np.isfinite(values)] * (1 - (3 / (4 * (n1 + n2) - 9)))
    if values.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]


def assumption_checks(treatment: np.ndarray, control: np.ndarray, seed: int) -> dict:
    """Normality and equal variance — reported, not used to pick a test.

    Both tests are run at every n, and at these sample sizes both reject on
    trivial departures; that is why the headline test is Welch (no equal
    variance assumption) and Mann-Whitney is reported beside it.
    """
    rng = np.random.default_rng(seed)

    def sample(values: np.ndarray) -> np.ndarray:
        return values if len(values) <= 5000 else rng.choice(values, 5000, replace=False)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        shapiro_a = scipy_stats.shapiro(sample(treatment)) if len(treatment) >= 3 else None
        shapiro_b = scipy_stats.shapiro(sample(control)) if len(control) >= 3 else None
        levene = scipy_stats.levene(treatment, control) if min(len(treatment), len(control)) >= 2 else None
    return {
        "shapiro_p_treatment": float(shapiro_a.pvalue) if shapiro_a else None,
        "shapiro_p_control": float(shapiro_b.pvalue) if shapiro_b else None,
        "levene_p": float(levene.pvalue) if levene else None,
        "note": "both reject on trivial departures at this n; Welch and Mann-Whitney "
                "are reported so neither result depends on them",
    }


def compare(treatment: np.ndarray, control: np.ndarray, seed: int, draws: int,
            checks: bool = True) -> dict:
    treatment = np.asarray(treatment, dtype=float)
    control = np.asarray(control, dtype=float)
    treatment = treatment[np.isfinite(treatment)]
    control = control[np.isfinite(control)]
    d, g = hedges_g(treatment, control)
    low, high = bootstrap_g(treatment, control, seed, draws)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        welch = scipy_stats.ttest_ind(treatment, control, equal_var=False)
        mann = scipy_stats.mannwhitneyu(treatment, control, alternative="two-sided")
    return {
        "n_treatment": int(len(treatment)), "n_control": int(len(control)),
        "mean_treatment": float(np.mean(treatment)) if len(treatment) else float("nan"),
        "mean_control": float(np.mean(control)) if len(control) else float("nan"),
        "difference": float(np.mean(treatment) - np.mean(control))
        if len(treatment) and len(control) else float("nan"),
        "cohens_d_unpaired": d, "hedges_g": g,
        "g_ci_low": low, "g_ci_high": high,
        "welch_t": float(welch.statistic), "p_welch": float(welch.pvalue),
        "mannwhitney_u": float(mann.statistic), "p_mannwhitney": float(mann.pvalue),
        "test": "Welch two-sample t (headline), Mann-Whitney U (rank), "
                f"bootstrap {draws} resamples for the CI on g",
        "assumptions": assumption_checks(treatment, control, seed) if checks else None,
    }


# ------------------------------------------------------------ mixed effects

def _subsample(frame: pd.DataFrame, learner: str, item: str, seed: int) -> pd.DataFrame:
    """Cap learners, items and rows so a crossed fit terminates (see the note above)."""
    rng = np.random.default_rng(seed)
    learners = frame[learner].dropna().unique()
    if len(learners) > MIXED_MAX_LEARNERS:
        keep = rng.choice(learners, MIXED_MAX_LEARNERS, replace=False)
        frame = frame[frame[learner].isin(keep)]
    counts = frame[item].value_counts()
    frame = frame[frame[item].isin(counts.index[:MIXED_MAX_ITEMS])]
    if len(frame) > MIXED_MAX_ROWS:
        frame = frame.sample(MIXED_MAX_ROWS, random_state=seed)
    return frame.copy()


def crossed_mixedlm(frame: pd.DataFrame, formula: str, learner: str, item: str) -> dict:
    """One MixedLM with crossed learner and item random intercepts.

    statsmodels has no crossed-effects syntax; the documented workaround is one
    artificial group covering every row with the two factors entered as
    variance components, which is what this does.
    """
    from statsmodels.regression.mixed_linear_model import MixedLM

    data = frame.copy()
    data["_all"] = 1
    variance = {"learner": f"0 + C({learner})", "item": f"0 + C({item})"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = MixedLM.from_formula(formula, groups="_all", vc_formula=variance,
                                     re_formula="0", data=data)
        result = model.fit(method="lbfgs", maxiter=300)
        if not result.converged or not np.isfinite(result.bse).all():
            # lbfgs stalls on a variance component near zero. Powell is slower
            # and derivative-free; taking it only on failure keeps the common
            # case fast and stops a non-converged fit being reported as one.
            fallback = model.fit(method="powell", maxiter=2000)
            if fallback.converged and np.isfinite(fallback.bse).all():
                result = fallback
        confidence = result.conf_int()
    fixed = {}
    for name in result.fe_params.index:
        fixed[name] = {
            "coefficient": float(result.fe_params[name]),
            "std_error": float(result.bse[name]),
            "ci_low": float(confidence.loc[name, 0]),
            "ci_high": float(confidence.loc[name, 1]),
            "z": float(result.tvalues[name]),
            "p_value": float(result.pvalues[name]),
        }
    components = {name: float(value) for name, value in result.vcomp.items()} \
        if hasattr(result.vcomp, "items") else \
        dict(zip(model.exog_vc.names, [float(value) for value in result.vcomp]))
    residual = float(result.scale)
    total = sum(components.values()) + residual
    return {
        "formula": formula, "n": int(len(data)),
        "levels": {"learner": int(data[learner].nunique()), "item": int(data[item].nunique())},
        "fixed_effects": fixed,
        "variance": {**components, "residual": residual},
        "variance_share": {name: value / total for name, value in
                           {**components, "residual": residual}.items()},
        "converged": bool(result.converged),
    }


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(clipped / (1 - clipped))


def correctness_models(seed: int) -> dict:
    """Does the behavioural rung still add value once clustering is accounted for?

    The two predictors are the out-of-fold probability of the lowest rung and
    the *increment* the highest live rung adds to it, both on the logit scale.
    A positive, CI-excludes-zero coefficient on the increment is the mixed-model
    version of the ablation ladder's ΔAUC.
    """
    predictions = pd.read_parquet(BENCH_DIR / "study1-predictions.parquet")
    output = {}
    for source, block in predictions.groupby("source", observed=True):
        # study1 writes its predictions in the order of the frame it fitted on,
        # minus test rows, and that frame is deterministic given the seed. Rather
        # than trust that, the join is asserted row for row before item ids are
        # taken from it: a silent misalignment here would attach the wrong item
        # to every attempt.
        frame = study1.load(source, seed)
        frame = frame[frame["split"].astype(str) != "test"].reset_index(drop=True)
        block = block.reset_index(drop=True)
        if len(frame) != len(block):
            output[source] = {"skipped": f"{len(frame)} feature rows vs {len(block)} "
                                         "prediction rows — rerun `make train-baselines`"}
            continue
        aligned = np.array_equal(frame["learner_id"].astype(str).to_numpy(),
                                 block["learner_id"].astype(str).to_numpy())
        assert aligned, f"{source}: prediction rows are not the feature rows in order"
        assert (frame[study1.LABEL].to_numpy() == block[study1.LABEL].to_numpy()).all()

        # A rung a source cannot supply features for is written as an all-null
        # column, so "the highest rung" is the highest one that was actually
        # scored — not the highest one in the file.
        rungs = [column for column in block.columns
                 if column.endswith("|lightgbm") and block[column].notna().mean() > 0.5]
        if len(rungs) < 2:
            output[source] = {"skipped": f"{len(rungs)} scored lightgbm rungs — "
                                         "nothing to compare"}
            continue
        low, high = rungs[0], rungs[-1]
        data = pd.DataFrame({
            "y": block[study1.LABEL].to_numpy(dtype=float),
            "base_logit": _logit(block[low].to_numpy()),
            "rung_increment": _logit(block[high].to_numpy()) - _logit(block[low].to_numpy()),
            "learner_id": frame["learner_id"].astype(str).to_numpy(),
            "item_id": frame["next_item_id"].astype(str).to_numpy(),
        }).dropna()
        # Standardised: the increment is a tenth the scale of the base logit, and
        # the two together push lbfgs into a flat region. Per-sd coefficients are
        # also the comparable ones across four datasets on one chart.
        scales = {}
        for column in ("base_logit", "rung_increment"):
            deviation = float(data[column].std(ddof=0)) or 1.0
            scales[column] = deviation
            data[column] = (data[column] - data[column].mean()) / deviation
        fit = crossed_mixedlm(_subsample(data, "learner_id", "item_id", seed),
                              "y ~ base_logit + rung_increment", "learner_id", "item_id")
        output[source] = {
            "low_rung": low, "high_rung": high,
            "rows_available": int(len(data)),
            "predictor_scale": {"standardised": True, "sd_of_raw_logit": scales},
            "interpretation": "rung_increment > 0 with a CI excluding zero means the "
                              "higher rung adds signal the lower rung does not, after "
                              "learner and item clustering",
            **fit,
        }
    return output


def closed_loop_item_model(seed: int) -> dict:
    """Per-item correctness under each arm, learner and item crossed."""
    path = EVAL_DIR / "decision-log-v0.parquet"
    if not path.exists():
        return {"skipped": "no decision log — run `make eval-policies`"}
    log = pd.read_parquet(path, columns=["policy", "learner_id", "item_id", "step", "correct"])
    log = log[log["policy"].isin({REFERENCE_ARM, "model_L0", "model_L4", "random"})].copy()
    log["policy"] = pd.Categorical(log["policy"],
                                   categories=[REFERENCE_ARM] + sorted(
                                       set(log["policy"]) - {REFERENCE_ARM}))
    log["step_100"] = log["step"] / 100.0
    fit = crossed_mixedlm(_subsample(log, "learner_id", "item_id", seed),
                          "correct ~ C(policy) + step_100", "learner_id", "item_id")
    return {"reference": REFERENCE_ARM,
            "note": "a linear probability model: the outcome is binary, the coefficients "
                    "are percentage-point differences in the chance of a correct answer, "
                    "and no logistic mixed model is fitted because statsmodels' GLMM "
                    "support does not cover crossed effects",
            **fit}


def outcome_model(learners: pd.DataFrame, outcome: str) -> dict:
    """Learner-level outcome by arm, with a random intercept per learner.

    Every arm sees the same cohort — the simulator draws it before any policy
    acts — so a learner contributes one row per arm and the repeated measure is
    the learner, not the item.
    """
    from statsmodels.regression.mixed_linear_model import MixedLM

    data = learners.copy()
    data["cohort_learner"] = (data["seed"].astype(str) + "-"
                              + data["learner_index"].astype(str))
    data["policy"] = pd.Categorical(
        data["policy"], categories=[REFERENCE_ARM]
        + sorted(set(data["policy"]) - {REFERENCE_ARM}))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = MixedLM.from_formula(f"{outcome} ~ C(policy)", groups="cohort_learner",
                                      data=data).fit(method="lbfgs", maxiter=300)
        confidence = result.conf_int()
    total = float(result.cov_re.iloc[0, 0]) + float(result.scale)
    return {
        "formula": f"{outcome} ~ C(policy) + (1 | learner)",
        "reference": REFERENCE_ARM,
        "n": int(len(data)), "learners": int(data["cohort_learner"].nunique()),
        "fixed_effects": {
            name: {"coefficient": float(result.fe_params[name]),
                   "std_error": float(result.bse[name]),
                   "ci_low": float(confidence.loc[name, 0]),
                   "ci_high": float(confidence.loc[name, 1]),
                   "p_value": float(result.pvalues[name])}
            for name in result.fe_params.index},
        "variance": {"learner": float(result.cov_re.iloc[0, 0]), "residual": float(result.scale)},
        "variance_share": {"learner": float(result.cov_re.iloc[0, 0]) / total,
                           "residual": float(result.scale) / total},
        "converged": bool(result.converged),
    }


# ------------------------------------------------------------------- power

def power_analysis(effects: list[dict]) -> dict:
    """Achieved power for what was observed, and the n an RCT would need."""
    from statsmodels.stats.power import TTestIndPower

    solver = TTestIndPower()
    achieved = []
    for entry in effects:
        g = abs(entry["hedges_g"])
        n = min(entry["n_treatment"], entry["n_control"])
        if not np.isfinite(g) or g == 0 or n < 2:
            continue
        achieved.append({
            "comparison": entry["comparison"], "outcome": entry["outcome"],
            "hedges_g": entry["hedges_g"], "n_per_arm": n,
            "power": float(solver.solve_power(effect_size=g, nobs1=n, alpha=ALPHA, ratio=1.0)),
        })
    required = {}
    for g in (0.2, RCT_TARGET_G, 0.5, 0.8):
        required[str(g)] = float(np.ceil(solver.solve_power(
            effect_size=g, power=0.8, alpha=ALPHA, ratio=1.0)))
    grid = np.unique(np.round(np.logspace(np.log10(10), np.log10(3000), 40)).astype(int))
    curves = {str(g): [{"n_per_arm": int(n),
                        "power": float(solver.solve_power(effect_size=g, nobs1=int(n),
                                                          alpha=ALPHA, ratio=1.0))}
                       for n in grid]
              for g in (0.2, RCT_TARGET_G, 0.5)}
    return {
        "alpha": ALPHA, "test": "two-sided independent-samples t",
        "achieved": achieved,
        "required_n_per_arm_at_80_percent": required,
        "rct_requirement": {
            "target_g": RCT_TARGET_G,
            "n_per_arm": required[str(RCT_TARGET_G)],
            "n_total": 2 * required[str(RCT_TARGET_G)],
            "note": "g = 0.3 is the effect this literature reports for adaptive "
                    "instruction; the simulated effects above are not evidence that a "
                    "real trial would see one, only that this design could detect it",
        },
        "curves": curves,
    }


# --------------------------------------------------------------- censoring

def survival_analysis(learners: pd.DataFrame, budget: int) -> dict:
    """Items-to-mastery as a time-to-event outcome, censored at the budget."""
    from statsmodels.duration.survfunc import SurvfuncRight, survdiff

    arms = {}
    for policy, rows in learners.groupby("policy", observed=True):
        time = rows["items_to_mastery"].to_numpy(dtype=float)
        event = rows["mastered"].to_numpy(dtype=bool).astype(int)
        function = SurvfuncRight(time, event)
        points = np.asarray(function.surv_times, dtype=float)
        survival = np.asarray(function.surv_prob, dtype=float)
        # Restricted mean survival time to the budget: the area under the KM
        # curve. Unlike the mean of a censored variable this is estimable, and
        # it is the honest "items spent before mastery" summary.
        edges = np.concatenate([[0.0], points, [float(budget)]])
        heights = np.concatenate([[1.0], survival])
        widths = np.diff(edges)
        rmst = float(np.sum(heights[:len(widths)] * widths))
        reached = points[survival <= 0.5]
        arms[str(policy)] = {
            "n": int(len(rows)),
            "events": int(event.sum()),
            "censored": int(len(rows) - event.sum()),
            "censoring_rate": float(1 - event.mean()),
            "mastery_rate": float(event.mean()),
            "median_items_to_mastery": float(reached[0]) if len(reached) else None,
            "restricted_mean_items_to_budget": rmst,
            "mean_ignoring_censoring": float(np.mean(time)),
            "mean_knowledge_gain": float(rows["knowledge_gain"].mean()),
        }
    ordered = sorted(arms, key=lambda name: -arms[name]["mastery_rate"])
    best = ordered[0]
    comparisons = []
    for policy in ordered[1:]:
        pair = learners[learners["policy"].isin([best, policy])]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            statistic, p_value = survdiff(
                pair["items_to_mastery"].to_numpy(dtype=float),
                pair["mastered"].to_numpy(dtype=bool).astype(int),
                pair["policy"].to_numpy())
        comparisons.append({"reference": best, "policy": policy,
                            "logrank_chi2": float(statistic), "p_value": float(p_value)})
    return {
        "budget": budget,
        "why": "most learners do not master inside the budget, so the mean of "
               "items_to_mastery is a lower bound on any difference, not the difference. "
               "The uncensored co-primary is knowledge_gain.",
        "arms": arms,
        "logrank_vs_best": comparisons,
        "highest_mastery_rate": best,
    }


# ------------------------------------------------------------------ tables

def _write_table(frame: pd.DataFrame, name: str, caption: str) -> None:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    csv = TABLE_DIR / f"{name}.csv"
    frame.to_csv(csv, index=False)
    label = f"tab:{name}"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        latex = frame.to_latex(index=False, escape=True, float_format="%.4f",
                               caption=caption, label=label, longtable=False)
    (TABLE_DIR / f"{name}.tex").write_text(latex, encoding="utf-8")
    print(f"wrote: {csv.relative_to(ROOT)} (+ .tex)")


def build_tables(report: dict, learners: pd.DataFrame) -> list[str]:
    written = []

    final = json.loads((BENCH_DIR / "study1-final.json").read_text(encoding="utf-8"))
    rows = [{"source": source, "cell": cell, "rung": cell.split("|")[0],
             "model": cell.split("|")[1],
             "roc_auc": values.get("roc_auc"), "ci_low": values.get("ci_low"),
             "ci_high": values.get("ci_high"), "ece": values.get("ece"),
             "brier": values.get("brier"), "n_learners": values.get("learners")}
            for source, block in final["sources"].items()
            for cell, values in block.get("cells", {}).items()]
    _write_table(pd.DataFrame(rows).sort_values(["source", "cell"]), "model-comparison",
                 "Study 1 held-out test performance by dataset, rung and model.")
    written.append("model-comparison")

    cv = json.loads((BENCH_DIR / "study1-cv.json").read_text(encoding="utf-8"))
    ladder = [{"source": source, "model": model, "step": step,
               "delta_auc": values["delta_auc"], "ci_low": values["ci_low"],
               "ci_high": values["ci_high"], "p_value": values["p_value"],
               "p_holm": values.get("p_holm"),
               "clears_protocol_band": values.get("clears_protocol_band")}
              for source, block in cv["sources"].items()
              for model, steps in block.get("ladder", {}).items()
              for step, values in steps.items()]
    _write_table(pd.DataFrame(ladder), "ablation-ladder",
                 "Ablation ladder: cross-validated ΔAUC per rung, Holm-corrected.")
    written.append("ablation-ladder")

    gate = json.loads((EVAL_DIR / "validation-gate.json").read_text(encoding="utf-8"))
    validity = [{"state": state, "admitted": state in gate.get("admitted", []),
                 **{key: value for key, value in measures.items()
                    if not isinstance(value, (dict, list))}}
                for state, measures in gate.get("measurements", {}).items()]
    _write_table(pd.DataFrame(validity), "state-validity",
                 "Study 2 validity gate: every state head, admitted or rejected.")
    written.append("state-validity")

    policies = []
    for path in sorted(EVAL_DIR.glob("policy-results-v*.json")):
        block = json.loads(path.read_text(encoding="utf-8"))
        for arm, values in block.get("policies", {}).items():
            policies.append({"variant": block["variant"], "policy": arm,
                             **{key: value for key, value in values.items()
                                if not isinstance(value, (dict, list))}})
    _write_table(pd.DataFrame(policies), "policy-comparison",
                 "Study 4 closed loop (SIMULATED): every arm on every simulator variant.")
    written.append("policy-comparison")

    ope_path = EVAL_DIR / "ope-results.json"
    if ope_path.exists():
        ope = json.loads(ope_path.read_text(encoding="utf-8"))
        _write_table(pd.DataFrame([
            {key: value for key, value in entry.items() if not isinstance(value, (dict, list))}
            for entry in ope["estimates"]]), "ope",
            "Study 3 off-policy evaluation: IPS, SNIPS and DR against the on-policy truth.")
        written.append("ope")

    # `test` and `assumptions` are the same for every row; they live in the
    # report's protocol block rather than in 27 identical table cells.
    tests = pd.DataFrame([{key: value for key, value in entry.items()
                           if key not in ("assumptions", "test")}
                          for entry in report["effect_sizes"]])
    _write_table(tests, "statistical-tests",
                 "Phase 11 effect sizes, unpaired, with bootstrap CIs and Holm correction.")
    written.append("statistical-tests")

    _write_table(pd.DataFrame([{"policy": name, **values} for name, values
                               in report["censoring"]["arms"].items()]), "censoring",
                 "Items to mastery as a censored outcome: Kaplan-Meier summaries.")
    written.append("censoring")
    return written


# -------------------------------------------------------------------- main

def load_learners() -> tuple[pd.DataFrame, list[str]]:
    paths = sorted(EVAL_DIR.glob("policy-per-learner-v*.csv"))
    if not paths:
        raise SystemExit("no per-learner outcomes — run `make eval-policies` first")
    frames = {path.stem.split("-")[-1].upper(): pd.read_csv(path) for path in paths}
    return frames, sorted(frames)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260821)
    parser.add_argument("--bootstrap", type=int, default=10_000,
                        help="resamples for every CI on an effect size")
    args = parser.parse_args()

    variants, names = load_learners()
    primary = variants["V0"]
    budget = int(json.loads((EVAL_DIR / "policy-results-v0.json")
                            .read_text(encoding="utf-8"))["budget_items"])
    arms = sorted(primary["policy"].unique())
    print(f"{len(arms)} arms, {len(primary):,} learner-rows, variants {', '.join(names)}, "
          f"budget {budget}, bootstrap {args.bootstrap:,}")

    # ---- effect sizes: every arm against the reference, on every outcome.
    effects = []
    for outcome in OUTCOMES:
        for arm in arms:
            if arm == REFERENCE_ARM:
                continue
            result = compare(primary.loc[primary["policy"] == arm, outcome].to_numpy(),
                             primary.loc[primary["policy"] == REFERENCE_ARM, outcome].to_numpy(),
                             args.seed, args.bootstrap)
            effects.append({"comparison": f"{arm} vs {REFERENCE_ARM}", "policy": arm,
                            "reference": REFERENCE_ARM, "outcome": outcome, **result})
    print(f"effect sizes: {len(effects)} comparisons")

    # ---- Holm, one family per outcome.
    corrected = {}
    for outcome in OUTCOMES:
        family = {entry["comparison"]: entry["p_welch"]
                  for entry in effects if entry["outcome"] == outcome}
        adjusted = study1.holm(family)
        corrected[outcome] = {
            "family": sorted(family), "size": len(family),
            "raw": family, "holm": adjusted,
            "significant_raw": sorted(name for name, value in family.items() if value < ALPHA),
            "significant_holm": sorted(name for name, value in adjusted.items() if value < ALPHA),
        }
        for entry in effects:
            if entry["outcome"] == outcome:
                entry["p_holm"] = adjusted[entry["comparison"]]
                entry["significant_holm"] = bool(adjusted[entry["comparison"]] < ALPHA)

    # ---- guardrail 5: any d over 3 stops the phase for investigation.
    outsized = [entry for entry in effects if abs(entry["cohens_d_unpaired"]) > 3]

    report = {
        "phase": 11,
        "seed": args.seed,
        "git_sha": build_features.git_sha(),
        "bootstrap_resamples": args.bootstrap,
        "alpha": ALPHA,
        "reference_arm": REFERENCE_ARM,
        "protocol": {
            "effect_size": "Hedges' g on independent samples; the paired d Phase 8 "
                           "reports is the correct significance test and the wrong "
                           "effect size, because it divides by the sd of the difference",
            "clustering": "crossed learner and item random intercepts on every per-item "
                          "model; a random learner intercept on every learner-level model",
            "correction": "Holm within each outcome family, raw and corrected both reported",
            "censoring": "Kaplan-Meier and log-rank for items-to-mastery; knowledge_gain "
                         "is the uncensored co-primary",
            "simulated": "every closed-loop number here is simulated",
        },
        "effect_sizes": effects,
        "multiple_comparisons": corrected,
        "effect_sizes_over_threshold": [entry["comparison"] for entry in outsized],
    }

    print("mixed effects: learner-level outcomes")
    report["mixed_effects"] = {
        "outcomes": {outcome: outcome_model(primary, outcome) for outcome in OUTCOMES},
    }
    print("mixed effects: per-item closed loop")
    report["mixed_effects"]["closed_loop_items"] = closed_loop_item_model(args.seed)
    print("mixed effects: correctness prediction (Study 1)")
    report["mixed_effects"]["correctness"] = correctness_models(args.seed)

    print("power analysis")
    report["power"] = power_analysis(effects)
    print("survival / censoring")
    report["censoring"] = survival_analysis(primary, budget)

    report["tables"] = build_tables(report, primary)
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2, default=float) + "\n", encoding="utf-8")
    print(f"wrote: {REPORT_PATH.relative_to(ROOT)}")

    required = report["power"]["rct_requirement"]
    print(f"\nRCT sample size for g = {RCT_TARGET_G} at 80 % power: "
          f"{required['n_per_arm']:.0f} per arm ({required['n_total']:.0f} total)")
    if outsized:
        print(f"\nHALT (BUILD.md §4.5): {len(outsized)} unpaired |d| > 3 — "
              f"{', '.join(entry['comparison'] for entry in outsized[:5])}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
