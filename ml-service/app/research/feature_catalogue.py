"""The feature catalogue — one definition of every modelling input.

    python -m app.research.feature_catalogue --write-doc     # regenerates docs/feature-catalogue.md
    python -m app.research.feature_catalogue --list L4       # names in one ladder rung

This module is **data, not analysis**, and it is deliberately standard-library
only so it can be imported by the host as well as the container. Three
consumers read it and none of them keeps its own copy of the list:

* ``build_features.py`` — builds exactly these columns, nothing else.
* ``audit_leakage.py``  — asserts no latent ground truth is in here, and that
  every history feature declares itself ``shift(1)``-guarded.
* ``eda.py``            — colours every figure by ``signal_class`` and slices
  the ablation ladder by ``level``.

BUILD.md Phase 5 step 1: *a feature with no rationale does not get built*. The
rationale field is not documentation, it is a gate — ``validate()`` refuses an
empty one and ``make features`` calls it before writing a single row.

Ladder rungs (BUILD.md Phase 5 step 2), which are also the consent classes of
`docs/preregistration.md` §4 and therefore the privacy–utility curve of RQ4:

===== =============== ==============================================
Rung  Signal class    Adds
===== =============== ==============================================
L0    correctness     outcome, attempts, prior accuracy, difficulty
L1    timing          response time, latency, lag, personal drift
L2    interaction     option changes, hints, idle, focus
L3    session         position, elapsed, matched-difficulty decay
L4    motor           the pre-registered trajectory aggregate
===== =============== ==============================================
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
DOC_PATH = ROOT / "docs" / "feature-catalogue.md"

LEVELS = ["L0", "L1", "L2", "L3", "L4"]
LEVEL_CLASS = {
    "L0": "correctness",
    "L1": "timing",
    "L2": "interaction",
    "L3": "session",
    "L4": "motor",
}

#: Column-name prefixes that are latent ground truth or a target. Never a
#: feature — this is the machine-readable form of BUILD.md Defect 2, and
#: `audit_leakage.py` enforces it against the catalogue and against every
#: feature matrix that reaches a model.
FORBIDDEN_PREFIXES = (
    "knowledge_", "fatigue_", "confidence_", "engagement_",
    "latent_", "probe_", "archetype", "guessed", "probability_correct",
    "next_", "target_",
)


@dataclass(frozen=True)
class Feature:
    """One modelling input. Every field is required; `validate()` says so."""

    name: str
    level: str
    definition: str
    #: ``real`` (archival only), ``sim`` (simulated only) or ``both``.
    source: str
    rationale: str
    privacy: str
    #: True when the value summarises the learner's own past. Such a feature
    #: must be computed over attempts **strictly before** its own row, so that
    #: a row is unchanged by anything that happens after it.
    history: bool = False
    #: Sources where the column is null throughout, with the reason. Written
    #: into the doc so a per-source null is a documented absence, not a
    #: surprise in Phase 6.
    unavailable: dict[str, str] = field(default_factory=dict)

    @property
    def signal_class(self) -> str:
        return LEVEL_CLASS[self.level]


# Availability notes reused below. Phase 3 recorded these gaps in
# `docs/PROGRESS.md` and figure f03-08; they are quoted here so the catalogue
# is the one place that says what a source cannot supply.
_NO_TIMESTAMP = {"assistments_2009": "no wall-clock column in the 2009-2010 release"}
_NO_ATTEMPTS = {"ednet_kt1": "KT1 stores one row per response, no attempt or hint counters"}
_SIM_ONLY = {
    "assistments_2009": "no interaction telemetry in an archival log",
    "assistments_2012": "no interaction telemetry in an archival log",
    "ednet_kt1": "no interaction telemetry in an archival log",
}

CATALOGUE: list[Feature] = [
    # ---------------------------------------------------------------- L0
    Feature(
        "correct", "L0", "1 if the attempt was scored correct, else 0.",
        "both",
        "The outcome every knowledge-tracing model since K1 (Corbett & Anderson 1994) "
        "conditions on. Present-tense: it describes attempt t, and the label is at t+1.",
        "low",
    ),
    Feature(
        "attempt_count", "L0",
        "Number of attempts the learner made on this item before it was scored.",
        "both",
        "Repeated attempts on one item are the classic wheel-spinning signal (B3, B4); "
        "PFA (K14's logistic family) counts successes and failures separately for this reason.",
        "low", unavailable=_NO_ATTEMPTS,
    ),
    Feature(
        "prior_accuracy", "L0",
        "Mean of `correct` over the learner's attempts strictly before this one.",
        "both",
        "The ability term of every logistic knowledge-tracing model; K13 and K14 both find "
        "that a well-specified logistic ability feature is hard for deep models to beat at this data size.",
        "low", history=True,
    ),
    Feature(
        "prior_attempts", "L0",
        "Count of the learner's attempts strictly before this one.",
        "both",
        "The denominator `prior_accuracy` must be read against: 1.0 from one attempt and "
        "1.0 from two hundred are different states. Also the practice-count term of PFA.",
        "low", history=True,
    ),
    Feature(
        "skill_prior_accuracy", "L0",
        "Mean of `correct` over the learner's earlier attempts on the same skill / concept.",
        "both",
        "Per-skill mastery is what BKT (K1, K4) tracks and what an adaptive policy actually "
        "selects on. A learner strong overall can be weak on the concept in front of them.",
        "low", history=True,
    ),
    Feature(
        "skill_prior_attempts", "L0",
        "Count of the learner's earlier attempts on the same skill / concept.",
        "both",
        "Opportunity count. The x-axis of a learning curve, and the exposure term PFA needs "
        "to separate a low mastery estimate from a thin one.",
        "low", history=True,
    ),
    Feature(
        "item_difficulty", "L0",
        "Item difficulty on the logit scale. Archival: `-logit(p)` from the item's accuracy "
        "among **training** learners, shrunk toward the source mean with a 20-response prior. "
        "Simulated: the generative `item_b`.",
        "both",
        "The Rasch b parameter. P1 and the CAT literature select items by ability-difficulty "
        "distance; without it, response time is uninterpretable because a slow response to an "
        "easy item and a slow response to a hard one mean opposite things.",
        "low",
    ),
    Feature(
        "n_options", "L0", "Number of answer options presented.",
        "sim",
        "Sets the chance level, which is what a rapid-guess accuracy check is read against "
        "(P7, P8). Phase 3 recorded that neither ASSISTments release stores it.",
        "low",
        unavailable={"assistments_2009": "option count not recorded",
                     "assistments_2012": "option count not recorded",
                     "ednet_kt1": "not carried through KT1's response table"},
    ),
    # ---------------------------------------------------------------- L1
    Feature(
        "log_response_time", "L1", "log1p of the winsorised response time in seconds.",
        "both",
        "SAINT+ (K8) reports +1.25 % AUC from elapsed time and is the honest reference point "
        "for this whole project. Logged because raw response time is heavy-tailed.",
        "low",
    ),
    Feature(
        "log_rt_z_item", "L1",
        "`log_response_time` standardised by the item's mean and sd among **training** learners.",
        "both",
        "The lit-review's 'include — core' list names *log response time standardised per item* "
        "explicitly (§7.5). Raw speed confounds a fast learner with an easy item; P1's joint "
        "speed-accuracy model separates them the same way.",
        "low",
    ),
    Feature(
        "rt_drift", "L1",
        "`log_response_time` minus the mean `log_response_time` over the learner's earlier attempts.",
        "both",
        "Speed relative to the learner's own baseline rather than the population's. P3 relaxes "
        "van der Linden's constant-speed assumption for exactly this reason; a fatigue-aware "
        "model has to break it (P1's stated limitation).",
        "low", history=True,
    ),
    Feature(
        "log_lag_time", "L1",
        "log1p of the seconds since the learner's previous attempt.",
        "both",
        "SAINT+'s second temporal feature (K8): the spacing between attempts carries forgetting "
        "and session structure. Needs a wall clock.",
        "low", unavailable=_NO_TIMESTAMP,
    ),
    Feature(
        "reading_time", "L1", "Seconds from item presentation to the first interaction with it.",
        "sim",
        "Cocea & Weibelzahl (B12) build disengagement detectors on reading time against stem "
        "length: too short to have read the question is the cleanest off-task evidence a log holds.",
        "medium", unavailable=_SIM_ONLY,
    ),
    Feature(
        "decision_latency", "L1", "Seconds from the first option selection to submission.",
        "sim",
        "Named in the lit-review's core include list (§7.5) and the RQ2 question. B17 links "
        "latency-to-commit to drift-diffusion decision uncertainty.",
        "medium", unavailable=_SIM_ONLY,
    ),
    Feature(
        "time_after_last_interaction", "L1",
        "Seconds between the last recorded interaction and submission.",
        "sim",
        "A long silent tail before submit is hesitation or absence, and it separates the two "
        "when read with `idle_time`. Off-task time features are B2's original ITS signal.",
        "medium", unavailable=_SIM_ONLY,
    ),
    Feature(
        "response_time_vs_estimate", "L1",
        "Response time divided by the item's authored time estimate.",
        "sim",
        "A normative rather than empirical speed reference — the same shape as the Wise & Kong "
        "threshold (P7), which is a fraction of an item's normative time, not of the learner's.",
        "low", unavailable=_SIM_ONLY,
    ),
    # ---------------------------------------------------------------- L2
    Feature(
        "option_changes", "L2", "Number of `OPTION_CHANGED` events before submission.",
        "sim",
        "Answer-changing is a measured metacognitive signal with a known asymmetry: B26 and B27 "
        "find wrong-to-right changes dominate and that who benefits depends on ability. The "
        "lit-review lists option-change sequences *conditioned on ability* as experimental-include.",
        "medium", unavailable=_SIM_ONLY,
    ),
    Feature(
        "option_change_entropy", "L2",
        "Shannon entropy (bits) of the distribution of options selected during the attempt.",
        "sim",
        "Separates one decisive correction from cycling through every option. Two changes with "
        "the same count carry different information; B20's catalogue of cursor measures makes "
        "the same distinction for paths.",
        "medium", unavailable=_SIM_ONLY,
    ),
    Feature(
        "option_revisits", "L2", "Times an already-selected option was selected again.",
        "sim",
        "The revisit term of the hierarchical speed-accuracy-revisits model (P4), which shows "
        "revisiting is not noise on top of response time but its own behavioural channel.",
        "medium", unavailable=_SIM_ONLY,
    ),
    Feature(
        "first_selection_changed", "L2", "1 if the first selected option was not the submitted one.",
        "sim",
        "The binary form of the B26/B27 answer-change literature, kept alongside the count "
        "because the first change is the one those papers analyse.",
        "medium", unavailable=_SIM_ONLY,
    ),
    Feature(
        "hint_count", "L2", "Hints requested on this item.",
        "both",
        "Help abuse is half of Baker's gaming-the-system detector (B1), and the intervention "
        "built on it reportedly halved gaming — one of the few closed loops in this literature.",
        "low", unavailable=_NO_ATTEMPTS,
    ),
    Feature(
        "skipped", "L2", "1 if the item was submitted with no answer.",
        "sim",
        "Non-response is data, not absence (`docs/preregistration.md` §3). A learner who starts "
        "skipping is the disengagement signal B13 detects from session logs.",
        "low", unavailable=_SIM_ONLY,
    ),
    Feature(
        "idle_count", "L2", "Number of idle spells (no input for the client's idle threshold).",
        "sim",
        "The lit-review's core include list names *idle gaps with browser focus/visibility state*. "
        "Count and duration separate one long absence from constant micro-interruption.",
        "medium", unavailable=_SIM_ONLY,
    ),
    Feature(
        "idle_time", "L2", "Total seconds spent idle during the attempt.",
        "sim",
        "The duration half of the above. It is also what makes `log_response_time` honest: time "
        "on an item is not time on task.",
        "medium", unavailable=_SIM_ONLY,
    ),
    Feature(
        "visibility_changes", "L2", "Times the tab lost or regained visibility during the attempt.",
        "sim",
        "Direct off-task evidence in a browser, the modern equivalent of B2's off-task detector, "
        "and cheap: the Page Visibility API needs no extra permission.",
        "medium", unavailable=_SIM_ONLY,
    ),
    # ---------------------------------------------------------------- L3
    Feature(
        "question_number", "L3", "1-based position of the attempt within its session.",
        "both",
        "The x-axis of every within-session decay curve, and the exposure term a fatigue claim "
        "is measured against (B32's state-space fatigue model).",
        "low", unavailable=_NO_TIMESTAMP,
    ),
    Feature(
        "session_duration", "L3", "Seconds elapsed in the session at the moment of submission.",
        "both",
        "Time-on-task, the second fatigue axis. B32 models fatigue against elapsed time rather "
        "than item count; both are kept because they disagree when items differ in length.",
        "low", unavailable=_NO_TIMESTAMP,
    ),
    Feature(
        "matched_difficulty_accuracy_slope", "L3",
        "OLS slope of difficulty-residualised correctness on within-session position, over the "
        "learner's earlier attempts in this session. Null before 4 attempts.",
        "both",
        "The lit-review's experimental-include list names *within-session matched-difficulty "
        "accuracy decay* — accuracy alone falls when a policy hands out harder items, so the "
        "item's own base rate is removed first. This is the fatigue construct's only observable.",
        "low", history=True, unavailable=_NO_TIMESTAMP,
    ),
    Feature(
        "matched_difficulty_speed_slope", "L3",
        "The same slope computed on item-residualised `log_response_time`.",
        "both",
        "Its companion in the same list. P1's constant-speed-across-a-test assumption is exactly "
        "what this measures, and Phase 4 found the real slope is negative — learners speed up.",
        "low", history=True, unavailable=_NO_TIMESTAMP,
    ),
    Feature(
        "idle_fraction", "L3", "`idle_time` divided by the attempt's total response time.",
        "sim",
        "Scale-free off-task share, comparable across a 5-second and a 5-minute item, which the "
        "raw duration is not.",
        "medium", unavailable=_SIM_ONLY,
    ),
    Feature(
        "focus_fraction", "L3", "Share of the attempt during which the tab was visible.",
        "sim",
        "The visibility counterpart of `idle_fraction`; the pair distinguishes present-but-stuck "
        "from absent, which B2 and B12 treat as different states with different interventions.",
        "medium", unavailable=_SIM_ONLY,
    ),
    # ---------------------------------------------------------------- L4
    # Definitions are pre-registered verbatim in docs/preregistration.md §1 and
    # computed by app.features.extract — never re-derived here.
    Feature(
        "auc_toward_nonchosen", "L4",
        "Signed area between the pointer path and the straight line to the chosen option, "
        "positive toward the most-hovered non-chosen option (pixel-seconds).",
        "sim",
        "The attraction measure of the mouse-tracking literature (B16, B20): a path that bows "
        "toward the alternative is the canonical index of competing response activation. RQ2's "
        "central feature.",
        "high", unavailable=_SIM_ONLY,
    ),
    Feature(
        "max_deviation", "L4", "Greatest perpendicular distance of the path from that line (px).",
        "sim",
        "The second standard attraction statistic in B16's methodological guidance; reported "
        "alongside AUC because the two dissociate when a path deviates late.",
        "high", unavailable=_SIM_ONLY,
    ),
    Feature(
        "x_flips", "L4", "Sign changes in horizontal pointer direction between samples.",
        "sim",
        "The zigzag / direction-change measure named in the lit-review's experimental include "
        "list and catalogued by B20 as a distinct construct from deviation magnitude.",
        "high", unavailable=_SIM_ONLY,
    ),
    Feature(
        "sample_entropy", "L4", "SampEn of the per-step speed series (m = 2, r = 0.2 sigma).",
        "sim",
        "Movement regularity. B18 predicts respondent difficulty in web surveys from exactly "
        "this family of mouse features; irregular movement marks effortful processing.",
        "high", unavailable=_SIM_ONLY,
    ),
    Feature(
        "velocity_peak", "L4", "Maximum pointer speed during the attempt (px/s).",
        "sim",
        "The velocity-profile measure of the lit-review's include list; B17 ties peak velocity "
        "to drift-diffusion decision parameters.",
        "high", unavailable=_SIM_ONLY,
    ),
    Feature(
        "velocity_mean", "L4", "Mean pointer speed during the attempt (px/s).",
        "sim",
        "Kept with the peak because their ratio separates one decisive dash from sustained "
        "movement — B19's reproducibility guidance asks for both rather than a single summary.",
        "high", unavailable=_SIM_ONLY,
    ),
    Feature(
        "pause_count", "L4", "Runs of speed < 0.05 px/ms lasting > 300 ms.",
        "sim",
        "Pause count is named in the lit-review's include list. Hesitation is demoted there to "
        "*a feature feeding confidence*, not a state of its own (§7.5) — which is how Phase 7 uses it.",
        "high", unavailable=_SIM_ONLY,
    ),
    Feature(
        "path_ratio", "L4", "Total path length divided by the straight-line distance. 1.0 is straight.",
        "sim",
        "The oldest and most robust cursor measure in B20's catalogue, and the least sensitive to "
        "sampling rate — the sanity check against which the fancier statistics are read.",
        "high", unavailable=_SIM_ONLY,
    ),
    Feature(
        "time_to_first_movement", "L4", "Seconds from presentation to the first pointer movement.",
        "sim",
        "Initiation time. B16 separates it from movement time because a long still period before "
        "moving is deliberation, whereas a slow path is competition during the movement itself.",
        "high", unavailable=_SIM_ONLY,
    ),
    Feature(
        "time_to_first_selection", "L4", "Seconds from presentation to the first option selection.",
        "sim",
        "The motor-channel companion of `decision_latency`, retained because the two disagree "
        "when the learner selects early and then deliberates about changing.",
        "high", unavailable=_SIM_ONLY,
    ),
    Feature(
        "hover_time_max", "L4", "Longest time spent hovering over any single option (s).",
        "sim",
        "Dwell on an option is the attraction measure that survives a one-column layout, where "
        "lateral geometry is weak — the interface limitation Phase 4 recorded for AUC and x-flips.",
        "high", unavailable=_SIM_ONLY,
    ),
    Feature(
        "hover_time_nonchosen", "L4", "Total hover time on options other than the submitted one (s).",
        "sim",
        "Direct evidence of a considered-and-rejected alternative, which is the mouse-tracking "
        "construct (B16) stated in a form that does not depend on where the options are drawn.",
        "high", unavailable=_SIM_ONLY,
    ),
    Feature(
        "cursor_samples", "L4", "Number of 50 ms pointer samples in the attempt.",
        "sim",
        "Provenance, and pre-registered as such: every trajectory statistic above must be read "
        "against how many samples produced it (`docs/preregistration.md` §1).",
        "high", unavailable=_SIM_ONLY,
    ),
]

#: Columns carried through `build_features.py` for grouping, splitting and
#: evaluation. **None of these is ever a model input** — `audit_leakage.py`
#: asserts the separation. The latent and probe columns exist only so Phase 7
#: can score a state estimate against the value that generated it.
#: ``next_item_id`` and ``next_skill_id`` are here rather than in the catalogue
#: on purpose. An adaptive system *chooses* the next item, so their identity is
#: genuinely known at decision time and the per-item base-rate floor in
#: `baselines.py` is defined by them. They are still not features: the ladder's
#: `item_difficulty` was pre-registered as the *current* item's, and quietly
#: upgrading it to the next item's would change what the ladder measures after
#: the fact. The effect is to make the floor stronger than the models it is a
#: floor for, which is the conservative direction.
PASSTHROUGH = [
    "learner_id", "source", "variant", "item_id", "skill_id", "order_index",
    "session_id", "split", "next_correct", "next_item_id", "next_skill_id",
    "learner_rte", "solution_behaviour",
    "archetype", "latent_knowledge", "latent_engagement", "latent_confidence",
    "latent_fatigue", "probe_confidence", "probe_effort", "propensity",
    "difficulty_score", "distracted",
]

BY_NAME = {feature.name: feature for feature in CATALOGUE}


def names(level: str | None = None, upto: str | None = None) -> list[str]:
    """Feature names in one rung (``level``) or in every rung up to ``upto``."""
    if level:
        return [f.name for f in CATALOGUE if f.level == level]
    if upto:
        allowed = LEVELS[: LEVELS.index(upto) + 1]
        return [f.name for f in CATALOGUE if f.level in allowed]
    return [f.name for f in CATALOGUE]


def ladder() -> dict[str, list[str]]:
    """The ablation ladder: rung -> every feature available at that rung."""
    return {level: names(upto=level) for level in LEVELS}


def available(source: str, upto: str = "L4") -> list[str]:
    """Feature names that carry a real value for ``source``."""
    return [name for name in names(upto=upto) if source not in BY_NAME[name].unavailable]


def validate() -> None:
    """The gate. A catalogue that fails this must not produce a dataset."""
    seen: set[str] = set()
    for feature in CATALOGUE:
        where = f"feature {feature.name!r}"
        assert feature.name not in seen, f"{where}: duplicated"
        seen.add(feature.name)
        assert feature.level in LEVELS, f"{where}: unknown level {feature.level!r}"
        assert feature.source in {"real", "sim", "both"}, f"{where}: bad source {feature.source!r}"
        assert feature.privacy in {"low", "medium", "high"}, f"{where}: bad privacy"
        # BUILD.md Phase 5 step 1 and its acceptance check, enforced rather than reviewed.
        assert feature.rationale.strip(), f"{where}: empty rationale — it does not get built"
        assert feature.definition.strip(), f"{where}: empty definition"
        for prefix in FORBIDDEN_PREFIXES:
            assert not feature.name.startswith(prefix), f"{where}: latent ground truth as a feature"
    overlap = seen & set(PASSTHROUGH)
    assert not overlap, f"columns are both feature and passthrough: {sorted(overlap)}"


def to_markdown() -> str:
    validate()
    lines = [
        "# Feature catalogue",
        "",
        "**Generated** — `python -m app.research.feature_catalogue --write-doc`. The source of",
        "truth is `ml-service/app/research/feature_catalogue.py`; edit that, not this file.",
        "",
        "One row per modelling input. A feature with an empty rationale cannot be built:",
        "`validate()` refuses it and `make features` calls `validate()` before writing a row",
        "(BUILD.md Phase 5 step 1).",
        "",
        "The **rung** column is the ablation ladder of Phase 6 and, read the other way, the",
        "consent classes of [`preregistration.md`](preregistration.md) §4 — so it is also the",
        "privacy–utility curve of RQ4. Citation keys (`K8`, `B16`, `P4`, …) are the",
        "[literature review](requirements/Literature-Review-Adaptive-Learning-Behavioural-Analytics.pdf)'s own.",
        "",
        "Latent state (`latent_*`), self-report probes (`probe_*`) and the label (`next_correct`)",
        "are **not in this table** and never enter a model. They are carried beside the features",
        "for evaluation only, and `audit_leakage.py` fails the build if one crosses over.",
        "",
    ]
    for level in LEVELS:
        rung = [f for f in CATALOGUE if f.level == level]
        lines += [
            f"## {level} — {LEVEL_CLASS[level]} ({len(rung)} features)",
            "",
            "| Feature | Definition | Source | Privacy | Rationale |",
            "|---|---|---|---|---|",
        ]
        for feature in rung:
            source = {"both": "real + simulated", "real": "real only", "sim": "simulated only"}[feature.source]
            if feature.history:
                source += " · history, `shift(1)`-guarded"
            lines.append(
                f"| `{feature.name}` | {feature.definition} | {source} | {feature.privacy} | {feature.rationale} |"
            )
        lines.append("")

    lines += ["## Per-source availability", "",
              "A source that cannot supply a feature yields **null**, never zero — absence and",
              "measurement stay distinguishable (`preregistration.md` §4).", "",
              "| Feature | Source | Why it is null |", "|---|---|---|"]
    for feature in CATALOGUE:
        for source, reason in feature.unavailable.items():
            lines.append(f"| `{feature.name}` | {source} | {reason} |")
    lines += ["",
              "## Carried, never modelled", "",
              "`" + "`, `".join(PASSTHROUGH) + "`", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-doc", action="store_true", help=f"regenerate {DOC_PATH.name}")
    parser.add_argument("--list", dest="level", choices=LEVELS, help="print the names in one rung")
    args = parser.parse_args()

    validate()
    if args.level:
        print("\n".join(names(level=args.level)))
        return
    if args.write_doc:
        DOC_PATH.write_text(to_markdown() + "\n", encoding="utf-8")
        print(f"wrote: {DOC_PATH.relative_to(ROOT)} ({len(CATALOGUE)} features, {len(LEVELS)} rungs)")
        return
    for level in LEVELS:
        rung = names(level=level)
        print(f"{level} {LEVEL_CLASS[level]:12s} {len(rung):2d} cumulative {len(names(upto=level)):2d}")


if __name__ == "__main__":
    main()
