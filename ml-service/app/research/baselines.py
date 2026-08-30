"""Study 1 baselines — the floors and the competitors.

The literature review is explicit that omitting two of these is individually
fatal to the paper: a **feature-engineered logistic knowledge-tracing model
(PFA)**, which K13 and K14 find hard for deep models to beat at this data size,
and a **timing-aware deep model (SAINT+/LPKT)**, which already does what this
project claims to do and is therefore the real competitor. Both are here.

    from app.research.baselines import BASELINES, fit_baseline

Every baseline has the same shape: ``fit(train, validate) -> probabilities for
validate``. They take the raw sequence frame rather than the feature matrix,
because BKT and the deep models consume ``(skill, correct)`` sequences and
fitting them from the tabular matrix would be reconstructing what
`build_features` already normalised.

Two upstream bugs are worked around at the top of this file rather than
scattered through it. Both are theirs, both are quarantined here, and both are
tested so a version bump that fixes them fails loudly instead of silently
keeping a dead shim.
"""
from __future__ import annotations

import sys
import time
import types
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# --------------------------------------------------------------- upstream shims

def _stub_turtle() -> None:
    """pykt-toolkit 0.0.38 cannot be imported without a GUI toolkit.

    ``pykt/models/qdkt.py`` line 2 is ``from turtle import forward`` — an IDE
    auto-import that was never removed. ``turtle`` imports ``tkinter``, which
    needs ``libtk8.6``, which a ``python:3.11-slim`` image does not ship. It is
    on the import path of ``pykt.models.__init__``, so **every** pykt model is
    unimportable, including the four this project needs.

    ponytail: a stub module rather than installing ``tk`` in the image. The
    symbol is imported and never called — pykt's own `QDKT.forward` shadows it
    — so a two-line stub is both smaller and more portable than adding a GUI
    toolkit to a headless research image. Delete this when pykt fixes line 2.
    """
    if "turtle" not in sys.modules:
        module = types.ModuleType("turtle")
        module.forward = None
        sys.modules["turtle"] = module


def _import_pybkt():
    """pyBKT 1.4.3 cannot be imported *or fitted* alongside modern sklearn/numpy.

    Two separate breakages, both upstream:

    1. **Import.** ``pyBKT/util/metrics.py`` probes every
       ``*_loss``/``*_score``/``*_error`` symbol in sklearn's private metric
       modules by calling it with two Python *lists*, keeping whatever does not
       raise ``TypeError``. sklearn 1.9's array-API rewrite raises
       ``AttributeError`` and ``AxisError`` on those inputs instead, so the
       probe crashes at import time.
    2. **Fit.** ``pyBKT/fit/EM_fit.py`` assigns ``result['total_loglike']`` —
       a 1-element array — into a scalar slot. NumPy 2 refuses the implicit
       conversion, so every ``Model.fit`` dies on the first EM iteration.

    ponytail: two surgical shims rather than pinning the whole numeric stack
    back to satisfy one baseline. The probe's *result* is optional — pyBKT
    defines the three metrics it actually uses before the probe runs and only
    *adds* to them — so hiding the two private modules makes it find nothing
    and succeed, and the modules are restored in a `finally`. The second is a
    one-line coercion of a value that was always meant to be a scalar; it
    changes no arithmetic. Delete both when pyBKT supports NumPy 2.
    """
    import sklearn.metrics as sk

    saved = sk._classification, sk._regression
    sk._classification = sk._regression = types.SimpleNamespace()
    try:
        from pyBKT.models import Model
    finally:
        sk._classification, sk._regression = saved

    from pyBKT.fit import EM_fit

    if not getattr(EM_fit.run, "_alp_scalar_loglike", False):
        inner = EM_fit.run

        def run(*args, **kwargs):
            result = inner(*args, **kwargs)
            result["total_loglike"] = float(np.asarray(result["total_loglike"]).reshape(-1)[0])
            return result

        run._alp_scalar_loglike = True
        EM_fit.run = run
    return Model


# --------------------------------------------------------------------- shared

@dataclass
class BaselineResult:
    name: str
    probabilities: np.ndarray
    train_seconds: float
    notes: dict


#: Sequences longer than this are windowed for the deep models. 200 is pyKT's
#: own default and the number every published DKT/SAKT/AKT comparison uses.
SEQUENCE_LENGTH = 200

#: Learners drawn for a deep-model fit. These are CPU-only transformer runs;
#: the full EdNet training set would take hours per rung to make a point that
#: `f06-10`'s learning curve already makes. The cap is recorded in every
#: result and quoted in the paper as the limitation it is.
DEEP_TRAIN_LEARNERS = 3_000
DEEP_EPOCHS = 3
DEEP_BATCH = 64


def _clip(probabilities: np.ndarray) -> np.ndarray:
    return np.clip(np.nan_to_num(probabilities, nan=0.5), 1e-6, 1 - 1e-6)


# ------------------------------------------------------------- trivial floors

def majority_class(train: pd.DataFrame, validate: pd.DataFrame) -> BaselineResult:
    """Predict the training base rate for everyone. AUC is 0.5 by construction."""
    start = time.perf_counter()
    rate = float(train["next_correct"].mean())
    return BaselineResult("majority_class", np.full(len(validate), rate),
                          time.perf_counter() - start, {"base_rate": round(rate, 4)})


def item_base_rate(train: pd.DataFrame, validate: pd.DataFrame) -> BaselineResult:
    """Predict each item's own training success rate.

    The floor that matters: a model that cannot beat "how hard is this item"
    has learned nothing about the *learner*. Items unseen in training fall back
    to the global rate.
    """
    start = time.perf_counter()
    # The label is the *next* attempt's outcome, so the item that matters is
    # the next item, not this one.
    rates = train.groupby("next_item_id", observed=True)["next_correct"].mean()
    overall = float(train["next_correct"].mean())
    predicted = validate["next_item_id"].map(rates).fillna(overall).to_numpy(dtype=float)
    return BaselineResult("item_base_rate", predicted, time.perf_counter() - start,
                          {"items_seen": int(len(rates)), "fallback_rate": round(overall, 4)})


# ------------------------------------------------------------------ IRT / PFA

def irt_ability(train: pd.DataFrame, validate: pd.DataFrame) -> BaselineResult:
    """Rasch: one ability per learner, one difficulty per item, nothing else.

    Fitted as a logistic regression on the learner's running ability estimate
    and the item's difficulty — the two-term model the CAT literature (P1)
    selects items with. A learner absent from training gets the population mean
    ability, which is what a cold-start really has.

    ponytail: joint marginal-maximum-likelihood estimation would give the same
    ordering here at a hundred times the cost. This is a *floor*, and its job
    is to say how much of the signal is "who is answering and how hard is it".
    """
    start = time.perf_counter()
    columns = ["prior_accuracy", "item_difficulty"]
    model = LogisticRegression(max_iter=500)
    fitted = _fit_logistic(model, train, validate, columns)
    return BaselineResult("irt_rasch_ability", fitted, time.perf_counter() - start,
                          {"terms": columns})


def pfa_logistic(train: pd.DataFrame, validate: pd.DataFrame) -> BaselineResult:
    """Performance Factors Analysis — the baseline that is fatal to omit.

    PFA models the log-odds of success from per-skill counts of prior successes
    and prior failures, plus a skill difficulty term. K14's automated logistic
    search found this family surpassing deep models in both accuracy and
    explainability, and K13 found the same at this data size. If a behavioural
    rung cannot beat PFA, the paper says so.
    """
    start = time.perf_counter()
    columns = ["skill_prior_successes", "skill_prior_failures", "prior_accuracy",
               "prior_attempts", "item_difficulty"]
    model = LogisticRegression(max_iter=800)
    fitted = _fit_logistic(model, train, validate, columns)
    return BaselineResult("pfa_logistic_kt", fitted, time.perf_counter() - start,
                          {"terms": columns})


def _fit_logistic(model, train: pd.DataFrame, validate: pd.DataFrame,
                  columns: list[str]) -> np.ndarray:
    """Median-imputed, standardised logistic fit over ``columns``."""
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", model),
    ])
    pipeline.fit(train[columns], train["next_correct"].astype(int))
    return _clip(pipeline.predict_proba(validate[columns])[:, 1])


# ------------------------------------------------------------------------ BKT

#: pyBKT's EM is single-threaded pure-Python-plus-numpy and fits one HMM per
#: skill, so it is the slowest thing in the whole pipeline by an order of
#: magnitude: 2,097 s on the *smallest* source, against 13 s for DKT on the same
#: rows. Four sources at that rate is a multi-hour run for one reference number.
#:
# ponytail: BKT is fitted on a subsample of learners rather than the full
# training split. Ceiling: the number is a reference point for the related-work
# comparison, not a headline result, and the subsample is recorded in the
# report so it cannot be quoted as a full-data figure. Upgrade path: raise or
# remove BKT_MAX_LEARNERS if BKT ever becomes a load-bearing comparison, and
# budget hours rather than minutes for it.
BKT_MAX_LEARNERS = 800


def bkt(train: pd.DataFrame, validate: pd.DataFrame) -> BaselineResult:
    """Bayesian Knowledge Tracing via pyBKT (K1, K4).

    pyBKT fits one four-parameter HMM per skill and predicts the probability
    that the learner has mastered it. Fitted on the *current* correctness
    sequence, then read one step ahead so its prediction lines up with this
    project's `next_correct` label.

    Fitted on at most :data:`BKT_MAX_LEARNERS` learners — see the note there.
    Learners are chosen by a seeded draw, not by taking the head of the frame,
    because the frames arrive ordered by learner and the head would be a
    systematically early cohort.
    """
    start = time.perf_counter()
    Model = _import_pybkt()

    learners = train["learner_id"].unique()
    subsample = len(learners)
    if subsample > BKT_MAX_LEARNERS:
        rng = np.random.default_rng(20260821)
        keep = set(rng.choice(learners, size=BKT_MAX_LEARNERS, replace=False))
        train = train[train["learner_id"].isin(keep)]
        subsample = BKT_MAX_LEARNERS

    def shape(frame: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({
            "user_id": frame["learner_id"].astype(str),
            "skill_name": frame["skill_id"].astype(str),
            "correct": frame["correct"].astype(int),
            "order_id": frame["order_index"].astype(int),
        })

    model = Model(seed=20260821, num_fits=1)
    model.fit(data=shape(train))
    predicted = model.predict(data=shape(validate))
    column = "correct_predictions" if "correct_predictions" in predicted else "state_predictions"
    return BaselineResult("bkt_pybkt", _clip(predicted[column].to_numpy(dtype=float)),
                          time.perf_counter() - start,
                          {"skills_fitted": int(train["skill_id"].nunique()),
                           "prediction_column": column,
                           "fitted_on_learners": int(subsample),
                           "available_learners": int(len(learners)),
                           "subsampled": bool(len(learners) > BKT_MAX_LEARNERS)})


# ------------------------------------------------------------------- deep KT

def _sequences(frame: pd.DataFrame, skill_index: dict, item_index: dict,
               length: int = SEQUENCE_LENGTH) -> dict:
    """Learner sequences as padded arrays, with the label aligned to `next_correct`.

    Returns concept ids, item ids, responses, the target and a mask, all
    ``(n_windows, length)``. Position *t* holds attempt *t*'s concept and
    response; the target at *t* is attempt *t+1*'s outcome, so the model is
    asked exactly the question the tabular models are asked.
    """
    frame = frame.sort_values(["learner_id", "order_index"], kind="stable")
    concepts, items, responses, targets, masks, rows = [], [], [], [], [], []
    for _, sequence in frame.groupby("learner_id", sort=False, observed=True):
        concept = sequence["skill_id"].astype(str).map(skill_index).fillna(0).to_numpy(dtype=np.int64)
        item = sequence["item_id"].astype(str).map(item_index).fillna(0).to_numpy(dtype=np.int64)
        response = sequence["correct"].to_numpy(dtype=np.int64)
        target = sequence["next_correct"].to_numpy(dtype=np.float32)
        index = sequence.index.to_numpy()
        for start in range(0, len(response), length):
            window = slice(start, start + length)
            pad = length - len(response[window])
            concepts.append(np.pad(concept[window], (0, pad)))
            items.append(np.pad(item[window], (0, pad)))
            responses.append(np.pad(response[window], (0, pad)))
            targets.append(np.pad(target[window], (0, pad)))
            masks.append(np.pad(np.ones(len(response[window]), dtype=bool), (0, pad)))
            rows.append(np.pad(index[window], (0, pad), constant_values=-1))
    return {
        "concepts": np.stack(concepts), "items": np.stack(items),
        "responses": np.stack(responses), "targets": np.stack(targets),
        "masks": np.stack(masks), "rows": np.stack(rows),
    }


def deep_kt(name: str, train: pd.DataFrame, validate: pd.DataFrame,
            seed: int = 20260821) -> BaselineResult:
    """DKT, SAKT, AKT or SAINT from `pykt-toolkit`, trained on our own splits.

    pyKT's loaders re-split internally and assume their own directory layout,
    so only the **model definitions** are used here — the protocol stays
    `splits.json`'s. That is the same decision Phase 3 recorded: the protocol
    is pyKT's, the code is not.

    SAINT is the timing-aware family the literature review names as the real
    competitor (K8). Note the honest caveat, recorded in every result: pykt's
    SAINT takes exercise, concept and response but **not** elapsed time, so
    what runs here is SAINT, not SAINT+. Its timing channel is supplied to the
    tabular models through the L1 rung instead, which is the comparison the
    ablation ladder is actually for.
    """
    _stub_turtle()
    import torch
    from torch import nn

    torch.manual_seed(seed)
    torch.set_num_threads(4)

    skills = sorted(set(train["skill_id"].astype(str)) | set(validate["skill_id"].astype(str)))
    items = sorted(set(train["item_id"].astype(str)) | set(validate["item_id"].astype(str)))
    skill_index = {value: index + 1 for index, value in enumerate(skills)}
    item_index = {value: index + 1 for index, value in enumerate(items)}
    n_concepts, n_items = len(skills) + 1, len(items) + 1

    model = _deep_model(name, n_concepts, n_items)
    start = time.perf_counter()

    fitted = _sequences(train, skill_index, item_index)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_function = nn.BCELoss(reduction="none")
    order = np.arange(len(fitted["concepts"]))
    generator = np.random.default_rng(seed)
    for _ in range(DEEP_EPOCHS):
        generator.shuffle(order)
        model.train()
        for batch in np.array_split(order, max(1, len(order) // DEEP_BATCH)):
            optimiser.zero_grad()
            predicted, mask, target = _deep_forward(model, name, fitted, batch, torch)
            if mask.sum() == 0:
                continue
            loss = (loss_function(predicted, target) * mask).sum() / mask.sum()
            loss.backward()
            optimiser.step()

    model.eval()
    held = _sequences(validate, skill_index, item_index)
    probabilities = np.full(len(validate), np.nan)
    position = {row: index for index, row in enumerate(validate.index)}
    with torch.no_grad():
        for batch in np.array_split(np.arange(len(held["concepts"])),
                                    max(1, len(held["concepts"]) // DEEP_BATCH)):
            predicted, mask, _ = _deep_forward(model, name, held, batch, torch)
            values = predicted.numpy()
            keep = mask.numpy().astype(bool)
            # Predictions are aligned to targets[:, :-1], so row r of the batch
            # carries the label of the *same* position — the shift already
            # happened inside the forward pass.
            rows = held["rows"][batch][:, :-1]
            for value, flag, row in zip(values.ravel(), keep.ravel(), rows.ravel()):
                if flag and row in position:
                    probabilities[position[row]] = value
    unscored = np.isnan(probabilities)

    return BaselineResult(name, _clip(probabilities), time.perf_counter() - start, {
        "rows_unscored": int(unscored.sum()),
        "concepts": n_concepts, "items": n_items,
        "sequence_length": SEQUENCE_LENGTH, "epochs": DEEP_EPOCHS,
        "train_learners": int(train["learner_id"].nunique()),
        "train_windows": int(len(fitted["concepts"])),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "caveat": ("pykt's SAINT consumes exercise/concept/response only, not elapsed time — "
                   "this is SAINT, not SAINT+" if name == "saint" else None),
    })


def _deep_model(name: str, n_concepts: int, n_items: int):
    _stub_turtle()
    if name == "dkt":
        from pykt.models.dkt import DKT
        return DKT(num_c=n_concepts, emb_size=64, dropout=0.2)
    if name == "sakt":
        from pykt.models.sakt import SAKT
        return SAKT(num_c=n_concepts, seq_len=SEQUENCE_LENGTH, emb_size=64,
                    num_attn_heads=4, dropout=0.2)
    if name == "akt":
        from pykt.models.akt import AKT
        return AKT(n_question=n_concepts, n_pid=0, d_model=64, n_blocks=1,
                   dropout=0.2, kq_same=1)
    if name == "saint":
        from pykt.models.saint import SAINT
        return SAINT(num_q=n_items, num_c=n_concepts, seq_len=SEQUENCE_LENGTH,
                     emb_size=64, num_attn_heads=4, dropout=0.2)
    raise ValueError(f"unknown deep model {name!r}")


def _deep_forward(model, name: str, data: dict, batch, torch):
    """One forward pass, normalised across four incompatible pykt signatures.

    Every model here is a *sequence* model: its output at position *i* predicts
    the response at position *i+1*, given everything up to and including *i*.
    That is exactly this project's `next_correct` label, so all four are made to
    return one array aligned to ``targets[:, :-1]`` — one prediction per row that
    has a successor. Each model wants its inputs shaped differently to get
    there, and this adapter is where those four conventions live so the driver
    can treat them as one.

    Getting this wrong is silent: a model fed off-by-one still trains, still
    converges, and still reports an AUC — just a chance-level one.
    `test_sequences_align_the_label_one_step_ahead` is the guard.
    """
    concepts = torch.as_tensor(data["concepts"][batch]).long()
    items = torch.as_tensor(data["items"][batch]).long()
    responses = torch.as_tensor(data["responses"][batch]).long()
    targets = torch.as_tensor(data["targets"][batch]).float()[:, :-1]
    # A position is scored only if both it and its successor are real.
    mask = (torch.as_tensor(data["masks"][batch])[:, :-1]
            & torch.as_tensor(data["masks"][batch])[:, 1:]).float()

    if name == "dkt":
        # LSTM over interactions 0..n-2; read the logit for the concept that is
        # asked next. pyKT's own loop gathers on q[:, 1:] for the same reason.
        output = model(concepts[:, :-1], responses[:, :-1])
        predicted = torch.gather(output, 2, concepts[:, 1:].unsqueeze(-1)).squeeze(-1)
    elif name == "sakt":
        # Causal self-attention over the history, queried with the next concept.
        predicted = model(concepts[:, :-1], responses[:, :-1], concepts[:, 1:])
    elif name == "akt":
        # AKT consumes the full sequence and shifts internally; position i of
        # its output is the prediction for response i, so drop position 0.
        output, _ = model(concepts, responses)
        predicted = output[:, 1:]
    else:  # saint
        # Encoder sees every exercise; the decoder's responses are shifted right
        # and it prepends its own start token, so in_res must be one shorter
        # than in_ex. Output position i predicts response i; drop position 0.
        output = model(items, concepts, responses[:, :-1])
        predicted = output[:, 1:]

    predicted = predicted.reshape(predicted.shape[0], -1)[:, :targets.shape[1]]
    if predicted.shape[1] < targets.shape[1]:
        pad = targets.shape[1] - predicted.shape[1]
        predicted = torch.cat([predicted, torch.full((predicted.shape[0], pad), 0.5)], dim=1)
    return predicted.clamp(1e-6, 1 - 1e-6), mask, targets


#: name -> callable(train, validate) -> BaselineResult. The driver iterates
#: this; adding a baseline means adding a row here and nothing else.
BASELINES = {
    "majority_class": majority_class,
    "item_base_rate": item_base_rate,
    "irt_rasch_ability": irt_ability,
    "pfa_logistic_kt": pfa_logistic,
    "bkt_pybkt": bkt,
    "dkt": lambda train, validate: deep_kt("dkt", train, validate),
    "sakt": lambda train, validate: deep_kt("sakt", train, validate),
    "akt": lambda train, validate: deep_kt("akt", train, validate),
    "saint": lambda train, validate: deep_kt("saint", train, validate),
}

DEEP_MODELS = ("dkt", "sakt", "akt", "saint")


def fit_baseline(name: str, train: pd.DataFrame, validate: pd.DataFrame) -> BaselineResult:
    """Run one baseline, returning a null-but-recorded result if it fails.

    A baseline that cannot run is reported as unavailable with its exception,
    never silently dropped: BUILD.md calls two of these individually fatal to
    omit, so their absence has to be visible in the output rather than in a
    log nobody reads.
    """
    try:
        return BASELINES[name](train, validate)
    except Exception as error:  # noqa: BLE001 - the message is the result
        return BaselineResult(name, np.full(len(validate), np.nan), 0.0,
                              {"failed": f"{type(error).__name__}: {error}".split(chr(10))[0]})


def auc_or_none(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    if np.isnan(probabilities).all() or len(np.unique(labels)) < 2:
        return None
    keep = ~np.isnan(probabilities)
    if keep.sum() < 100 or len(np.unique(labels[keep])) < 2:
        return None
    return float(roc_auc_score(labels[keep], probabilities[keep]))
