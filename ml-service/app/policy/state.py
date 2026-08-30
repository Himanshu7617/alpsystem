"""The Phase 7 state encoder, run online and gated.

A policy in Phase 8 needs a state estimate *before* it picks the next item, from
the attempts it has seen so far and nothing else. This module steps Study 2's
GRU one attempt at a time over a whole cohort at once — batched, because a
single-learner GRU step costs 2 ms of framework overhead and a 1,000-learner
step costs 2.5 ms in total.

Two properties this module exists to hold:

1. **Only admitted states leave it.** :meth:`StateEstimator.states` runs every
   estimate through ``validation_gate.Gate.filter_states``, which *drops* a
   rejected state rather than zeroing it. Phase 7's gate rejected engagement
   (Wise & Kong's negative criterion) and confidence (discriminant validity),
   so a policy built on this module can condition on knowledge and nothing else
   — and the same code admits them again if a future run passes them.
2. **Each rung gets its own encoder.** ``model_L0`` reads the encoder that was
   trained on correctness features only. Handing every arm the top-rung
   estimate would make the ladder a relabelling exercise.

Dropout sits after the GRU trunk in Study 2's architecture, so the recurrence
is deterministic and only the head is stochastic: the MC-dropout standard error
is computed by resampling the head's dropout mask on the current hidden state,
which is exactly ``study2.predict`` without re-running the sequence.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from app.model import validation_gate
from app.policy.online_features import LearnerFeatures, item_response_stats, matrix

ROOT = Path(__file__).resolve().parents[3]
MODEL_DIR = ROOT / "artifacts" / "models"
RUNG_ENCODER_PATH = MODEL_DIR / "state-encoders-by-rung.pt"
RUNG_CALIBRATOR_PATH = MODEL_DIR / "state-calibrators-by-rung.joblib"

#: Study 2's MC-dropout budget (pre-registration §8.3).
MC_PASSES = 20


def available_rungs() -> list[str]:
    if not RUNG_ENCODER_PATH.exists():
        return []
    return list(torch.load(RUNG_ENCODER_PATH, weights_only=False, map_location="cpu"))


class StateEstimator:
    """One rung's encoder, stepped over a cohort of learners in lockstep.

    ``observe(attempts)`` takes one attempt record per learner (``None`` for a
    learner who did not act this step), advances the GRU, and stores the
    calibrated state probabilities. ``states(index)`` returns the gated state
    dict for one learner.
    """

    def __init__(self, rung: str, cohort: int, gate: validation_gate.Gate | None = None,
                 seed: int = 0, mc_passes: int = MC_PASSES):
        if not RUNG_ENCODER_PATH.exists():
            raise SystemExit(f"{RUNG_ENCODER_PATH.relative_to(ROOT)} does not exist. "
                             "Run `make train-states` first.")
        import joblib

        checkpoints = torch.load(RUNG_ENCODER_PATH, weights_only=False, map_location="cpu")
        if rung not in checkpoints:
            raise SystemExit(f"rung {rung} has no encoder. Trained: {sorted(checkpoints)}")
        checkpoint = checkpoints[rung]
        calibrators = joblib.load(RUNG_CALIBRATOR_PATH).get(rung, {})

        from app.research.study2 import StateEncoder  # local: keeps torch off the import path

        self.rung = rung
        self.columns = list(checkpoint["columns"])
        self.centre = np.array([checkpoint["centre"][name] for name in self.columns], dtype="float32")
        self.scale = np.array([checkpoint["scale"][name] for name in self.columns], dtype="float32")
        self.chunk = int(checkpoint["chunk"])
        self.dropout = float(checkpoint["dropout"])
        self.heads = list(checkpoint["heads"])
        self.model = StateEncoder(2 * len(self.columns), checkpoint["hidden"], self.dropout)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.calibrators = calibrators
        self.gate = gate if gate is not None else validation_gate.load()
        self.mc_passes = mc_passes
        self.generator = torch.Generator().manual_seed(seed)

        self.hidden_size = int(checkpoint["hidden"])
        self.cohort = cohort
        self.features = [LearnerFeatures(item_response_stats()) for _ in range(cohort)]
        self.hidden = torch.zeros(1, cohort, self.hidden_size)
        self.steps = np.zeros(cohort, dtype=int)
        self.probability = {head: np.full(cohort, np.nan, dtype="float32") for head in self.heads}
        self.error = {head: np.full(cohort, np.nan, dtype="float32") for head in self.heads}
        self.rows: list[dict | None] = [None] * cohort

    def reset(self) -> None:
        """Forget the cohort and start again, without reloading the encoder.

        The RL environment builds a fresh cohort every episode; reloading a
        checkpoint each time costs more than every other part of a step
        combined. The weights are unchanged, so only the per-learner buffers
        need clearing.
        """
        self.features = [LearnerFeatures(item_response_stats()) for _ in range(self.cohort)]
        self.hidden = torch.zeros(1, self.cohort, self.hidden_size)
        self.steps = np.zeros(self.cohort, dtype=int)
        for head in self.heads:
            self.probability[head][:] = np.nan
            self.error[head][:] = np.nan
        self.rows = [None] * self.cohort

    def reset_one(self, index: int) -> None:
        """Forget one learner's history, leaving the rest of the cohort alone.

        The live service hands each session a slot out of a fixed pool; a slot
        that is recycled must not inherit the previous learner's hidden state.
        """
        self.features[index] = LearnerFeatures(item_response_stats())
        self.hidden[:, index, :] = 0.0
        self.steps[index] = 0
        for head in self.heads:
            self.probability[head][index] = np.nan
            self.error[head][index] = np.nan
        self.rows[index] = None

    # ------------------------------------------------------------------ input

    def rows_for(self, attempts: list[dict | None]) -> list[dict | None]:
        """Modelling rows for one step, history features included."""
        rows: list[dict | None] = [None] * self.cohort
        for index, attempt in enumerate(attempts):
            if attempt is not None:
                rows[index] = self.features[index].row(attempt)
        return rows

    def pack(self, rows: list[dict | None]) -> torch.Tensor:
        """Rows → the encoder's input, standardised the way Study 2 packed it."""
        values = matrix(rows, self.columns)
        present = np.isfinite(values).astype("float32")
        centred = (values - self.centre) / self.scale
        centred = np.nan_to_num(np.clip(centred, -5, 5), nan=0.0, posinf=0.0, neginf=0.0)
        return torch.from_numpy(np.concatenate([centred, present], axis=1)[:, None, :])

    # ------------------------------------------------------------------ step

    @torch.no_grad()
    def observe(self, attempts: list[dict | None]) -> list[dict | None]:
        """Advance the estimate by one attempt per learner. Returns the rows."""
        rows = self.rows_for(attempts)
        acted = [index for index, row in enumerate(rows) if row is not None]
        if not acted:
            return rows
        self.rows = rows

        # Study 2 trained on `chunk`-long sequences with the hidden state reset
        # at each boundary. Carrying it further at decision time would feed the
        # model a state it never saw in training.
        reset = [index for index in acted if self.steps[index] and self.steps[index] % self.chunk == 0]
        if reset:
            self.hidden[:, reset, :] = 0.0

        index = torch.tensor(acted)
        trunk, hidden = self.model.gru(self.pack([rows[i] for i in acted]),
                                       self.hidden[:, index, :].contiguous())
        self.hidden[:, index, :] = hidden
        trunk = trunk[:, 0, :]

        # MC dropout on the trunk output only — the recurrence above is
        # deterministic, exactly as in `study2.predict`.
        keep = 1.0 - self.dropout
        mask = torch.bernoulli(torch.full((self.mc_passes, *trunk.shape), keep),
                               generator=self.generator) / keep
        draws = trunk[None, :, :] * mask
        for head in self.heads:
            probabilities = torch.sigmoid(self.model.heads[head](draws).squeeze(-1)).numpy()
            mean = probabilities.mean(axis=0)
            calibrator = self.calibrators.get(head)
            self.probability[head][acted] = (calibrator.predict(mean) if calibrator is not None
                                             else mean)
            self.error[head][acted] = probabilities.std(axis=0)
        self.steps[acted] += 1
        return rows

    # ----------------------------------------------------------------- output

    def states(self, index: int) -> dict:
        """The gated state estimate for one learner.

        A state that failed Phase 7's gate is **absent** from this dict. So is a
        state that has not been measured yet, because the learner has made no
        attempt this session.
        """
        estimate = {head: float(self.probability[head][index]) for head in self.heads
                    if np.isfinite(self.probability[head][index])}
        return self.gate.filter_states(estimate)

    def standard_error(self, index: int) -> dict:
        errors = {head: float(self.error[head][index]) for head in self.heads
                  if np.isfinite(self.error[head][index])}
        return self.gate.filter_states(errors)
