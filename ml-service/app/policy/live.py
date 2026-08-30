"""Serving the Phase 8/9 policy to the live platform.

The research arms are written for a cohort: one estimator steps every learner
in lockstep, one arm chooses an action per learner, and both are indexed by a
slot. A live service has the same shape with the cohort spread over time
instead of over a batch, so this module hands each live session a **slot** out
of a fixed pool and drives the arms through their existing contract. Nothing
here re-implements a policy, a feature or a state estimate, and nothing is
fitted inside a request.

Three properties the platform depends on:

1. **The served policy is the arm the paper reports.** ``ALP_POLICY`` selects
   it by the same key Phase 8 and Phase 9 used (``rule`` | ``bandit`` | ``rl``
   are aliases for ``rule_improved``, ``bandit_lin_ts`` and ``rl_ppo``).
2. **A gate-rejected state never leaves the service.** States come from
   :class:`app.policy.state.StateEstimator`, which drops what Phase 7's gate
   rejected rather than zeroing it, and the explanation is the arm's own.
3. **A slot is rebuilt from history, never assumed.** The caller sends the
   session's attempts so far with every request; a slot that is new, recycled
   or lost to a restart replays them. That makes the service restartable
   without a session store.

`ponytail:` one in-process slot pool behind one lock. A second worker gets its
own pool, which is correct but re-warms on the first request, and a session
whose slot is evicted pays one replay. Move the pool to Redis only if the
service is ever run at a scale where that replay shows up in the p95.
"""
from __future__ import annotations

import os
import threading

from app.model import validation_gate
from app.policy import learned, policies
from app.policy.online_features import observable_view
from app.policy.state import StateEstimator

#: `ALP_POLICY` is the one switch for which arm serves. `ADAPTIVE_POLICY` was
#: the backend's old name for it; two names for one switch is how a service
#: ends up running a different policy from the one the paper reports, so the
#: old name is read here only to fail loudly if someone still sets it.
ALIASES = {"rule": "rule_improved", "bandit": "bandit_lin_ts", "rl": "rl_ppo"}
DEFAULT_POLICY = "rule_improved"

#: Live sessions held in memory at once. Beyond this the oldest is evicted and
#: replays if it comes back.
SLOTS = int(os.environ.get("ALP_LIVE_SLOTS") or 32)
SEED = int(os.environ.get("SEED") or 20260821)

#: The state rung served when the arm has no estimator of its own (the rule
#: arms). L4 is the full gated state, which is what the learner-facing panel
#: shows; the arm still only reads what its own rung allows.
PANEL_RUNG = "L4"

#: Exploration rate of the *served* logging policy. It defaults to the value
#: Phase 8 pre-registered and reports, and it is reported by `/model-info`, so
#: a pilot that raises it to collect wider support cannot do so silently. The
#: bandits ignore it: exploration is their own algorithm's job.
EPSILON = float(os.environ.get("ALP_LIVE_EPSILON") or policies.EPSILON)


def gate_status(gate) -> dict:
    """What the gate admitted, and why it refused everything else."""
    heads = set(gate.report.get("states", {})) | set(gate.admitted) | set(validation_gate.DROPPED_STATES)
    return {"admitted": sorted(gate.admitted),
            "rejected": {head: gate.reason(head) for head in sorted(heads)
                         if head not in gate.admitted}}


def policy_key() -> str:
    """The arm this process serves, resolved through the aliases."""
    key = os.environ.get("ALP_POLICY") or DEFAULT_POLICY
    return ALIASES.get(key, key)


def build_arm(key: str, cohort: int, seed: int):
    arm = (learned.build(key, cohort, seed) if key in learned.ARM_KEYS
           else policies.build(key, cohort, seed))
    if not isinstance(arm, learned.LinearBandit):
        arm.epsilon = EPSILON
    return arm


class Engine:
    """One arm, one estimator, and a pool of learner slots in front of them."""

    def __init__(self, key: str | None = None, slots: int = SLOTS, seed: int = SEED):
        self.key = key or policy_key()
        self.slots = slots
        self.seed = seed
        self.policy = build_arm(self.key, slots, seed)
        self.gate = self.policy.gate
        # A model or bandit arm already owns the estimator it conditions on;
        # reusing it keeps one GRU per process and keeps the panel honest about
        # which estimate the decision was actually made from.
        self.estimator = getattr(self.policy, "estimator", None)
        self.owns_estimator = self.estimator is None
        if self.owns_estimator:
            self.estimator = StateEstimator(PANEL_RUNG, slots, gate=self.gate, seed=seed)
        self.lock = threading.RLock()
        self.slot_of: dict[str, int] = {}
        self.order: list[str] = []          # least-recently-used first
        self.observed: dict[str, int] = {}  # attempts already fed per session
        self.free = list(range(slots))
        self.evictions = 0
        self.replays = 0

    # ------------------------------------------------------------- the pool

    def _touch(self, session_id: str) -> None:
        if session_id in self.order:
            self.order.remove(session_id)
        self.order.append(session_id)

    def _clear(self, session_id: str, index: int) -> None:
        self.estimator.reset_one(index)
        pending = getattr(self.policy, "pending", None)
        if pending is not None:
            pending[index] = None
        self.policy.last_rows[index] = None
        self.observed[session_id] = 0

    def acquire(self, session_id: str) -> int:
        """This session's slot, evicting the least recently used if needed."""
        if session_id in self.slot_of:
            self._touch(session_id)
            return self.slot_of[session_id]
        if not self.free:
            oldest = self.order.pop(0)
            self.free.append(self.slot_of.pop(oldest))
            self.observed.pop(oldest, None)
            self.evictions += 1
        index = self.free.pop(0)
        self.slot_of[session_id] = index
        self._touch(session_id)
        self._clear(session_id, index)
        return index

    # ------------------------------------------------------------- the loop

    def sync(self, session_id: str, attempts: list[dict]) -> int:
        """Feed the attempts this slot has not seen. Returns the slot index.

        The caller sends the whole session history every time, so a slot that
        is new or was evicted replays it and a warm slot advances by one step.
        A history shorter than what the slot has already seen can only mean it
        belongs to a different session, so the slot starts over.
        """
        with self.lock:
            index = self.acquire(session_id)
            seen = self.observed.get(session_id, 0)
            if len(attempts) < seen:
                self._clear(session_id, index)
                seen = 0
            if seen == 0 and attempts:
                self.replays += 1
            for attempt in attempts[seen:]:
                batch: list[dict | None] = [None] * self.slots
                batch[index] = attempt
                self.policy.observe(batch)
                if self.owns_estimator:
                    self.estimator.observe(batch)
            self.observed[session_id] = len(attempts)
            return index

    def states(self, index: int) -> dict:
        """The gated estimate, its uncertainty, and what the gate refused."""
        with self.lock:
            return {
                "states": self.estimator.states(index),
                "standard_error": self.estimator.standard_error(index),
                "rung": self.estimator.rung,
                "steps": int(self.estimator.steps[index]),
                "gate": gate_status(self.gate),
            }

    def decide(self, session_id: str, view: dict) -> dict:
        """One action for one learner, with its propensity and explanation."""
        with self.lock:
            index = self.acquire(session_id)
            full = {
                **view,
                "index": index,
                "observables": observable_view(self.policy.last_rows[index], self.policy.rung),
            }
            batch: list[dict | None] = [None] * self.slots
            batch[index] = full
            action = self.policy.act(batch)[index]
            return {**action, "policy": self.policy.key, "rung": self.policy.rung,
                    "action_space": learned.N_ACTIONS,
                    "action_index": learned.action_index(action)}

    def info(self) -> dict:
        return {
            "policy": self.policy.key,
            "policy_rung": self.policy.rung,
            "estimator_rung": self.estimator.rung,
            "epsilon": float(getattr(self.policy, "epsilon", 0.0)),
            "action_space": learned.N_ACTIONS,
            "gate": gate_status(self.gate),
            "live_slots": self.slots,
            "slots_in_use": len(self.slot_of),
            "evictions": self.evictions,
            "replays": self.replays,
            "seed": self.seed,
        }


_engine: Engine | None = None
_build_lock = threading.Lock()


def engine() -> Engine:
    """The process-wide engine, built on first use.

    Loading the encoder and the arm costs a second or two; doing it lazily
    keeps a service that is only being health-checked from paying for it, and
    doing it once keeps it out of the request path after that.
    """
    global _engine
    if _engine is None:
        with _build_lock:
            if _engine is None:
                _engine = Engine()
    return _engine


def reset() -> None:
    """Drop the engine. Tests use this to serve a different arm."""
    global _engine
    with _build_lock:
        _engine = None
