"""Adaptive policies — the arms compared in Phase 8's closed loop.

Four modules, one job each:

* :mod:`app.policy.online_features` — the catalogue's history features computed
  one attempt at a time, so a policy sees exactly what Study 1 and Study 2 were
  fitted on.
* :mod:`app.policy.state` — the Phase 7 encoder run online, gated by
  ``app.model.validation_gate``. A state that failed the gate is absent here,
  not zeroed.
* :mod:`app.policy.rules` — the state-to-action rules, each with a citation and
  a declared signal-class requirement.
* :mod:`app.policy.policies` — the arms themselves, all emitting the same
  action space and a propensity.

**Nothing in this package may import the simulator.** The environment calls the
policy; the policy never reads the environment's response function. That was
Defect 1 in ``BUILD.md`` and ``app.research.test_policies`` greps for it.
"""
