"""Run profiles and resource limits — `smoke` | `dev` | `full`.

Every expensive research script resolves its workload here rather than
defaulting to "every core, every learner". A phase that pins ten cores for an
hour is how the Phase 8 run got interrupted the first time, and an interrupted
run that cannot resume is a run that has to be paid for twice.

Precedence, highest first:

1. an explicit command-line flag (``--learners 500``),
2. an ``ALP_*`` environment variable,
3. the profile default.

Profiles

* ``smoke`` — smallest end-to-end run that still exercises every branch. Fast,
  writes nothing publishable. This is the default, deliberately: `make all`
  must not launch a full experiment as a side effect.
* ``dev`` — enough learners and variants to read a real effect and check
  resource use before committing to the full grid.
* ``full`` — the pre-registered experiment. Never the default; `make full`
  asks for it by name and `docs/full-run-justification.md` records why.
"""
from __future__ import annotations

import os

#: The knobs every long-running research script shares, and the environment
#: variable that overrides each one.
ENV = {
    "workers": "ALP_MAX_WORKERS",
    "learners": "ALP_MAX_STUDENTS",
    "budget": "ALP_MAX_INTERACTIONS",
    "seeds": "ALP_SEEDS",
    "variants": "ALP_VARIANTS",
    "rl_timesteps": "ALP_RL_TIMESTEPS",
    "checkpoint_interval": "ALP_CHECKPOINT_INTERVAL",
}

PROFILES = {
    "smoke": {"workers": 2, "learners": 40, "budget": 60, "seeds": 1,
              "variants": "V0", "rl_timesteps": 2_000, "checkpoint_interval": 1},
    "dev": {"workers": 4, "learners": 250, "budget": 200, "seeds": 1,
            "variants": "V0,V1", "rl_timesteps": 50_000, "checkpoint_interval": 5},
    "full": {"workers": 5, "learners": 1000, "budget": 200, "seeds": 3,
             "variants": "V0,V1,V2,V3,V4", "rl_timesteps": 300_000,
             "checkpoint_interval": 10},
}

DEFAULT = "smoke"
#: `full` leaves cores free deliberately. **A worker count is only half of the
#: cap**: each worker's numpy/BLAS will happily spawn one thread per core on its
#: own, so six workers measured at 997 % CPU on an eight-core allocation — the
#: worker count said six and the machine ran flat out. `THREAD_LIMITS` below is
#: the other half, and both are needed.
# ponytail: fixed caps, not a thermal governor. Read the SMC sensors and
# throttle adaptively only if a capped run still overheats.

#: One thread per worker. Set in the environment before numpy is imported —
#: `docker-compose.yml` and the Makefile's `LIMITS` both pass these, and
#: `apply_thread_limits()` is the belt-and-braces for a bare `python -m` call.
THREAD_LIMITS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                 "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS")


def apply_thread_limits(threads: int = 1) -> None:
    """Pin every BLAS backend to `threads`. Call before importing numpy."""
    for variable in THREAD_LIMITS:
        os.environ.setdefault(variable, str(threads))


def name() -> str:
    """The active profile name."""
    chosen = os.environ.get("ALP_RUN_PROFILE", DEFAULT).strip().lower()
    if chosen not in PROFILES:
        raise SystemExit(f"ALP_RUN_PROFILE={chosen!r} is not one of {sorted(PROFILES)}")
    return chosen


def load(profile: str | None = None) -> dict:
    """Profile defaults with any ``ALP_*`` environment override applied."""
    chosen = profile or name()
    if chosen not in PROFILES:
        raise SystemExit(f"--profile {chosen!r} is not one of {sorted(PROFILES)}")
    settings = dict(PROFILES[chosen])
    for key, variable in ENV.items():
        raw = os.environ.get(variable)
        if raw is None or not raw.strip():
            continue
        settings[key] = raw.strip() if isinstance(settings[key], str) else int(raw)
    settings["profile"] = chosen
    # Never hand out more workers than the machine has, however it was asked for.
    settings["workers"] = max(1, min(int(settings["workers"]), os.cpu_count() or 1))
    return settings


def peek(argv: list[str]) -> str | None:
    """``--profile NAME`` read before argparse, so it can set the defaults."""
    for index, token in enumerate(argv):
        if token == "--profile" and index + 1 < len(argv):
            return argv[index + 1]
        if token.startswith("--profile="):
            return token.split("=", 1)[1]
    return None


def banner(settings: dict, extra: str = "") -> str:
    parts = [f"profile={settings['profile']}"]
    parts += [f"{key}={settings[key]}" for key in
              ("workers", "learners", "budget", "seeds", "variants") if key in settings]
    return "run: " + " ".join(parts) + (f" {extra}" if extra else "")


def demo() -> None:
    """Self-check: precedence is flag > environment > profile default."""
    os.environ.pop("ALP_MAX_WORKERS", None)
    os.environ["ALP_RUN_PROFILE"] = "dev"
    assert name() == "dev"
    assert load()["learners"] == PROFILES["dev"]["learners"]
    os.environ["ALP_MAX_STUDENTS"] = "7"
    assert load()["learners"] == 7
    assert load("smoke")["learners"] == 7, "environment overrides the profile it is given"
    del os.environ["ALP_MAX_STUDENTS"]
    assert load("smoke")["learners"] == PROFILES["smoke"]["learners"]
    os.environ["ALP_MAX_WORKERS"] = "9999"
    assert load()["workers"] <= (os.cpu_count() or 1), "worker cap is never exceeded"
    del os.environ["ALP_MAX_WORKERS"]
    apply_thread_limits()
    assert all(os.environ[name] == "1" for name in THREAD_LIMITS), "BLAS threads pinned"
    os.environ["ALP_RUN_PROFILE"] = "nonsense"
    try:
        name()
    except SystemExit:
        pass
    else:
        raise AssertionError("an unknown profile must fail loudly")
    del os.environ["ALP_RUN_PROFILE"]
    assert peek(["--profile", "full"]) == "full"
    assert peek(["--profile=full"]) == "full"
    assert peek(["--seed", "1"]) is None
    print("runprofile: ok")


if __name__ == "__main__":
    demo()
