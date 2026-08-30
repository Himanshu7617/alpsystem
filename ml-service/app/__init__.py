"""Package init for the ml-service.

**Import order is load-bearing: LightGBM must be imported before PyTorch.**
This import is the first statement in the package for that reason. Nothing may
be added above it.

torch and the LightGBM wheel ship separate copies of the OpenMP runtime (torch
bundles its own libomp; the LightGBM wheel dlopens the Homebrew one). If torch's
copy initialises first, unpickling a LightGBM booster later kills the process:
SIGSEGV in ``__kmp_suspend_initialize_thread``, exit 139, no traceback and
nothing to catch. It is a hard crash, not an exception, which is why it cannot
be handled where the model is loaded — it has to be prevented here.

Two facts that make this easy to get wrong:

* Importing lightgbm *after* torch does not help. ``import torch; import
  lightgbm; joblib.load(<lgbm model>)`` still segfaults. Only importing
  lightgbm first works.
* scikit-learn's bundled copy is harmless in this environment — ``import
  sklearn`` before lightgbm loads a booster fine. torch is the one that matters.

Importing lightgbm here makes its runtime the incumbent and torch attaches to
it. This runs before any ``app.*`` submodule, so it covers both entry points:
``uvicorn app.main:app`` and ``python -m app.research.<script>``.

DO NOT remove, reorder or "tidy" this import, and do not move a torch, sklearn
or lightgbm import above it. To see the failure it prevents::

    python -c "import torch, joblib; joblib.load('artifacts/models/study1-sim-V0-lightgbm/model.joblib')"

The same guard is duplicated in ``ml-service/conftest.py`` because pytest loads
conftest before it imports anything under ``app``.

ponytail: import-order workaround, not a fix. Upgrade path: a lightgbm wheel
built against the same OpenMP runtime torch bundles, or one built with
USE_OPENMP=OFF, removes the need for this.
"""
import sys
import warnings

#: True when something imported torch before this package, i.e. before the
#: guard below could run. Nothing here can undo that — importing lightgbm after
#: torch does not help — so the only useful move is to say so loudly while
#: there is still a readable stderr, rather than let the process die at
#: ``joblib.load`` with no traceback.
_TORCH_WON_THE_RACE = "torch" in sys.modules

try:  # pragma: no cover - environment guard, not logic
    import lightgbm  # noqa: F401  # MUST stay first; see module docstring
except ImportError:  # lightgbm not installed: nothing to protect
    pass
else:
    if _TORCH_WON_THE_RACE:
        warnings.warn(
            "torch was imported before app/__init__.py, so LightGBM's OpenMP "
            "runtime is not the incumbent. Loading a LightGBM model in this "
            "process will segfault (SIGSEGV in __kmp_suspend_initialize_thread, "
            "exit 139). Import `app` before torch. See app/__init__.py.",
            RuntimeWarning,
            stacklevel=2,
        )
