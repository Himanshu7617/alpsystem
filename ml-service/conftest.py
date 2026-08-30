"""Test-session setup for the ml-service and research suites.

macOS/arm64 ships three OpenMP runtimes in this environment (torch's bundled
libomp, scikit-learn's bundled copy, and the Homebrew libomp that the LightGBM
wheel dlopens). Whichever one loads second while another is already
initialised takes down the process: unpickling a LightGBM booster after
``import torch`` segfaults inside ``Booster.__setstate__``. Importing lightgbm
first makes its runtime the incumbent and the others attach to it.

The failure is a hard SIGSEGV in ``__kmp_suspend_initialize_thread`` (exit
139), not an exception, so it cannot be caught at the call site. Do not remove
or reorder this import.

This duplicates the guard in ``app/__init__.py``, which is the one that covers
the application: uvicorn and ``python -m app.research.*`` never load conftest,
and pytest loads conftest before it imports anything under ``app``. Both paths
need their own copy; deleting either one leaves the other path unprotected.

ponytail: import-order workaround, not a fix. Upgrade path: a lightgbm wheel
built against the same OpenMP runtime torch bundles, or a lightgbm built with
USE_OPENMP=OFF, removes the need for this file.
"""

try:  # pragma: no cover - environment guard, not logic
    import lightgbm  # noqa: F401
except ImportError:
    pass
