# Adaptive Learning Platform — task DAG.
#
# Every target is idempotent and prints what it wrote. Targets belonging to a
# phase that has not been built yet fail loudly rather than silently doing
# nothing: a research pipeline that quietly skips a step is worse than one that
# stops. See BUILD.md for the phase each stub belongs to.

SHELL := /bin/bash
.DEFAULT_GOAL := help

SEED ?= 20260821
# Workload size for every expensive research target: smoke | dev | full.
# `smoke` is the default on purpose — `make all` must not launch a full
# experiment as a side effect of being typed. See ml-service/app/research/
# runprofile.py for what each profile sets and which ALP_* variable overrides it.
PROFILE ?= smoke
# Threads for single-process fitting targets (train-baselines, train-states).
# Matches the worker cap used by the multi-process targets: same total, one
# knob per shape of parallelism.
FIT_THREADS ?= 5
COMPOSE ?= docker compose
BACKEND_DIR := backend
ML_DIR := ml-service
# Research scripts resolve `artifacts/` relative to the working directory and
# predictor.py walks up to the repository root, so both only agree when python
# runs from here with ml-service on the path.
PY ?= PYTHONPATH=$(ML_DIR) python3

DB_URL ?= postgresql://adaptive_learning:adaptive_learning@127.0.0.1:54329/adaptive_learning?schema=public

.PHONY: help up down logs seed test lint items telemetry fetch-data prep-data simulate features \
        train-baselines train-states eval-policies ope rl figures stats report \
        integrate paper all clean smoke dev full status test-report reproduce

help:
	@echo "Infrastructure : up down logs seed test lint clean"
	@echo "Data           : items telemetry fetch-data prep-data simulate features"
	@echo "Models         : train-baselines train-states"
	@echo "Evaluation     : eval-policies integrate ope rl"
	@echo "Reporting      : figures stats report paper"
	@echo "Verification   : test test-report reproduce"
	@echo "Run profiles   : smoke dev full status  (or PROFILE=dev make eval-policies)"
	@echo "Everything     : all   (runs at PROFILE=$(PROFILE))"
	@echo
	@echo "Seed: $(SEED). Profile: $(PROFILE) — smoke validates, dev develops, full is"
	@echo "the pre-registered experiment and must be asked for by name."
	@echo "Overrides: ALP_MAX_WORKERS ALP_MAX_STUDENTS ALP_MAX_INTERACTIONS ALP_SEEDS"
	@echo "           ALP_VARIANTS ALP_RL_TIMESTEPS ALP_CHECKPOINT_INTERVAL"
	@echo
	@echo "Targets marked PHASE N are not built yet (see BUILD.md)."

# ---------------------------------------------------------------- infrastructure

up:
	$(COMPOSE) up -d --build
	@echo "wrote: running containers ->"
	@$(COMPOSE) ps

down:
	$(COMPOSE) down
	@echo "wrote: nothing (containers stopped; the postgres volume is kept)"

logs:
	$(COMPOSE) logs -f --tail=100

seed:
	cd $(BACKEND_DIR) && DATABASE_URL="$(DB_URL)" npx prisma migrate deploy
	cd $(BACKEND_DIR) && DATABASE_URL="$(DB_URL)" npx prisma db seed

# OMP_NUM_THREADS=1 on the host path for the same reason app/__init__.py
# imports lightgbm first: torch and lightgbm bring their own OpenMP runtimes.
# The import-order guard stops the segfault; it does not stop the deadlock the
# two runtimes hit with multiple threads, which hangs test_study2.py forever
# rather than failing. The container already sets this in docker-compose.yml.
test:
	cd $(BACKEND_DIR) && DATABASE_URL="$(DB_URL)" npm test
	@if $(COMPOSE) ps --status running --services 2>/dev/null | grep -qx ml-service; then \
	    $(COMPOSE) exec -T ml-service sh -c "cd $(ML_DIR) && pytest -q"; \
	elif command -v pytest >/dev/null 2>&1; then \
	    cd $(ML_DIR) && OMP_NUM_THREADS=1 pytest -q; \
	else \
	    echo "ml-service suite SKIPPED: no running container and no pytest on PATH. Run 'make up' first."; \
	    exit 1; \
	fi

# No linter is installed and none is worth adding for this project: a syntax
# gate over both languages catches what actually breaks a pipeline run.
lint:
	@find $(BACKEND_DIR)/src $(BACKEND_DIR)/server.js $(BACKEND_DIR)/prisma \
	    -name '*.js' -not -path '*/node_modules/*' -print0 \
	    | xargs -0 -n1 node --check
	@echo "checked: backend javascript syntax"
	@python3 -m compileall -q $(ML_DIR)/app >/dev/null
	@echo "checked: ml-service python syntax"

clean:
	rm -rf $(ML_DIR)/app/**/__pycache__ $(ML_DIR)/app/__pycache__ .pytest_cache
	@echo "removed: python bytecode caches"
	@echo "kept: artifacts/ (regenerate deliberately, never as a side effect of clean)"

# ------------------------------------------------------------------- pipeline
# Research scripts need the ml-service dependency set (numpy, sklearn,
# matplotlib). Prefer the running container, whose image is the pinned
# environment; fall back to host python only if no container is up.
# The ALP_* limits are forwarded on every call: docker compose resolves the
# environment: block when the container is *created*, so a profile chosen at
# `make` time would otherwise be ignored by an already-running container.
# One thread per worker: a worker cap alone does not bound CPU, because each
# worker's BLAS spawns a thread per core underneath it.
# Phase 10 load test: the running stack, and how hard to push it.
BASE_URL ?= http://127.0.0.1:4000
ML_URL ?= http://127.0.0.1:8000
LOAD_LEARNERS ?= 12
LOAD_ITEMS ?= 20
LOAD_SWEEP ?= 1,4,8,16

THREADS = -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 -e MKL_NUM_THREADS=1 \
          -e NUMEXPR_NUM_THREADS=1 -e VECLIB_MAXIMUM_THREADS=1

LIMITS = -e ALP_RUN_PROFILE=$(PROFILE) -e ALP_MAX_WORKERS -e ALP_MAX_STUDENTS \
         -e ALP_MAX_INTERACTIONS -e ALP_SEEDS -e ALP_VARIANTS -e ALP_RL_TIMESTEPS \
         -e ALP_CHECKPOINT_INTERVAL -e PYTHONHASHSEED=0 $(THREADS)

define RUN_PY_SCRIPT
@if $(COMPOSE) ps --status running --services 2>/dev/null | grep -qx ml-service; then \
    $(COMPOSE) exec -T -w /workspace $(LIMITS) ml-service python $(1); \
else \
    ALP_RUN_PROFILE=$(PROFILE) PYTHONHASHSEED=0 $(PY) $(1); \
fi
endef

define RUN_PY
@if $(COMPOSE) ps --status running --services 2>/dev/null | grep -qx ml-service; then \
    $(COMPOSE) exec -T $(LIMITS) ml-service python -m $(1); \
else \
    ALP_RUN_PROFILE=$(PROFILE) PYTHONHASHSEED=0 $(PY) -m $(1); \
fi
endef

# PHASE 1 — item bank, validation, Rasch calibration, content figures.
# Re-run after Phase 3: calibrate_items replaces the author priors with fitted
# b once archival responses mapped to bank items exist.
items:
	$(call RUN_PY,app.research.build_item_bank --seed $(SEED))
	$(call RUN_PY,app.research.validate_bank)
	$(call RUN_PY,app.research.calibrate_items --seed $(SEED))
	$(call RUN_PY,app.research.figures_phase1)

# PHASE 2 — telemetry figures. The instrument itself (schema, frontend, /v1
# endpoints, extractor) is exercised by `make test`; this target only rebuilds
# the four figures that document it.
telemetry:
	$(call RUN_PY,app.research.figures_phase2 --seed $(SEED))

# Each stub below names the phase that fills it in.

define PHASE_STUB
@echo "$(1) is not built yet — BUILD.md $(2) implements it."; exit 1
endef

# PHASE 3 — real archival data. `fetch-data` downloads ASSISTments 2009/2012 and
# an EdNet KT1 shard into data/raw/ and verifies each against a pinned SHA-256;
# it needs only the standard library, so it runs on the host rather than in the
# container. `prep-data` normalises them to one schema, splits by learner and
# draws the figure pack.
fetch-data:
	python3 scripts/fetch_data.py --seed $(SEED)

prep-data:
	$(call RUN_PY,app.research.prep_archival --seed $(SEED))
	$(call RUN_PY,app.research.figures_phase3 --seed $(SEED))

# PHASE 4 — calibrated simulator. Calibration fits the generative model against
# the Phase 3 archival tables and exits non-zero on a distribution mismatch
# (KS D > 0.15); generation writes the calibrated baseline and its four
# deliberate mis-specifications; the figures document the fit and the
# before/after of Defect 3.
simulate:
	$(call RUN_PY,app.research.calibrate_simulator --seed $(SEED))
	$(call RUN_PY,app.research.generate_sim --seed $(SEED))
	$(call RUN_PY,app.research.figures_phase4 --seed $(SEED))

# PHASE 5 — feature engineering and EDA. `build_features` writes one matrix per
# source from the catalogue; `audit_leakage` is the standing guard against
# Defect 2 and exits non-zero on a leak (it also runs inside `make test`); `eda`
# draws the thirteen figures and writes the summary and shortlist JSON. The
# catalogue doc is generated on the host because docs/ is mounted read-only.
features:
	$(call RUN_PY,app.research.build_features --seed $(SEED))
	$(call RUN_PY,app.research.audit_leakage)
	$(call RUN_PY,app.research.eda --seed $(SEED))
	$(PY) -m app.research.feature_catalogue --write-doc

# PHASE 6 — Study 1. `study1` runs the ablation ladder over every model and
# every live rung, cross-validated on the CV folds, logs each cell to MLflow,
# then touches the test set exactly once and writes study1-final.json.
# `figures_phase6` draws the fourteen charts from those files and appends the
# three findings only a figure computes. Expect ~40 minutes.
# study1 is ONE process, not a worker pool, so the one-thread-per-worker cap
# that keeps eval-policies from taking the machine makes it serial instead.
# Give it the same total budget the pool gets: threads here, workers there.
train-baselines: THREADS = -e OMP_NUM_THREADS=$(FIT_THREADS) -e OPENBLAS_NUM_THREADS=$(FIT_THREADS) \
                           -e MKL_NUM_THREADS=$(FIT_THREADS) -e NUMEXPR_NUM_THREADS=$(FIT_THREADS) \
                           -e VECLIB_MAXIMUM_THREADS=$(FIT_THREADS)
train-baselines:
	$(call RUN_PY,app.research.study1 --seed $(SEED))
	$(call RUN_PY,app.research.figures_phase6 --seed $(SEED))

# PHASE 7 — Study 2. `study2` trains the shared encoder with one supervised
# head per state at every live rung, calibrates each head on half of fold4,
# scores it on the other half, and writes the validation gate's verdict —
# passes and failures alike. `figures_phase7` draws the ten charts from those
# files. The test set is not touched by this target.
train-states:
	$(call RUN_PY,app.research.study2 --seed $(SEED))
	$(call RUN_PY,app.research.figures_phase7 --seed $(SEED))

# PHASE 8 — the closed loop. `evaluate_policies` runs ten arms against the
# profile's simulator variants × seeds × learners, writes the per-variant
# results, the per-learner CSVs, the decision log with propensities (Study 3
# needs them and they cannot be backfilled) and the L0-vs-L4 divergence report;
# it exits non-zero if any Cohen's d exceeds 3 without a documented exemption.
# `figures_phase8` draws the fourteen charts from those files.
#
#   make smoke                        10 cells, ~5 s      — does it run at all
#   PROFILE=dev make eval-policies    20 cells, ~1.5 min  — does it say anything
#   make full                        150 cells, ~30 min   — the experiment
#
# Every finished cell is checkpointed, so an interrupted run resumes; `make
# status` reports where it got to.
eval-policies:
	$(call RUN_PY,app.research.evaluate_policies --seed $(SEED))
	$(call RUN_PY,app.research.figures_phase8 --seed $(SEED))

# PHASE 8.5 — integration. `validate_phase85` runs the end-to-end chain at the
# smoke size and writes artifacts/evaluation/phase-8.5-validation.json, pass or
# fail; `figures_phase85` draws the four integration figures, each annotated
# with numbers read from the artifact tree at render time.
integrate:
	$(call RUN_PY,app.research.validate_phase85 --seed $(SEED))
	$(call RUN_PY,app.research.figures_phase85 --seed $(SEED))

# PHASE 9 — Study 3. `ope` writes the logged dataset with propensities, trains
# the two bandits online and reports IPS/SNIPS/DR against the true on-policy
# value; `rl` trains PPO on V0 and evaluates it on V0–V4. Both draw the figure
# pack, so `make ope` alone gives f09-01 to f09-05 and `make rl` completes it.
#
# PPO resumes from artifacts/models/rl/checkpoints/, so an interrupted run
# continues rather than restarting; `make status` reports where it got to.
#   make ope                 ~4 s at smoke, ~1 min at dev
#   PROFILE=dev make rl      50k timesteps, minutes not hours
ope:
	$(call RUN_PY,app.research.ope --seed $(SEED))
	$(call RUN_PY,app.research.figures_phase9 --seed $(SEED))

rl:
	$(call RUN_PY,app.research.train_rl --seed $(SEED))
	$(call RUN_PY,app.research.figures_phase9 --seed $(SEED))

# PHASE 10 — the live platform. `bank` regenerates the item bank the running
# app serves from the canonical Phase 1 bank, so the platform and the
# experiments cannot drift on to different items. `loadtest` and `system`
# measure the deployed loop and need the stack up (`make up`).
bank:
	$(call RUN_PY,app.research.export_bank)

loadtest:
	@python3 scripts/loadtest.py --base-url $(BASE_URL) --ml-url $(ML_URL) \
	    --learners $(LOAD_LEARNERS) --items $(LOAD_ITEMS) --sweep $(LOAD_SWEEP)

system: loadtest
	$(call RUN_PY,app.research.figures_phase10 --dashboard $(BASE_URL))

# PHASE 11 — statistics and the figure pack. `stats` writes the mixed-effects
# models, the unpaired effect sizes with bootstrap CIs, the Holm corrections,
# the power analysis and the censored-outcome treatment, plus the six
# sensitivity analyses and the LaTeX/CSV table pack. `figures` redraws every
# figure in the project from artifacts and rebuilds artifacts/figures/INDEX.md
# from the provenance each figure records as it is written.
#
# The sensitivity sweep over the mastery threshold is the one analysis that
# reruns the closed loop, so it obeys the run profile like every other
# expensive target: PROFILE=dev make stats.
# Phase 5's figures are drawn by `eda`, not by a `figures_phase5` — it is in the
# list because `make figures` must rebuild every figure, not most of them.
FIGURE_SCRIPTS = figures_phase1 figures_phase2 figures_phase3 figures_phase4 eda \
                 figures_phase6 figures_phase7 figures_phase8 figures_phase85 \
                 figures_phase9 figures_phase10 figures_phase11 figures_phase12

stats:
	$(call RUN_PY,app.research.sensitivity --seed $(SEED))
	$(call RUN_PY,app.research.stats --seed $(SEED))

figures: stats
	@rm -f artifacts/figures/manifest.jsonl
	@for script in $(FIGURE_SCRIPTS); do \
	    echo "--- $$script"; \
	    if $(COMPOSE) ps --status running --services 2>/dev/null | grep -qx ml-service; then \
	        $(COMPOSE) exec -T $(LIMITS) ml-service python -m app.research.$$script --seed $(SEED) || exit 1; \
	    else \
	        ALP_RUN_PROFILE=$(PROFILE) PYTHONHASHSEED=0 $(PY) -m app.research.$$script --seed $(SEED) || exit 1; \
	    fi; \
	done
	$(call RUN_PY,app.research.figure_index)

# PHASE 12 — verification and the two PDFs. `test-report` runs both suites and
# records the counts; `reproduce` runs this whole chain one target at a time in
# a clean checkout and times each; `report` and `paper` render the markdown
# through weasyprint inside the container, where the pango stack lives.
#
# `reproduce` is deliberately not part of `all`: `all` is what it measures.
test-report:
	@python3 scripts/run_tests.py

reproduce:
	@python3 scripts/reproduce.py --profile $(PROFILE)

report:
	$(call RUN_PY,app.research.figures_phase12 --seed $(SEED))
	$(call RUN_PY_SCRIPT,scripts/build_pdf.py docs/reports/ALP-System-Report.md \
	    --out docs/reports/ALP-System-Report.pdf)

paper:
	$(call RUN_PY_SCRIPT,scripts/build_pdf.py docs/paper-draft.md \
	    --out docs/reports/ALP-Research-Paper-Draft.pdf)

# ------------------------------------------------------------- run profiles
# The progressive ladder BUILD.md §0.3 requires: prove the pipeline at the
# smallest useful size, read the resource cost, and only then pay for the
# full grid. `full` is never reached by typing `make`.

# `smoke` validates and writes nothing: it must never be able to overwrite a
# published result with a 40-learner one. `dev` does write, because reading a
# real effect is the point of it — and it therefore *replaces* the evaluation
# artifacts and figures. `make full` puts them back in about a minute, because
# every cell of the last full run is still checkpointed.
smoke:
	$(call RUN_PY,app.research.evaluate_policies --profile smoke --quick --seed $(SEED))

dev:
	@echo "note: dev overwrites artifacts/evaluation/* and artifacts/figures/08-policies/*."
	@echo "      'make full' restores them from the cell checkpoints."
	$(MAKE) PROFILE=dev eval-policies

full:
	@test -f docs/full-run-justification.md || { \
	    echo "refusing: write docs/full-run-justification.md first (BUILD.md §0.3)."; exit 1; }
	@echo "full profile: the pre-registered grid. Justification:"
	@sed -n '1,12p' docs/full-run-justification.md
	$(MAKE) PROFILE=full eval-policies

# Progress of the current or last expensive run, read from the artifacts the
# run itself writes. Safe to call while a run is in flight.
status:
	@python3 scripts/run_status.py

all: items telemetry fetch-data prep-data simulate features train-baselines train-states \
     eval-policies integrate ope rl bank figures stats report paper
