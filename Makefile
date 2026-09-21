.PHONY: help up down logs data-dirs fix-data env env-interactive venv venv-cuda venv-pip venv-rebuild-xformers docker-check test test-no-gpu test-ci test-unit test-heavy test-unit-no-cv2 test-dir lint lint-spec-plan lint-doc-links lint-imports plan-status ci ci-github cvat-up cvat-down cvat-logs cvat-admin mapper-logs utlz-install utlz utlz-endpoints export-openapi frigate-up

# Base data directory — overridden by DATA_DIR in .env or shell environment.
DATA_DIR ?= .data
export DATA_DIR
# Host wheel cache for Docker additional context `wheels` (AGENTS.md: $DATA_DIR/wheels).
export WHEELS_DIR := $(abspath $(DATA_DIR))/wheels

# Project interpreter: the venv when present, otherwise python3 on PATH.
PY := $(if $(wildcard .venv/bin/python),.venv/bin/python,python3)
RUFF := $(if $(wildcard .venv/bin/ruff),.venv/bin/ruff,ruff)
# Quality checks re-export ss_kit.quality; the interpreter must have ss-common installed.
QUALITY := PYTHONPATH=src $(PY) -m selfsuvis.scripts.quality
# ss-fusion packages (ss-perception, ss-mapping, ss-fusion, fusion-rt) are installed
# from the published git tag; lint-imports and OpenAPI export use src plus the venv.
# Docker vision images install torch from the same CUDA extra as the host venv
# (cu128, cu126, cpu) so cached flash-attn wheels can match.
TORCH_CUDA_INDEX ?= $(shell $(PY) -c "import torch; c=torch.version.cuda or ''; print('cu'+c.replace('.','')) if c else 'cpu'" 2>/dev/null || echo cpu)
export TORCH_CUDA_INDEX

# Default target: show help when no target is given
help:
	@echo "=============================================="
	@echo "  Video Semantic Search - Make targets"
	@echo "=============================================="
	@echo ""
	@echo "  Stack (Docker)"
	@echo "  ---------------"
	@echo "  make up              Start main stack + mapper ICP service (docker-compose.override.yml auto-loaded)"
	@echo "  make frigate-up              Start Frigate from docker/core (profile frigate)"
	@echo "  make cvat-up         Start CVAT annotation service (http://localhost:8091)"
	@echo "  make cvat-down       Stop CVAT services"
	@echo "  make cvat-admin      Create CVAT superuser (first-time setup)"
	@echo "  make mapper-logs     Stream mapper (ICP fusion) container logs"
	@echo "  make down            Stop all containers"
	@echo "  make logs            Stream container logs (last 100 lines)"
	@echo "  make docker-check    Check that Docker daemon is reachable (run if you get permission denied)"
	@echo ""
	@echo "  Local dev (venv)"
	@echo "  -----------------"
	@echo "  make env             Generate .data/.env (auto-detects GPU/RAM, picks models)"
	@echo "  make env-interactive Generate .data/.env with interactive prompts (profile, sidecars, models)"
	@echo "  make venv                    Create .venv and install deps; if .venv exists, prompts to recreate or update"
	@echo "  make venv-cuda               Same as venv but forces CUDA wheel install (use if nvidia-smi is absent but GPU present)"
	@echo "  make venv-pip                Install pip into an existing .venv (e.g. after uv venv .venv)"
	@echo "  make venv-rebuild-xformers   Rebuild xformers from source for common GPU arches (RTX 2000/3000/4000, H100)"
	@echo "  make utlz-install            Install optional Utilyze GPU profiler (Linux amd64, NVIDIA Ampere+)"
	@echo "  make utlz                    Run Utilyze with selfsuvis-safe defaults (disables upstream metrics by default)"
	@echo "  make utlz-endpoints          Show Utilyze-discovered inference endpoints per GPU"
	@echo ""
	@echo "  Tests"
	@echo "  -----"
	@echo "  make test            Full integration tests in Docker (API + worker + Qdrant; needs GPU or test-no-gpu)"
	@echo "  make test-no-gpu     Same as test but without GPU (use if NVIDIA Container Toolkit is not installed)"
	@echo "  make test-dir        Same as test; set INDEX_DIR_PATH for directory-indexing tests"
	@echo "  make test-ci         Light unit tests (GitHub CI; no torch/cv2/ffmpeg)"
	@echo "  make test-unit       Full unit tests on host (needs the local venv with vision deps)"
	@echo "  make test-heavy      Heavy/slow/benchmark/gpu unit tests only"
	@echo "  make test-unit-no-cv2  Unit tests skipping cv2-dependent tests (if numpy/opencv version mismatch)"
	@echo ""
	@echo "  Code quality"
	@echo "  --------------"
	@echo "  make lint            Run ruff check, ruff format --check, import-linter, lint-spec-plan, and lint-doc-links"
	@echo "  make lint-imports    Run import-linter contracts"
	@echo "  make lint-spec-plan  Check that the spec capability registry and docs/impl/plan.md agree"
	@echo "  make lint-doc-links  Check that relative Markdown links and anchors resolve"
	@echo "  make plan-status     Count plan tasks by lane/status and show the next eligible task per lane"
	@echo "  make ci              lint plus test-unit (local agent gate)"
	@echo "  make ci-github       lint plus test-ci (GitHub light job)"
	@echo ""
	@echo "  Sibling repositories"
	@echo "  --------------------"
	@echo "  ss-fusion: clone volod/ss-fusion tag v0.2.0"
	@echo "  ss-sens / ss-control / ss-mlab: clone the volod/<name> tag v0.1.0 repo"
	@echo ""
	@echo "  Troubleshooting"
	@echo "  ----------------"
	@echo "  Docker permission denied:  sudo usermod -aG docker \$$USER  then log out and back in (or newgrp docker)"
	@echo "  GPU driver error:           sudo ./scripts/install/install_nvidia_docker.sh  or  make test-no-gpu"
	@echo "  Unable to open database:   sudo chown -R \$$(id -u):\$$(id -g) .data"
	@echo "  Root-owned data:           make fix-data"
	@echo ""
	@echo "  Run  make <target>  or  make help  to show this again."

# Ensure data dirs exist and are owned by current user (avoids root-owned files from containers)
# Pre-create Qdrant Snapshots dir to avoid "Permission denied" when running as non-root
data-dirs:
	@mkdir -p "$(DATA_DIR)/postgres" "$(DATA_DIR)/qdrant/Snapshots" "$(DATA_DIR)/videos" "$(DATA_DIR)/.cache" "$(DATA_DIR)/wheels" && chown -R $$(id -u):$$(id -g) "$(DATA_DIR)" 2>/dev/null && echo "Data directories ready ($(DATA_DIR))." || echo "Created $(DATA_DIR). If Qdrant fails with Permission denied, run: make fix-data"

up: docker-check data-dirs
	UID=$$(id -u) GID=$$(id -g) docker compose -f docker/core/docker-compose.yml up --build

down: docker-check
	UID=$$(id -u) GID=$$(id -g) docker compose -f docker/core/docker-compose.yml down

logs: docker-check
	UID=$$(id -u) GID=$$(id -g) docker compose -f docker/core/docker-compose.yml logs -f --tail=100

env:
	$(if $(wildcard .venv/bin/python),.venv/bin/python -m selfsuvis.scripts.generate_env --env dev,python -m selfsuvis.scripts.generate_env --env dev)

env-interactive:
	$(if $(wildcard .venv/bin/python),.venv/bin/python -m selfsuvis.scripts.generate_env --interactive,python -m selfsuvis.scripts.generate_env --interactive)

venv:
	@if [ -d .venv ]; then \
		printf "\n  .venv already exists.\n"; \
		printf "  [r] Recreate — remove and create a fresh .venv\n"; \
		printf "  [u] Update   — install/upgrade requirements in the existing .venv\n"; \
		printf "\n  Choice [r/u]: "; \
		read choice; \
		case "$$choice" in \
			r|R) \
				echo "Removing existing .venv..."; \
				rm -rf .venv; \
				uv venv .venv; \
				./scripts/install/install_requirements.sh vision,dev .venv \
				;; \
			u|U) \
				echo "Updating requirements in existing .venv..."; \
				./scripts/install/install_requirements.sh vision,dev .venv \
				;; \
			*) \
				echo "Invalid choice '$$choice'. Run  make venv  again and enter r or u."; \
				exit 1 \
				;; \
		esac \
	else \
		uv venv .venv; \
		./scripts/install/install_requirements.sh vision,dev .venv; \
	fi

# Force CUDA torch wheels regardless of nvidia-smi detection (use when GPU is present but nvidia-smi absent)
venv-cuda:
	uv venv .venv
	FORCE_CUDA=1 ./scripts/install/install_requirements.sh vision,dev .venv

# Rebuild xformers from source targeting the GPU present on this machine.
# Auto-detects compute capability via nvidia-smi; falls back to a safe multi-arch
# list (up to sm_90) when no GPU is found, avoiding compute_120 failures on older nvcc.
# Run when python -m xformers.info shows your GPU arch as unavailable.
# Expected build time: 20-60 min.
venv-rebuild-xformers:
	@echo "Rebuilding xformers from source (20-60 min)..."
	@uv pip install --python .venv pip
	@_CC=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' '); \
	_ARCH="$${_CC:+$${_CC}+PTX}"; \
	_ARCH="$${_ARCH:-7.5;8.0;8.6;8.9;9.0+PTX}"; \
	echo "  TORCH_CUDA_ARCH_LIST=$${_ARCH}"; \
	TORCH_CUDA_ARCH_LIST="$${_ARCH}" \
	MAX_JOBS=$$(.venv/bin/python -m ss_kit max-jobs) \
	.venv/bin/python -m pip install xformers \
	  --no-build-isolation --no-deps --no-binary xformers --force-reinstall --no-cache-dir
	@echo "Done. Verify:  .venv/bin/python -m xformers.info"

# Install pip into existing .venv (when uv created it without pip)
venv-pip:
	uv pip install --python .venv pip

# Fix ownership of data dir (run if Qdrant fails with "Permission denied" on Snapshots)
fix-data:
	@echo "Fixing ownership of $(DATA_DIR)/..."
	@sudo chown -R $$(id -u):$$(id -g) "$(DATA_DIR)" 2>/dev/null && echo "Done. Run make up again." || echo "Run: sudo chown -R $$(id -u):$$(id -g) $(DATA_DIR)"

# Verify Docker daemon is reachable (fixes permission-denied before running test/up)
docker-check:
	@if ! docker info >/dev/null 2>&1; then \
		echo ""; \
		echo "Docker is not accessible (permission denied or daemon not running)."; \
		echo ""; \
		echo "Safe fix: add your user to the docker group:"; \
		echo "  sudo usermod -aG docker $$USER"; \
		echo "Then log out and back in, or in this terminal run: newgrp docker"; \
		echo ""; \
		exit 1; \
	fi
	@echo "Docker access OK."

# Ensure test data dirs exist and are owned by current user (avoids "unable to open database file")
test-dirs:
	@mkdir -p "$(DATA_DIR)/postgres" "$(DATA_DIR)/postgres-test" "$(DATA_DIR)/qdrant/Snapshots" "$(DATA_DIR)/qdrant-test/Snapshots" "$(DATA_DIR)/videos" "$(DATA_DIR)/.cache" "$(DATA_DIR)/cache_test" "$(DATA_DIR)/redis" "$(DATA_DIR)/wheels" && chown -R $$(id -u):$$(id -g) "$(DATA_DIR)" 2>/dev/null && echo "Test directories ready ($(DATA_DIR))." || echo "Created $(DATA_DIR). If api/worker fail with 'unable to open database file', run: sudo chown -R $$(id -u):$$(id -g) $(DATA_DIR)"

# Wipe integration-only Postgres/Qdrant so leftover processed_files cannot skip
# indexing after a Qdrant reset (duplicate hash -> finished job, empty query).
test-reset-state:
	@rm -rf "$(DATA_DIR)/postgres-test" "$(DATA_DIR)/qdrant-test"
	@mkdir -p "$(DATA_DIR)/postgres-test" "$(DATA_DIR)/qdrant-test/Snapshots"
	@chown -R $$(id -u):$$(id -g) "$(DATA_DIR)/postgres-test" "$(DATA_DIR)/qdrant-test" 2>/dev/null || true

# Integration tests (require API + worker + Qdrant). Runs docker-check first. Uses GPU by default.
test: docker-check test-dirs test-reset-state
	UID=$$(id -u) GID=$$(id -g) INDEX_DIR_PATH=/app/tests/assets DATA_DIR=$(abspath $(DATA_DIR)) docker compose -f docker/core/docker-compose.yml -f docker/test/docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from tests
	UID=$$(id -u) GID=$$(id -g) INDEX_DIR_PATH=/app/tests/assets DATA_DIR=$(abspath $(DATA_DIR)) docker compose -f docker/core/docker-compose.yml -f docker/test/docker-compose.test.yml down --remove-orphans

# Integration tests without GPU (use if NVIDIA Container Toolkit is not installed)
test-no-gpu: docker-check test-dirs test-reset-state
	UID=$$(id -u) GID=$$(id -g) INDEX_DIR_PATH=/app/tests/assets DATA_DIR=$(abspath $(DATA_DIR)) docker compose -f docker/core/docker-compose.yml -f docker/core/docker-compose.no-gpu.yml -f docker/test/docker-compose.test.yml up --build --abort-on-container-exit --exit-code-from tests
	UID=$$(id -u) GID=$$(id -g) INDEX_DIR_PATH=/app/tests/assets DATA_DIR=$(abspath $(DATA_DIR)) docker compose -f docker/core/docker-compose.yml -f docker/core/docker-compose.no-gpu.yml -f docker/test/docker-compose.test.yml down --remove-orphans

# Directory integration test (same as test; set INDEX_DIR_PATH for custom path)
test-dir:
	$(MAKE) test

export-openapi:
	PYTHONPATH=src $(PY) -c "import json; from selfsuvis.app.main import app; print(json.dumps(app.openapi(), indent=2))" > docs/api/video-openapi.json
	PYTHONPATH=src $(PY) -c "import json; from selfsuvis.fusion_rt.app import app; print(json.dumps(app.openapi(), indent=2))" > docs/api/fusion-rt-openapi.json
	@echo "OpenAPI specs written to docs/api/video-openapi.json and docs/api/fusion-rt-openapi.json"

# Light unit tests for GitHub CI: no torch, opencv, ffmpeg, or model loads.
test-ci:
	$(if $(wildcard .venv/bin/python),.venv/bin/python -m pytest tests/unit/ -q --ci-light,pytest tests/unit/ -q --ci-light)

# Full unit tests (no services). Needs the local venv with vision deps.
test-unit:
	$(if $(wildcard .venv/bin/python),.venv/bin/python -m pytest tests/unit/ -v,pytest tests/unit/ -v)

# Heavy/slow/benchmark/gpu unit tests (full local venv).
test-heavy:
	$(if $(wildcard .venv/bin/python),.venv/bin/python -m pytest tests/unit/ -v -m "heavy or slow or benchmark or gpu",pytest tests/unit/ -v -m "heavy or slow or benchmark or gpu")

# Unit tests excluding cv2-dependent tests (use when numpy 2.x breaks opencv)
test-unit-no-cv2:
	$(if $(wildcard .venv/bin/python),.venv/bin/python -m pytest tests/unit/ -v --ignore=tests/unit/pipeline/media/test_rtsp_captioner.py,pytest tests/unit/ -v --ignore=tests/unit/pipeline/media/test_rtsp_captioner.py)

frigate-up: docker-check
	UID=$$(id -u) GID=$$(id -g) DATA_DIR=$(abspath $(DATA_DIR)) docker compose -f docker/core/docker-compose.yml --profile frigate up -d frigate

cvat-up: docker-check
	docker compose -f docker/cvat/docker-compose.cvat.yml up -d
	@echo ""
	@echo "CVAT starting at http://localhost:8090"
	@echo "First time? Run: make cvat-admin"

cvat-down: docker-check
	docker compose -f docker/cvat/docker-compose.cvat.yml down

cvat-logs: docker-check
	docker compose -f docker/cvat/docker-compose.cvat.yml logs -f --tail=100

cvat-admin: docker-check
	docker compose -f docker/cvat/docker-compose.cvat.yml exec cvat_server python manage.py createsuperuser

mapper-logs: docker-check
	UID=$$(id -u) GID=$$(id -g) docker compose -f docker/core/docker-compose.yml -f docker/core/docker-compose.override.yml logs -f --tail=100 mapper

# Lint (requires: pip install ruff)
lint:
	$(RUFF) check .
	$(RUFF) format --check .
	@$(MAKE) --no-print-directory lint-imports
	@$(MAKE) --no-print-directory lint-spec-plan
	@$(MAKE) --no-print-directory lint-doc-links

lint-imports:
	PYTHONPATH=src $(if $(wildcard .venv/bin/lint-imports),.venv/bin/lint-imports,lint-imports) --cache-dir .data/.import-linter

lint-spec-plan:
	@$(QUALITY).plan_integrity --root "$(CURDIR)"

lint-doc-links:
	@$(QUALITY).doc_links --root "$(CURDIR)"

plan-status:
	@$(QUALITY).plan_summary --root "$(CURDIR)"

ci: lint test-unit

ci-github: lint test-ci

utlz-install:
	./scripts/install/install_utilyze.sh

utlz:
	./scripts/ssv/ssv-utilyze.sh

utlz-endpoints:
	./scripts/ssv/ssv-utilyze.sh --endpoints

