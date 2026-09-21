# AGENTS.md project rules

This file is the canonical instruction source for every coding agent in this repository.
Tool-specific files (`CLAUDE.md`, `GEMINI.md`, `.codex`) link here and keep only
integration-specific routing.

## Project guardrails

- Do not create git commits unless explicitly asked.
- Do not revert user changes or unrelated dirty files while fixing an issue.
- Do not add `from __future__ import annotations`; use normal annotations and `TYPE_CHECKING` imports when needed.
- Keep top-level `scripts/` as shell entrypoints. Put production Python implementations under `src/selfsuvis/...`.
- Reuse `scripts/shared/common.sh` for shared shell root/env/bootstrap behavior.
- Runtime data belongs under `$DATA_DIR` (default `.data/`). Never write to a module-local `.data/` inside `src/`.
- The shared `.data/wheels/` directory is reserved exclusively for compiled wheel artifacts.
- Never hardcode absolute directories. Resolve every path from the project base directory and honor `.env`/`DATA_DIR`.
- Use ASCII in logs, docs, comments, and generated shell output.

## Heavy compilation (ninja / cmake / CUDA)

`MAX_JOBS=$(.venv/bin/python -m ss_kit max-jobs)` (`ss-kit max-jobs`;
`src/selfsuvis/scripts/shell_helpers.py max-jobs` re-exports it). Do not inline the formula.

Compiled wheels MUST be cached under `.data/wheels/<package-name>_<key>/`.

## Current layout

- API: `src/selfsuvis/app/`
- Worker: `src/selfsuvis/worker/`
- UI: `src/selfsuvis/ui/`
- Video remainder of `pipeline/` (realtime, workflows, ICP mapper, video media/storage)
- Perception and mapping: [volod/ss-fusion](https://github.com/volod/ss-fusion) tag `v0.2.0`
- Contracts and `ss_kit`: [volod/ss-common](https://github.com/volod/ss-common) tag `v0.2.1`
- Runtime config: `selfsuvis.pipeline.core.config` (ss-perception package)
- Docker and shell ops: `docker/`, `scripts/`

## Usual commands

- `make venv` — installs selfsuvis plus ss-fusion packages from git tag `v0.2.0`
- `make test-unit`, `make test-ci`, `make lint`
- `make ci` — lint plus test-unit; `make ci-github` — lint plus test-ci
- `make plan-status` — next eligible task per plan lane
- `make up`, `make down`, `make logs`
- `python -m selfsuvis.scripts.migrate_postgres`

## Documentation lifecycle

| Question | Source of truth |
| --- | --- |
| What should the product do? | `docs/design/spec.md` |
| What work remains? | `docs/impl/plan.md` |
| What exists and where? | `docs/impl/current.md` |
| What happened to a finished task? | `docs/impl/records/` |
| How is work performed? | `docs/guide/` and this file |
| How do I study the stack? | `docs/learning_path/` |
