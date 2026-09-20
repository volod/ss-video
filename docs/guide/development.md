# Development

Contributing guide, code conventions, project workflow, and tooling references.

## Workflow

Issues are tracked in Linear. Commit format:

```
SS-NN | short description in imperative mood
```

Example: `SS-42 | add GPS sidecar extraction for DJI videos`

Standard development cycle using Claude Code skills:

```
/office-hours → /plan-ceo-review → /plan-eng-review → [build] → /review → /qa → /ship
```

Count lines of code across tracked files:

```bash
cloc $(git ls-files)
```

## Code style

- **Python**: `ruff` for linting and formatting. Run `make lint` before committing.
- **Line length**: 100 characters.
- **Type hints**: required on all public functions and class methods.
- **Docstrings**: Google style. Required on public API surfaces; skip on private helpers.
- **Imports**: `isort`-compatible grouping (stdlib → third-party → local). `ruff` enforces this.

```bash
make lint          # ruff check + ruff format --check + lint-spec-plan + lint-doc-links
make plan-status   # next eligible plan task per lane
```

Auto-fix:

```bash
.venv/bin/ruff check --fix .
.venv/bin/ruff format .
```

## Project conventions

- **Never commit to `main` directly.** Open a PR; linear issue number in branch name.
- **Never commit secrets.** `.env` is git-ignored. Generate it with `make env` or `python -m selfsuvis.scripts.generate_env`.
- **Data directories are git-ignored.** `.data/` is excluded.
  Add large assets via DVC or document manual download steps in scripts.
- **Integration tests use `.data/` and `.data/cache_test/`.** Videos go in `.data/videos/`.
- **One migration per PR** that touches the DB schema. Migration script goes in `scripts/`.

## PM tooling

Install project management CLI tools (pandoc, glow, mermaid, markdownlint, Linear CLI):

```bash
bash scripts/install_pm_tools.sh
```

## Useful Claude Code skills

| Skill | When to use |
|---|---|
| `/plan-ceo-review` | High-level product and priority review of a plan |
| `/plan-eng-review` | Engineering feasibility and architecture review |
| `/review` | Code review of a PR or diff |
| `/qa` | QA pass — test coverage, edge cases, regressions |
| `/ship` | Final pre-merge checklist |
| `/browse` | Web research — always use this instead of raw browser tools |
| `/retro` | Post-iteration retrospective |

## External resources

| Resource | Link |
|---|---|
| Claude Code skills index | [antigravity-awesome-skills](https://github.com/sickn33/antigravity-awesome-skills) |
| Awesome Claude Code | [hesreallyhim/awesome-claude-code](https://github.com/hesreallyhim/awesome-claude-code) |
| Everything Claude Code | [affaan-m/everything-claude-code](https://github.com/affaan-m/everything-claude-code) |
| kasetto (skill runner) | [pivoshenko/kasetto](https://github.com/pivoshenko/kasetto) |
| gstack | [garrytan/gstack](https://github.com/garrytan/gstack) |
| Build your own X | [codecrafters-io/build-your-own-x](https://github.com/codecrafters-io/build-your-own-x) |
