# CLAUDE.md

## Project rules

Read [AGENTS.md](AGENTS.md)

## GStack

- Use GStack slash skills when the user asks for that workflow or the task clearly needs it.
- Use `/browse` from GStack for browser-backed web work; do not use `mcp__claude-in-chrome__*` tools.
- Useful routing: `/investigate` for bugs, `/review` for diff review, `/qa` or `/qa-only` for browser QA, `/plan-eng-review` for architecture, `/plan-ceo-review` for scope, `/design-review` for visual polish, `/ship` or `/land-and-deploy` for release flow, `/context-save` and `/context-restore` for handoff.
- Speckit slash commands live under `.claude/commands/`; use them only when the user invokes the Speckit flow.
