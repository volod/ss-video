# Documentation

This directory separates product intent, forward work, current behavior, and contributor guidance.

| Question | Source of truth |
| --- | --- |
| What should the product do? | [Specification](design/spec.md) |
| What work remains? | [Implementation plan](impl/plan.md) |
| What exists and where? | [Current implementation](impl/current.md) |
| How should work be performed? | [Guide](guide/README.md) and [AGENTS.md](../AGENTS.md) |
| How is a component configured or called? | [Reference](reference/README.md) |
| How is it operated? | [Operations](operations/README.md) and [runbooks](runbooks/README.md) |

`make lint-spec-plan` and `make lint-doc-links` run in `make lint`.
