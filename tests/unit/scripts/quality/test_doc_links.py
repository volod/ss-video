from pathlib import Path

from selfsuvis.scripts.quality.doc_links import (
    anchors,
    broken_links,
    documentation_files,
    heading_anchor,
    main,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]


def _write(root: Path, name: str, text: str) -> Path:
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def test_repository_documentation_links_resolve() -> None:
    assert broken_links(PROJECT_ROOT) == []


def test_heading_slugs_drop_punctuation_and_number_duplicates(tmp_path: Path) -> None:
    document = _write(tmp_path, "docs/topic.md", "# API + CLI\n\n## Result\n\n## Result\n")

    assert heading_anchor("API + CLI") == "api--cli"
    assert anchors(document) == {"api--cli", "result", "result-1"}


def test_heading_slugs_follow_git_host_rules(tmp_path: Path) -> None:
    assert heading_anchor("Optional Step 7 — Run coop Steps 37-43") == (
        "optional-step-7--run-coop-steps-37-43"
    )
    assert heading_anchor("Use `make lint` [here](x.md)") == "use-make-lint-here"
    document = _write(tmp_path, "docs/a.md", '<a name="custom"></a>\n\n```\n# not a heading\n```\n')
    assert anchors(document) == {"custom"}


def test_broken_link_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "# Root\n\n[missing](docs/gone.md)\n")

    assert broken_links(tmp_path) == ["README.md:3: missing file -> docs/gone.md"]


def test_broken_anchor_is_reported(tmp_path: Path) -> None:
    _write(tmp_path, "docs/index.md", "# Index\n\n[stale](topic.md#old)\n[self](#nowhere)\n")
    _write(tmp_path, "docs/topic.md", "# Current heading\n")

    assert broken_links(tmp_path) == [
        "docs/index.md:3: missing anchor -> topic.md#old",
        "docs/index.md:4: missing anchor -> #nowhere",
    ]


def test_valid_file_anchor_and_directory_links_pass(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "docs/index.md",
        "# Index\n\n[topic](topic.md#current-heading)\n[self](#index)\n[code](../src/)\n"
        '[titled](topic.md "Topic")\n',
    )
    _write(tmp_path, "docs/topic.md", "# Current heading\n")
    _write(tmp_path, "src/module.py", "")

    assert broken_links(tmp_path) == []


def test_external_fenced_and_inline_code_links_are_ignored(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "README.md",
        "# Root\n\n[web](https://example.com)\n`[x](gone.md)`\n\n```text\n[x](gone.md)\n```\n",
    )

    assert broken_links(tmp_path) == []


def test_documentation_sources_include_adapters_and_cursor_rules(tmp_path: Path) -> None:
    codex = _write(tmp_path, ".codex", "# Codex\n")
    gemini = _write(tmp_path, "GEMINI.md", "# Gemini\n")
    cursor_rule = _write(tmp_path, ".cursor/rules/project.mdc", "# Rule\n")
    outside = _write(tmp_path, "src/README.md", "# Not checked\n")

    files = documentation_files(tmp_path)

    assert files == sorted([codex, gemini, cursor_rule])
    assert outside not in files


def test_main_returns_failure_for_a_broken_link(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "# Root\n\n[x](missing.md)\n")

    assert main(["--root", str(tmp_path)]) == 1


def test_main_returns_success_without_broken_links(tmp_path: Path) -> None:
    _write(tmp_path, "README.md", "# Root\n")

    assert main(["--root", str(tmp_path)]) == 0
