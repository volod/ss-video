"""Every src module belongs to exactly one ownership group in current.md."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
OWNERSHIP_DOC = REPO_ROOT / "docs" / "impl" / "current.md"
SRC_PACKAGES = ("selfsuvis", "ssv_vdp")
GROUPS = frozenset({"ss-video", "ss-fusion", "ss-mapping", "ss-perception"})
TABLE_HEADING = "### Ownership prefixes"
SRC_ROOTS = (REPO_ROOT / "src",)


def _parse_prefix_table(text: str) -> list[tuple[str, str]]:
    lines = text.splitlines()
    try:
        start = next(i for i, line in enumerate(lines) if line.strip() == TABLE_HEADING)
    except StopIteration as exc:
        raise AssertionError(f"missing {TABLE_HEADING} in {OWNERSHIP_DOC}") from exc
    rows: list[tuple[str, str]] = []
    in_table = False
    for line in lines[start + 1 :]:
        stripped = line.strip()
        if stripped.startswith("| Prefix"):
            in_table = True
            continue
        if not in_table:
            continue
        if stripped.startswith("| ---"):
            continue
        if not stripped.startswith("|"):
            break
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if len(cells) != 2:
            raise AssertionError(f"ownership row must have two cells: {stripped}")
        prefix, group = cells
        if prefix.startswith("`") and prefix.endswith("`"):
            prefix = prefix[1:-1]
        if group not in GROUPS:
            raise AssertionError(f"unknown ownership group {group!r} for {prefix}")
        rows.append((prefix, group))
    if not rows:
        raise AssertionError("ownership prefix table is empty")
    return rows


def _module_from_path(path: Path, src_root: Path) -> str:
    rel = path.relative_to(src_root)
    parts = list(rel.with_suffix("").parts)
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


def _classify(module: str, prefixes: list[tuple[str, str]]) -> str:
    best_group = None
    best_len = -1
    for prefix, group in prefixes:
        if module == prefix or module.startswith(prefix + "."):
            if len(prefix) > best_len:
                best_group = group
                best_len = len(prefix)
    if best_group is None:
        raise AssertionError(f"no ownership prefix for {module}")
    return best_group


def _src_modules() -> list[str]:
    modules = []
    for src_root in SRC_ROOTS:
        if not src_root.is_dir():
            continue
        for package in SRC_PACKAGES:
            root = src_root / package
            if not root.is_dir():
                continue
            for path in sorted(root.rglob("*.py")):
                if "__pycache__" in path.parts:
                    continue
                modules.append(_module_from_path(path, src_root))
    return modules


def test_ownership_prefixes_cover_every_src_module() -> None:
    prefixes = _parse_prefix_table(OWNERSHIP_DOC.read_text(encoding="utf-8"))
    listed = [prefix for prefix, _group in prefixes]
    assert len(listed) == len(set(listed)), f"duplicate ownership prefixes: {listed}"
    counts = {group: 0 for group in GROUPS}
    for module in _src_modules():
        counts[_classify(module, prefixes)] += 1
    assert counts["ss-video"] > 0, counts


def test_ownership_longest_prefix_wins_for_mixed_packages() -> None:
    prefixes = _parse_prefix_table(OWNERSHIP_DOC.read_text(encoding="utf-8"))
    assert _classify("selfsuvis.pipeline.mapping.icp", prefixes) == "ss-video"
    assert _classify("selfsuvis.pipeline.mapping.sfm", prefixes) == "ss-mapping"
    assert _classify("selfsuvis.pipeline.media.frames", prefixes) == "ss-perception"
    assert _classify("selfsuvis.pipeline.media.download", prefixes) == "ss-video"
    assert _classify("selfsuvis.fusion_rt.correlator", prefixes) == "ss-fusion"
    assert _classify("ssv_vdp.cli", prefixes) == "ss-fusion"
    assert _classify("selfsuvis.app.main", prefixes) == "ss-video"
