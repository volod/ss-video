"""Read and write ``HF_TOKEN`` for host setup.

Interactive prompts live in ``scripts/install/ensure_prereqs.sh``. This module
never prints the token value. A non-empty process environment wins. A blank
``HF_TOKEN`` in the environment is ignored so the layered ``.env`` files can
supply the token after the shell unsets the blank value.
"""

import argparse
import os
import re
import sys
from collections.abc import Mapping
from pathlib import Path

_ENV_LAYERS = (".env", ".data/.env", ".data/.env.local")
_LINE_RE = re.compile(r"^(?:export[ \t]+)?HF_TOKEN[ \t]*=[ \t]*(.*)$")
_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{8,}")
_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", '"': '"', "\\": "\\"}


def validate_hf_token(token: str) -> str:
    """Return ``token`` when it matches a Hugging Face token.

    Args:
        token: Raw pasted token.

    Returns:
        The stripped token.

    Raises:
        ValueError: The value is empty or does not match ``hf_`` plus 8
            characters.
    """
    cleaned = token.strip()
    if _TOKEN_RE.fullmatch(cleaned) is None:
        raise ValueError(
            "HF_TOKEN format was not accepted. "
            "Use a token from https://huggingface.co/settings/tokens."
        )
    return cleaned


def layered_hf_token(root: Path) -> tuple[str | None, Path | None]:
    """Return the winning file value and the file that set it.

    Later layers override earlier ones, including an empty assignment.
    ``None`` means no layer assigns ``HF_TOKEN``.

    Args:
        root: Repository root that contains ``.env`` and ``.data/``.

    Returns:
        ``(value, path)``. ``value`` is ``None`` when the key is absent.
    """
    value: str | None = None
    source: Path | None = None
    for relative in _ENV_LAYERS:
        path = root / relative
        if not path.is_file():
            continue
        parsed = _parse_file(path.read_text(encoding="utf-8"))
        if parsed is None:
            continue
        value = parsed
        source = path
    return value, source


def effective_hf_token(root: Path, environ: Mapping[str, str]) -> str:
    """Return the token a run will see after a blank environment value is unset.

    Args:
        root: Repository root.
        environ: Process environment. A non-empty ``HF_TOKEN`` wins over files.

    Returns:
        The stripped token, or ``""`` when none is configured.
    """
    raw = environ.get("HF_TOKEN", "")
    if raw.strip():
        return raw.strip()
    layered, _source = layered_hf_token(root)
    if layered is None:
        return ""
    return layered.strip()


def token_status(root: Path, environ: Mapping[str, str]) -> str:
    """Return ``present`` or ``missing`` without the token text."""
    if effective_hf_token(root, environ).strip():
        return "present"
    return "missing"


def save_hf_token(root: Path, token: str) -> Path:
    """Write ``token`` into the env file that wins the layered load.

    Args:
        root: Repository root.
        token: Token to store. Validated before any file is changed.

    Returns:
        The file that was written.

    Raises:
        ValueError: ``token`` failed ``validate_hf_token``.
    """
    cleaned = validate_hf_token(token)
    _value, source = layered_hf_token(root)
    path = source if source is not None else root / ".env"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        updated = _upsert_text(path.read_text(encoding="utf-8"), cleaned)
    else:
        updated = f"# Hugging Face token for gated model downloads.\nHF_TOKEN={cleaned}\n"
    path.write_text(updated, encoding="utf-8")
    return path


def display_path(root: Path, path: Path) -> str:
    """Return ``path`` relative to ``root`` when it lives under that root."""
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def main(argv: list[str] | None = None) -> int:
    """Print token presence or save a token passed in the environment."""
    parser = argparse.ArgumentParser(description="Check or save HF_TOKEN without printing it.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    status = subparsers.add_parser("token-status", help="Print present or missing")
    status.add_argument("--root", type=Path, default=Path.cwd())
    save = subparsers.add_parser("save-token", help="Write HF_TOKEN from the environment")
    save.add_argument("--root", type=Path, default=Path.cwd())
    args = parser.parse_args(argv)
    root = args.root
    if args.command == "token-status":
        print(token_status(root, os.environ))
        return 0
    token = os.environ.get("HF_TOKEN", "")
    try:
        path = save_hf_token(root, token)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"Saved HF_TOKEN to {display_path(root, path)}")
    return 0


def _parse_file(text: str) -> str | None:
    found = False
    value = ""
    for line in text.splitlines():
        parsed = _parse_line(line)
        if parsed is None:
            continue
        found = True
        value = parsed
    if not found:
        return None
    return value


def _parse_line(line: str) -> str | None:
    match = _LINE_RE.match(line.strip())
    if match is None:
        return None
    return _parse_value(match.group(1))


def _parse_value(raw: str) -> str:
    if raw.startswith("'"):
        end = raw.find("'", 1)
        if end < 0:
            return raw[1:]
        return raw[1:end]
    if raw.startswith('"'):
        return _double_quoted(raw)
    comment = re.search(r"\s#", raw)
    if comment:
        raw = raw[: comment.start()]
    return raw.strip()


def _double_quoted(raw: str) -> str:
    out: list[str] = []
    index = 1
    while index < len(raw):
        char = raw[index]
        if char == "\\" and index + 1 < len(raw):
            out.append(_ESCAPES.get(raw[index + 1], raw[index + 1]))
            index += 2
            continue
        if char == '"':
            return "".join(out)
        out.append(char)
        index += 1
    return "".join(out)


def _upsert_text(text: str, token: str) -> str:
    out: list[str] = []
    replaced = False
    for line in text.splitlines():
        if _parse_line(line) is None:
            out.append(line)
            continue
        if not replaced:
            out.append(f"HF_TOKEN={token}")
            replaced = True
    if not replaced:
        out.append(f"HF_TOKEN={token}")
    return "\n".join(out) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
