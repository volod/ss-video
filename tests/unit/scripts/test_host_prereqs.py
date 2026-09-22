from pathlib import Path

import pytest

from selfsuvis.scripts.host_prereqs import (
    effective_hf_token,
    layered_hf_token,
    main,
    save_hf_token,
    token_status,
    validate_hf_token,
)

_TOKEN = "hf_unittesttoken"


def test_validate_hf_token_rejects_short_values():
    with pytest.raises(ValueError, match="HF_TOKEN format"):
        validate_hf_token("hf_short")


def test_layered_later_file_wins_including_empty(tmp_path: Path):
    (tmp_path / ".env").write_text(f"API_KEY=abc\nHF_TOKEN={_TOKEN}\n", encoding="utf-8")
    data = tmp_path / ".data"
    data.mkdir()
    (data / ".env").write_text("HF_TOKEN=\n", encoding="utf-8")

    value, source = layered_hf_token(tmp_path)

    assert value == ""
    assert source == data / ".env"


def test_environment_token_wins_over_files(tmp_path: Path):
    (tmp_path / ".env").write_text("HF_TOKEN=hf_fromfiletoken\n", encoding="utf-8")

    assert effective_hf_token(tmp_path, {"HF_TOKEN": _TOKEN}) == _TOKEN
    assert token_status(tmp_path, {"HF_TOKEN": _TOKEN}) == "present"


def test_blank_environment_falls_through_to_file(tmp_path: Path):
    (tmp_path / ".env").write_text(f'export HF_TOKEN="{_TOKEN}"\n', encoding="utf-8")

    assert effective_hf_token(tmp_path, {"HF_TOKEN": "  "}) == _TOKEN
    assert token_status(tmp_path, {}) == "present"


def test_inline_comment_is_stripped(tmp_path: Path):
    (tmp_path / ".env").write_text(f"HF_TOKEN={_TOKEN} # gated\n", encoding="utf-8")

    value, _source = layered_hf_token(tmp_path)

    assert value == _TOKEN


def test_save_hf_token_updates_winning_file_and_keeps_neighbors(tmp_path: Path):
    env_file = tmp_path / ".env"
    env_file.write_text("API_KEY=abc\nHF_TOKEN=hf_oldtokenvalue\n", encoding="utf-8")
    data = tmp_path / ".data"
    data.mkdir()
    (data / ".env").write_text("HF_TOKEN=\nQDRANT_HOST=localhost\n", encoding="utf-8")

    written = save_hf_token(tmp_path, f"  {_TOKEN}  ")

    assert written == data / ".env"
    text = written.read_text(encoding="utf-8")
    assert text == f"HF_TOKEN={_TOKEN}\nQDRANT_HOST=localhost\n"
    assert env_file.read_text(encoding="utf-8").startswith("API_KEY=abc\n")


def test_save_hf_token_creates_root_env(tmp_path: Path):
    written = save_hf_token(tmp_path, _TOKEN)

    assert written == tmp_path / ".env"
    assert f"HF_TOKEN={_TOKEN}\n" in written.read_text(encoding="utf-8")


def test_main_status_does_not_print_the_token(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("HF_TOKEN", raising=False)
    (tmp_path / ".env").write_text(f"HF_TOKEN={_TOKEN}\n", encoding="utf-8")

    code = main(["token-status", "--root", str(tmp_path)])

    captured = capsys.readouterr()
    assert code == 0
    assert captured.out.strip() == "present"
    assert _TOKEN not in captured.out
    assert _TOKEN not in captured.err
