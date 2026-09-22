"""Tests for the local-pipeline model cache helper."""

from selfsuvis.scripts.cache_models import (
    model_invocations,
    pending_cache_commands,
    prepare_models_argv,
)


def test_pending_cache_commands_skips_models_that_are_already_cached() -> None:
    required = [
        "ssv-models --clip",
        "ssv-models --dino",
        "ssv-models --whisper --whisper-model facebook/seamless-m4t-v2-large",
    ]
    missing = ["ssv-models --dino"]

    skipped, needed = pending_cache_commands(required, missing)

    assert skipped == [
        "ssv-models --clip",
        "ssv-models --whisper --whisper-model facebook/seamless-m4t-v2-large",
    ]
    assert needed == ["ssv-models --dino"]


def test_model_invocations_keeps_non_ssv_commands_separate() -> None:
    flags, other = model_invocations(
        [
            "ssv-models --clip",
            "ssv-models --whisper --whisper-model facebook/seamless-m4t-v2-large",
            "ollama pull gemma3:4b",
        ]
    )

    assert flags == [
        "--clip",
        "--whisper",
        "--whisper-model",
        "facebook/seamless-m4t-v2-large",
    ]
    assert other == [["ollama", "pull", "gemma3:4b"]]


def test_prepare_models_argv_uses_the_ssv_models_script() -> None:
    argv = prepare_models_argv(["--clip", "--dino"], "/tmp/venv/bin/python")

    assert argv == ["/tmp/venv/bin/ssv-models", "--clip", "--dino"]
