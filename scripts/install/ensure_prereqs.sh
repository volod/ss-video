#!/usr/bin/env bash
# Prompt for ffmpeg and HF_TOKEN before a local run.
# Usage:
#   ensure_prereqs.sh ffmpeg [--require]
#   ensure_prereqs.sh hf-token [--require]
#   ensure_prereqs.sh start -- COMMAND...
set -euo pipefail

SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="$SCRIPT_ROOT"
if [[ -n "${SELFSUVIS_ROOT:-}" ]]; then
  ROOT="$SELFSUVIS_ROOT"
fi

prereq_python() {
  if [[ -x "$SCRIPT_ROOT/.venv/bin/python" ]]; then
    printf '%s\n' "$SCRIPT_ROOT/.venv/bin/python"
  else
    printf '%s\n' python3
  fi
}

token_status() {
  local py
  py="$(prereq_python)"
  PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$py" -m selfsuvis.scripts.host_prereqs token-status --root "$ROOT"
}

save_token() {
  local py token
  py="$(prereq_python)"
  token="$1"
  HF_TOKEN="$token" PYTHONPATH="$SCRIPT_ROOT/src${PYTHONPATH:+:$PYTHONPATH}" \
    "$py" -m selfsuvis.scripts.host_prereqs save-token --root "$ROOT"
}

check_ffmpeg() {
  local require="$1"
  if command -v ffmpeg >/dev/null 2>&1; then
    return 0
  fi
  if [[ ! -t 0 ]]; then
    printf 'ffmpeg is not installed.\n' >&2
    printf 'Install it with: sudo ./scripts/install/install_system_deps.sh\n' >&2
    if [[ "$require" -eq 1 ]]; then
      exit 1
    fi
    return 0
  fi
  printf '\n'
  printf 'ffmpeg is not installed. Video decoding needs it.\n'
  printf '  [y] Install now (sudo ./scripts/install/install_system_deps.sh)\n'
  printf '  [n] Skip\n'
  local choice
  while true; do
    printf 'Choice [y/n]: '
    choice=""
    read -r choice || true
    case "$choice" in
      y|Y)
        sudo "$SCRIPT_ROOT/scripts/install/install_system_deps.sh"
        hash -r
        if ! command -v ffmpeg >/dev/null 2>&1; then
          printf 'ffmpeg is still not on PATH after install.\n' >&2
          exit 1
        fi
        printf 'ffmpeg is installed.\n'
        return 0
        ;;
      n|N)
        if [[ "$require" -eq 1 ]]; then
          printf 'Install ffmpeg, then run the command again.\n' >&2
          printf '  sudo ./scripts/install/install_system_deps.sh\n' >&2
          exit 1
        fi
        printf 'Skipped ffmpeg install.\n'
        printf 'Install later with: sudo ./scripts/install/install_system_deps.sh\n'
        return 0
        ;;
      *)
        printf 'Enter y or n.\n'
        ;;
    esac
  done
}

check_hf_token() {
  local require="$1"
  local status
  status="$(token_status)"
  if [[ "$status" == "present" ]]; then
    return 0
  fi
  if [[ ! -t 0 ]]; then
    printf 'HF_TOKEN is not set.\n' >&2
    printf 'Add HF_TOKEN to .env, then run the command again.\n' >&2
    if [[ "$require" -eq 1 ]]; then
      exit 1
    fi
    return 0
  fi
  printf '\n'
  printf 'HF_TOKEN is not set.\n'
  printf 'Gated Hugging Face models need it in .env before a run.\n'
  printf '  [s] Save a token into .env\n'
  printf '  [n] Skip\n'
  local choice token
  while true; do
    printf 'Choice [s/n]: '
    choice=""
    read -r choice || true
    case "$choice" in
      s|S)
        while true; do
          printf 'Paste the Hugging Face token (input is hidden): '
          token=""
          read -r -s token || true
          printf '\n'
          if [[ -z "$token" ]]; then
            if [[ "$require" -eq 1 ]]; then
              printf 'Set HF_TOKEN in .env, then run the command again.\n' >&2
              exit 1
            fi
            printf 'HF_TOKEN was left unset.\n'
            return 0
          fi
          if save_token "$token"; then
            token=""
            return 0
          fi
          token=""
        done
        ;;
      n|N)
        if [[ "$require" -eq 1 ]]; then
          printf 'Set HF_TOKEN in .env, then run the command again.\n' >&2
          exit 1
        fi
        printf 'HF_TOKEN was left unset. A run will ask again.\n'
        return 0
        ;;
      *)
        printf 'Enter s or n.\n'
        ;;
    esac
  done
}

require_venv() {
  if [[ ! -x "$SCRIPT_ROOT/.venv/bin/python" ]]; then
    printf 'Project venv is missing. Run: make venv\n' >&2
    exit 1
  fi
}

usage() {
  printf 'Usage: %s {ffmpeg|hf-token|start} [--require] [-- COMMAND...]\n' "$0" >&2
  exit 2
}

main() {
  local cmd="${1:-}"
  local require=0
  shift || true
  if [[ "${1:-}" == "--require" ]]; then
    require=1
    shift
  fi
  case "$cmd" in
    ffmpeg)
      check_ffmpeg "$require"
      ;;
    hf-token)
      check_hf_token "$require"
      ;;
    start)
      require_venv
      check_ffmpeg 1
      check_hf_token 1
      if [[ "${1:-}" == "--" ]]; then
        shift
      fi
      if [[ -z "${HF_TOKEN:-}" ]]; then
        unset HF_TOKEN
      fi
      if [[ $# -eq 0 ]]; then
        exit 0
      fi
      cd "$SCRIPT_ROOT"
      exec "$@"
      ;;
    *)
      usage
      ;;
  esac
}

main "$@"
