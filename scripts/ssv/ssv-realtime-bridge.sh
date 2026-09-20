#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck disable=SC1091
source "$SCRIPT_DIR/../shared/common.sh"

usage() {
  cat <<'EOF'
Usage: ./scripts/ssv/ssv-realtime-bridge.sh [--backend mavsdk|ros]

Run the packaged realtime telemetry bridge runtime. The bridge writes normalized
sensor packets into the configured realtime session using DATABASE_URL.
EOF
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi

project_cd_root
project_run_python_module selfsuvis.realtime.bridge_runtime "$@"
