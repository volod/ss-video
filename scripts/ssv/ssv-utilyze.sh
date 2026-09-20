#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Run Utilyze with defaults suited to selfsuvis local profiling.

Defaults:
- disables Utilyze upstream workload metrics API unless explicitly overridden
- writes logs to DATA_DIR/reports/utilyze.log
- forwards any additional utlz flags verbatim

Examples:
  ./scripts/ssv/ssv-utilyze.sh
  ./scripts/ssv/ssv-utilyze.sh --devices 0
  ./scripts/ssv/ssv-utilyze.sh --endpoints
  UTLZ_DISABLE_METRICS=0 ./scripts/ssv/ssv-utilyze.sh
EOF
  exit 0
fi

if ! command -v utlz >/dev/null 2>&1; then
  echo "utlz is not installed. Run ./scripts/install/install_utilyze.sh first." >&2
  exit 1
fi

# shellcheck source=scripts/shared/common.sh
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/../shared/common.sh"

log_dir="$(project_data_dir)/reports"
mkdir -p "${log_dir}"

export UTLZ_DISABLE_METRICS="${UTLZ_DISABLE_METRICS:-1}"
export UTLZ_LOG="${UTLZ_LOG:-${log_dir}/utilyze.log}"
export UTLZ_LOG_LEVEL="${UTLZ_LOG_LEVEL:-INFO}"

echo "Starting Utilyze..."
echo "  log: ${UTLZ_LOG}"
echo "  upstream metrics: ${UTLZ_DISABLE_METRICS}"
echo "  note: profiling may require sudo or CAP_SYS_ADMIN on this host"

exec utlz "$@"
