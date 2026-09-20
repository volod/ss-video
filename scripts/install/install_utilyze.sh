#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Install Utilyze, an optional NVIDIA GPU efficiency profiler used for operator-side
inspection of live selfsuvis workloads.

This is intentionally not part of the Python dependency graph:
- upstream installation is shell-based
- profiling requires Linux amd64 + NVIDIA Ampere or newer
- root / CAP_SYS_ADMIN is typically required

Usage:
  ./scripts/install_utilyze.sh
EOF
  exit 0
fi

if [[ "$(uname -s)" != "Linux" ]]; then
  echo "Utilyze install is only supported on Linux hosts." >&2
  exit 1
fi

arch="$(uname -m)"
if [[ "$arch" != "x86_64" && "$arch" != "amd64" ]]; then
  echo "Utilyze currently targets Linux amd64; detected architecture: $arch" >&2
  exit 1
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. Install the NVIDIA driver before installing Utilyze." >&2
  exit 1
fi

cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')"
cc_major="${cc%%.*}"
if [[ -z "$cc_major" || "$cc_major" -lt 8 ]]; then
  echo "Utilyze requires NVIDIA Ampere or newer (compute capability >= 8.0)." >&2
  echo "Detected compute capability: ${cc:-unknown}" >&2
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required to install Utilyze." >&2
  exit 1
fi

echo "Installing Utilyze via the upstream installer..."
echo "Host checks passed: Linux amd64, NVIDIA compute capability ${cc}."
echo "The installer may prompt for sudo because GPU profiling counters are typically admin-gated."
curl -sSfL https://systalyze.com/utilyze/install.sh | sh
