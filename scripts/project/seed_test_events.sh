#!/usr/bin/env bash
# Seed test zones and events for Phase 3A correlator development.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/../shared/common.sh"
python -m selfsuvis.scripts.seed_test_events "$@"
