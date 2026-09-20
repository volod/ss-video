#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  cat <<'EOF'
Usage: ./scripts/ssv/ssv-reset-qdrant.sh

Deletes the configured Qdrant collection after interactive confirmation.
Uses:
  QDRANT_URL   default http://localhost:6333
  COLLECTION   default video_semantic
EOF
  exit 0
fi

QDRANT_URL=${QDRANT_URL:-http://localhost:6333}
COLLECTION=${COLLECTION:-video_semantic}

read -p "Delete Qdrant collection '${COLLECTION}' at ${QDRANT_URL}? [y/N] " -r
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
  echo "Canceled"
  exit 0
fi

curl -s -X DELETE "${QDRANT_URL}/collections/${COLLECTION}" | python -m json.tool
