#!/bin/sh
set -e
# Fix storage ownership so Qdrant can run as non-root (UID:GID)
# snapshots_path is ./storage/snapshots (inside our mount) via config override
if [ -n "${UID}" ] && [ -n "${GID}" ]; then
  mkdir -p /qdrant/storage/snapshots
  chown -R "${UID}:${GID}" /qdrant/storage
  exec gosu "${UID}:${GID}" "$@"
else
  exec "$@"
fi
