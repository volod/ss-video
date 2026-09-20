#!/usr/bin/env bash
set -euo pipefail

containers=$(docker ps -aq)
if [[ -n "$containers" ]]; then
    docker stop $containers
    docker rm $containers
fi

docker rmi $(docker images -q) 2>/dev/null || true
docker volume rm $(docker volume ls -q) 2>/dev/null || true
docker network ls --format '{{.Name}}' \
    | grep -vE '^(bridge|host|none)$' \
    | xargs -r docker network rm 2>/dev/null || true
docker system prune -a --volumes --force
