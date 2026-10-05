#!/usr/bin/env bash
# Runs Label Studio locally for the MANUAL annotation step.
#
# Requires Docker. The `docker compose` plugin is not available on every
# machine, so this script uses `docker run` directly and is equivalent to
# annotation/docker-compose.yml.
#
# Usage:
#   ./scripts/label_studio.sh start    # start and wait until healthy
#   ./scripts/label_studio.sh stop     # stop the container
#   ./scripts/label_studio.sh status   # show the container state
#   ./scripts/label_studio.sh logs     # follow the server logs
#   ./scripts/label_studio.sh reset    # delete container AND local state

set -euo pipefail

IMAGE="heartexlabs/label-studio:1.23.2"
NAME="annotation-qc-label-studio"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_DIR="$ROOT/.labelstudio"

usage() {
    sed -n '/^# Usage:/,/^$/p' "$0" | sed 's/^# \{0,1\}//'
}

wait_for_server() {
    printf "waiting for Label Studio"
    for _ in $(seq 1 60); do
        if curl -sf -o /dev/null http://localhost:8080/health; then
            printf "\nLabel Studio is ready: http://localhost:8080\n"
            return 0
        fi
        printf "."
        sleep 2
    done
    printf "\nserver did not become healthy in time; check: %s logs\n" "$0"
    return 1
}

case "${1:-}" in
start)
    mkdir -p "$DATA_DIR"
    # The container runs as UID 1001, which is not the host user.
    chmod o+rwx "$DATA_DIR"
    if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
        docker start "$NAME" >/dev/null
    else
        docker run -d --name "$NAME" \
            -p 8080:8080 \
            -e LABEL_STUDIO_LOCAL_FILES_SERVING_ENABLED=true \
            -e LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT=/label-studio/data \
            -v "$DATA_DIR:/label-studio/data" \
            -v "$ROOT/data:/label-studio/data/data:ro" \
            "$IMAGE"
    fi
    wait_for_server
    ;;
stop)
    docker stop "$NAME"
    ;;
status)
    docker ps -a --filter "name=$NAME"
    ;;
logs)
    docker logs -f "$NAME"
    ;;
reset)
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run --rm -v "$DATA_DIR:/cleanup" busybox \
        sh -c 'rm -rf /cleanup/* /cleanup/.[!.]* 2>/dev/null || true'
    echo "Removed the container and all local Label Studio state."
    ;;
*)
    usage
    exit 1
    ;;
esac
